from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from finquery_agent.config import LLMSettings
from finquery_agent.llm import LLMClient
from finquery_agent.nl2sql.answer import AnswerComposer
from finquery_agent.nl2sql.charting import ChartRenderer
from finquery_agent.nl2sql.intent import StructuredIntent
from finquery_agent.nl2sql.intent_engine import RuleBasedIntentEngine
from finquery_agent.nl2sql.llm_intent_engine import HybridIntentEngine
from finquery_agent.nl2sql.planner import split_intent_by_table
from finquery_agent.nl2sql.session import QuerySessionStore
from finquery_agent.nl2sql.sql_builder import SQLBuildError, SQLBuilder
from finquery_agent.rag.models import SearchResult, make_snippet
from finquery_agent.rag.service import RAGService
from finquery_agent.schema.registry import SchemaRegistry


@dataclass(frozen=True)
class AnalysisResult:
    status: str
    answer_text: str = ""
    answer_source: str = "deterministic"
    llm_used: bool = False
    intent: dict[str, Any] | None = None
    financial: dict[str, Any] | None = None
    rag: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    chart_images: list[dict[str, Any]] | None = None
    clarification: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    # Agent 模式的执行轨迹(每步工具调用的摘要);非 Agent 链路为 None。
    execution_trace: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "answer_text": self.answer_text,
            "answer_source": self.answer_source,
            "llm_used": self.llm_used,
            "financial": self.financial,
            "rag": self.rag,
            "sources": self.sources or [],
            "warnings": list(self.warnings),
        }
        if self.intent is not None:
            payload["intent"] = self.intent
        if self.clarification is not None:
            payload["clarification"] = self.clarification
        if self.chart_images:
            payload["chart_images"] = self.chart_images
            payload["chart_image"] = self.chart_images[0]
        if self.execution_trace is not None:
            payload["execution_trace"] = self.execution_trace
        financial = self.financial or {}
        if financial.get("queries"):
            payload["queries"] = financial["queries"]
        if financial.get("result"):
            payload["result"] = financial["result"]
        return payload


class AnalysisService:
    def __init__(
        self,
        registry: SchemaRegistry,
        intent_engine: RuleBasedIntentEngine | HybridIntentEngine,
        sql_builder: SQLBuilder,
        query_executor: Any,
        answer_composer: AnswerComposer,
        chart_renderer: ChartRenderer,
        rag_service: RAGService,
        session_store: QuerySessionStore | None = None,
        llm_settings: LLMSettings | None = None,
    ):
        self.registry = registry
        self.intent_engine = intent_engine
        self.sql_builder = sql_builder
        self.query_executor = query_executor
        self.answer_composer = answer_composer
        self.chart_renderer = chart_renderer
        self.rag_service = rag_service
        self.session_store = session_store or QuerySessionStore()
        self.llm_settings = llm_settings or LLMSettings()
        self._llm_client = LLMClient(self.llm_settings)

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        use_llm: bool = True,
        use_rag: bool = True,
        use_vector: bool = True,
        rag_top_k: int = 8,
    ) -> AnalysisResult:
        parsed = self.intent_engine.parse(question)
        use_financial_session = bool(session_id and (self.session_store.has_pending(session_id) or _looks_like_financial_question(question, parsed)))
        intent = self.session_store.resolve(session_id, question, self.intent_engine) if use_financial_session else parsed

        if _should_return_financial_clarification(intent):
            return AnalysisResult(
                status="clarification",
                intent=intent.to_dict(),
                clarification=intent.clarification.__dict__ if intent.clarification else None,
            )

        financial = self._run_financial_query(question, intent)
        rag_results = []
        if use_rag:
            rag_results = self._retrieve_sources(question, intent, rag_top_k, use_vector)
        sources = [result.to_dict() for result in rag_results]
        rag_payload = {"count": len(sources), "sources": sources}
        chart_images = financial.get("chart_images") if financial else []
        deterministic = self._deterministic_answer(question, financial, rag_results)

        if use_llm and self._llm_available():
            llm_answer = self._compose_with_llm(question, financial, rag_results, deterministic)
            if llm_answer:
                return AnalysisResult(
                    status="answer",
                    answer_text=llm_answer,
                    answer_source="analysis_llm",
                    llm_used=True,
                    intent=intent.to_dict(),
                    financial=financial,
                    rag=rag_payload,
                    sources=sources,
                    chart_images=chart_images,
                    warnings=tuple(financial.get("warnings", ())) if financial else (),
                )

        return AnalysisResult(
            status="answer",
            answer_text=deterministic,
            answer_source="analysis_deterministic",
            llm_used=False,
            intent=intent.to_dict(),
            financial=financial,
            rag=rag_payload,
            sources=sources,
            chart_images=chart_images,
            warnings=tuple(financial.get("warnings", ())) if financial else (),
        )

    def _run_financial_query(self, question: str, intent: StructuredIntent) -> dict[str, Any]:
        if intent.needs_clarification or not intent.metrics:
            return {"status": "skipped", "reason": "未识别到可执行的财务查询意图。", "queries": []}
        queries = []
        combined_warnings = list(intent.warnings)
        try:
            sub_intents = split_intent_by_table(intent, self.registry)
            for sub_intent in sub_intents:
                query = self.sql_builder.build(sub_intent.to_dsl())
                item: dict[str, Any] = {
                    "intent": sub_intent.to_dict(),
                    "sql": query.sql,
                    "params": query.params,
                    "table_name": query.table_name,
                    "metric_columns": query.metric_columns,
                    "warnings": query.warnings,
                    "result": self.query_executor.execute(query).to_dict(),
                }
                queries.append(item)
                combined_warnings.extend(query.warnings)
        except SQLBuildError as exc:
            return {"status": "error", "reason": str(exc), "queries": [], "warnings": tuple(dict.fromkeys(combined_warnings))}

        answer = self.answer_composer.compose(question, intent, queries, use_llm=False)
        chart_images = [chart_image.to_dict() for chart_image in self.chart_renderer.render_all(intent, queries)]
        payload: dict[str, Any] = {
            "status": "answer",
            "intent": intent.to_dict(),
            "queries": queries,
            "answer_text": answer.answer_text,
            "answer_source": answer.answer_source,
            "warnings": tuple(dict.fromkeys(combined_warnings)),
            "chart_images": chart_images,
        }
        if len(queries) == 1:
            payload["result"] = queries[0].get("result")
        return payload

    def _deterministic_answer(self, question: str, financial: dict[str, Any], rag_results: list[SearchResult]) -> str:
        parts = []
        if financial and financial.get("status") == "answer" and financial.get("answer_text"):
            parts.append(f"财务数据结果：{financial['answer_text']}")
        elif financial and financial.get("status") == "error":
            parts.append(f"财务数据部分暂未完成：{financial.get('reason')}。")

        if rag_results:
            lines = [f"研报依据：检索到 {len(rag_results)} 条相关片段。"]
            for index, result in enumerate(rag_results[:5], start=1):
                chunk = result.chunk
                meta = "，".join(item for item in (chunk.org_name, chunk.publish_date, chunk.section_title) if item)
                lines.append(f"{index}. 《{chunk.title}》{'（' + meta + '）' if meta else ''}：{make_snippet(chunk.text, 160)}")
            parts.append("\n".join(lines))
        else:
            parts.append("研报依据：当前知识库未检索到足够相关的研报片段。")

        if not parts:
            return "未能根据当前财务数据库和研报知识库生成回答。"
        return "\n".join(parts)

    def _llm_available(self) -> bool:
        return self._llm_client.is_available()

    def _retrieve_sources(
        self,
        question: str,
        intent: StructuredIntent,
        top_k: int,
        use_vector: bool,
    ) -> list[SearchResult]:
        """问题指定了公司时优先取该公司的研报(检索层 stock_codes 过滤),
        无命中再回退全库检索 —— 行业类/宏观类问题不该因公司过滤而无结果。"""
        if intent.company_codes:
            filtered = self.rag_service.search(
                question, top_k=top_k, use_vector=use_vector, stock_codes=tuple(intent.company_codes)
            )
            if filtered:
                return filtered
        return self.rag_service.search(question, top_k=top_k, use_vector=use_vector)

    def _compose_with_llm(
        self,
        question: str,
        financial: dict[str, Any],
        rag_results: list[SearchResult],
        deterministic_answer: str,
    ) -> str | None:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是上市公司财务与研报融合分析助手。只能基于提供的财务查询结果和研报证据回答。"
                    "不要编造未提供的数据、研报观点或来源。财务数据与研报观点要区分表达。"
                    "如果证据不足，请说明缺口。回答尽量简洁，并在使用研报观点处标注[来源1]、[来源2]。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "financial": _compact_financial(financial),
                        "research_evidence": _compact_sources(rag_results),
                        "fallback_answer": deterministic_answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self._llm_client.chat(messages, temperature=0.2)


def _looks_like_financial_question(question: str, intent: StructuredIntent) -> bool:
    if intent.metrics or intent.years or intent.periods or intent.metric_filters:
        return True
    normalized = re.sub(r"\s+", "", question)
    return bool(re.search(r"20\d{2}|年报|季度|营收|收入|净利润|资产|负债|现金流|毛利率|ROE|费用|同比|环比", normalized, re.IGNORECASE))


def _should_return_financial_clarification(intent: StructuredIntent) -> bool:
    if not intent.needs_clarification:
        return False
    if intent.metrics:
        return True
    missing = tuple(intent.clarification.missing_slots) if intent.clarification else ()
    return missing != ("metric",)


def _compact_financial(financial: dict[str, Any]) -> dict[str, Any]:
    if not financial:
        return {}
    compact_queries = []
    for query in financial.get("queries", [])[:4]:
        result = query.get("result") or {}
        compact_queries.append(
            {
                "table_name": query.get("table_name"),
                "metric_columns": query.get("metric_columns"),
                "units": result.get("units") or {},
                "rows": (result.get("rows") or [])[:12],
                "row_count": result.get("row_count", len(result.get("rows") or [])),
                "warnings": query.get("warnings") or [],
            }
        )
    return {
        "status": financial.get("status"),
        "answer_text": financial.get("answer_text"),
        "warnings": financial.get("warnings") or [],
        "queries": compact_queries,
    }


def _compact_sources(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "index": index + 1,
            "title": result.chunk.title,
            "report_type": result.chunk.report_type,
            "stock_name": result.chunk.stock_name,
            "industry_name": result.chunk.industry_name,
            "org_name": result.chunk.org_name,
            "publish_date": result.chunk.publish_date,
            "section_title": result.chunk.section_title,
            "snippet": make_snippet(result.chunk.text, 900),
        }
        for index, result in enumerate(results[:8])
    ]