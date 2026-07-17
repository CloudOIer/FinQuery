from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from finquery_agent.config import LLMSettings
from finquery_agent.llm import LLMClient
from finquery_agent.nl2sql.intent import StructuredIntent
from finquery_agent.schema.models import FieldDefinition
from finquery_agent.schema.registry import SchemaRegistry


@dataclass(frozen=True)
class AnswerResult:
    answer_text: str
    answer_source: str
    llm_used: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"answer_text": self.answer_text, "answer_source": self.answer_source, "llm_used": self.llm_used}


class AnswerComposer:
    """Convert query results into user-facing answers.

    The deterministic answer is always generated first. LLM polishing is optional and
    only runs when both the request and config explicitly allow it, so normal queries do
    not spend LLM quota accidentally.
    """

    def __init__(self, registry: SchemaRegistry, llm_settings: LLMSettings | None = None):
        self.registry = registry
        self.llm_settings = llm_settings or LLMSettings()
        self._llm_client = LLMClient(self.llm_settings)

    def compose(
        self,
        question: str,
        intent: StructuredIntent,
        queries: list[dict[str, Any]],
        use_llm: bool = False,
    ) -> AnswerResult:
        deterministic = self._deterministic_answer(question, queries)
        if not use_llm or not self._llm_available():
            return AnswerResult(answer_text=deterministic, answer_source="deterministic", llm_used=False)
        llm_answer = self._compose_with_llm(question, intent, queries, deterministic)
        return AnswerResult(answer_text=llm_answer or deterministic, answer_source="llm" if llm_answer else "deterministic", llm_used=bool(llm_answer))

    def _llm_available(self) -> bool:
        return self._llm_client.is_available()

    def _deterministic_answer(self, question: str, queries: list[dict[str, Any]]) -> str:
        if not queries:
            return "没有生成可执行查询。"
        parts = [self._answer_for_query(query) for query in queries]
        return "\n".join(part for part in parts if part).strip() or "查询已完成，但没有可展示的数据。"

    def _answer_for_query(self, query: dict[str, Any]) -> str:
        result = query.get("result")
        table_name = str(query.get("table_name") or "")
        metric_columns = tuple(query.get("metric_columns") or ())
        if not result:
            return "已生成 SQL，尚未执行查询。"
        rows = result.get("rows") or []
        units = result.get("units") or {}
        if not rows:
            return "未查询到符合条件的记录；按当前数据口径，数量为 0。"
        if len(rows) == 1 and len(metric_columns) == 1:
            row = rows[0]
            metric = metric_columns[0]
            label = self._field_label(table_name, metric)
            value = _format_value(row.get(metric), units.get(metric))
            subject = self._format_subject(row)
            return f"{subject}的{label}为{value}。"
        snippets = []
        for row in rows[:8]:
            subject = self._format_subject(row)
            metrics = []
            for metric in metric_columns[:3]:
                metrics.append(f"{self._field_label(table_name, metric)}={_format_value(row.get(metric), units.get(metric))}")
            snippets.append(f"{subject}：{'，'.join(metrics)}")
        suffix = "" if len(rows) <= 8 else f"（仅展示前 8 条，共 {len(rows)} 条）"
        return f"查询到 {len(rows)} 条记录。" + "；".join(snippets) + suffix + "。"

    def _field_label(self, table_name: str, field_name: str) -> str:
        table = self.registry.tables.get(table_name)
        if not table:
            return field_name
        field_map: dict[str, FieldDefinition] = {field.name: field for field in table.fields}
        field = field_map.get(field_name)
        return field.chinese_name if field else field_name

    def _format_subject(self, row: dict[str, Any]) -> str:
        stock_code = str(row.get("stock_code") or "")
        company = row.get("stock_abbr") or (self.registry.companies.get(stock_code).stock_abbr if stock_code in self.registry.companies else None) or stock_code or "该公司"
        year = row.get("report_year")
        period = _period_label(row.get("report_period"))
        if year and period:
            return f"{company}{year}年{period}"
        if year:
            return f"{company}{year}年"
        return str(company)

    def _compose_with_llm(
        self,
        question: str,
        intent: StructuredIntent,
        queries: list[dict[str, Any]],
        deterministic_answer: str,
    ) -> str | None:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是财报问数结果表达助手。只根据用户问题和提供的数据回答。"
                    "不要解释SQL、系统流程或推理过程。数据为空就说明未查询到。"
                    "保留单位，回答不超过3句话。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "intent": intent.to_dict(),
                        "query_results": _compact_queries_for_llm(queries),
                        "fallback_answer": deterministic_answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return self._llm_client.chat(messages, temperature=0.1)


def _compact_queries_for_llm(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for query in queries:
        result = query.get("result") or {}
        rows = result.get("rows") or []
        compact.append(
            {
                "table_name": query.get("table_name"),
                "metric_columns": query.get("metric_columns"),
                "units": result.get("units") or {},
                "rows": rows[:12],
                "row_count": result.get("row_count", len(rows)),
            }
        )
    return compact


def _period_label(period: Any) -> str:
    mapping = {"FY": "年报", "Q1": "第一季度", "HY": "半年度", "Q3": "第三季度"}
    raw = "".join(str(period or "").split())
    upper = raw.upper()
    if upper in mapping:
        return mapping[upper]
    if "一季" in raw or "第1季" in raw or "1季" in raw:
        return "第一季度"
    if "半" in raw or "中期" in raw:
        return "半年度"
    if "三季" in raw or "第3季" in raw or "3季" in raw:
        return "第三季度"
    if "年报" in raw or "年度" in raw or raw == "年":
        return "年报"
    return raw


def _format_value(value: Any, unit: str | None = None) -> str:
    if value is None:
        return "暂无数据"
    if isinstance(value, float):
        text = f"{value:,.6f}".rstrip("0").rstrip(".")
    elif isinstance(value, int):
        text = f"{value:,}"
    elif isinstance(value, Decimal):
        text = f"{float(value):,.6f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return f"{text}{unit or ''}"
