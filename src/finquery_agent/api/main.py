from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from finquery_agent.agent import AgentService
from finquery_agent.analysis import AnalysisService
from finquery_agent.config import load_llm_settings, load_rag_settings
from finquery_agent.db import create_database_engine
from finquery_agent.nl2sql import (
    HybridIntentEngine,
    LlmIntentEngine,
    QueryDSL,
    RuleBasedIntentEngine,
    SQLBuildError,
    SQLBuilder,
)
from finquery_agent.nl2sql.answer import AnswerComposer
from finquery_agent.nl2sql.charting import ChartRenderer
from finquery_agent.nl2sql.executor import QueryExecutor
from finquery_agent.nl2sql.planner import split_intent_by_table
from finquery_agent.nl2sql.session import QuerySessionStore
from finquery_agent.rag.service import RAGService
from finquery_agent.schema import load_default_registry


class GenerateSqlRequest(BaseModel):
    intent: str = Field(default="metric_query")
    metrics: list[str]
    company_codes: list[str] = Field(default_factory=list)
    company_names: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    limit: int = 100
    allow_all_periods: bool = False


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None
    execute: bool = False
    use_llm: bool = True


class RAGSearchRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=30)
    use_vector: bool = True


class RAGAskRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=30)
    use_vector: bool = True
    use_llm: bool = True


class AnalysisAskRequest(BaseModel):
    question: str
    session_id: str | None = None
    use_llm: bool = True
    use_rag: bool = True
    use_vector: bool = True
    rag_top_k: int = Field(default=8, ge=1, le=30)
    # Agent 模式:LLM 自主规划工具调用(多步查数/计算/检索);失败自动降级回单轮管道。
    use_agent: bool = False


app = FastAPI(title="FinQuery Agent API", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
registry = load_default_registry()
sql_builder = SQLBuilder(registry)
# LLM 意图识别由 config/llm.json 的 intent_enabled 控制;关闭或 LLM 不可用时
# HybridIntentEngine 自动降级到规则引擎,因此这里无条件地装配混合引擎即可。
_llm_settings = load_llm_settings()
rule_intent_engine = RuleBasedIntentEngine(registry)
intent_engine = HybridIntentEngine(rule_intent_engine, LlmIntentEngine(registry, _llm_settings))
query_executor = QueryExecutor(create_database_engine(), registry)
session_store = QuerySessionStore()
answer_composer = AnswerComposer(registry, _llm_settings)
chart_renderer = ChartRenderer(registry)
_rag_service: RAGService | None = None
_rag_lock = Lock()
_analysis_service: AnalysisService | None = None
_analysis_lock = Lock()
_agent_service: AgentService | None = None
_agent_lock = Lock()


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is not None:
        return _rag_service
    with _rag_lock:
        if _rag_service is None:
            _rag_service = RAGService(load_rag_settings(), _llm_settings)
        return _rag_service


def get_analysis_service() -> AnalysisService:
    global _analysis_service
    if _analysis_service is not None:
        return _analysis_service
    with _analysis_lock:
        if _analysis_service is None:
            _analysis_service = AnalysisService(
                registry=registry,
                intent_engine=intent_engine,
                sql_builder=sql_builder,
                query_executor=query_executor,
                answer_composer=answer_composer,
                chart_renderer=chart_renderer,
                rag_service=get_rag_service(),
                session_store=session_store,
                llm_settings=_llm_settings,
            )
        return _analysis_service


def get_agent_service() -> AgentService:
    global _agent_service
    if _agent_service is not None:
        return _agent_service
    with _agent_lock:
        if _agent_service is None:
            _agent_service = AgentService(get_analysis_service(), _llm_settings)
        return _agent_service


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema/tables")
def list_tables() -> dict[str, object]:
    return {
        "tables": [
            {
                "name": table.name,
                "chinese_name": table.chinese_name,
                "field_count": len(table.fields),
            }
            for table in registry.tables.values()
        ]
    }


@app.post("/rag/search")
def rag_search(request: RAGSearchRequest) -> dict[str, object]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    try:
        results = get_rag_service().search(question, top_k=request.top_k, use_vector=request.use_vector)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG 检索失败: {exc}") from exc
    return {"status": "ok", "count": len(results), "results": [result.to_dict() for result in results]}


@app.post("/rag/ask")
def rag_ask(request: RAGAskRequest) -> dict[str, object]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    try:
        answer = get_rag_service().answer(
            question,
            top_k=request.top_k,
            use_vector=request.use_vector,
            use_llm=request.use_llm,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG 问答失败: {exc}") from exc
    payload = answer.to_dict()
    payload["status"] = "answer"
    return payload


@app.post("/analysis/ask")
def analysis_ask(request: AnalysisAskRequest) -> dict[str, object]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question 不能为空")
    analysis_kwargs = {
        "session_id": request.session_id,
        "use_llm": request.use_llm,
        "use_rag": request.use_rag,
        "use_vector": request.use_vector,
        "rag_top_k": request.rag_top_k,
    }
    try:
        if request.use_agent:
            result = get_agent_service().ask(question, **analysis_kwargs)
        else:
            result = get_analysis_service().ask(question, **analysis_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"融合分析失败: {exc}") from exc
    return result.to_dict()


@app.post("/nl2sql/generate")
def generate_sql(request: GenerateSqlRequest) -> dict[str, object]:
    dsl = QueryDSL(
        intent=request.intent,
        metrics=tuple(request.metrics),
        company_codes=tuple(request.company_codes),
        company_names=tuple(request.company_names),
        years=tuple(request.years),
        periods=tuple(request.periods),
        limit=request.limit,
        allow_all_periods=request.allow_all_periods,
    )
    try:
        query = sql_builder.build(dsl)
    except SQLBuildError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "sql": query.sql,
        "params": query.params,
        "table_name": query.table_name,
        "metric_columns": query.metric_columns,
        "warnings": query.warnings,
    }


@app.post("/query/intent")
def parse_query_intent(request: QueryRequest) -> dict[str, object]:
    intent = session_store.resolve(request.session_id, request.question, intent_engine)
    return intent.to_dict()


@app.post("/query/ask")
def ask_query(request: QueryRequest) -> dict[str, object]:
    intent = session_store.resolve(request.session_id, request.question, intent_engine)
    if intent.needs_clarification:
        return {"status": "clarification", "intent": intent.to_dict(), "clarification": intent.clarification.__dict__ if intent.clarification else None}
    sub_intents = split_intent_by_table(intent, registry)
    queries = []
    combined_warnings = list(intent.warnings)
    for sub_intent in sub_intents:
        try:
            query = sql_builder.build(sub_intent.to_dsl())
        except SQLBuildError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        item = {
            "intent": sub_intent.to_dict(),
            "sql": query.sql,
            "params": query.params,
            "table_name": query.table_name,
            "metric_columns": query.metric_columns,
            "warnings": query.warnings,
        }
        if request.execute:
            item["result"] = query_executor.execute(query).to_dict()
        queries.append(item)
        combined_warnings.extend(query.warnings)

    first_query = queries[0]
    response = {
        "status": "answer",
        "intent": intent.to_dict(),
        "queries": queries,
        "warnings": tuple(dict.fromkeys(combined_warnings)),
        "chart": intent.chart.__dict__ if intent.chart else None,
    }
    if request.execute:
        answer = answer_composer.compose(request.question, intent, queries, use_llm=request.use_llm)
        response.update(answer.to_dict())
        chart_images = chart_renderer.render_all(intent, queries)
        if chart_images:
            response["chart_images"] = [chart_image.to_dict() for chart_image in chart_images]
            response["chart_image"] = chart_images[0].to_dict()
    # Backward-compatible fields for the common single-query case.
    if len(queries) == 1:
        response.update(
            {
                "sql": first_query["sql"],
                "params": first_query["params"],
                "table_name": first_query["table_name"],
                "metric_columns": first_query["metric_columns"],
            }
        )
        if request.execute:
            response["result"] = first_query.get("result")
    return response
