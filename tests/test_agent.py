"""Agent 循环、工具校验与降级测试(全部 mock LLM)。"""

from __future__ import annotations

import json

import pytest

from finquery_agent.agent.planner import AgentPlanError, AgentPlanner
from finquery_agent.agent.tools import AgentToolbox, _safe_eval, tool_schemas
from finquery_agent.config import LLMSettings
from finquery_agent.llm import LLMClient
from finquery_agent.schema import load_default_registry

REGISTRY = load_default_registry()


class FakeExecutor:
    def __init__(self, rows=None):
        self.rows = rows or [{"stock_code": "600332", "report_year": 2024, "report_period": "FY", "total_operating_revenue": 100.0}]

    def execute(self, query):
        from finquery_agent.nl2sql.executor import QueryResult

        return QueryResult(columns=("stock_code",), rows=tuple(self.rows), units={}, row_count=len(self.rows))


class FakeRAG:
    def search(self, question, top_k=None, use_vector=None, **kwargs):
        return []


def _toolbox(executor=None):
    from finquery_agent.nl2sql.charting import ChartRenderer
    from finquery_agent.nl2sql.sql_builder import SQLBuilder

    return AgentToolbox(
        registry=REGISTRY,
        sql_builder=SQLBuilder(REGISTRY),
        query_executor=executor or FakeExecutor(),
        rag_service=FakeRAG(),
        chart_renderer=ChartRenderer(REGISTRY),
    )


def _client(script: list[dict | None]) -> LLMClient:
    """按脚本依次返回 message 的假客户端;chat() 用于强制总结场景。"""
    client = LLMClient(LLMSettings(enabled=True, model="m", api_key="k", base_url="https://x"))
    replies = iter(script)

    client.chat_with_tools = lambda *args, **kwargs: next(replies)  # type: ignore[method-assign]
    client.chat = lambda *args, **kwargs: "强制总结回答"  # type: ignore[method-assign]
    return client


def _tool_call(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}],
    }


# ----------------------------------------------------------------------
# 工具校验
# ----------------------------------------------------------------------

def test_query_tool_rejects_unknown_metric_and_company():
    toolbox = _toolbox()

    assert "error" in toolbox.execute("query_financial_data", {"metrics": ["不存在指标"], "years": [2024], "periods": ["FY"]})
    assert "error" in toolbox.execute("query_financial_data", {"metrics": ["净利润"], "companies": ["特斯拉"], "years": [2024], "periods": ["FY"]})
    assert "error" in toolbox.execute("query_financial_data", {"metrics": ["净利润"], "periods": ["FY"]})  # 缺 years


def test_query_tool_executes_and_truncates_rows():
    rows = [{"stock_code": f"60{i:04d}", "report_year": 2024, "report_period": "FY", "net_profit_10k_yuan": float(i)} for i in range(30)]
    toolbox = _toolbox(FakeExecutor(rows))

    result = toolbox.execute("query_financial_data", {"metrics": ["净利润"], "years": [2024], "periods": ["FY"]})

    assert "error" not in result
    query = result["queries"][0]
    assert query["row_count"] == 30
    assert len(query["rows"]) == 20  # MAX_ROWS_TO_LLM 截断
    assert query["truncated"] is True
    assert toolbox.last_queries  # 完整结果留给 render_chart/响应


def test_render_chart_requires_prior_query():
    toolbox = _toolbox()

    assert "error" in toolbox.execute("render_chart", {"chart_type": "line"})


def test_calculate_tool_and_safe_eval():
    toolbox = _toolbox()

    growth = toolbox.execute("calculate", {"expression": "(a-b)/b*100", "variables": {"a": 120, "b": 100}})
    assert growth["result"] == pytest.approx(20.0)
    med = toolbox.execute("calculate", {"expression": "median(xs)", "variables": {"xs": [3, 1, 2]}})
    assert med["result"] == 2

    # 危险表达式必须被拒绝(白名单外节点/函数)。
    assert "error" in toolbox.execute("calculate", {"expression": "__import__('os').system('ls')"})
    assert "error" in toolbox.execute("calculate", {"expression": "open('/etc/passwd')"})
    with pytest.raises(ValueError):
        _safe_eval("(lambda: 1)()", {})


# ----------------------------------------------------------------------
# 循环与降级
# ----------------------------------------------------------------------

def test_planner_executes_tool_then_answers():
    toolbox = _toolbox()
    client = _client([
        _tool_call("query_financial_data", {"metrics": ["营业总收入"], "companies": ["白云山"], "years": [2024], "periods": ["FY"]}),
        {"role": "assistant", "content": "白云山2024年报营业总收入为100万元。"},
    ])
    planner = AgentPlanner(client, toolbox, max_steps=4)

    run = planner.run("白云山2024年报营收")

    assert run.completed is True
    assert "100" in run.answer_text
    assert len(run.steps) == 1
    assert run.steps[0]["tool"] == "query_financial_data"
    assert run.steps[0]["status"] == "ok"
    assert run.financial_queries


def test_planner_feeds_tool_error_back_and_llm_recovers():
    toolbox = _toolbox()
    client = _client([
        _tool_call("query_financial_data", {"metrics": ["净利润"], "periods": ["FY"]}),  # 缺 years → error
        _tool_call("query_financial_data", {"metrics": ["净利润"], "companies": ["白云山"], "years": [2024], "periods": ["FY"]}),
        {"role": "assistant", "content": "修正参数后查询成功。"},
    ])
    planner = AgentPlanner(client, toolbox, max_steps=4)

    run = planner.run("白云山净利润")

    assert run.completed is True
    assert [s["status"] for s in run.steps] == ["error", "ok"]


def test_planner_forces_summary_at_max_steps():
    toolbox = _toolbox()
    same_call = _tool_call("query_financial_data", {"metrics": ["净利润"], "companies": ["白云山"], "years": [2024], "periods": ["FY"]})
    client = _client([same_call, same_call])
    planner = AgentPlanner(client, toolbox, max_steps=2)

    run = planner.run("白云山净利润")

    assert run.completed is False
    assert run.answer_text == "强制总结回答"
    assert len(run.steps) == 2


def test_planner_raises_when_llm_unavailable():
    toolbox = _toolbox()
    client = _client([None])
    planner = AgentPlanner(client, toolbox, max_steps=2)

    with pytest.raises(AgentPlanError):
        planner.run("任意问题")


def test_agent_service_falls_back_to_pipeline():
    from finquery_agent.agent.service import AgentService

    class FakeAnalysis:
        registry = REGISTRY
        sql_builder = None
        query_executor = None
        rag_service = FakeRAG()
        chart_renderer = None

        def ask(self, question, **kwargs):
            from finquery_agent.analysis.service import AnalysisResult

            return AnalysisResult(status="answer", answer_text="旧管道回答", answer_source="analysis_deterministic")

    # LLM 配置不可用 → 不进循环,直接降级。
    service = AgentService.__new__(AgentService)
    service.analysis_service = FakeAnalysis()
    service._client = LLMClient(LLMSettings())  # disabled
    result = service.ask("问题")
    assert result.answer_source == "analysis_deterministic"


def test_tool_schemas_shape():
    schemas = tool_schemas()

    names = {schema["function"]["name"] for schema in schemas}
    assert names == {"query_financial_data", "search_research_reports", "calculate", "render_chart"}
    for schema in schemas:
        assert schema["function"]["parameters"]["type"] == "object"
