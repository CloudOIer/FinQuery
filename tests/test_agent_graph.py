"""LangGraph 执行引擎测试(全部 mock LLM)。

重点验证三件事:与自研循环的行为等价(轨迹字段、步数保护、降级语义)、
并发请求之间的状态隔离、多轮追问的上下文延续。
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from finquery_agent.agent.graph import GraphAgentRunner
from finquery_agent.agent.planner import AgentPlanError, AgentPlanner
from finquery_agent.agent.tools import AgentToolbox
from finquery_agent.config import LLMSettings
from finquery_agent.llm import LLMClient
from finquery_agent.schema import load_default_registry

REGISTRY = load_default_registry()


class FakeExecutor:
    def __init__(self, rows=None):
        self.rows = rows or [
            {"stock_code": "600332", "report_year": 2024, "report_period": "FY", "total_operating_revenue": 100.0}
        ]

    def execute(self, query):
        from finquery_agent.nl2sql.executor import QueryResult

        return QueryResult(columns=("stock_code",), rows=tuple(self.rows), units={}, row_count=len(self.rows))


class FakeRAG:
    def search(self, question, top_k=None, use_vector=None, **kwargs):
        return []


def _toolbox(executor=None) -> AgentToolbox:
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
    client = LLMClient(LLMSettings(enabled=True, model="m", api_key="k", base_url="https://x"))
    replies = iter(script)

    client.chat_with_tools = lambda *args, **kwargs: next(replies)  # type: ignore[method-assign]
    client.chat = lambda *args, **kwargs: "强制总结回答"  # type: ignore[method-assign]
    return client


def _tool_call(name: str, arguments: dict, call_id: str = "call-1") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


def _runner(script: list[dict | None], max_steps: int = 4, boxes: list | None = None) -> GraphAgentRunner:
    def factory() -> AgentToolbox:
        box = _toolbox()
        if boxes is not None:
            boxes.append(box)
        return box

    return GraphAgentRunner(client=_client(script), toolbox_factory=factory, max_steps=max_steps)


QUERY_ARGS = {"metrics": ["营业总收入"], "companies": ["白云山"], "years": [2024], "periods": ["FY"]}


# ----------------------------------------------------------------------
# 与自研循环的行为等价
# ----------------------------------------------------------------------

def test_graph_executes_tool_then_answers():
    runner = _runner(
        [
            _tool_call("query_financial_data", QUERY_ARGS),
            {"role": "assistant", "content": "白云山2024年报营业总收入为100万元。"},
        ]
    )

    run = runner.run("白云山2024年报营收")

    assert run.completed is True
    assert "100" in run.answer_text
    assert len(run.steps) == 1
    assert run.steps[0]["tool"] == "query_financial_data"
    assert run.steps[0]["status"] == "ok"
    assert run.financial_queries


def test_graph_feeds_tool_error_back_and_llm_recovers():
    runner = _runner(
        [
            _tool_call("query_financial_data", {"metrics": ["净利润"], "periods": ["FY"]}),  # 缺 years → error
            _tool_call("query_financial_data", {"metrics": ["净利润"], "companies": ["白云山"], "years": [2024], "periods": ["FY"]}),
            {"role": "assistant", "content": "修正参数后查询成功。"},
        ]
    )

    run = runner.run("白云山净利润")

    assert run.completed is True
    assert [step["status"] for step in run.steps] == ["error", "ok"]


def test_graph_forces_summary_at_max_steps():
    same_call = _tool_call("query_financial_data", QUERY_ARGS)
    runner = _runner([same_call, same_call], max_steps=2)

    run = runner.run("白云山净利润")

    assert run.completed is False
    assert run.answer_text == "强制总结回答"
    assert len(run.steps) == 2


def test_graph_raises_when_llm_unavailable():
    runner = _runner([None], max_steps=2)

    with pytest.raises(AgentPlanError):
        runner.run("任意问题")


def test_graph_trace_fields_match_loop_engine():
    """轨迹字段是前端面板与评测脚本共用的契约,两种引擎必须逐字段一致。"""
    script = [
        _tool_call("query_financial_data", QUERY_ARGS),
        {"role": "assistant", "content": "回答。"},
    ]
    loop_run = AgentPlanner(_client(list(script)), _toolbox(), max_steps=4).run("白云山营收")
    graph_run = _runner(list(script)).run("白云山营收")

    assert set(graph_run.steps[0]) == set(loop_run.steps[0])
    for field in ("step", "tool", "arguments", "status", "summary"):
        assert graph_run.steps[0][field] == loop_run.steps[0][field]
    assert graph_run.answer_text == loop_run.answer_text
    assert graph_run.completed == loop_run.completed


# ----------------------------------------------------------------------
# 并发隔离与多轮追问
# ----------------------------------------------------------------------

def test_each_run_gets_its_own_toolbox():
    """工具箱按运行新建,是并发请求之间不串数据的前提。"""
    boxes: list[AgentToolbox] = []
    script = [_tool_call("query_financial_data", QUERY_ARGS), {"role": "assistant", "content": "答"}] * 2
    runner = _runner(script, boxes=boxes)

    runner.run("第一次")
    runner.run("第二次")

    assert len(boxes) == 2
    assert boxes[0] is not boxes[1]


def _capture(runner: GraphAgentRunner, reply: str) -> dict:
    """接管下一次模型调用,记下它实际看到的消息列表。"""
    seen: dict = {}

    def fake(messages, *args, **kwargs):
        seen["messages"] = messages
        return {"role": "assistant", "content": reply}

    runner.client.chat_with_tools = fake  # type: ignore[method-assign]
    return seen


def test_session_carries_history_into_followup():
    runner = _runner(
        [
            {"role": "assistant", "content": "白云山2024年营收100万元。"},
        ]
    )

    runner.run("白云山2024年营收", session_id="s-1")
    seen = _capture(runner, "它的净利润是20万元。")
    run = runner.run("那净利润呢", session_id="s-1")

    contents = [str(message.get("content") or "") for message in seen["messages"]]
    assert any("白云山2024年营收" in text for text in contents), "追问应带上首轮提问"
    assert any("100万元" in text for text in contents), "追问应带上首轮回答"
    assert sum(1 for message in seen["messages"] if message.get("role") == "system") == 1, "系统提示词不应重复注入"
    assert run.answer_text == "它的净利润是20万元。"


def test_sessions_are_isolated_from_each_other():
    runner = _runner([{"role": "assistant", "content": "答一"}])
    runner.run("会话一的问题", session_id="s-a")

    seen = _capture(runner, "答二")
    runner.run("会话二的问题", session_id="s-b")

    contents = [str(message.get("content") or "") for message in seen["messages"]]
    assert not any("会话一的问题" in text for text in contents)


def test_trace_resets_between_turns_in_same_session():
    """执行轨迹按轮独立:追问的轨迹不应带上首轮的步骤。"""
    runner = _runner(
        [
            _tool_call("query_financial_data", QUERY_ARGS),
            {"role": "assistant", "content": "首轮回答"},
            _tool_call("query_financial_data", QUERY_ARGS),
            {"role": "assistant", "content": "追问回答"},
        ]
    )

    first = runner.run("首轮", session_id="s-2")
    second = runner.run("追问", session_id="s-2")

    assert len(first.steps) == 1
    assert len(second.steps) == 1
    assert second.steps[0]["step"] == 1


# ----------------------------------------------------------------------
# 流式
# ----------------------------------------------------------------------

def test_stream_emits_progress_before_result():
    """进度事件要覆盖"模型思考"和"工具执行中",而不是只在步骤完成后才有动静。"""
    runner = _runner(
        [
            _tool_call("query_financial_data", QUERY_ARGS),
            {"role": "assistant", "content": "最终回答"},
        ]
    )

    events = list(runner.stream("白云山营收"))
    kinds = [event["event"] for event in events]

    assert kinds == ["thinking", "tool_start", "step", "thinking", "result"]
    assert events[1]["tool"] == "query_financial_data"
    assert events[2]["step"]["tool"] == "query_financial_data"
    assert events[-1]["result"].answer_text == "最终回答"


def test_stream_reports_progress_even_without_tool_calls():
    """模型一步作答时也要有进度事件,否则前端全程静默。"""
    runner = _runner([{"role": "assistant", "content": "直接回答"}])

    events = list(runner.stream("白云山是做什么的"))

    assert [event["event"] for event in events] == ["thinking", "result"]
    assert events[-1]["result"].answer_text == "直接回答"


# ----------------------------------------------------------------------
# 并发隔离:两种引擎都必须成立
# ----------------------------------------------------------------------

class _CompanyAwareExecutor:
    """按查询命中的公司返回对应数据,用于识别结果是否串到了另一个请求。"""

    def execute(self, query):
        from finquery_agent.nl2sql.executor import QueryResult

        blob = json.dumps(query.params, ensure_ascii=False, default=str)
        code = "600332" if "600332" in blob else "603259"
        return QueryResult(
            columns=("stock_code", "total_operating_revenue"),
            rows=({"stock_code": code, "report_year": 2024, "report_period": "FY", "total_operating_revenue": 100.0},),
            units={},
            row_count=1,
        )


def _make_service(engine: str, barrier):
    """构造一个只依赖假执行器的 AgentService,并让两个请求在查询后、画图前会合。"""
    from finquery_agent.agent.service import AgentService
    from finquery_agent.nl2sql.charting import ChartRenderer
    from finquery_agent.nl2sql.sql_builder import SQLBuilder

    class FakeAnalysis:
        registry = REGISTRY
        sql_builder = SQLBuilder(REGISTRY)
        query_executor = _CompanyAwareExecutor()
        rag_service = FakeRAG()
        chart_renderer = ChartRenderer(REGISTRY)

    service = AgentService(FakeAnalysis(), LLMSettings(enabled=True, model="m", api_key="k", base_url="https://x"), engine=engine)

    phases: dict[str, int] = {}
    lock = threading.Lock()

    def chat_with_tools(messages, tools, **kwargs):
        question = next(m["content"] for m in messages if m.get("role") == "user")
        with lock:
            phase = phases.get(question, 0)
            phases[question] = phase + 1
        company = "白云山" if "白云山" in question else "药明康德"
        if phase == 0:
            return _tool_call("query_financial_data", {**QUERY_ARGS, "companies": [company]}, call_id=f"q-{company}")
        if phase == 1:
            barrier.wait(timeout=5)  # 两个请求都取完数后才继续,最大化暴露共享状态
            return _tool_call("render_chart", {"chart_type": "bar"}, call_id=f"c-{company}")
        return {"role": "assistant", "content": f"{company}的回答"}

    service._client.chat_with_tools = chat_with_tools  # type: ignore[method-assign]
    service._client.chat = lambda *a, **kw: "总结"  # type: ignore[method-assign]
    return service


@pytest.mark.parametrize("engine", ["loop", "graph"])
def test_concurrent_requests_do_not_share_query_state(engine):
    """工具箱曾由服务长期持有,并发请求会互相覆盖最近一次查询结果,
    导致图表画出另一个请求的数据。两种引擎都必须隔离。"""
    barrier = threading.Barrier(2)
    service = _make_service(engine, barrier)
    results: dict[str, Any] = {}

    def ask(question: str, key: str) -> None:
        results[key] = service.ask(question, session_id=f"session-{key}")

    threads = [
        threading.Thread(target=ask, args=("白云山2024年营收并画图", "a")),
        threading.Thread(target=ask, args=("药明康德2024年营收并画图", "b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    def codes_of(result) -> set[str]:
        rows = result.financial["queries"][0]["result"]["rows"]
        return {str(row["stock_code"]) for row in rows}

    assert codes_of(results["a"]) == {"600332"}, "白云山的请求拿到了另一个请求的数据"
    assert codes_of(results["b"]) == {"603259"}, "药明康德的请求拿到了另一个请求的数据"
    assert results["a"].chart_images and results["b"].chart_images
    assert results["a"].answer_text != results["b"].answer_text
