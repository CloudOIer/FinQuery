"""LangGraph 版执行引擎:与自研循环并存的另一种编排实现。

与 planner.py 的关系:两者行为等价、互不依赖。本模块复用 planner.py 的系统提示词、
结果数据类与异常类型(而非另抄一份),以保证两种引擎的提示词、响应结构和降级语义
不会随时间漂移;自研循环本身不做任何修改。

相比自研循环额外提供:
- 每次运行独立持有工具箱,由调用方经运行时配置注入,请求之间不再共享中间产物;
- 检查点持久化,同一会话的后续提问自动带上此前的对话与工具观察;
- 步进事件流,调用方可在运行过程中实时获取每一步的执行轨迹。

步数保护不使用框架的递归上限:递归上限触发时抛异常终止,而本系统要求到达上限时
去掉工具做一次强制总结,给出部分答案并声明信息缺口。因此在状态里自行计数并经
条件边路由到收尾节点,递归上限仅作为彻底失控时的兜底。
"""

from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, Iterator, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from finquery_agent.agent.planner import (
    DEFAULT_MAX_STEPS,
    AgentPlanError,
    AgentRunResult,
    _summarize,
    _SYSTEM_PROMPT,
)
from finquery_agent.agent.tools import AgentToolbox, tool_schemas
from finquery_agent.llm import LLMClient

FORCE_SUMMARY_PROMPT = (
    "已达到最大工具调用步数。请基于以上已获取的信息直接给出最终回答;信息不足的部分明确说明。"
)


class AgentState(TypedDict):
    """messages 跨轮累积以支撑追问;其余字段每轮覆盖,使执行轨迹按轮独立。"""

    messages: Annotated[list[dict[str, Any]], operator.add]
    trace: list[dict[str, Any]]
    step: int
    max_steps: int
    completed: bool


def _toolbox_from(config: RunnableConfig) -> AgentToolbox:
    toolbox = (config.get("configurable") or {}).get("toolbox")
    if toolbox is None:
        raise AgentPlanError("运行配置缺少 toolbox。")
    return toolbox


def _client_from(config: RunnableConfig) -> LLMClient:
    client = (config.get("configurable") or {}).get("client")
    if client is None:
        raise AgentPlanError("运行配置缺少 LLM 客户端。")
    return client


def _emit(event: dict[str, Any]) -> None:
    """向自定义事件流推送进度。

    建立在节点"开始"而非"完成"时:模型决策和检索都是秒级耗时,
    只在完成后推送会让前端在最漫长的等待里保持静默。
    非流式调用时没有流写入器,此时静默跳过。
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    if writer is not None:
        writer(event)


def _decide(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _emit({"event": "thinking", "step": state["step"] + 1})
    message = _client_from(config).chat_with_tools(state["messages"], tool_schemas(), temperature=0.1)
    if message is None:
        raise AgentPlanError("LLM 请求失败。")
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        answer = str(message.get("content") or "").strip()
        if not answer:
            raise AgentPlanError("LLM 返回空回答。")
        return {"messages": [message], "completed": True}
    # 带 tool_calls 的助手消息必须原样入状态,且先于随后的 tool 结果消息。
    return {"messages": [message], "step": state["step"] + 1}


def _run_tools(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """串行执行本轮工具调用。

    不做并行:工具箱维护着跨调用的共享产物(最近一次查询结果、研报来源编号),
    并发执行会让图表引用哪份数据、来源如何编号变得不确定。
    """
    toolbox = _toolbox_from(config)
    tool_calls = state["messages"][-1].get("tool_calls") or []
    trace = list(state["trace"])
    new_messages: list[dict[str, Any]] = []
    for call in tool_calls:
        _emit({"event": "tool_start", "tool": str((call.get("function") or {}).get("name") or "")})
        observation, entry = _execute_call(toolbox, call, len(trace) + 1)
        trace.append(entry)
        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": json.dumps(observation, ensure_ascii=False, default=str),
            }
        )
    return {"messages": new_messages, "trace": trace}


def _summarize_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _emit({"event": "summarizing"})
    messages = state["messages"] + [{"role": "user", "content": FORCE_SUMMARY_PROMPT}]
    answer = _client_from(config).chat(messages, temperature=0.1)
    if not answer:
        raise AgentPlanError(f"达到最大步数({state['max_steps']})且无法生成总结。")
    return {"messages": [{"role": "assistant", "content": answer}], "completed": False}


def _route_after_agent(state: AgentState) -> str:
    return END if state["completed"] else "tools"


def _route_after_tools(state: AgentState) -> str:
    """步数在工具执行后才判断:自研循环的每一步是"调模型 + 执行工具",
    若在执行前判断会吞掉最后一步的工具调用,两种引擎的轨迹长度就不一致了。"""
    return "summarize" if state["step"] >= state["max_steps"] else "agent"


def _execute_call(
    toolbox: AgentToolbox, call: dict[str, Any], step_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
        if not isinstance(arguments, dict):
            arguments = {}
    except json.JSONDecodeError:
        arguments = {}
        # 给出具体格式示例:实测 LLM 会连续用同样的非法格式重试,
        # 只说"不是合法 JSON"不足以让它纠正。
        observation = {
            "error": (
                "工具参数不是合法 JSON。arguments 必须是标准 JSON 对象(双引号、无表达式),"
                '例如 {"expression": "(a-b)/b*100", "variables": {"a": 120.5, "b": 100.0}};'
                "数值请先在 variables 里给出字面量,不要在 JSON 里写算式。"
            )
        }
        return observation, _trace_entry(step_index, name, {}, observation, 0.0)

    started = time.time()
    observation = toolbox.execute(name, arguments)
    elapsed = time.time() - started
    return observation, _trace_entry(step_index, name, arguments, observation, elapsed)


def _trace_entry(
    step: int,
    tool: str,
    arguments: dict[str, Any],
    observation: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    """字段与自研循环逐一对齐:前端执行过程面板与评测脚本共用同一份格式。"""
    return {
        "step": step,
        "tool": tool,
        "arguments": arguments,
        "status": "error" if "error" in observation else "ok",
        "summary": _summarize(observation),
        "elapsed_seconds": round(elapsed, 2),
    }


def build_graph(checkpointer: Any = None) -> Any:
    graph = StateGraph(AgentState)
    graph.add_node("agent", _decide)
    graph.add_node("tools", _run_tools)
    graph.add_node("summarize", _summarize_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", "summarize": "summarize"})
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer)


class GraphAgentRunner:
    """状态图的调用封装。

    工具箱由 factory 每次运行新建:工具箱记录着最近一次查询结果供图表引用,
    进程内共享一份会让并发请求互相覆盖,画出上一个请求的数据。
    """

    def __init__(
        self,
        client: LLMClient,
        toolbox_factory: Any,
        max_steps: int = DEFAULT_MAX_STEPS,
        checkpointer: Any = None,
    ):
        self.client = client
        self.toolbox_factory = toolbox_factory
        self.max_steps = max_steps
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = build_graph(self.checkpointer)

    def run(self, question: str, session_id: str | None = None) -> AgentRunResult:
        toolbox, state_input, config = self._prepare(question, session_id)
        final = self.graph.invoke(state_input, config=config)
        return self._result(final, toolbox)

    def stream(self, question: str, session_id: str | None = None) -> Iterator[dict[str, Any]]:
        """逐步产出执行事件,末尾产出最终结果。

        供接口层做增量推送:执行过程本身有信息量,等全流程结束再一次性展示,
        等于把最有价值的进度信息留到了最没用的时刻。
        """
        toolbox, state_input, config = self._prepare(question, session_id)
        emitted = 0
        for mode, chunk in self.graph.stream(
            state_input, config=config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                yield chunk
                continue
            for payload in chunk.values():
                if not isinstance(payload, dict):
                    continue
                trace = payload.get("trace") or []
                for entry in trace[emitted:]:
                    yield {"event": "step", "step": entry}
                emitted = max(emitted, len(trace))
        yield {"event": "result", "result": self._result(self.graph.get_state(config).values, toolbox)}

    def _prepare(
        self, question: str, session_id: str | None
    ) -> tuple[AgentToolbox, dict[str, Any], dict[str, Any]]:
        toolbox = self.toolbox_factory()
        thread_id = session_id or f"once-{time.time_ns()}"
        config = {
            "configurable": {"thread_id": thread_id, "toolbox": toolbox, "client": self.client},
            "recursion_limit": 2 * self.max_steps + 8,
        }
        # 已有历史的会话只追加提问,系统提示词不重复注入。
        has_history = bool(self.graph.get_state(config).values.get("messages"))
        messages: list[dict[str, Any]] = [] if has_history else [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": question})
        state_input: dict[str, Any] = {
            "messages": messages,
            "trace": [],
            "step": 0,
            "max_steps": self.max_steps,
            "completed": False,
        }
        return toolbox, state_input, config

    def _result(self, state: dict[str, Any], toolbox: AgentToolbox) -> AgentRunResult:
        messages = state.get("messages") or []
        answer = ""
        for message in reversed(messages):
            if message.get("role") == "assistant" and not message.get("tool_calls"):
                answer = str(message.get("content") or "").strip()
                break
        if not answer:
            raise AgentPlanError("状态图未产出最终回答。")
        return AgentRunResult(
            answer_text=answer,
            steps=list(state.get("trace") or []),
            sources=list(toolbox.collected_sources),
            chart_images=list(toolbox.collected_charts),
            financial_queries=list(toolbox.last_queries),
            completed=bool(state.get("completed")),
        )
