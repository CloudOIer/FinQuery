"""Agent 规划器:计划-执行-观察循环。

循环协议(OpenAI function-calling):LLM 返回 tool_calls 则逐个执行工具并把
结果以 tool 消息回填;返回纯文本则视为最终回答,循环结束。

两个保护边界:
- max_steps:LLM 可能陷入"反复调同一工具"的循环,步数上限保证请求必然终止;
  到达上限时携带已收集的工具结果做一次强制总结(去掉 tools 参数),
  尽力给出部分答案而不是报错。
- 每步记录 execution_trace(工具名/参数/结果摘要/耗时):前端可视化、
  评测归因、线上审计共用同一份记录。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from finquery_agent.agent.tools import AgentToolbox, tool_schemas
from finquery_agent.llm import LLMClient

# 附件6 四段式多意图题(查数×2 + 计算×3 + 检索 + 图表)实测需 7~9 步,
# 8 步在"能完成复杂题"与"防止失控循环的成本上限"之间取平衡。
DEFAULT_MAX_STEPS = 8

_SYSTEM_PROMPT = """你是上市公司财务分析 Agent。通过调用工具回答用户问题,规则:
1. 财务数字必须来自 query_financial_data 的结果,研报观点必须来自 search_research_reports 的结果,不允许凭记忆编造。
2. 衍生计算(增长率/比值/中位数等)用 calculate 工具,不要心算。
3. 多步问题按依赖顺序拆解:先取数,再计算,再检索观点,需要时最后 render_chart。
4. 数据库只有 76 家医药类公司、2022-2025 年、报告期 FY/Q1/HY/Q3;金额单位万元。
5. 工具返回 error 时,根据错误信息修正参数重试;同一错误不要重复超过 2 次。
6. 最终回答:中文、简洁、区分"财务数据"与"研报观点",引用研报处标注[来源N];数据缺口要明确说明。"""


@dataclass
class AgentRunResult:
    answer_text: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    chart_images: list[dict[str, Any]] = field(default_factory=list)
    financial_queries: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = True


class AgentPlanError(RuntimeError):
    """LLM 不可用或循环中断且无法给出任何回答(由上层降级)。"""


class AgentPlanner:
    def __init__(self, client: LLMClient, toolbox: AgentToolbox, max_steps: int = DEFAULT_MAX_STEPS):
        self.client = client
        self.toolbox = toolbox
        self.max_steps = max_steps
        self._schemas = tool_schemas()

    def run(self, question: str) -> AgentRunResult:
        self.toolbox.reset()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            message = self.client.chat_with_tools(messages, self._schemas, temperature=0.1)
            if message is None:
                raise AgentPlanError("LLM 请求失败。")
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                answer = str(message.get("content") or "").strip()
                if not answer:
                    raise AgentPlanError("LLM 返回空回答。")
                return self._result(answer, trace, completed=True)

            # assistant 消息必须原样回填(含 tool_calls),协议要求与 tool 结果一一对应。
            messages.append(message)
            for call in tool_calls:
                observation, trace_entry = self._execute_call(call, len(trace) + 1)
                trace.append(trace_entry)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(observation, ensure_ascii=False, default=str),
                    }
                )

        # 步数耗尽:强制总结(不给 tools,LLM 只能基于已有观察作答)。
        messages.append(
            {
                "role": "user",
                "content": "已达到最大工具调用步数。请基于以上已获取的信息直接给出最终回答;信息不足的部分明确说明。",
            }
        )
        answer = self.client.chat(messages, temperature=0.1)
        if not answer:
            raise AgentPlanError(f"达到最大步数({self.max_steps})且无法生成总结。")
        return self._result(answer, trace, completed=False)

    def _execute_call(self, call: dict[str, Any], step_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
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
            trace_entry = self._trace_entry(step_index, name, {}, observation, 0.0)
            return observation, trace_entry

        started = time.time()
        observation = self.toolbox.execute(name, arguments)
        elapsed = time.time() - started
        return observation, self._trace_entry(step_index, name, arguments, observation, elapsed)

    def _trace_entry(
        self,
        step: int,
        tool: str,
        arguments: dict[str, Any],
        observation: dict[str, Any],
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "tool": tool,
            "arguments": arguments,
            "status": "error" if "error" in observation else "ok",
            "summary": _summarize(observation),
            "elapsed_seconds": round(elapsed, 2),
        }

    def _result(self, answer: str, trace: list[dict[str, Any]], completed: bool) -> AgentRunResult:
        return AgentRunResult(
            answer_text=answer,
            steps=trace,
            sources=list(self.toolbox.collected_sources),
            chart_images=list(self.toolbox.collected_charts),
            financial_queries=list(self.toolbox.last_queries),
            completed=completed,
        )


def _summarize(observation: dict[str, Any]) -> str:
    """trace 里存摘要而非完整结果:完整查询结果可能有几十行,
    trace 是给人看和给 judge 评的,过长反而掩盖关键信息。"""
    if "error" in observation:
        return str(observation["error"])[:200]
    if "queries" in observation:
        parts = [f"{q['table']} {q['row_count']}行" for q in observation["queries"]]
        return "查询返回:" + ";".join(parts)
    if "results" in observation:
        count = len(observation["results"])
        if count == 0:
            return "未检索到相关研报"
        titles = "、".join(str(item.get("title", ""))[:20] for item in observation["results"][:3])
        return f"检索到 {count} 条研报片段:{titles}"
    if "result" in observation:
        return f"计算结果 = {observation['result']}"
    if "charts" in observation:
        return "已生成图表:" + "、".join(str(item.get("title", "")) for item in observation["charts"])
    return json.dumps(observation, ensure_ascii=False, default=str)[:200]
