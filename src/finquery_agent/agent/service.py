"""AgentService:Agent 模式入口与降级控制。

输出与 AnalysisService 相同的 AnalysisResult:对 API 层与前端而言,
Agent 只是同一问答能力的另一种执行策略,响应结构不变(多一个 execution_trace)。

降级链:LLM 不可用 / Agent 循环失败(AgentPlanError)→ 原单轮固定管道
(AnalysisService.ask)。answer_source 标记 agent_llm / 降级后的原有取值,
调用方与评测都能区分答案由哪条链路产生。
"""

from __future__ import annotations

from typing import Any

from finquery_agent.agent.planner import AgentPlanError, AgentPlanner
from finquery_agent.agent.tools import AgentToolbox
from finquery_agent.analysis.service import AnalysisResult, AnalysisService
from finquery_agent.config import LLMSettings
from finquery_agent.llm import LLMClient


class AgentService:
    def __init__(
        self,
        analysis_service: AnalysisService,
        llm_settings: LLMSettings,
        max_steps: int = 8,
    ):
        self.analysis_service = analysis_service
        self.llm_settings = llm_settings
        self._client = LLMClient(llm_settings)
        toolbox = AgentToolbox(
            registry=analysis_service.registry,
            sql_builder=analysis_service.sql_builder,
            query_executor=analysis_service.query_executor,
            rag_service=analysis_service.rag_service,
            chart_renderer=analysis_service.chart_renderer,
        )
        self.planner = AgentPlanner(self._client, toolbox, max_steps=max_steps)

    def ask(self, question: str, **analysis_kwargs: Any) -> AnalysisResult:
        """Agent 优先;失败降级到单轮管道(参数原样透传)。"""
        if not self._client.is_available():
            return self.analysis_service.ask(question, **analysis_kwargs)
        try:
            run = self.planner.run(question)
        except AgentPlanError:
            return self.analysis_service.ask(question, **analysis_kwargs)

        financial: dict[str, Any] | None = None
        if run.financial_queries:
            financial = {"status": "answer", "queries": run.financial_queries}
        return AnalysisResult(
            status="answer",
            answer_text=run.answer_text,
            answer_source="agent_llm",
            llm_used=True,
            financial=financial,
            rag={"count": len(run.sources), "sources": run.sources},
            sources=run.sources,
            chart_images=run.chart_images,
            warnings=() if run.completed else ("已达到最大工具调用步数,回答基于已获取的部分信息。",),
            execution_trace=run.steps,
        )
