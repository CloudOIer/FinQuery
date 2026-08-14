"""AgentService:Agent 模式入口与降级控制。

输出与 AnalysisService 相同的 AnalysisResult:对 API 层与前端而言,
Agent 只是同一问答能力的另一种执行策略,响应结构不变(多一个 execution_trace)。

降级链:LLM 不可用 / Agent 循环失败(AgentPlanError)→ 原单轮固定管道
(AnalysisService.ask)。answer_source 标记 agent_llm / 降级后的原有取值,
调用方与评测都能区分答案由哪条链路产生。

两种执行引擎可切换,响应结构等价:loop 为自研 function-calling 循环(默认),
graph 为 LangGraph 状态图,后者额外支持多轮追问与执行过程流式推送。

工具箱按请求新建而非进程内长期持有:它记录着最近一次查询的完整结果供图表工具引用,
共享一份时并发请求会互相覆盖,导致图表画出另一个请求的数据、研报来源串号。
"""

from __future__ import annotations

from typing import Any, Iterator

from finquery_agent.agent.planner import AgentPlanError, AgentPlanner, AgentRunResult
from finquery_agent.agent.tools import AgentToolbox
from finquery_agent.analysis.service import AnalysisResult, AnalysisService
from finquery_agent.config import LLMSettings
from finquery_agent.llm import LLMClient

MAX_STEPS_WARNING = "已达到最大工具调用步数,回答基于已获取的部分信息。"


class AgentService:
    def __init__(
        self,
        analysis_service: AnalysisService,
        llm_settings: LLMSettings,
        max_steps: int = 8,
        engine: str = "loop",
    ):
        self.analysis_service = analysis_service
        self.llm_settings = llm_settings
        self.max_steps = max_steps
        self.engine = engine
        self._client = LLMClient(llm_settings)
        self._runner: Any = None

    def new_toolbox(self) -> AgentToolbox:
        analysis = self.analysis_service
        return AgentToolbox(
            registry=analysis.registry,
            sql_builder=analysis.sql_builder,
            query_executor=analysis.query_executor,
            rag_service=analysis.rag_service,
            chart_renderer=analysis.chart_renderer,
        )

    def ask(
        self,
        question: str,
        session_id: str | None = None,
        engine: str | None = None,
        **analysis_kwargs: Any,
    ) -> AnalysisResult:
        """Agent 优先;失败降级到单轮管道。

        session_id 既是状态图的会话标识,也是降级后单轮管道的会话参数,两边都要给。
        """
        if not self._client.is_available():
            return self.analysis_service.ask(question, session_id=session_id, **analysis_kwargs)
        try:
            run = self._execute(question, session_id, engine or self.engine)
        except AgentPlanError:
            return self.analysis_service.ask(question, session_id=session_id, **analysis_kwargs)
        return self.to_result(run)

    def stream(self, question: str, session_id: str | None = None) -> Iterator[dict[str, Any]]:
        """逐步推送执行轨迹,末尾推送完整结果;仅状态图引擎支持。"""
        if not self._client.is_available():
            raise AgentPlanError("LLM 不可用。")
        for event in self._graph_runner().stream(question, session_id):
            if event.get("event") == "result":
                yield {"event": "result", "result": self.to_result(event["result"]).to_dict()}
            else:
                yield event

    def to_result(self, run: AgentRunResult) -> AnalysisResult:
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
            warnings=() if run.completed else (MAX_STEPS_WARNING,),
            execution_trace=run.steps,
        )

    def _execute(self, question: str, session_id: str | None, engine: str) -> AgentRunResult:
        if engine == "graph":
            return self._graph_runner().run(question, session_id)
        # 自研循环:每次新建工具箱与规划器,循环实现本身保持不变。
        return AgentPlanner(self._client, self.new_toolbox(), max_steps=self.max_steps).run(question)

    def _graph_runner(self) -> Any:
        if self._runner is None:
            from finquery_agent.agent.graph import GraphAgentRunner

            self._runner = GraphAgentRunner(
                client=self._client,
                toolbox_factory=self.new_toolbox,
                max_steps=self.max_steps,
            )
        return self._runner

