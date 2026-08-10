"""MCP 服务的依赖装配。

工具在这里被设计成无状态的:MCP 服务端可能以 stateless HTTP 部署或被负载
均衡到多个实例,请求之间没有可靠的会话归属;同一进程同时服务多个客户端时,
任何跨调用的服务端记忆还会把一个客户端的数据泄露给另一个。因此进程内共享的
只有只读依赖,每次调用现取一个工具箱承载本次请求的中间产物。

RAG 的向量模型与索引加载需要数十秒,而 MCP 服务由客户端按需拉起,启动耗时
会直接表现为握手延迟,故推迟到首次检索时才构建。
"""

from __future__ import annotations

import threading
from typing import Any

from finquery_agent.agent.tools import AgentToolbox
from finquery_agent.config import load_llm_settings, load_rag_settings
from finquery_agent.db import create_database_engine
from finquery_agent.nl2sql.charting import ChartRenderer
from finquery_agent.nl2sql.executor import QueryExecutor
from finquery_agent.nl2sql.sql_builder import SQLBuilder
from finquery_agent.rag.service import RAGService
from finquery_agent.schema import load_default_registry
from finquery_agent.schema.registry import SchemaRegistry


class LazyRAGService:
    """把 RAGService 的构建推迟到首次检索。

    AgentToolbox 只经 search 使用 RAG,转发该方法即可满足工具契约。
    """

    def __init__(self) -> None:
        self._service: RAGService | None = None
        self._lock = threading.Lock()

    def search(self, *args: Any, **kwargs: Any) -> Any:
        return self.resolve().search(*args, **kwargs)

    def resolve(self) -> RAGService:
        if self._service is not None:
            return self._service
        with self._lock:
            if self._service is None:
                self._service = RAGService(load_rag_settings(), load_llm_settings())
            return self._service


class ServiceContainer:
    """进程内共享的只读依赖,以及按调用产出的一次性工具箱。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registry: SchemaRegistry | None = None
        self._sql_builder: SQLBuilder | None = None
        self._query_executor: QueryExecutor | None = None
        self._chart_renderer: ChartRenderer | None = None
        self._rag_service = LazyRAGService()

    @property
    def registry(self) -> SchemaRegistry:
        self._ensure_shared()
        assert self._registry is not None
        return self._registry

    def _ensure_shared(self) -> None:
        if self._registry is not None:
            return
        with self._lock:
            if self._registry is not None:
                return
            registry = load_default_registry()
            self._sql_builder = SQLBuilder(registry)
            self._query_executor = QueryExecutor(create_database_engine(), registry)
            self._chart_renderer = ChartRenderer(registry)
            self._registry = registry

    def new_toolbox(self) -> AgentToolbox:
        self._ensure_shared()
        assert self._registry is not None and self._sql_builder is not None
        assert self._query_executor is not None and self._chart_renderer is not None
        return AgentToolbox(
            registry=self._registry,
            sql_builder=self._sql_builder,
            query_executor=self._query_executor,
            rag_service=self._rag_service,  # type: ignore[arg-type]
            chart_renderer=self._chart_renderer,
        )
