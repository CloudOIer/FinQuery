"""MCP 服务的注册契约、无状态语义与安全边界测试(不依赖真实数据库与 RAG)。"""

from __future__ import annotations

import json

import anyio
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from finquery_agent.mcp_server import deps as deps_module  # noqa: E402
from finquery_agent.mcp_server import server as server_module  # noqa: E402
from finquery_agent.nl2sql.charting import ChartRenderer  # noqa: E402
from finquery_agent.nl2sql.executor import QueryResult  # noqa: E402
from finquery_agent.nl2sql.sql_builder import SQLBuilder  # noqa: E402
from finquery_agent.schema import load_default_registry  # noqa: E402

REGISTRY = load_default_registry()


class FakeExecutor:
    def execute(self, query):
        rows = [
            {
                "stock_code": "600332",
                "report_year": 2023,
                "report_period": "FY",
                "total_operating_revenue": 100.0,
            },
            {
                "stock_code": "600332",
                "report_year": 2024,
                "report_period": "FY",
                "total_operating_revenue": 120.0,
            },
        ]
        return QueryResult(columns=tuple(rows[0]), rows=tuple(rows), units={"total_operating_revenue": "万元"}, row_count=len(rows))


class FakeRAG:
    def search(self, question, top_k=None, **kwargs):
        return []


@pytest.fixture(autouse=True)
def container(monkeypatch):
    instance = deps_module.ServiceContainer()
    instance._registry = REGISTRY
    instance._sql_builder = SQLBuilder(REGISTRY)
    instance._query_executor = FakeExecutor()
    instance._chart_renderer = ChartRenderer(REGISTRY)
    instance._rag_service = FakeRAG()
    monkeypatch.setattr(server_module, "CONTAINER", instance)
    return instance


def _run(coro_factory):
    return anyio.run(coro_factory)


def _payload(result):
    return json.loads(result.content[0].text)


def test_registers_four_tools_with_shared_descriptions():
    async def main():
        async with Client(server_module.mcp) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in _run(main).tools}
    assert set(tools) == {"query_financial_data", "search_research_reports", "calculate", "render_chart"}
    assert tools["query_financial_data"].input_schema["required"] == ["metrics", "years", "periods"]
    # 描述取自 Agent 的同一份工具定义,避免两处措辞漂移。
    assert "76 家医药类公司" in tools["query_financial_data"].description


def test_exposes_context_resources_and_prompts():
    async def main():
        async with Client(server_module.mcp) as client:
            resources = await client.list_resources()
            templates = await client.list_resource_templates()
            prompts = await client.list_prompts()
            companies = await client.read_resource("finquery://companies")
            schema = await client.read_resource("finquery://schema/income_sheet")
            return resources, templates, prompts, companies, schema

    resources, templates, prompts, companies, schema = _run(main)
    assert {str(item.uri) for item in resources.resources} == {"finquery://companies", "finquery://schema"}
    assert {item.uri_template for item in templates.resource_templates} == {"finquery://schema/{table}"}
    assert {item.name for item in prompts.prompts} == {"compare_companies", "analyze_trend"}
    assert companies.contents[0].text.startswith("stock_code,stock_abbr,company_name")
    assert "income_sheet" in schema.contents[0].text


def test_unknown_table_resource_is_rejected():
    async def main():
        async with Client(server_module.mcp) as client:
            return await client.read_resource("finquery://schema/pg_catalog")

    with pytest.raises(Exception):
        _run(main)


def test_calculate_rejects_non_whitelisted_calls():
    async def main():
        async with Client(server_module.mcp) as client:
            ok = await client.call_tool("calculate", {"expression": "(a-b)/b*100", "variables": {"a": 120, "b": 100}})
            blocked = await client.call_tool("calculate", {"expression": "__import__('os').system('id')"})
            return ok, blocked

    ok, blocked = _run(main)
    assert _payload(ok)["result"] == pytest.approx(20.0)
    assert "只允许调用" in _payload(blocked)["error"]


def test_unknown_metric_is_rejected_before_sql():
    async def main():
        async with Client(server_module.mcp) as client:
            return await client.call_tool(
                "query_financial_data",
                {"metrics": ["'; DROP TABLE income_sheet; --"], "years": [2024], "periods": ["FY"]},
            )

    assert "没有可识别的指标" in _payload(_run(main))["error"]


def test_chart_tool_takes_its_own_query_parameters():
    async def main():
        async with Client(server_module.mcp) as client:
            return await client.call_tool(
                "render_chart",
                {
                    "chart_type": "line",
                    "metrics": ["营业总收入"],
                    "companies": ["600332"],
                    "years": [2023, 2024],
                    "periods": ["FY"],
                    "title": "营收趋势",
                },
            )

    chart = _payload(_run(main))["charts"][0]
    assert chart["title"] == "营收趋势"
    assert chart["svg"].startswith("<svg")


def test_chart_reports_unknown_company_instead_of_drawing():
    async def main():
        async with Client(server_module.mcp) as client:
            return await client.call_tool(
                "render_chart",
                {
                    "chart_type": "line",
                    "metrics": ["营业总收入"],
                    "companies": ["不存在的公司"],
                    "years": [2024],
                    "periods": ["FY"],
                },
            )

    assert "公司不在数据库中" in _payload(_run(main))["error"]
