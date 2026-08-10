"""FinQuery MCP 服务:把 Agent 的四个受控能力开放给外部 MCP 客户端。

服务端直接复用内置 Agent 的 AgentToolbox,而不是另写一套查询逻辑,因此
指标白名单、公司实体解析、参数化 SQL、AST 受限求值这些安全约束对外部
客户端同样生效 —— MCP 只是换了一层传输协议,不是绕过校验的第二条数据
通道。工具描述也取自同一份定义,避免两处措辞漂移导致模型行为不一致。

与内置 Agent 的一个关键差异是 render_chart:Agent 内部版本引用上一步查询的
结果快照,而 MCP 请求之间没有可靠的会话归属(可能是 stateless HTTP,也
可能多客户端共用一个进程),因此这里把它改成自带取数参数的单次调用。

除工具外还提供两类只读上下文:
- Resource:公司名单与表结构。客户端在提问前先对齐可用实体,可显著减少
  因公司名、指标名臆造而产生的无效调用。
- Prompt:把"多家公司横向对比"这类需要固定调用顺序的套路沉淀为模板,
  由用户显式触发,而不是指望模型每次自行推导出正确的编排。
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError
from mcp.types import ToolAnnotations
from pydantic import Field

from finquery_agent.agent.tools import tool_schemas
from finquery_agent.mcp_server.deps import ServiceContainer

CONTAINER = ServiceContainer()

_DESCRIPTIONS: dict[str, str] = {
    schema["function"]["name"]: schema["function"]["description"] for schema in tool_schemas()
}

# 四个工具都不写库,声明只读可让客户端跳过逐次授权确认。
_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)

mcp = MCPServer(
    name="finquery",
    title="FinQuery 财报问答",
    version="0.1.0",
    instructions=(
        "FinQuery 提供 76 家医药类上市公司 2022-2025 年的财务数据与券商研报知识库。"
        "回答涉及具体公司前,先读取 finquery://companies 确认公司在库内;"
        "需要精确数字用 query_financial_data,需要观点与定性判断用 search_research_reports,"
        "任何衍生指标(增长率、占比、中位数)一律用 calculate 计算,不要心算。"
    ),
)


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------


@mcp.tool(
    name="query_financial_data",
    title="查询财务数据",
    description=_DESCRIPTIONS["query_financial_data"],
    annotations=_READ_ONLY,
)
async def query_financial_data(
    metrics: Annotated[list[str], Field(description="财务指标中文名,如 营业总收入、净利润、研发费用,最多 4 个")],
    years: Annotated[list[int], Field(description="年份,如 [2024, 2025]")],
    periods: Annotated[list[Literal["FY", "Q1", "HY", "Q3"]], Field(description="报告期:FY 年报、Q1 一季报、HY 半年报、Q3 三季报")],
    companies: Annotated[list[str] | None, Field(description="公司简称或股票代码;排名/筛选类问题留空表示全库")] = None,
    limit: Annotated[int, Field(ge=1, le=200, description="返回行数上限,排名题常用 3/5/10")] = 100,
    order_by_metric: Annotated[str | None, Field(description="排序依据指标,必须出现在 metrics 中")] = None,
    sort_direction: Annotated[Literal["desc", "asc"], Field(description="排序方向")] = "desc",
) -> dict[str, Any]:
    return await _execute(
        "query_financial_data",
        {
            "metrics": metrics,
            "years": years,
            "periods": periods,
            "companies": companies or [],
            "limit": limit,
            "order_by_metric": order_by_metric,
            "sort_direction": sort_direction,
        },
    )


@mcp.tool(
    name="search_research_reports",
    title="检索券商研报",
    description=_DESCRIPTIONS["search_research_reports"],
    annotations=_READ_ONLY,
)
async def search_research_reports(
    query: Annotated[str, Field(description="检索问题,中文自然语言")],
    stock_codes: Annotated[list[str] | None, Field(description="只看这些公司的研报(简称或代码);跨公司、行业类问题留空")] = None,
    report_type: Annotated[Literal["stock", "industry"] | None, Field(description="限定研报类型;不确定时留空")] = None,
    top_k: Annotated[int, Field(ge=1, le=10, description="返回片段数")] = 5,
) -> dict[str, Any]:
    return await _execute(
        "search_research_reports",
        {
            "query": query,
            "stock_codes": stock_codes or [],
            "report_type": report_type,
            "top_k": top_k,
        },
    )


@mcp.tool(
    name="calculate",
    title="安全计算",
    description=_DESCRIPTIONS["calculate"],
    annotations=_READ_ONLY,
)
async def calculate(

    expression: Annotated[str, Field(description="数学表达式,变量用字母名,如 (a-b)/b*100")],
    variables: Annotated[
        dict[str, float | list[float]] | None,
        Field(description="变量名到数值的映射;数组可配合 median/mean/sum 使用"),
    ] = None,
) -> dict[str, Any]:
    return await _execute("calculate", {"expression": expression, "variables": variables or {}})


@mcp.tool(
    name="render_chart",
    title="查询并渲染图表",
    description=(
        "把财务数据渲染成折线图或柱状图,用于展示趋势与对比。"
        "取数参数与 query_financial_data 一致,本工具内部完成取数与绘图,"
        "无需先调用 query_financial_data。返回 SVG 源码与文字描述。"
    ),
    annotations=_READ_ONLY,
)
async def render_chart(
    chart_type: Annotated[Literal["line", "bar"], Field(description="line=趋势,bar=对比")],
    metrics: Annotated[list[str], Field(description="财务指标中文名,最多 4 个")],
    years: Annotated[list[int], Field(description="年份,如 [2022, 2023, 2024]")],
    periods: Annotated[list[Literal["FY", "Q1", "HY", "Q3"]], Field(description="报告期")],
    companies: Annotated[list[str] | None, Field(description="公司简称或股票代码")] = None,
    title: Annotated[str | None, Field(description="图表标题,可选")] = None,
) -> dict[str, Any]:
    def run() -> dict[str, Any]:
        toolbox = CONTAINER.new_toolbox()
        query_result = toolbox.execute(
            "query_financial_data",
            {
                "metrics": metrics,
                "years": years,
                "periods": periods,
                "companies": companies or [],
                "limit": 200,
            },
        )
        if "error" in query_result:
            return query_result
        result = toolbox.execute("render_chart", {"chart_type": chart_type, "title": title})
        if "error" in result:
            return result
        return {
            "charts": [
                {
                    "chart_type": chart["chart_type"],
                    "title": chart["title"],
                    "x_axis_label": chart["x_axis_label"],
                    "y_axis_label": chart["y_axis_label"],
                    "alt_text": chart["alt_text"],
                    "svg": _decode_svg(chart["image_data_url"]),
                }
                for chart in toolbox.collected_charts
            ]
        }

    return await anyio.to_thread.run_sync(run)


def _decode_svg(image_data_url: str) -> str:
    """图表以 data URL 形式产出,这里还原为 SVG 源码,省去客户端二次解码。"""
    _, _, encoded = image_data_url.partition("base64,")
    return base64.b64decode(encoded).decode("utf-8")


async def _execute(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """在工作线程内执行同步工具,避免数据库与向量检索阻塞事件循环。"""
    toolbox = CONTAINER.new_toolbox()
    return await anyio.to_thread.run_sync(toolbox.execute, tool_name, arguments)


# ----------------------------------------------------------------------
# 资源
# ----------------------------------------------------------------------


@mcp.resource(
    "finquery://companies",
    name="公司名单",
    description="财务库覆盖的全部上市公司(股票代码、简称、全称)。提问涉及具体公司前应先核对。",
    mime_type="text/csv",
)
def list_companies() -> str:
    registry = CONTAINER.registry
    lines = ["stock_code,stock_abbr,company_name"]
    for company in sorted(registry.companies.values(), key=lambda item: item.stock_code):
        lines.append(f"{company.stock_code},{company.stock_abbr},{company.company_name}")
    return "\n".join(lines)


@mcp.resource(
    "finquery://schema",
    name="数据表清单",
    description="财务库的表名与中文含义,用于挑选 finquery://schema/{table} 进一步查看字段。",
    mime_type="text/plain",
)
def list_tables() -> str:
    registry = CONTAINER.registry
    return "\n".join(f"{name}\t{table.chinese_name}" for name, table in registry.tables.items())


@mcp.resource(
    "finquery://schema/{table}",
    name="数据表字段说明",
    description="指定数据表的字段名、中文名与单位,用于确认指标是否可查。",
    mime_type="text/plain",
)
def get_table_schema(table: str) -> str:
    registry = CONTAINER.registry
    definition = registry.tables.get(table)
    if definition is None:
        raise ResourceError(f"未知数据表:{table}。可用表:{sorted(registry.tables)}")
    lines = [f"# {definition.name}({definition.chinese_name})", "field\tchinese_name\tunit\tdescription"]
    for item in definition.fields:
        lines.append(f"{item.name}\t{item.chinese_name}\t{item.unit or '-'}\t{item.description}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# 提示词
# ----------------------------------------------------------------------


@mcp.prompt(
    name="compare_companies",
    title="横向对比两家公司",
    description="生成一份对比两家公司同期财务表现的分析指令,固定先取数、再计算、后归因的顺序。",
)
def compare_companies(
    company_a: Annotated[str, Field(description="公司甲的简称或股票代码")],
    company_b: Annotated[str, Field(description="公司乙的简称或股票代码")],
    year: Annotated[int, Field(description="对比年份")],
    period: Annotated[Literal["FY", "Q1", "HY", "Q3"], Field(description="报告期")] = "FY",
) -> str:
    return (
        f"对比 {company_a} 与 {company_b} 在 {year} 年 {period} 的经营表现,按以下步骤执行:\n"
        f"1. 调用 query_financial_data 一次性取回两家公司的营业总收入、净利润、研发费用;\n"
        f"2. 调用 calculate 计算净利率与研发费用率,不要心算;\n"
        f"3. 调用 search_research_reports 各检索一次,补充差异背后的业务原因;\n"
        f"4. 给出三条结论,每条都要标注数据来自哪家公司的哪个报告期,观点类表述须标注研报出处。"
    )


@mcp.prompt(
    name="analyze_trend",
    title="分析单指标多年趋势",
    description="生成一份单公司指标趋势分析指令,包含取数、增长率计算与配图。",
)
def analyze_trend(
    company: Annotated[str, Field(description="公司简称或股票代码")],
    metric: Annotated[str, Field(description="财务指标中文名,如 营业总收入")],
    start_year: Annotated[int, Field(description="起始年份")] = 2022,
    end_year: Annotated[int, Field(description="结束年份")] = 2025,
) -> str:
    return (
        f"分析 {company} 在 {start_year}-{end_year} 年 {metric} 的变化趋势,按以下步骤执行:\n"
        f"1. 调用 query_financial_data 取回该区间各年年报(FY)的 {metric};\n"
        f"2. 调用 calculate 逐年计算同比增长率;\n"
        f"3. 调用 render_chart 生成同参数的折线图;\n"
        f"4. 用不超过 200 字说明趋势拐点,并注明缺失年份(若有)。"
    )
