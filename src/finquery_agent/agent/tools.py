"""Agent 工具集:LLM 可调用的四个受控能力。

工具是 LLM 与数据之间唯一的通道,每个工具内部沿用既有安全链路:
- query_financial_data:槽位白名单校验 → DSL → SQLBuilder(参数化)→ 执行;
- search_research_reports:混合检索 → 元数据过滤 → 精排 → 配额;
- calculate:AST 白名单求值,不使用 eval;
- render_chart:引用最近一次财务查询的结果绘图,数据不经 LLM 转手,
  避免模型在传参时篡改/幻觉数值。

工具执行永不抛异常:参数不合法时返回 {"error": ...} 文本给 LLM,
让它有机会修正参数重试 —— 错误信息本身就是 Agent 自愈的依据。
返回给 LLM 的结果做行数/长度截断,防止大结果撑爆上下文。
"""

from __future__ import annotations

import ast
import json
import statistics
from typing import Any, Callable

from finquery_agent.nl2sql.charting import ChartRenderer
from finquery_agent.nl2sql.intent import ChartSpec, StructuredIntent
from finquery_agent.nl2sql.planner import split_intent_by_table
from finquery_agent.nl2sql.sql_builder import SQLBuildError, SQLBuilder
from finquery_agent.rag.models import make_snippet
from finquery_agent.rag.service import RAGService
from finquery_agent.schema.metrics import resolve_metric_with_policy
from finquery_agent.schema.registry import SchemaRegistry

VALID_PERIODS = {"FY", "Q1", "HY", "Q3"}
MAX_ROWS_TO_LLM = 20


class AgentToolbox:
    """工具注册表 + 执行器。保存最近一次财务查询供 render_chart 引用。"""

    def __init__(
        self,
        registry: SchemaRegistry,
        sql_builder: SQLBuilder,
        query_executor: Any,
        rag_service: RAGService,
        chart_renderer: ChartRenderer,
    ):
        self.registry = registry
        self.sql_builder = sql_builder
        self.query_executor = query_executor
        self.rag_service = rag_service
        self.chart_renderer = chart_renderer
        # 跨工具共享的运行期产物:查询明细给最终回答用,图表给响应用。
        self.last_queries: list[dict[str, Any]] = []
        self.collected_sources: list[dict[str, Any]] = []
        self.collected_charts: list[dict[str, Any]] = []
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "query_financial_data": self._query_financial_data,
            "search_research_reports": self._search_research_reports,
            "calculate": self._calculate,
            "render_chart": self._render_chart,
        }

    def reset(self) -> None:
        self.last_queries = []
        self.collected_sources = []
        self.collected_charts = []

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return {"error": f"未知工具:{name}"}
        try:
            return handler(arguments)
        except Exception as exc:  # 工具内部异常折叠为错误消息,循环不中断。
            return {"error": f"工具执行失败:{exc}"}

    # ------------------------------------------------------------------
    # 工具 1:财务数据查询
    # ------------------------------------------------------------------

    def _query_financial_data(self, args: dict[str, Any]) -> dict[str, Any]:
        metrics: list[str] = []
        rejected: list[str] = []
        for raw in _str_list(args.get("metrics"))[:4]:
            resolution = resolve_metric_with_policy(self.registry, raw)
            if resolution.default_field is None:
                rejected.append(raw)
            else:
                metrics.append(raw)
        if not metrics:
            return {"error": f"没有可识别的指标:{rejected}。请使用财务指标的规范中文名。"}

        company_codes: list[str] = []
        unknown_companies: list[str] = []
        for raw in _str_list(args.get("companies")):
            code = self.registry.resolve_company_code(raw)
            if code is None:
                unknown_companies.append(raw)
            elif code not in company_codes:
                company_codes.append(code)
        if unknown_companies:
            return {"error": f"公司不在数据库中:{unknown_companies}。数据库仅含 76 家医药类上市公司。"}

        years = sorted({int(y) for y in args.get("years") or [] if str(y).isdigit() and 1990 <= int(y) <= 2100})
        periods = [p for p in _str_list(args.get("periods")) if p in VALID_PERIODS]
        if not years:
            return {"error": "缺少 years 参数(如 [2025])。"}
        if not periods:
            return {"error": "缺少 periods 参数(FY/Q1/HY/Q3)。"}

        limit = max(1, min(int(args.get("limit") or 100), 200))
        sort_direction = args.get("sort_direction") if args.get("sort_direction") in {"asc", "desc"} else "desc"
        order_by = args.get("order_by_metric") if args.get("order_by_metric") in metrics else None
        intent_type = "metric_query" if company_codes else "ranking_query"
        if intent_type == "ranking_query" and order_by is None:
            order_by = metrics[0]

        intent = StructuredIntent(
            original_question=json.dumps(args, ensure_ascii=False),
            intent_type=intent_type,
            metrics=tuple(metrics),
            company_codes=tuple(company_codes),
            years=tuple(years),
            periods=tuple(periods),
            limit=limit,
            order_by_metric=order_by,
            sort_direction=str(sort_direction),
            intent_source="agent",
        )
        try:
            sub_intents = split_intent_by_table(intent, self.registry)
        except Exception as exc:
            return {"error": f"查询规划失败:{exc}"}

        payload_queries = []
        combined: list[dict[str, Any]] = []
        for sub_intent in sub_intents:
            try:
                query = self.sql_builder.build(sub_intent.to_dsl())
            except SQLBuildError as exc:
                return {"error": f"SQL 构建失败:{exc}"}
            result = self.query_executor.execute(query).to_dict()
            combined.append(
                {
                    "intent": sub_intent.to_dict(),
                    "sql": query.sql,
                    "params": query.params,
                    "table_name": query.table_name,
                    "metric_columns": query.metric_columns,
                    "warnings": query.warnings,
                    "result": result,
                }
            )
            rows = result.get("rows") or []
            payload_queries.append(
                {
                    "table": query.table_name,
                    "metric_columns": list(query.metric_columns),
                    "units": result.get("units") or {},
                    "row_count": len(rows),
                    "rows": rows[:MAX_ROWS_TO_LLM],
                    "truncated": len(rows) > MAX_ROWS_TO_LLM,
                }
            )
        self.last_queries = combined
        return {"queries": payload_queries}

    # ------------------------------------------------------------------
    # 工具 2:研报检索
    # ------------------------------------------------------------------

    def _search_research_reports(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return {"error": "缺少 query 参数。"}
        stock_codes = []
        for raw in _str_list(args.get("stock_codes")):
            code = self.registry.resolve_company_code(raw)
            if code:
                stock_codes.append(code)
        report_type = args.get("report_type") if args.get("report_type") in {"stock", "industry"} else None
        top_k = max(1, min(int(args.get("top_k") or 5), 10))
        results = self.rag_service.search(query, top_k=top_k, stock_codes=tuple(stock_codes), report_type=report_type)
        evidence = []
        for index, result in enumerate(results, start=len(self.collected_sources) + 1):
            evidence.append(
                {
                    "source_index": index,
                    "title": result.chunk.title,
                    "stock_name": result.chunk.stock_name,
                    "org_name": result.chunk.org_name,
                    "publish_date": result.chunk.publish_date,
                    "section": result.chunk.section_title,
                    "snippet": make_snippet(result.chunk.text, max_chars=400),
                }
            )
        self.collected_sources.extend(result.to_dict() for result in results)
        if not evidence:
            return {"results": [], "note": "未检索到相关研报,可尝试更换关键词或去掉过滤条件。"}
        return {"results": evidence}

    # ------------------------------------------------------------------
    # 工具 3:安全计算
    # ------------------------------------------------------------------

    def _calculate(self, args: dict[str, Any]) -> dict[str, Any]:
        expression = str(args.get("expression") or "").strip()
        if not expression:
            return {"error": "缺少 expression 参数。"}
        variables = args.get("variables") or {}
        if not isinstance(variables, dict):
            return {"error": "variables 必须是对象,如 {\"a\": 1.5}。"}
        try:
            value = _safe_eval(expression, {str(k): v for k, v in variables.items()})
        except Exception as exc:
            return {"error": f"表达式求值失败:{exc}"}
        return {"result": value}

    # ------------------------------------------------------------------
    # 工具 4:图表渲染(引用最近一次查询结果,数据不经 LLM)
    # ------------------------------------------------------------------

    def _render_chart(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.last_queries:
            return {"error": "尚无可绘图的数据:请先调用 query_financial_data。"}
        chart_type = args.get("chart_type") if args.get("chart_type") in {"line", "bar"} else "line"
        title = str(args.get("title") or "") or None
        base_intent: dict[str, Any] = self.last_queries[0].get("intent") or {}
        intent = StructuredIntent(
            original_question="agent-chart",
            intent_type=str(base_intent.get("intent_type") or "metric_query"),
            metrics=tuple(base_intent.get("metrics") or ()),
            company_codes=tuple(base_intent.get("company_codes") or ()),
            years=tuple(base_intent.get("years") or ()),
            periods=tuple(base_intent.get("periods") or ()),
            chart=ChartSpec(chart_type=str(chart_type), x="report_year", y=None, title=title),
            intent_source="agent",
        )
        images = self.chart_renderer.render_all(intent, self.last_queries)
        if not images:
            return {"error": "查询结果中没有可绘制的数值数据。"}
        rendered = [image.to_dict() for image in images]
        self.collected_charts.extend(rendered)
        return {"charts": [{"title": item["title"], "chart_type": item["chart_type"]} for item in rendered]}


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI function-calling 格式的工具描述。描述面向 LLM 编写:
    说明什么时候用、参数语义与取值约定(单位、枚举),这直接影响调用质量。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "query_financial_data",
                "description": (
                    "查询上市公司财务数据库(76 家医药类公司,2022-2025 年,报告期 FY/Q1/HY/Q3)。"
                    "用于获取精确财务数字。不指定 companies 时在全部公司中排序/筛选(排名场景)。"
                    "金额单位为万元。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metrics": {"type": "array", "items": {"type": "string"}, "description": "财务指标中文名,如 营业总收入、净利润、研发费用,最多4个"},
                        "companies": {"type": "array", "items": {"type": "string"}, "description": "公司简称或股票代码;排名/筛选题留空"},
                        "years": {"type": "array", "items": {"type": "integer"}, "description": "年份,如 [2024, 2025]"},
                        "periods": {"type": "array", "items": {"type": "string", "enum": ["FY", "Q1", "HY", "Q3"]}, "description": "报告期"},
                        "limit": {"type": "integer", "description": "返回行数上限,排名题常用 3/5/10"},
                        "order_by_metric": {"type": "string", "description": "排序依据指标(须在 metrics 中)"},
                        "sort_direction": {"type": "string", "enum": ["desc", "asc"]},
                    },
                    "required": ["metrics", "years", "periods"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_research_reports",
                "description": "检索券商研报知识库(97 篇个股研报 + 67 篇行业研报)。用于获取分析观点、行业判断、业务驱动因素等定性证据。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "检索问题,中文自然语言"},
                        "stock_codes": {"type": "array", "items": {"type": "string"}, "description": "只看这些公司的研报(简称或代码);跨公司/行业问题留空"},
                        "report_type": {"type": "string", "enum": ["stock", "industry"], "description": "限定研报类型;不确定时留空"},
                        "top_k": {"type": "integer", "description": "返回片段数,默认 5,最大 10"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": (
                    "数值计算器。支持四则运算、比较、min/max/abs/round/sum/median/mean。"
                    "用于增长率、比值、占比、中位数等衍生计算,避免心算错误。"
                    "示例:expression='(a-b)/b*100', variables={'a': 120, 'b': 100}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式,变量用字母名"},
                        "variables": {"type": "object", "description": "变量名到数值的映射,数组可用于 median/mean/sum"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "render_chart",
                "description": "把最近一次 query_financial_data 的结果渲染成图表(自动取查询数据,无需传数值)。需要展示趋势/对比时使用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chart_type": {"type": "string", "enum": ["line", "bar"], "description": "line=趋势,bar=对比"},
                        "title": {"type": "string", "description": "图表标题,可选"},
                    },
                    "required": ["chart_type"],
                },
            },
        },
    ]


# ----------------------------------------------------------------------
# 受限表达式求值
# ----------------------------------------------------------------------

_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "sum": sum,
    "median": statistics.median,
    "mean": statistics.mean,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Compare,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.Load,
)


def _safe_eval(expression: str, variables: dict[str, Any]) -> Any:
    """AST 白名单求值:只允许算术/比较/白名单函数,杜绝任意代码执行。"""
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"不允许的表达式节点:{type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("只允许调用 min/max/abs/round/sum/median/mean")
    return _eval_node(tree.body, variables)


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只允许数值常量")
    if isinstance(node, ast.Name):
        if node.id in variables:
            return _as_number(variables[node.id])
        raise ValueError(f"未定义变量:{node.id}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(item, variables) for item in node.elts]
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, variables)
        return -operand if isinstance(node.op, ast.USub) else +operand
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        ops = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right, ast.Div: lambda: left / right, ast.Pow: lambda: left**right, ast.Mod: lambda: left % right}
        for op_type, func in ops.items():
            if isinstance(node.op, op_type):
                return func()
        raise ValueError("不支持的运算符")
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, variables)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, variables)
            checks = {ast.Lt: left < right, ast.LtE: left <= right, ast.Gt: left > right, ast.GtE: left >= right, ast.Eq: left == right, ast.NotEq: left != right}
            result = result and checks[type(op)]
            left = right
        return result
    if isinstance(node, ast.Call):
        func = _ALLOWED_FUNCS[node.func.id]  # 白名单已在 _safe_eval 校验
        args = [_eval_node(arg, variables) for arg in node.args]
        return func(*args)
    raise ValueError(f"不允许的表达式节点:{type(node).__name__}")


def _as_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_as_number(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"变量值不是数值:{value!r}")


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
