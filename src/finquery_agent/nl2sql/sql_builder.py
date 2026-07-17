from __future__ import annotations

from dataclasses import dataclass


from finquery_agent.nl2sql.dsl import QueryDSL
from finquery_agent.nl2sql.validator import validate_select_sql
from finquery_agent.schema.metrics import resolve_metric_with_policy
from finquery_agent.schema.registry import SchemaRegistry, normalize_stock_code


class SQLBuildError(ValueError):
    pass


@dataclass(frozen=True)
class SQLQuery:
    sql: str
    params: dict[str, object]
    table_name: str
    metric_columns: tuple[str, ...]
    warnings: tuple[str, ...] = ()


class SQLBuilder:
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def build(self, dsl: QueryDSL) -> SQLQuery:
        if not dsl.metrics:
            raise SQLBuildError("至少需要一个查询指标")
        if not dsl.periods and not dsl.allow_all_periods:
            raise SQLBuildError("财务查询必须指定 report_period，或显式允许跨报告期查询")

        fields = []
        warnings = []
        for metric in dsl.metrics:
            resolution = resolve_metric_with_policy(self.registry, metric)
            field = resolution.default_field
            if field is None:
                raise SQLBuildError(f"无法识别指标: {metric}")
            if resolution.needs_clarification:
                raise SQLBuildError(resolution.explanation)
            if resolution.explanation != "指标可直接解析。":
                warnings.append(resolution.explanation)
            fields.append(field)

        table_names = {field.table_name for field in fields}
        if len(table_names) != 1:
            raise SQLBuildError("第一版 SQL builder 暂不支持跨财务表指标混查")
        table_name = next(iter(table_names))

        company_codes = self._resolve_company_codes(dsl)
        order_metric = dsl.order_by_metric or fields[0].name
        order_field = self.registry.resolve_metric(order_metric)
        select_columns = ["stock_code", "stock_abbr", "report_year", "report_period"]
        select_columns.extend(field.name for field in fields)

        where_clauses = []
        params: dict[str, object] = {}
        if company_codes:
            placeholders = []
            for index, code in enumerate(company_codes):
                key = f"stock_code_{index}"
                placeholders.append(f":{key}")
                params[key] = code
            where_clauses.append(f"stock_code IN ({', '.join(placeholders)})")
        if dsl.years:
            placeholders = []
            for index, year in enumerate(dsl.years):
                key = f"report_year_{index}"
                placeholders.append(f":{key}")
                params[key] = int(year)
            where_clauses.append(f"report_year IN ({', '.join(placeholders)})")
        if dsl.periods:
            placeholders = []
            for index, period in enumerate(dsl.periods):
                key = f"report_period_{index}"
                placeholders.append(f":{key}")
                params[key] = period.upper()
            where_clauses.append(f"report_period IN ({', '.join(placeholders)})")

        for index, metric_filter in enumerate(dsl.metric_filters):
            filter_field = self.registry.resolve_metric(metric_filter.metric)
            if filter_field is None:
                raise SQLBuildError(f"无法识别筛选指标: {metric_filter.metric}")
            if filter_field.table_name != table_name:
                raise SQLBuildError("筛选指标必须与查询指标属于同一张财务表")
            operator = _normalize_filter_operator(metric_filter.operator)
            key = f"metric_filter_{index}"
            where_clauses.append(f"{filter_field.name} {operator} :{key}")
            params[key] = metric_filter.value

        if dsl.intent == "ranking_query" and order_field and order_field.table_name == table_name:
            where_clauses.append(f"{order_field.name} IS NOT NULL")

        sql = f"SELECT {', '.join(select_columns)} FROM {table_name}"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        if order_field and order_field.table_name == table_name:
            direction = "ASC" if dsl.sort_direction.lower() == "asc" else "DESC"
            sql += f" ORDER BY report_year ASC, report_period ASC, {order_field.name} {direction} NULLS LAST"
        else:
            sql += " ORDER BY report_year ASC, report_period ASC, stock_code ASC"

        limit = max(1, min(int(dsl.limit), 500))
        sql += " LIMIT :limit"
        params["limit"] = limit

        validate_select_sql(sql, allowed_tables=set(self.registry.tables))
        return SQLQuery(
            sql=sql,
            params=params,
            table_name=table_name,
            metric_columns=tuple(field.name for field in fields),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def _resolve_company_codes(self, dsl: QueryDSL) -> tuple[str, ...]:
        codes = {normalize_stock_code(code) for code in dsl.company_codes if code}
        for name in dsl.company_names:
            code = self.registry.resolve_company_code(name)
            if code is None:
                raise SQLBuildError(f"无法识别公司: {name}")
            codes.add(code)
        return tuple(sorted(codes))


def _normalize_filter_operator(operator: str) -> str:
    allowed = {">", ">=", "<", "<=", "=", "!="}
    if operator not in allowed:
        raise SQLBuildError(f"不支持的筛选运算符: {operator}")
    return operator
