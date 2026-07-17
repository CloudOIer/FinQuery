from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from finquery_agent.nl2sql.dsl import MetricFilter
from finquery_agent.nl2sql.intent import ChartSpec, ClarificationRequest, StructuredIntent
from finquery_agent.schema.metrics import resolve_metric_with_policy
from finquery_agent.schema.registry import SchemaRegistry, normalize_text


@dataclass(frozen=True)
class TimeParseResult:
    years: tuple[int, ...]
    periods: tuple[str, ...]
    allow_all_periods: bool = False


def build_clarification(
    intent_type: str,
    has_companies: bool,
    has_metrics: bool,
    years: tuple[int, ...],
    periods: tuple[str, ...],
    allow_all_periods: bool,
) -> ClarificationRequest | None:
    """槽位缺失 → 澄清问题。提取为模块级函数是为了让规则引擎和 LLM 引擎
    共享同一套澄清策略:无论哪条链路解析,缺同样的槽位就问同样的问题,
    前端/评测无需关心意图来自哪个引擎。"""
    if intent_type == "unsupported" or not has_metrics:
        return ClarificationRequest(("metric",), "你想查询哪个财务指标？例如营业收入、净利润、总资产或经营现金流。")
    if not years:
        return ClarificationRequest(("year",), "你想查询哪个年份或时间范围？例如 2024 年、2023-2025 年或近三年。")
    if not periods and not allow_all_periods:
        return ClarificationRequest(("report_period",), "你想查询哪个报告期？例如年报、第一季度、半年度或三季度。")
    if intent_type in {"metric_query", "trend_query"} and not has_companies:
        return ClarificationRequest(("company",), "你想查询哪家公司？可以输入股票简称、公司全称或股票代码。")
    return None


def metric_warnings(registry: SchemaRegistry, metrics: list[str] | tuple[str, ...]) -> list[str]:
    """歧义指标(如“利润”)的口径说明,两个引擎共享,保证提示文案一致。"""
    warnings: list[str] = []
    for metric in metrics:
        resolution = resolve_metric_with_policy(registry, metric)
        if resolution.explanation != "指标可直接解析。":
            warnings.append(resolution.explanation)
    return list(dict.fromkeys(warnings))


class RuleBasedIntentEngine:
    """Deterministic baseline for Task 2.

    与 LlmIntentEngine 输出同样的 StructuredIntent;SQL 统一由本项目的
    DSL/SQLBuilder 生成,任何引擎都不直接产出自由 SQL。
    """

    def __init__(self, registry: SchemaRegistry, reference_date: date | None = None):
        self.registry = registry
        self.reference_date = reference_date or date.today()

    def parse(self, question: str) -> StructuredIntent:
        normalized_question = normalize_text(question)
        companies = self._extract_companies(question)
        metrics = self._extract_metrics(question)
        time_result = self._extract_time(question)
        intent_type = self._classify_intent(normalized_question, companies, metrics)
        metric_filters = self._extract_metric_filters(question, metrics)
        limit = self._extract_limit(question, intent_type)
        order_metric = metrics[0] if intent_type == "ranking_query" and metrics else None
        sort_direction = "asc" if any(token in normalized_question for token in ("最低", "最少", "倒数", "升序")) else "desc"
        chart = self._extract_chart(question, intent_type, metrics)
        warnings = self._metric_warnings(metrics)

        clarification = self._build_clarification(intent_type, companies, metrics, time_result)
        return StructuredIntent(
            original_question=question,
            intent_type=intent_type,
            metrics=tuple(metrics),
            company_codes=tuple(code for code, _ in companies),
            company_names=tuple(name for _, name in companies),
            years=time_result.years,
            periods=time_result.periods,
            limit=limit,
            order_by_metric=order_metric,
            sort_direction=sort_direction,
            metric_filters=tuple(metric_filters),
            allow_all_periods=time_result.allow_all_periods,
            needs_clarification=clarification is not None,
            clarification=clarification,
            warnings=tuple(warnings),
            chart=chart,
        )

    def _classify_intent(self, normalized_question: str, companies: list[tuple[str, str]], metrics: list[str]) -> str:
        if not metrics:
            return "unsupported"
        if not companies and _is_all_company_analysis(normalized_question):
            return "ranking_query"
        if any(token in normalized_question for token in ("趋势", "变化", "走势", "近三年", "最近三年", "历年")):
            return "trend_query"
        if len(companies) >= 2 or any(token in normalized_question for token in ("对比", "比较", "相比")):
            return "comparison_query"
        return "metric_query"

    def _extract_companies(self, question: str) -> list[tuple[str, str]]:
        normalized_question = normalize_text(question)
        matches: dict[str, str] = {}
        for company in self.registry.companies.values():
            aliases = (company.stock_code, company.stock_abbr, company.company_name)
            if any(alias and normalize_text(alias) in normalized_question for alias in aliases):
                matches[company.stock_code] = company.stock_abbr
        return sorted(matches.items())

    def _extract_metrics(self, question: str) -> list[str]:
        normalized_question = normalize_text(question)
        aliases: list[tuple[int, str, str]] = []
        for table in self.registry.tables.values():
            for field in table.fields:
                if field.is_dimension:
                    continue
                for alias in {field.name, field.chinese_name, *field.aliases}:
                    normalized_alias = normalize_text(alias)
                    if normalized_alias and normalized_alias in normalized_question:
                        aliases.append((len(normalized_alias), alias, field.name))
        if not aliases:
            # 一些常用词在中文问题中短而高频，单独给一层兜底，避免首版解析过窄。
            for alias in ("营收", "收入", "净利润", "利润", "总资产", "经营现金流", "ROE"):
                if normalize_text(alias) in normalized_question:
                    aliases.append((len(normalize_text(alias)), alias, alias))
        aliases.sort(reverse=True)
        seen_fields: set[str] = set()
        selected_aliases: list[str] = []
        metrics: list[str] = []
        for _, alias, _ in aliases:
            normalized_alias = normalize_text(alias)
            if any(normalized_alias and normalized_alias in selected for selected in selected_aliases):
                continue
            resolution = resolve_metric_with_policy(self.registry, alias)
            if resolution.default_field is None:
                continue
            key = f"{resolution.default_field.table_name}.{resolution.default_field.name}"
            if key in seen_fields:
                continue
            seen_fields.add(key)
            selected_aliases.append(normalized_alias)
            metrics.append(alias)
            if len(metrics) >= 4:
                break
        return metrics

    def _extract_time(self, question: str) -> TimeParseResult:
        normalized_question = normalize_text(question)
        years = self._extract_years(question)
        if any(token in normalized_question for token in ("近三年", "最近三年")):
            years = tuple(range(self.reference_date.year - 3, self.reference_date.year))
        elif any(token in normalized_question for token in ("近两年", "最近两年")):
            years = tuple(range(self.reference_date.year - 2, self.reference_date.year))
        elif "去年" in normalized_question:
            years = (self.reference_date.year - 1,)
        elif "前年" in normalized_question:
            years = (self.reference_date.year - 2,)

        periods: list[str] = []
        if any(token in normalized_question for token in ("年报", "年度", "全年", "fy")):
            periods.append("FY")
        if any(token in normalized_question for token in ("一季", "第一季度", "q1")):
            periods.append("Q1")
        if any(token in normalized_question for token in ("半年", "半年度", "中报", "hy", "h1")):
            periods.append("HY")
        if any(token in normalized_question for token in ("三季", "第三季度", "前三季度", "q3")):
            periods.append("Q3")
        if any(token in normalized_question for token in ("各报告期", "所有报告期", "全部报告期")):
            return TimeParseResult(years=years, periods=(), allow_all_periods=True)
        if not periods and years and any(token in normalized_question for token in ("近三年", "最近三年", "近两年", "最近两年", "趋势", "历年")):
            periods.append("FY")
        if not periods and years:
            periods.append("FY")
        return TimeParseResult(years=years, periods=tuple(dict.fromkeys(periods)), allow_all_periods=False)

    def _extract_years(self, question: str) -> tuple[int, ...]:
        years: set[int] = set()
        for start, end in re.findall(r"(20\d{2})\s*[-—至到]\s*(20\d{2})", question):
            start_year, end_year = int(start), int(end)
            if start_year > end_year:
                start_year, end_year = end_year, start_year
            years.update(range(start_year, end_year + 1))
        years.update(int(year) for year in re.findall(r"20\d{2}", question))
        return tuple(sorted(years))

    def _extract_limit(self, question: str, intent_type: str) -> int:
        if intent_type != "ranking_query":
            return 100
        match = re.search(r"(?:前|top\s*|最高的|最低的)(\d+)", question, flags=re.IGNORECASE)
        return int(match.group(1)) if match else 10

    def _extract_metric_filters(self, question: str, metrics: list[str]) -> list[MetricFilter]:
        if not metrics:
            return []
        normalized_question = normalize_text(question)
        if ("/" in question or any(token in normalized_question for token in ("比值", "占比", "比例"))) and len(metrics) > 1:
            return []
        if any(token in normalized_question for token in ("为负", "负数", "小于0")):
            return [MetricFilter(metric=metric, operator="<", value=0) for metric in metrics]
        if any(token in normalized_question for token in ("为正", "均为正", "大于0")):
            return [MetricFilter(metric=metric, operator=">", value=0) for metric in metrics]
        patterns = (
            (r"(?:超过|大于|高于)\s*([0-9]+(?:\.[0-9]+)?)\s*(万|万元|亿|亿元|%)?", ">"),
            (r"(?:不低于|至少|大于等于)\s*([0-9]+(?:\.[0-9]+)?)\s*(万|万元|亿|亿元|%)?", ">="),
            (r"(?:低于|小于|少于)\s*([0-9]+(?:\.[0-9]+)?)\s*(万|万元|亿|亿元|%)?", "<"),
        )
        filters: list[MetricFilter] = []
        for pattern, operator in patterns:
            match = re.search(pattern, question)
            if not match:
                continue
            value = Decimal(match.group(1))
            unit = match.group(2) or ""
            # 大多数财务金额字段以万元为标准单位。用户说“2亿”时换成 20000 万元；
            # 用户说“200万/万元”时保持 200。百分比字段保持原值。
            if unit in {"亿", "亿元"}:
                value *= Decimal("10000")
            filters.append(MetricFilter(metric=metrics[0], operator=operator, value=float(value)))
            break
        return filters

    def _extract_chart(self, question: str, intent_type: str, metrics: list[str]) -> ChartSpec | None:
        normalized_question = normalize_text(question)
        if not any(token in normalized_question for token in ("画图", "图", "可视化", "趋势")):
            return None
        chart_type = "line" if intent_type == "trend_query" else "bar"
        return ChartSpec(chart_type=chart_type, x="report_year", y=metrics[0] if metrics else None, title=None)

    def _metric_warnings(self, metrics: list[str]) -> list[str]:
        return metric_warnings(self.registry, metrics)

    def _build_clarification(
        self,
        intent_type: str,
        companies: list[tuple[str, str]],
        metrics: list[str],
        time_result: TimeParseResult,
    ) -> ClarificationRequest | None:
        return build_clarification(
            intent_type,
            has_companies=bool(companies),
            has_metrics=bool(metrics),
            years=time_result.years,
            periods=time_result.periods,
            allow_all_periods=time_result.allow_all_periods,
        )


def _is_all_company_analysis(normalized_question: str) -> bool:
    return any(
        token in normalized_question
        for token in (
            "最高",
            "最低",
            "排名",
            "前",
            "top",
            "超过",
            "大于",
            "小于",
            "哪些公司",
            "哪家公司",
            "公司有哪些",
            "各公司",
            "10家公司",
            "平均",
            "均值",
            "中位数",
            "数量",
            "连续",
            "最大",
            "最小",
            "总额",
            "为负",
            "负数",
            "为正",
            "均为正",
        )
    )
