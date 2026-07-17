from __future__ import annotations

from dataclasses import dataclass, field

from finquery_agent.nl2sql.dsl import MetricFilter, QueryDSL


@dataclass(frozen=True)
class ChartSpec:
    chart_type: str
    x: str | None = None
    y: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class ClarificationRequest:
    missing_slots: tuple[str, ...]
    question: str


@dataclass(frozen=True)
class StructuredIntent:
    original_question: str
    intent_type: str
    metrics: tuple[str, ...]
    company_codes: tuple[str, ...] = field(default_factory=tuple)
    company_names: tuple[str, ...] = field(default_factory=tuple)
    years: tuple[int, ...] = field(default_factory=tuple)
    periods: tuple[str, ...] = field(default_factory=tuple)
    limit: int = 100
    order_by_metric: str | None = None
    sort_direction: str = "desc"
    metric_filters: tuple[MetricFilter, ...] = field(default_factory=tuple)
    allow_all_periods: bool = False
    needs_clarification: bool = False
    clarification: ClarificationRequest | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    chart: ChartSpec | None = None
    sub_tasks: tuple["StructuredIntent", ...] = field(default_factory=tuple)
    # 意图来源追踪:rule(规则引擎) / llm(LLM 解析) / rule_fallback(LLM 失败后降级)。
    # 用于评测对比与线上问题定位:回答出错时能快速判断是哪条解析链路产生的。
    intent_source: str = "rule"

    def to_dsl(self) -> QueryDSL:
        return QueryDSL(
            intent=self.intent_type,
            metrics=self.metrics,
            company_codes=self.company_codes,
            company_names=self.company_names,
            years=self.years,
            periods=self.periods,
            limit=self.limit,
            order_by_metric=self.order_by_metric,
            sort_direction=self.sort_direction,
            metric_filters=self.metric_filters,
            allow_all_periods=self.allow_all_periods,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "original_question": self.original_question,
            "intent_type": self.intent_type,
            "metrics": list(self.metrics),
            "company_codes": list(self.company_codes),
            "company_names": list(self.company_names),
            "years": list(self.years),
            "periods": list(self.periods),
            "limit": self.limit,
            "order_by_metric": self.order_by_metric,
            "sort_direction": self.sort_direction,
            "metric_filters": [filter_item.__dict__ for filter_item in self.metric_filters],
            "allow_all_periods": self.allow_all_periods,
            "needs_clarification": self.needs_clarification,
            "clarification": self.clarification.__dict__ if self.clarification else None,
            "warnings": list(self.warnings),
            "chart": self.chart.__dict__ if self.chart else None,
            "sub_tasks": [task.to_dict() for task in self.sub_tasks],
            "intent_source": self.intent_source,
        }
