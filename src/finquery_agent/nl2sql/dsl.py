from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricFilter:
    metric: str
    operator: str
    value: float


@dataclass(frozen=True)
class QueryDSL:
    intent: str
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
