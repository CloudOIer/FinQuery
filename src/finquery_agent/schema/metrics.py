from __future__ import annotations

from dataclasses import dataclass

from finquery_agent.schema.models import FieldDefinition
from finquery_agent.schema.registry import SchemaRegistry, normalize_text


@dataclass(frozen=True)
class MetricResolution:
    query: str
    default_field: FieldDefinition | None
    candidates: tuple[FieldDefinition, ...]
    needs_clarification: bool
    explanation: str


AMBIGUOUS_TERMS = {
    "利润": "利润是非精确金融口径；已先按上市公司财报问数中常用的归属于上市公司股东的净利润理解。也可进一步指定利润总额、利润表净利润合计或扣非净利润。",
    "净利润": "默认按上市公司财报问数口径解释为归属于上市公司股东的净利润；如需利润表净利润合计或扣非净利润，应在回答中提示可切换口径。",
}


def resolve_metric_with_policy(registry: SchemaRegistry, query: str) -> MetricResolution:
    normalized = normalize_text(query)
    candidates = tuple(registry.resolve_metric_candidates(query))
    default_field = registry.resolve_metric(query)

    if normalized == "利润":
        return MetricResolution(
            query=query,
            default_field=default_field,
            candidates=candidates,
            needs_clarification=False,
            explanation=AMBIGUOUS_TERMS["利润"],
        )

    if normalized == "净利润":
        return MetricResolution(
            query=query,
            default_field=default_field,
            candidates=candidates,
            needs_clarification=False,
            explanation=AMBIGUOUS_TERMS["净利润"],
        )

    return MetricResolution(
        query=query,
        default_field=default_field,
        candidates=candidates,
        needs_clarification=False,
        explanation="指标可直接解析。" if default_field else "未找到匹配指标。",
    )
