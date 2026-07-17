from __future__ import annotations

from dataclasses import replace

from finquery_agent.nl2sql.dsl import MetricFilter
from finquery_agent.nl2sql.intent import StructuredIntent
from finquery_agent.schema.registry import SchemaRegistry


def split_intent_by_table(intent: StructuredIntent, registry: SchemaRegistry) -> tuple[StructuredIntent, ...]:
    """Split one user intent into table-local sub intents.

    SQLBuilder intentionally avoids cross-table SQL. When an utterance mentions metrics
    from multiple financial tables, we keep the user's one intent but execute several
    table-local DSLs and merge results at the application layer later.
    """
    grouped_metrics: dict[str, list[str]] = {}
    for metric in intent.metrics:
        field = registry.resolve_metric(metric)
        if field is None:
            continue
        grouped_metrics.setdefault(field.table_name, []).append(metric)
    if len(grouped_metrics) <= 1:
        return (intent,)

    sub_tasks: list[StructuredIntent] = []
    for table_name, metrics in grouped_metrics.items():
        filters = tuple(_filter_for_table(metric_filter, table_name, registry) for metric_filter in intent.metric_filters)
        filters = tuple(filter_item for filter_item in filters if filter_item is not None)
        sub_tasks.append(
            replace(
                intent,
                intent_type="metric_query" if intent.intent_type == "comparison_query" else intent.intent_type,
                metrics=tuple(metrics),
                metric_filters=filters,
                order_by_metric=metrics[0] if intent.order_by_metric else None,
                sub_tasks=(),
            )
        )
    return tuple(sub_tasks)


def _filter_for_table(metric_filter: MetricFilter, table_name: str, registry: SchemaRegistry) -> MetricFilter | None:
    field = registry.resolve_metric(metric_filter.metric)
    if field is None or field.table_name != table_name:
        return None
    return metric_filter
