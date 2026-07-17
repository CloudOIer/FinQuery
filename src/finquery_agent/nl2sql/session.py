from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock
from typing import Protocol

from finquery_agent.nl2sql.intent import ClarificationRequest, StructuredIntent


class IntentEngineLike(Protocol):
    """意图引擎接口:规则版/LLM版/混合版都只需提供 parse()。"""

    def parse(self, question: str) -> StructuredIntent: ...


@dataclass
class QuerySessionState:
    pending_intent: StructuredIntent | None = None
    last_intent: StructuredIntent | None = None


class QuerySessionStore:
    """In-memory session store for clarification and short follow-up turns.

    This is intentionally simple for MVP. It can later be replaced by Redis or a DB table
    without changing the API contract because callers only depend on resolve().
    """

    def __init__(self):
        self._states: dict[str, QuerySessionState] = {}
        self._lock = Lock()

    def has_pending(self, session_id: str | None) -> bool:
        if not session_id:
            return False
        with self._lock:
            state = self._states.get(session_id)
            return bool(state and state.pending_intent)

    def resolve(self, session_id: str | None, question: str, engine: "IntentEngineLike") -> StructuredIntent:
        parsed = engine.parse(question)
        if not session_id:
            return parsed
        with self._lock:
            state = self._states.setdefault(session_id, QuerySessionState())
            base = state.pending_intent or (state.last_intent if _looks_like_follow_up(parsed) else None)
            resolved = _merge_intents(base, parsed) if base else parsed
            resolved = _refresh_clarification(resolved)
            if resolved.needs_clarification:
                state.pending_intent = resolved
            else:
                state.pending_intent = None
                state.last_intent = resolved
            return resolved


def _looks_like_follow_up(intent: StructuredIntent) -> bool:
    has_new_slot = bool(intent.metrics or intent.company_codes or intent.company_names or intent.years or intent.periods or intent.metric_filters or intent.chart)
    return intent.needs_clarification and has_new_slot


def _merge_intents(base: StructuredIntent | None, update: StructuredIntent) -> StructuredIntent:
    if base is None:
        return update
    # Clarification replies often contain just one missing slot, e.g. only a company name.
    # Missing values are inherited from the pending or last intent.
    return replace(
        base,
        original_question=update.original_question,
        metrics=update.metrics or base.metrics,
        company_codes=update.company_codes or base.company_codes,
        company_names=update.company_names or base.company_names,
        years=update.years or base.years,
        periods=update.periods or base.periods,
        limit=update.limit if update.limit != 100 else base.limit,
        order_by_metric=update.order_by_metric or base.order_by_metric,
        sort_direction=update.sort_direction or base.sort_direction,
        metric_filters=update.metric_filters or base.metric_filters,
        chart=update.chart or base.chart,
        warnings=tuple(dict.fromkeys((*base.warnings, *update.warnings))),
        sub_tasks=(),
    )


def _refresh_clarification(intent: StructuredIntent) -> StructuredIntent:
    clarification = _clarification_for(intent)
    return replace(intent, needs_clarification=clarification is not None, clarification=clarification)


def _clarification_for(intent: StructuredIntent) -> ClarificationRequest | None:
    if not intent.metrics:
        return ClarificationRequest(("metric",), "你想查询哪个财务指标？例如营业收入、净利润、总资产或经营现金流。")
    if not intent.years:
        return ClarificationRequest(("year",), "你想查询哪个年份或时间范围？例如 2024 年、2023-2025 年或近三年。")
    if not intent.periods:
        return ClarificationRequest(("report_period",), "你想查询哪个报告期？例如年报、第一季度、半年度或三季度。")
    if intent.intent_type in {"metric_query", "trend_query"} and not intent.company_codes and not intent.company_names:
        return ClarificationRequest(("company",), "你想查询哪家公司？可以输入股票简称、公司全称或股票代码。")
    return None
