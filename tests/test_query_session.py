from datetime import date

from finquery_agent.nl2sql import RuleBasedIntentEngine
from finquery_agent.nl2sql.session import QuerySessionStore
from finquery_agent.schema import load_default_registry


def test_session_merges_clarification_reply_into_pending_intent():
    engine = RuleBasedIntentEngine(load_default_registry(), reference_date=date(2026, 6, 18))
    store = QuerySessionStore()

    first = store.resolve("s1", "2024年年报营收是多少", engine)
    second = store.resolve("s1", "白云山", engine)

    assert first.needs_clarification is True
    assert first.clarification is not None
    assert first.clarification.missing_slots == ("company",)
    assert second.needs_clarification is False
    assert second.company_codes == ("600332",)
    assert second.metrics == first.metrics
    assert second.years == (2024,)
    assert second.periods == ("FY",)


def test_session_uses_last_intent_for_short_follow_up():
    engine = RuleBasedIntentEngine(load_default_registry(), reference_date=date(2026, 6, 18))
    store = QuerySessionStore()

    first = store.resolve("s2", "白云山2024年年报营收是多少", engine)
    second = store.resolve("s2", "换成净利润", engine)

    assert first.needs_clarification is False
    assert second.needs_clarification is False
    assert second.company_codes == ("600332",)
    assert second.years == (2024,)
    assert second.periods == ("FY",)
    assert second.metrics != first.metrics
