from datetime import date

from finquery_agent.nl2sql import RuleBasedIntentEngine
from finquery_agent.nl2sql.planner import split_intent_by_table
from finquery_agent.schema import load_default_registry


def test_split_intent_by_table_splits_cross_table_metrics():
    registry = load_default_registry()
    intent = RuleBasedIntentEngine(registry, reference_date=date(2026, 6, 18)).parse("白云山2024年年报营收和总资产")

    sub_intents = split_intent_by_table(intent, registry)

    assert len(sub_intents) == 2
    assert {task.metrics for task in sub_intents} == {("营收",), ("总资产",)}
    assert all(task.company_codes == ("600332",) for task in sub_intents)
    assert all(task.years == (2024,) for task in sub_intents)
    assert all(task.periods == ("FY",) for task in sub_intents)
