from finquery_agent.schema import load_default_registry
from finquery_agent.schema.metrics import resolve_metric_with_policy


def test_profit_total_resolves_to_income_sheet_total_profit():
    registry = load_default_registry()
    resolution = resolve_metric_with_policy(registry, "利润总额")

    assert resolution.default_field is not None
    assert resolution.default_field.name == "total_profit"
    assert resolution.needs_clarification is False


def test_plain_profit_defaults_with_explanation_instead_of_blocking():
    registry = load_default_registry()
    resolution = resolve_metric_with_policy(registry, "利润")

    assert resolution.default_field is not None
    assert resolution.default_field.name == "net_profit_10k_yuan"
    assert resolution.needs_clarification is False
    assert "已先按" in resolution.explanation
    assert "利润总额" in resolution.explanation


def test_net_profit_defaults_to_parent_net_profit_policy():
    registry = load_default_registry()
    resolution = resolve_metric_with_policy(registry, "净利润")

    assert resolution.default_field is not None
    assert resolution.default_field.name == "net_profit_10k_yuan"
    assert resolution.needs_clarification is False
