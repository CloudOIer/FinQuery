from finquery_agent.schema import load_default_registry


def test_loads_financial_tables_and_companies():
    registry = load_default_registry()

    assert set(registry.tables) == {
        "core_performance_indicators_sheet",
        "balance_sheet",
        "income_sheet",
        "cash_flow_sheet",
    }
    assert len(registry.companies) >= 10
    assert registry.resolve_company_code("华润三九") == "000999"
    assert registry.resolve_company_code("999") == "000999"


def test_resolves_common_metric_aliases():
    registry = load_default_registry()

    revenue = registry.resolve_metric("营收")
    assert revenue is not None
    assert revenue.name == "total_operating_revenue"

    total_revenue = registry.resolve_metric("营业总收入")
    assert total_revenue is not None
    assert total_revenue.name == "total_operating_revenue"

    roe = registry.resolve_metric("ROE")
    assert roe is not None
    assert roe.name == "roe"
