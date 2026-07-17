import pytest

from finquery_agent.nl2sql import QueryDSL, SQLBuildError, SQLBuilder
from finquery_agent.schema import load_default_registry


def test_builds_safe_metric_query():
    registry = load_default_registry()
    builder = SQLBuilder(registry)

    query = builder.build(
        QueryDSL(
            intent="metric_query",
            metrics=("营收",),
            company_names=("白云山",),
            years=(2024,),
            periods=("FY",),
        )
    )

    assert query.table_name == "core_performance_indicators_sheet"
    assert "SELECT" in query.sql
    assert "total_operating_revenue" in query.sql
    assert "report_period IN" in query.sql
    assert query.params["stock_code_0"] == "600332"
    assert query.params["report_year_0"] == 2024
    assert query.params["report_period_0"] == "FY"


def test_requires_report_period_by_default():
    registry = load_default_registry()
    builder = SQLBuilder(registry)

    with pytest.raises(SQLBuildError, match="report_period"):
        builder.build(QueryDSL(intent="metric_query", metrics=("营收",), years=(2024,)))


def test_rejects_cross_table_metric_mix_in_first_version():
    registry = load_default_registry()
    builder = SQLBuilder(registry)

    with pytest.raises(SQLBuildError, match="跨财务表"):
        builder.build(
            QueryDSL(
                intent="metric_query",
                metrics=("营收", "总资产"),
                years=(2024,),
                periods=("FY",),
            )
        )


def test_ambiguous_financial_metric_defaults_and_warns():
    registry = load_default_registry()
    builder = SQLBuilder(registry)

    query = builder.build(QueryDSL(intent="metric_query", metrics=("利润",), years=(2024,), periods=("FY",)))

    assert query.metric_columns == ("net_profit_10k_yuan",)
    assert any("已先按" in warning for warning in query.warnings)


def test_ranking_query_excludes_null_order_values():
    registry = load_default_registry()
    builder = SQLBuilder(registry)

    query = builder.build(
        QueryDSL(
            intent="ranking_query",
            metrics=("营业总收入",),
            years=(2025,),
            periods=("Q3",),
            order_by_metric="营业总收入",
            limit=2,
        )
    )

    assert "total_operating_revenue IS NOT NULL" in query.sql
    assert "total_operating_revenue DESC NULLS LAST" in query.sql
    assert query.params["limit"] == 2
