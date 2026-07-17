from datetime import date

from finquery_agent.nl2sql import RuleBasedIntentEngine, SQLBuilder
from finquery_agent.schema import load_default_registry


def _engine():
    return RuleBasedIntentEngine(load_default_registry(), reference_date=date(2026, 6, 18))


def test_parses_metric_query_to_dsl_and_sql():
    registry = load_default_registry()
    intent = RuleBasedIntentEngine(registry, reference_date=date(2026, 6, 18)).parse("白云山2024年年报营收是多少")

    assert intent.needs_clarification is False
    assert intent.intent_type == "metric_query"
    assert intent.company_codes == ("600332",)
    assert intent.years == (2024,)
    assert intent.periods == ("FY",)

    query = SQLBuilder(registry).build(intent.to_dsl())

    assert query.table_name == "core_performance_indicators_sheet"
    assert "total_operating_revenue" in query.sql
    assert query.params["stock_code_0"] == "600332"


def test_missing_company_triggers_clarification():
    intent = _engine().parse("2024年年报营收是多少")

    assert intent.needs_clarification is True
    assert intent.clarification is not None
    assert intent.clarification.missing_slots == ("company",)
    assert "哪家公司" in intent.clarification.question


def test_ranking_query_does_not_require_company():
    intent = _engine().parse("2024年年报净利润最高的5家公司")

    assert intent.needs_clarification is False
    assert intent.intent_type == "ranking_query"
    assert intent.limit == 5
    assert intent.order_by_metric is not None
    assert intent.sort_direction == "desc"


def test_metric_filter_is_parsed_for_threshold_query():
    intent = _engine().parse("2024年年报净利润超过200万的公司有哪些")

    assert intent.needs_clarification is False
    assert intent.intent_type == "ranking_query"
    assert len(intent.metric_filters) == 1
    assert intent.metric_filters[0].operator == ">"
    assert intent.metric_filters[0].value == 200

    query = SQLBuilder(load_default_registry()).build(intent.to_dsl())

    assert "net_profit_10k_yuan > :metric_filter_0" in query.sql
    assert query.params["metric_filter_0"] == 200


def test_trend_query_parses_recent_three_years_and_chart():
    intent = _engine().parse("白云山近三年营收趋势图")

    assert intent.intent_type == "trend_query"
    assert intent.years == (2023, 2024, 2025)
    assert intent.periods == ("FY",)
    assert intent.chart is not None
    assert intent.chart.chart_type == "line"


def test_year_only_defaults_to_full_year():
    intent = _engine().parse("泰格医药2024年研发费用是多少")

    assert intent.needs_clarification is False
    assert intent.years == (2024,)
    assert intent.periods == ("FY",)


def test_year_range_and_all_periods_are_parsed():
    intent = _engine().parse("泰格医药2022-2025年各报告期扣非净利润")

    assert intent.years == (2022, 2023, 2024, 2025)
    assert intent.periods == ()
    assert intent.allow_all_periods is True


def test_all_company_negative_filter_does_not_require_company():
    intent = _engine().parse("2025年第三季度经营性现金流量净额为负数的公司有哪些")

    assert intent.needs_clarification is False
    assert intent.intent_type == "ranking_query"
    assert intent.company_codes == ()
    assert len(intent.metric_filters) >= 1
    assert all(filter_item.operator == "<" and filter_item.value == 0 for filter_item in intent.metric_filters)


def test_all_company_positive_filter_for_multiple_metrics():
    intent = _engine().parse("2025年第三季度，净利润与经营性现金流量净额均为正的公司有哪些")

    assert intent.needs_clarification is False
    assert intent.intent_type == "ranking_query"
    assert len(intent.metric_filters) >= 2
    assert all(filter_item.operator == ">" and filter_item.value == 0 for filter_item in intent.metric_filters)


def test_operating_cash_flow_net_amount_alias_is_preferred():
    intent = _engine().parse("2025年第三季度，经营性现金流量净额为负数的公司有哪些")

    assert intent.metrics == ("经营性现金流量净额",)
    assert intent.metric_filters[0].metric == "经营性现金流量净额"


def test_ratio_question_does_not_push_ratio_threshold_to_single_metric_filter():
    intent = _engine().parse("2025年第三季度，经营性现金流量净额/净利润比值小于0.5的公司有哪些")

    assert intent.needs_clarification is False
    assert len(intent.metrics) >= 2
    assert intent.metric_filters == ()


def test_all_periods_query_does_not_require_specific_period():
    intent = _engine().parse("泰格医药2022-2025年各报告期扣非净利润")

    assert intent.needs_clarification is False
    assert intent.allow_all_periods is True
    assert intent.periods == ()
