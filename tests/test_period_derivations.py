from decimal import Decimal
from pathlib import Path

from finquery_agent.ingestion.financial_staging import FinancialStagingRecord
from finquery_agent.ingestion.models import ReportMetadata
from finquery_agent.ingestion.period_derivations import (
    PeriodGrowthRule,
    PriorPeriodValue,
    build_period_growth_records,
    previous_period_for_qoq,
    safe_growth_rate,
)
from finquery_agent.schema import load_default_registry


def _record(
    table: str,
    field: str,
    value: str,
    period_scope: str = "full_year",
    is_derived: bool = False,
) -> FinancialStagingRecord:
    return FinancialStagingRecord(
        target_table=table,
        target_field=field,
        source_label=field,
        raw_value=value,
        raw_unit="万元",
        standard_value=Decimal(value),
        standard_unit="万元",
        period_scope=period_scope,
        source_period_label="2024/FY",
        page_number=1,
        table_index=1,
        confidence=Decimal("0.90"),
        is_derived=is_derived,
        derivation_formula=None,
    )


def test_safe_growth_rate_uses_absolute_prior_value():
    assert safe_growth_rate(Decimal("150"), Decimal("100")) == Decimal("50.0")
    assert safe_growth_rate(Decimal("-30"), Decimal("-60")) == Decimal("50.0")
    assert safe_growth_rate(Decimal("10"), Decimal("0")) is None


def test_build_period_growth_records_derives_missing_yoy_fields():
    registry = load_default_registry()
    metadata = ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY")
    current_records = (
        _record("core_performance_indicators_sheet", "total_operating_revenue", "200"),
        _record("core_performance_indicators_sheet", "net_profit_10k_yuan", "60"),
        _record("balance_sheet", "liability_total_liabilities", "120", period_scope="point_in_time"),
        _record("cash_flow_sheet", "net_cash_flow", "80"),
    )
    prior_values = {
        ("core_performance_indicators_sheet", "total_operating_revenue"): Decimal("100"),
        ("core_performance_indicators_sheet", "net_profit_10k_yuan"): Decimal("40"),
        ("balance_sheet", "liability_total_liabilities"): Decimal("100"),
        ("cash_flow_sheet", "net_cash_flow"): Decimal("-40"),
    }

    def lookup(rule: PeriodGrowthRule, current_record: FinancialStagingRecord) -> PriorPeriodValue | None:
        value = prior_values.get((rule.base_table, rule.base_field))
        if value is None or rule.comparison != "yoy":
            return None
        return PriorPeriodValue(value, run_id=1, staging_id=10, report_year=2023, report_period="FY")

    derived = build_period_growth_records(metadata, current_records, registry, lookup)
    by_key = {(record.target_table, record.target_field): record for record in derived}

    assert by_key[("core_performance_indicators_sheet", "operating_revenue_yoy_growth")].standard_value == Decimal("100")
    assert by_key[("core_performance_indicators_sheet", "net_profit_yoy_growth")].standard_value == Decimal("50.0")
    assert by_key[("balance_sheet", "liability_total_liabilities_yoy_growth")].standard_value == Decimal("20.0")
    assert by_key[("cash_flow_sheet", "net_cash_flow_yoy_growth")].standard_value == Decimal("300")
    assert all(record.is_derived for record in derived)
    assert all("prior_run_id=1" in (record.derivation_formula or "") for record in derived)


def test_build_period_growth_records_does_not_override_existing_growth_fields():
    registry = load_default_registry()
    metadata = ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="FY")
    current_records = (
        _record("core_performance_indicators_sheet", "total_operating_revenue", "200"),
        _record("core_performance_indicators_sheet", "operating_revenue_yoy_growth", "88"),
    )

    def lookup(rule: PeriodGrowthRule, current_record: FinancialStagingRecord) -> PriorPeriodValue | None:
        return PriorPeriodValue(Decimal("100"), run_id=1, staging_id=10, report_year=2023, report_period="FY")

    derived = build_period_growth_records(metadata, current_records, registry, lookup)

    assert ("core_performance_indicators_sheet", "operating_revenue_yoy_growth") not in {
        (record.target_table, record.target_field) for record in derived
    }


def test_build_period_growth_records_derives_qoq_only_for_single_period_records():
    registry = load_default_registry()
    metadata = ReportMetadata(source_path=Path("600332.pdf"), stock_code="600332", report_year=2024, report_period="Q3")
    current_records = (_record("core_performance_indicators_sheet", "total_operating_revenue", "150", period_scope="single_period"),)

    def lookup(rule: PeriodGrowthRule, current_record: FinancialStagingRecord) -> PriorPeriodValue | None:
        if rule.comparison != "qoq":
            return None
        return PriorPeriodValue(Decimal("100"), run_id=2, staging_id=20, report_year=2024, report_period="Q2")

    derived = build_period_growth_records(metadata, current_records, registry, lookup)
    by_key = {(record.target_table, record.target_field): record for record in derived}

    assert previous_period_for_qoq(2024, "Q3") == (2024, "Q2")
    assert by_key[("core_performance_indicators_sheet", "operating_revenue_qoq_growth")].standard_value == Decimal("50.0")

    ytd_records = (_record("core_performance_indicators_sheet", "total_operating_revenue", "150", period_scope="year_to_date"),)
    assert build_period_growth_records(metadata, ytd_records, registry, lookup) == ()
