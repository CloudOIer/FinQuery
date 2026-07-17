from decimal import Decimal

from finquery_agent.ingestion.evaluation import EvaluationIssue, RunEvaluationReport
from finquery_agent.ingestion.promotion import _coerce_formal_value, _upsert_table_row, assess_promotion_quality
from finquery_agent.schema.models import FieldDefinition, TableDefinition


def _report(
    covered: int = 54,
    total: int = 60,
    core_present: int = 17,
    core_total: int = 17,
    issues: tuple[EvaluationIssue, ...] = (),
) -> RunEvaluationReport:
    return RunEvaluationReport(
        run_id=1,
        document_id=1,
        stock_code="688222",
        report_year=2023,
        report_period="FY",
        page_count_expected=10,
        page_count_extracted=10,
        table_count=5,
        staging_count=covered,
        mapping_log_count=covered,
        target_field_count=total,
        covered_field_count=covered,
        core_required_count=core_total,
        core_present_count=core_present,
        issues=issues,
    )


def test_assess_promotion_quality_passes_when_coverage_thresholds_met():
    result = assess_promotion_quality(_report())

    assert result.status == "pass"
    assert result.promotable is True
    assert result.core_coverage_ratio == Decimal("1")
    assert result.field_coverage_ratio == Decimal("0.9")


def test_assess_promotion_quality_passes_when_coverage_thresholds_not_met():
    result = assess_promotion_quality(_report(covered=30, core_present=10))

    assert result.status == "pass"
    assert result.promotable is True
    assert result.issues == ()
    assert result.message == "no blocking validation failures; coverage thresholds are informational"


def test_assess_promotion_quality_fails_on_validation_failures():
    result = assess_promotion_quality(_report(issues=(EvaluationIssue("raw_tables_present", "fail", "No tables"),)))

    assert result.status == "fail"
    assert result.promotable is False
    assert result.issues == ("No tables",)


def test_upsert_table_row_preserves_existing_values_when_new_value_is_null():
    class FakeConnection:
        def __init__(self):
            self.statement = ""

        def execute(self, statement, values):
            self.statement = str(statement)
            self.values = values

    table = TableDefinition(
        name="core_performance_indicators_sheet",
        chinese_name="业绩指标表",
        fields=(
            FieldDefinition("stock_code", "股票代码", "varchar(20)", "", "core_performance_indicators_sheet"),
            FieldDefinition("report_year", "报告期-年份", "int", "", "core_performance_indicators_sheet"),
            FieldDefinition("report_period", "报告期", "varchar(20)", "", "core_performance_indicators_sheet"),
            FieldDefinition("total_operating_revenue", "营业总收入", "decimal(20,2)", "", "core_performance_indicators_sheet"),
        ),
    )
    connection = FakeConnection()

    _upsert_table_row(
        connection,
        table,
        {"stock_code": "688222", "report_year": 2023, "report_period": "FY", "total_operating_revenue": None},
    )

    assert "total_operating_revenue = COALESCE(EXCLUDED.total_operating_revenue, core_performance_indicators_sheet.total_operating_revenue)" in connection.statement


def test_coerce_formal_value_drops_values_that_do_not_fit_numeric_precision():
    field = FieldDefinition(
        "operating_revenue_yoy_growth",
        "营业总收入-同比增长",
        "decimal(10,4)",
        "",
        "core_performance_indicators_sheet",
    )

    assert _coerce_formal_value(field, Decimal("999999.9999")) == Decimal("999999.9999")
    assert _coerce_formal_value(field, Decimal("1000000")) is None
    assert _coerce_formal_value(field, Decimal("270975486.87")) is None
