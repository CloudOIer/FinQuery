from decimal import Decimal

from finquery_agent.ingestion.evaluation import EvaluationIssue, RunEvaluationReport


def test_evaluation_report_status_and_ratios():
    report = RunEvaluationReport(
        run_id=1,
        document_id=1,
        stock_code="688222",
        report_year=2023,
        report_period="FY",
        page_count_expected=10,
        page_count_extracted=10,
        table_count=5,
        staging_count=2,
        mapping_log_count=2,
        target_field_count=10,
        covered_field_count=2,
        core_required_count=4,
        core_present_count=1,
        issues=(EvaluationIssue("core_field_present", "warn", "missing"),),
    )

    assert report.overall_status == "warn"
    assert report.field_coverage_ratio == Decimal("0.2")
    assert report.core_coverage_ratio == Decimal("0.25")
