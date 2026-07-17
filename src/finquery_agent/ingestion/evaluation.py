from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.engine import Engine

from finquery_agent.schema import load_default_registry
from finquery_agent.schema.models import FieldDefinition


CORE_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "core_performance_indicators_sheet": (
        "total_operating_revenue",
        "net_profit_10k_yuan",
        "net_profit_excl_non_recurring",
        "eps",
        "roe",
    ),
    "income_sheet": (
        "total_operating_revenue",
        "total_profit",
        "net_profit",
        "operating_profit",
    ),
    "balance_sheet": (
        "asset_total_assets",
        "liability_total_liabilities",
        "equity_total_equity",
        "asset_cash_and_cash_equivalents",
    ),
    "cash_flow_sheet": (
        "operating_cf_net_amount",
        "investing_cf_net_amount",
        "financing_cf_net_amount",
        "net_cash_flow",
    ),
}

EXPECTED_PERIOD_LABEL_TOKENS = (
    "年",
    "年度",
    "本期",
    "本年",
    "本报告期",
    "期末",
    "期初",
    "金额",
    "余额",
)


@dataclass(frozen=True)
class EvaluationIssue:
    rule_name: str
    status: str
    message: str
    staging_id: int | None = None


@dataclass(frozen=True)
class RunEvaluationReport:
    run_id: int
    document_id: int
    stock_code: str | None
    report_year: int | None
    report_period: str | None
    page_count_expected: int | None
    page_count_extracted: int
    table_count: int
    staging_count: int
    mapping_log_count: int
    target_field_count: int
    covered_field_count: int
    core_required_count: int
    core_present_count: int
    issues: tuple[EvaluationIssue, ...] = field(default_factory=tuple)

    @property
    def field_coverage_ratio(self) -> Decimal:
        if self.target_field_count == 0:
            return Decimal("0")
        return Decimal(self.covered_field_count) / Decimal(self.target_field_count)

    @property
    def core_coverage_ratio(self) -> Decimal:
        if self.core_required_count == 0:
            return Decimal("0")
        return Decimal(self.core_present_count) / Decimal(self.core_required_count)

    @property
    def overall_status(self) -> str:
        statuses = {issue.status for issue in self.issues}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        return "pass"

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "document_id": self.document_id,
            "stock_code": self.stock_code,
            "report_year": self.report_year,
            "report_period": self.report_period,
            "page_count_expected": self.page_count_expected,
            "page_count_extracted": self.page_count_extracted,
            "table_count": self.table_count,
            "staging_count": self.staging_count,
            "mapping_log_count": self.mapping_log_count,
            "target_field_count": self.target_field_count,
            "covered_field_count": self.covered_field_count,
            "field_coverage_ratio": f"{self.field_coverage_ratio:.2%}",
            "core_required_count": self.core_required_count,
            "core_present_count": self.core_present_count,
            "core_coverage_ratio": f"{self.core_coverage_ratio:.2%}",
            "overall_status": self.overall_status,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def evaluate_run(engine: Engine, run_id: int, write_validation: bool = False) -> RunEvaluationReport:
    registry = load_default_registry()
    target_fields = _target_financial_fields(registry)
    core_required = {(table, field) for table, fields in CORE_REQUIRED_FIELDS.items() for field in fields}

    with engine.begin() as connection:
        run = connection.execute(
            text(
                """
                SELECT r.run_id, r.document_id, d.stock_code, d.report_year, d.report_period,
                       NULLIF(r.metadata->>'page_count', '')::int AS page_count_expected
                FROM extraction_runs r
                JOIN source_documents d ON d.document_id = r.document_id
                WHERE r.run_id = :run_id
                """
            ),
            {"run_id": run_id},
        ).mappings().one()
        page_count_extracted = connection.execute(
            text("SELECT count(*) FROM extracted_pages WHERE run_id = :run_id"), {"run_id": run_id}
        ).scalar_one()
        table_count = connection.execute(
            text("SELECT count(*) FROM extracted_tables WHERE run_id = :run_id"), {"run_id": run_id}
        ).scalar_one()
        mapping_log_count = connection.execute(
            text("SELECT count(*) FROM field_mapping_logs WHERE run_id = :run_id"), {"run_id": run_id}
        ).scalar_one()
        staging_rows = connection.execute(
            text(
                """
                SELECT staging_id, target_table, target_field, source_label, raw_value, raw_unit,
                       standard_value, standard_unit, period_scope, source_period_label,
                       page_number, table_id, confidence
                FROM financial_staging
                WHERE run_id = :run_id
                ORDER BY target_table, target_field, staging_id
                """
            ),
            {"run_id": run_id},
        ).mappings().all()

        covered = {(row["target_table"], row["target_field"]) for row in staging_rows}
        issues = _evaluate_document(
            run=dict(run),
            page_count_extracted=page_count_extracted,
            table_count=table_count,
            staging_rows=staging_rows,
            target_fields=target_fields,
            core_required=core_required,
        )
        report = RunEvaluationReport(
            run_id=run_id,
            document_id=run["document_id"],
            stock_code=run["stock_code"],
            report_year=run["report_year"],
            report_period=run["report_period"],
            page_count_expected=run["page_count_expected"],
            page_count_extracted=page_count_extracted,
            table_count=table_count,
            staging_count=len(staging_rows),
            mapping_log_count=mapping_log_count,
            target_field_count=len(target_fields),
            covered_field_count=len(covered),
            core_required_count=len(core_required),
            core_present_count=len(covered & core_required),
            issues=tuple(issues),
        )
        if write_validation:
            _write_validation_results(connection, report)
        return report


def _evaluate_document(
    run: dict[str, object],
    page_count_extracted: int,
    table_count: int,
    staging_rows,
    target_fields: dict[tuple[str, str], FieldDefinition],
    core_required: set[tuple[str, str]],
) -> list[EvaluationIssue]:
    issues: list[EvaluationIssue] = []
    expected_pages = run.get("page_count_expected")
    if expected_pages is not None and int(expected_pages) != page_count_extracted:
        issues.append(
            EvaluationIssue(
                "raw_page_count_match",
                "fail",
                f"metadata page_count={expected_pages}, but extracted_pages={page_count_extracted}",
            )
        )
    if table_count == 0:
        issues.append(EvaluationIssue("raw_tables_present", "fail", "No tables were extracted from the PDF."))
    if not staging_rows:
        issues.append(EvaluationIssue("staging_rows_present", "fail", "No financial_staging rows were generated."))

    covered = {(row["target_table"], row["target_field"]) for row in staging_rows}
    for target_table, target_field in sorted(core_required - covered):
        issues.append(
            EvaluationIssue(
                "core_field_present",
                "warn",
                f"Missing core field {target_table}.{target_field} for run_id={run['run_id']}",
            )
        )

    for row in staging_rows:
        field = target_fields.get((row["target_table"], row["target_field"]))
        issues.extend(_evaluate_staging_row(row, field))
    return issues


def _evaluate_staging_row(row, field: FieldDefinition | None) -> list[EvaluationIssue]:
    issues: list[EvaluationIssue] = []
    staging_id = row["staging_id"]
    if row["standard_value"] is None:
        issues.append(EvaluationIssue("standard_value_present", "fail", "standard_value is NULL", staging_id))
    if row["page_number"] is None or row["table_id"] is None:
        issues.append(EvaluationIssue("source_trace_present", "fail", "page_number or table_id is missing", staging_id))
    if row["confidence"] is not None and row["confidence"] < Decimal("0.70"):
        issues.append(EvaluationIssue("confidence_threshold", "warn", f"low confidence={row['confidence']}", staging_id))
    expected_unit = _normalize_unit(field.unit if field else None)
    actual_unit = _normalize_unit(row["standard_unit"])
    if expected_unit and expected_unit != actual_unit:
        issues.append(
            EvaluationIssue(
                "standard_unit_match",
                "warn",
                f"expected standard_unit={expected_unit}, got {row['standard_unit']}",
                staging_id,
            )
        )
    label = str(row["source_period_label"] or "")
    if label and not any(token in label for token in EXPECTED_PERIOD_LABEL_TOKENS):
        issues.append(
            EvaluationIssue(
                "source_period_label_plausible",
                "warn",
                f"source_period_label looks non-period-like: {label}",
                staging_id,
            )
        )
    return issues


def _write_validation_results(connection, report: RunEvaluationReport) -> None:
    connection.execute(text("DELETE FROM validation_results WHERE run_id = :run_id"), {"run_id": report.run_id})
    for issue in report.issues:
        connection.execute(
            text(
                """
                INSERT INTO validation_results (run_id, document_id, staging_id, rule_name, status, message)
                VALUES (:run_id, :document_id, :staging_id, :rule_name, :status, :message)
                """
            ),
            {
                "run_id": report.run_id,
                "document_id": report.document_id,
                "staging_id": issue.staging_id,
                "rule_name": issue.rule_name,
                "status": issue.status,
                "message": issue.message,
            },
        )


def _target_financial_fields(registry) -> dict[tuple[str, str], FieldDefinition]:
    fields: dict[tuple[str, str], FieldDefinition] = {}
    for table in registry.tables.values():
        for field in table.fields:
            if not field.is_dimension:
                fields[(table.name, field.name)] = field
    return fields


def _normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    text_value = str(unit).replace(" ", "")
    if "万元" in text_value:
        return "万元"
    if "元" in text_value:
        return "元"
    if "%" in text_value:
        return "%"
    return text_value