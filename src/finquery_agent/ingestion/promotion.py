from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from finquery_agent.ingestion.evaluation import RunEvaluationReport, evaluate_run
from finquery_agent.schema import load_default_registry
from finquery_agent.schema.models import FieldDefinition, TableDefinition


DEFAULT_CORE_COVERAGE_THRESHOLD = Decimal("0.90")
DEFAULT_FIELD_COVERAGE_THRESHOLD = Decimal("0.60")


@dataclass(frozen=True)
class PromotionResult:
    run_id: int
    status: str
    promotable: bool
    promoted: bool
    core_coverage_ratio: Decimal
    field_coverage_ratio: Decimal
    core_threshold: Decimal
    field_threshold: Decimal
    promoted_tables: tuple[str, ...] = field(default_factory=tuple)
    message: str = ""
    issues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "promotable": self.promotable,
            "promoted": self.promoted,
            "core_coverage_ratio": f"{self.core_coverage_ratio:.2%}",
            "field_coverage_ratio": f"{self.field_coverage_ratio:.2%}",
            "core_threshold": f"{self.core_threshold:.2%}",
            "field_threshold": f"{self.field_threshold:.2%}",
            "promoted_tables": list(self.promoted_tables),
            "message": self.message,
            "issues": list(self.issues),
        }


def assess_promotion_quality(
    report: RunEvaluationReport,
    core_threshold: Decimal = DEFAULT_CORE_COVERAGE_THRESHOLD,
    field_threshold: Decimal = DEFAULT_FIELD_COVERAGE_THRESHOLD,
) -> PromotionResult:
    fail_issues = tuple(issue.message for issue in report.issues if issue.status == "fail")
    if fail_issues:
        return PromotionResult(
            run_id=report.run_id,
            status="fail",
            promotable=False,
            promoted=False,
            core_coverage_ratio=report.core_coverage_ratio,
            field_coverage_ratio=report.field_coverage_ratio,
            core_threshold=core_threshold,
            field_threshold=field_threshold,
            message="staging has blocking validation failures",
            issues=fail_issues,
        )
    return PromotionResult(
        run_id=report.run_id,
        status="pass",
        promotable=True,
        promoted=False,
        core_coverage_ratio=report.core_coverage_ratio,
        field_coverage_ratio=report.field_coverage_ratio,
        core_threshold=core_threshold,
        field_threshold=field_threshold,
        message=(
            "coverage thresholds met"
            if report.core_coverage_ratio >= core_threshold and report.field_coverage_ratio >= field_threshold
            else "no blocking validation failures; coverage thresholds are informational"
        ),
    )


def promote_run_to_formal_tables(
    engine: Engine,
    run_id: int,
    core_threshold: Decimal = DEFAULT_CORE_COVERAGE_THRESHOLD,
    field_threshold: Decimal = DEFAULT_FIELD_COVERAGE_THRESHOLD,
    force: bool = False,
) -> PromotionResult:
    report = evaluate_run(engine, run_id, write_validation=True)
    assessment = assess_promotion_quality(report, core_threshold=core_threshold, field_threshold=field_threshold)
    if not assessment.promotable:
        with engine.begin() as connection:
            _write_promotion_metadata(connection, run_id, assessment)
        return assessment

    registry = load_default_registry()
    with engine.begin() as connection:
        staging_rows = _load_preferred_staging_rows(connection, run_id)
        promoted_tables = _upsert_formal_tables(connection, registry.tables, report, staging_rows)
        result = PromotionResult(
            run_id=run_id,
            status="pass" if assessment.promotable else assessment.status,
            promotable=assessment.promotable,
            promoted=True,
            core_coverage_ratio=assessment.core_coverage_ratio,
            field_coverage_ratio=assessment.field_coverage_ratio,
            core_threshold=core_threshold,
            field_threshold=field_threshold,
            promoted_tables=tuple(promoted_tables),
            message="promoted to formal tables" if assessment.promotable else "force promoted despite quality gate",
            issues=assessment.issues,
        )
        _write_promotion_metadata(connection, run_id, result)
        return result


def _load_preferred_staging_rows(connection, run_id: int) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        text(
            """
            SELECT DISTINCT ON (target_table, target_field)
                target_table,
                target_field,
                standard_value,
                is_derived,
                confidence,
                staging_id
            FROM financial_staging
            WHERE run_id = :run_id
              AND standard_value IS NOT NULL
              AND validation_status <> 'rejected'
            ORDER BY target_table, target_field, is_derived ASC, confidence DESC NULLS LAST, staging_id DESC
            """
        ),
        {"run_id": run_id},
    ).mappings().all()
    values: dict[str, dict[str, Any]] = {}
    for row in rows:
        values.setdefault(row["target_table"], {})[row["target_field"]] = row["standard_value"]
    return values


def _upsert_formal_tables(
    connection,
    tables: dict[str, TableDefinition],
    report: RunEvaluationReport,
    staging_rows: dict[str, dict[str, Any]],
) -> list[str]:
    promoted: list[str] = []
    for table_name, table in tables.items():
        values = staging_rows.get(table_name, {})
        if not values:
            continue
        row_values = _formal_row_values(table, report, values)
        _upsert_table_row(connection, table, row_values)
        promoted.append(table_name)
    return promoted


def _formal_row_values(table: TableDefinition, report: RunEvaluationReport, staging_values: dict[str, Any]) -> dict[str, Any]:
    row_values: dict[str, Any] = {}
    for field in table.fields:
        if field.name == "serial_number":
            row_values[field.name] = 1
        elif field.name == "stock_code":
            row_values[field.name] = report.stock_code
        elif field.name == "stock_abbr":
            row_values[field.name] = ""
        elif field.name == "report_year":
            row_values[field.name] = report.report_year
        elif field.name == "report_period":
            row_values[field.name] = report.report_period
        else:
            row_values[field.name] = _coerce_formal_value(field, staging_values.get(field.name))
    return row_values


def _coerce_formal_value(field: FieldDefinition, value: Any) -> Any:
    if value is None:
        return None
    data_type = field.data_type.strip().lower()
    decimal_match = re.match(r"decimal\((\d+),(\d+)\)", data_type)
    if not decimal_match:
        return value
    precision = int(decimal_match.group(1))
    scale = int(decimal_match.group(2))
    limit = Decimal(10) ** (precision - scale)
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    if abs(decimal_value) >= limit:
        return None
    return decimal_value


def _upsert_table_row(connection, table: TableDefinition, row_values: dict[str, Any]) -> None:
    columns = [field.name for field in table.fields]
    insert_columns = ", ".join(columns)
    insert_values = ", ".join(f":{column}" for column in columns)
    update_columns = [column for column in columns if column not in {"stock_code", "report_year", "report_period"}]
    update_clause = ", ".join(f"{column} = COALESCE(EXCLUDED.{column}, {table.name}.{column})" for column in update_columns)
    connection.execute(
        text(
            f"""
            INSERT INTO {table.name} ({insert_columns})
            VALUES ({insert_values})
            ON CONFLICT (stock_code, report_year, report_period)
            DO UPDATE SET {update_clause}
            """
        ),
        row_values,
    )


def _write_promotion_metadata(connection, run_id: int, result: PromotionResult) -> None:
    connection.execute(
        text(
            """
            UPDATE extraction_runs
            SET metadata = metadata || jsonb_build_object('promotion', CAST(:promotion AS jsonb))
            WHERE run_id = :run_id
            """
        ),
        {"run_id": run_id, "promotion": json.dumps(result.to_dict(), ensure_ascii=False)},
    )
