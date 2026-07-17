from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from sqlalchemy import text

from finquery_agent.ingestion.financial_staging import FinancialStagingRecord, infer_period_scope
from finquery_agent.ingestion.models import ReportMetadata
from finquery_agent.schema.models import FieldDefinition
from finquery_agent.schema.registry import SchemaRegistry


@dataclass(frozen=True)
class PeriodGrowthRule:
    target_table: str
    target_field: str
    base_table: str
    base_field: str
    comparison: str


@dataclass(frozen=True)
class PriorPeriodValue:
    standard_value: Decimal
    run_id: int | None
    staging_id: int | None
    report_year: int
    report_period: str


PriorLookup = Callable[[PeriodGrowthRule, FinancialStagingRecord], PriorPeriodValue | None]

PERIOD_GROWTH_RULES: tuple[PeriodGrowthRule, ...] = (
    PeriodGrowthRule("core_performance_indicators_sheet", "operating_revenue_yoy_growth", "core_performance_indicators_sheet", "total_operating_revenue", "yoy"),
    PeriodGrowthRule("core_performance_indicators_sheet", "net_profit_yoy_growth", "core_performance_indicators_sheet", "net_profit_10k_yuan", "yoy"),
    PeriodGrowthRule("core_performance_indicators_sheet", "net_profit_excl_non_recurring_yoy", "core_performance_indicators_sheet", "net_profit_excl_non_recurring", "yoy"),
    PeriodGrowthRule("income_sheet", "operating_revenue_yoy_growth", "income_sheet", "total_operating_revenue", "yoy"),
    PeriodGrowthRule("income_sheet", "net_profit_yoy_growth", "income_sheet", "net_profit", "yoy"),
    PeriodGrowthRule("balance_sheet", "asset_total_assets_yoy_growth", "balance_sheet", "asset_total_assets", "yoy"),
    PeriodGrowthRule("balance_sheet", "liability_total_liabilities_yoy_growth", "balance_sheet", "liability_total_liabilities", "yoy"),
    PeriodGrowthRule("cash_flow_sheet", "net_cash_flow_yoy_growth", "cash_flow_sheet", "net_cash_flow", "yoy"),
    PeriodGrowthRule("core_performance_indicators_sheet", "operating_revenue_qoq_growth", "core_performance_indicators_sheet", "total_operating_revenue", "qoq"),
    PeriodGrowthRule("core_performance_indicators_sheet", "net_profit_qoq_growth", "core_performance_indicators_sheet", "net_profit_10k_yuan", "qoq"),
)

QOQ_PREVIOUS_PERIOD: dict[str, tuple[int, str]] = {
    "Q1": (-1, "Q4"),
    "Q2": (0, "Q1"),
    "Q3": (0, "Q2"),
    "Q4": (0, "Q3"),
}


def derive_period_growth_records(connection, run_id: int, metadata: ReportMetadata, registry: SchemaRegistry) -> tuple[FinancialStagingRecord, ...]:
    if not metadata.stock_code or metadata.report_year is None or not metadata.report_period:
        return ()
    current_records = _load_current_staging_records(connection, run_id)

    def lookup(rule: PeriodGrowthRule, current_record: FinancialStagingRecord) -> PriorPeriodValue | None:
        return _lookup_prior_period_value(connection, run_id, metadata, rule, current_record.period_scope)

    return build_period_growth_records(metadata, current_records, registry, lookup)


def build_period_growth_records(
    metadata: ReportMetadata,
    current_records: tuple[FinancialStagingRecord, ...],
    registry: SchemaRegistry,
    prior_lookup: PriorLookup,
) -> tuple[FinancialStagingRecord, ...]:
    by_field = _preferred_records(current_records)
    present_targets = {(record.target_table, record.target_field) for record in current_records}
    derived: list[FinancialStagingRecord] = []
    for rule in PERIOD_GROWTH_RULES:
        target_key = (rule.target_table, rule.target_field)
        if target_key in present_targets:
            continue
        current_base = by_field.get((rule.base_table, rule.base_field))
        if current_base is None or current_base.standard_value is None:
            continue
        if rule.comparison == "qoq" and not _can_calculate_qoq(metadata, current_base):
            continue
        prior = prior_lookup(rule, current_base)
        if prior is None:
            continue
        growth = safe_growth_rate(current_base.standard_value, prior.standard_value)
        if growth is None:
            continue
        record = _period_growth_record(rule, current_base, prior, growth, metadata, registry)
        if record is not None:
            derived.append(record)
            present_targets.add(target_key)
    return tuple(derived)


def safe_growth_rate(current_value: Decimal, prior_value: Decimal) -> Decimal | None:
    if prior_value == 0:
        return None
    return (current_value - prior_value) / abs(prior_value) * Decimal("100")


def previous_period_for_qoq(report_year: int, report_period: str) -> tuple[int, str] | None:
    offset = QOQ_PREVIOUS_PERIOD.get(report_period.upper())
    if offset is None:
        return None
    year_delta, prior_period = offset
    return report_year + year_delta, prior_period


def _preferred_records(records: tuple[FinancialStagingRecord, ...]) -> dict[tuple[str, str], FinancialStagingRecord]:
    selected: dict[tuple[str, str], FinancialStagingRecord] = {}
    for record in records:
        key = (record.target_table, record.target_field)
        previous = selected.get(key)
        if previous is None or _record_rank(record) > _record_rank(previous):
            selected[key] = record
    return selected


def _record_rank(record: FinancialStagingRecord) -> tuple[int, Decimal]:
    return (0 if record.is_derived else 1, record.confidence)


def _can_calculate_qoq(metadata: ReportMetadata, current_base: FinancialStagingRecord) -> bool:
    if metadata.report_year is None or not metadata.report_period:
        return False
    if current_base.period_scope != "single_period":
        return False
    return previous_period_for_qoq(metadata.report_year, metadata.report_period) is not None


def _period_growth_record(
    rule: PeriodGrowthRule,
    current_base: FinancialStagingRecord,
    prior: PriorPeriodValue,
    growth: Decimal,
    metadata: ReportMetadata,
    registry: SchemaRegistry,
) -> FinancialStagingRecord | None:
    field = _get_field(registry, rule.target_table, rule.target_field)
    if field is None:
        return None
    source_label = f"period_derived:{rule.base_table}.{rule.base_field}:{rule.comparison}"
    formula = (
        f"({rule.base_table}.{rule.base_field}[current]-{rule.base_table}.{rule.base_field}[prior])"
        f"/abs({rule.base_table}.{rule.base_field}[prior])*100; "
        f"prior={prior.report_year}/{prior.report_period}; prior_run_id={prior.run_id}; prior_staging_id={prior.staging_id}"
    )
    return FinancialStagingRecord(
        target_table=rule.target_table,
        target_field=rule.target_field,
        source_label=source_label,
        raw_value=str(growth),
        raw_unit="%",
        standard_value=growth,
        standard_unit="%" if _get_field_unit(field) == "%" else _get_field_unit(field),
        period_scope=infer_period_scope(metadata, rule.target_table),
        source_period_label=f"{metadata.report_year}/{metadata.report_period} vs {prior.report_year}/{prior.report_period}",
        page_number=current_base.page_number,
        table_index=current_base.table_index,
        confidence=Decimal("0.86"),
        is_derived=True,
        derivation_formula=formula,
    )


def _get_field_unit(field: FieldDefinition) -> str | None:
    if not field.unit:
        return None
    if "%" in field.unit or "百分" in field.unit:
        return "%"
    return field.unit


def _get_field(registry: SchemaRegistry, table_name: str, field_name: str) -> FieldDefinition | None:
    return next((field for field in registry.tables[table_name].fields if field.name == field_name), None)


def _load_current_staging_records(connection, run_id: int) -> tuple[FinancialStagingRecord, ...]:
    rows = connection.execute(
        text(
            """
            SELECT
                fs.target_table,
                fs.target_field,
                fs.source_label,
                fs.raw_value,
                fs.raw_unit,
                fs.standard_value,
                fs.standard_unit,
                fs.period_scope,
                fs.source_period_label,
                fs.page_number,
                COALESCE(et.table_index, 0) AS table_index,
                fs.confidence,
                fs.is_derived,
                fs.derivation_formula
            FROM financial_staging fs
            LEFT JOIN extracted_tables et ON et.table_id = fs.table_id
            WHERE fs.run_id = :run_id
            ORDER BY fs.target_table, fs.target_field, fs.is_derived, fs.confidence DESC NULLS LAST
            """
        ),
        {"run_id": run_id},
    ).mappings().all()
    return tuple(
        FinancialStagingRecord(
            target_table=row["target_table"],
            target_field=row["target_field"],
            source_label=row["source_label"],
            raw_value=row["raw_value"],
            raw_unit=row["raw_unit"],
            standard_value=row["standard_value"],
            standard_unit=row["standard_unit"],
            period_scope=row["period_scope"],
            source_period_label=row["source_period_label"],
            page_number=row["page_number"],
            table_index=row["table_index"],
            confidence=row["confidence"] or Decimal("0"),
            is_derived=bool(row["is_derived"]),
            derivation_formula=row["derivation_formula"],
        )
        for row in rows
    )


def _lookup_prior_period_value(
    connection,
    run_id: int,
    metadata: ReportMetadata,
    rule: PeriodGrowthRule,
    period_scope: str,
) -> PriorPeriodValue | None:
    prior_period = _prior_period(metadata, rule.comparison)
    if prior_period is None or not metadata.stock_code:
        return None
    prior_year, prior_report_period = prior_period
    row = connection.execute(
        text(
            """
            SELECT
                fs.standard_value,
                fs.run_id,
                fs.staging_id,
                fs.report_year,
                fs.report_period
            FROM financial_staging fs
            JOIN extraction_runs er ON er.run_id = fs.run_id
            WHERE fs.run_id <> :run_id
              AND fs.stock_code = :stock_code
              AND fs.report_year = :report_year
              AND fs.report_period = :report_period
              AND fs.target_table = :target_table
              AND fs.target_field = :target_field
              AND fs.period_scope = :period_scope
              AND fs.standard_value IS NOT NULL
            ORDER BY fs.is_derived ASC, er.finished_at DESC NULLS LAST, er.started_at DESC, fs.staging_id DESC
            LIMIT 1
            """
        ),
        {
            "run_id": run_id,
            "stock_code": metadata.stock_code,
            "report_year": prior_year,
            "report_period": prior_report_period,
            "target_table": rule.base_table,
            "target_field": rule.base_field,
            "period_scope": period_scope,
        },
    ).mappings().first()
    if row is None:
        return None
    return PriorPeriodValue(
        standard_value=row["standard_value"],
        run_id=row["run_id"],
        staging_id=row["staging_id"],
        report_year=row["report_year"],
        report_period=row["report_period"],
    )


def _prior_period(metadata: ReportMetadata, comparison: str) -> tuple[int, str] | None:
    if metadata.report_year is None or not metadata.report_period:
        return None
    if comparison == "yoy":
        return metadata.report_year - 1, metadata.report_period
    if comparison == "qoq":
        return previous_period_for_qoq(metadata.report_year, metadata.report_period)
    return None
