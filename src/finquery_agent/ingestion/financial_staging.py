from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from finquery_agent.ingestion.models import ExtractedTable, ParsedPdfDocument, ReportMetadata
from finquery_agent.schema.models import FieldDefinition
from finquery_agent.schema.registry import SchemaRegistry, normalize_text


@dataclass(frozen=True)
class FinancialStagingRecord:
    target_table: str
    target_field: str
    source_label: str
    raw_value: str
    raw_unit: str | None
    standard_value: Decimal | None
    standard_unit: str | None
    period_scope: str
    source_period_label: str | None
    page_number: int
    table_index: int
    confidence: Decimal
    is_derived: bool = False
    derivation_formula: str | None = None


@dataclass(frozen=True)
class FieldRule:
    target_table: str
    target_field: str
    labels: tuple[str, ...]
    table_types: tuple[str, ...]
    value_kind: str = "value"
    priority: int = 50


HEADER_LABELS = {"项目", "指标", "科目", "主要会计数据", "主要财务指标"}
NULL_VALUES = {"", "-", "--", "—", "不适用", "无", "nan", "n/a"}
CURRENT_TOKENS = ("本期", "本年", "本报告期", "期末", "本期发生额", "本期金额", "本年度")
PRIOR_TOKENS = ("上期", "上年", "上年度", "上年同期", "期初", "上期发生额", "上期金额")
NOTE_TOKENS = ("附注", "注释", "注")

EQUIVALENT_FIELD_RULES: tuple[tuple[tuple[str, str], tuple[tuple[str, str], ...]], ...] = (
    (
        ("core_performance_indicators_sheet", "total_operating_revenue"),
        (("income_sheet", "total_operating_revenue"),),
    ),
    (
        ("core_performance_indicators_sheet", "operating_revenue_yoy_growth"),
        (("income_sheet", "operating_revenue_yoy_growth"),),
    ),
    (
        ("core_performance_indicators_sheet", "net_profit_10k_yuan"),
        (("income_sheet", "net_profit"),),
    ),
    (
        ("core_performance_indicators_sheet", "net_profit_yoy_growth"),
        (("income_sheet", "net_profit_yoy_growth"),),
    ),
)

FALLBACK_EQUIVALENT_FIELD_RULES: tuple[tuple[tuple[str, str], tuple[tuple[str, str], ...]], ...] = (
    (
        ("income_sheet", "total_operating_revenue"),
        (("core_performance_indicators_sheet", "total_operating_revenue"),),
    ),
    (
        ("income_sheet", "operating_revenue_yoy_growth"),
        (("core_performance_indicators_sheet", "operating_revenue_yoy_growth"),),
    ),
)

FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule("core_performance_indicators_sheet", "total_operating_revenue", ("营业收入", "营业总收入"), ("core",), "value", 90),
    FieldRule("core_performance_indicators_sheet", "operating_revenue_yoy_growth", ("营业收入", "营业总收入"), ("core",), "growth", 90),
    FieldRule("core_performance_indicators_sheet", "net_profit_10k_yuan", ("归属于上市公司股东的净利润", "归属于母公司所有者的净利润"), ("core",), "value", 90),
    FieldRule("core_performance_indicators_sheet", "net_profit_yoy_growth", ("归属于上市公司股东的净利润", "归属于母公司所有者的净利润"), ("core",), "growth", 90),
    FieldRule("core_performance_indicators_sheet", "net_profit_excl_non_recurring", ("归属于上市公司股东的扣除非经常性损益的净利润", "扣除非经常性损益的净利润", "归母扣非净利润"), ("core",), "value", 90),
    FieldRule("core_performance_indicators_sheet", "net_profit_excl_non_recurring_yoy", ("归属于上市公司股东的扣除非经常性损益的净利润", "扣除非经常性损益的净利润", "归母扣非净利润"), ("core",), "growth", 90),
    FieldRule("core_performance_indicators_sheet", "eps", ("基本每股收益", "基本每股收益(元/股)", "基本每股收益（元/股）"), ("core", "income"), "value", 80),
    FieldRule("core_performance_indicators_sheet", "roe", ("加权平均净资产收益率",), ("core",), "value", 90),
    FieldRule("core_performance_indicators_sheet", "roe_weighted_excl_non_recurring", ("扣除非经常性损益后的加权平均净资产收益率",), ("core",), "value", 90),
    FieldRule("balance_sheet", "asset_total_assets", ("总资产", "资产总计", "资产合计"), ("core", "balance"), "value", 85),
    FieldRule("balance_sheet", "asset_total_assets_yoy_growth", ("总资产", "资产总计", "资产合计"), ("core",), "growth", 85),
    FieldRule("balance_sheet", "equity_total_equity", ("归属于上市公司股东的净资产", "归属于上市公司股东的所有者权益", "归属于母公司所有者权益合计", "归属于母公司所有者权益", "所有者权益合计", "股东权益合计"), ("core", "balance"), "value", 85),
    FieldRule("balance_sheet", "asset_cash_and_cash_equivalents", ("货币资金",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "asset_accounts_receivable", ("应收账款",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "asset_inventory", ("存货",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "asset_trading_financial_assets", ("交易性金融资产",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "asset_construction_in_progress", ("在建工程",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "liability_accounts_payable", ("应付账款",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "liability_advance_from_customers", ("预收款项", "预收账款"), ("balance",), "value", 80),
    FieldRule("balance_sheet", "liability_total_liabilities", ("负债合计", "总负债", "负债总计"), ("balance",), "value", 80),
    FieldRule("balance_sheet", "liability_contract_liabilities", ("合同负债",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "liability_short_term_loans", ("短期借款",), ("balance",), "value", 80),
    FieldRule("balance_sheet", "equity_unappropriated_profit", ("未分配利润",), ("balance",), "value", 80),
    FieldRule("income_sheet", "total_operating_revenue", ("营业总收入", "营业收入"), ("income",), "value", 80),
    FieldRule("income_sheet", "operating_expense_cost_of_sales", ("营业成本", "营业支出"), ("income",), "value", 80),
    FieldRule("income_sheet", "operating_expense_selling_expenses", ("销售费用",), ("income",), "value", 80),
    FieldRule("income_sheet", "operating_expense_administrative_expenses", ("管理费用",), ("income",), "value", 80),
    FieldRule("income_sheet", "operating_expense_financial_expenses", ("财务费用",), ("income",), "value", 80),
    FieldRule("income_sheet", "operating_expense_rnd_expenses", ("研发费用",), ("income",), "value", 80),
    FieldRule("income_sheet", "operating_expense_taxes_and_surcharges", ("税金及附加",), ("income",), "value", 80),
    FieldRule("income_sheet", "total_operating_expenses", ("营业总成本",), ("income",), "value", 80),
    FieldRule("income_sheet", "other_income", ("其他收益",), ("income",), "value", 80),
    FieldRule("income_sheet", "operating_profit", ("营业利润",), ("income",), "value", 80),
    FieldRule("income_sheet", "total_profit", ("利润总额",), ("income",), "value", 80),
    FieldRule("income_sheet", "net_profit", ("净利润",), ("income",), "value", 80),
    FieldRule("income_sheet", "credit_impairment_loss", ("信用减值损失",), ("income",), "value", 80),
    FieldRule("income_sheet", "asset_impairment_loss", ("资产减值损失",), ("income",), "value", 80),
    FieldRule("cash_flow_sheet", "operating_cf_cash_from_sales", ("销售商品、提供劳务收到的现金", "销售商品提供劳务收到的现金"), ("cash_flow",), "value", 80),
    FieldRule("cash_flow_sheet", "operating_cf_net_amount", ("经营活动产生的现金流量净额",), ("core", "cash_flow"), "value", 80),
    FieldRule("cash_flow_sheet", "investing_cf_cash_from_investment_recovery", ("收回投资收到的现金",), ("cash_flow",), "value", 80),
    FieldRule("cash_flow_sheet", "investing_cf_cash_for_investments", ("投资支付的现金",), ("cash_flow",), "value", 80),
    FieldRule("cash_flow_sheet", "investing_cf_net_amount", ("投资活动产生的现金流量净额",), ("cash_flow",), "value", 80),
    FieldRule("cash_flow_sheet", "financing_cf_cash_from_borrowing", ("取得借款收到的现金",), ("cash_flow",), "value", 80),
    FieldRule("cash_flow_sheet", "financing_cf_cash_for_debt_repayment", ("偿还债务支付的现金",), ("cash_flow",), "value", 80),
    FieldRule("cash_flow_sheet", "financing_cf_net_amount", ("筹资活动产生的现金流量净额", "融资活动产生的现金流量净额", "筹资活动产生的现金流量净增加额"), ("cash_flow",), "value", 80),
    FieldRule("cash_flow_sheet", "net_cash_flow", ("现金及现金等价物净增加额",), ("cash_flow",), "value", 80),
)


def extract_financial_staging_records(parsed: ParsedPdfDocument, registry: SchemaRegistry) -> tuple[FinancialStagingRecord, ...]:
    metadata = parsed.metadata
    if not _metadata_can_enter_staging(metadata, registry):
        return ()

    candidates: dict[tuple[str, str, str], tuple[int, FinancialStagingRecord]] = {}
    hidden: dict[str, Decimal] = {}
    for page in parsed.pages:
        for table in page.tables:
            raw_unit = infer_table_unit(table, page.text_content)
            for priority, record in _records_from_table(table, metadata, registry, raw_unit, hidden):
                key = (record.target_table, record.target_field, record.period_scope)
                previous = candidates.get(key)
                if previous is None or priority > previous[0]:
                    candidates[key] = (priority, record)

    records = [record for _, record in sorted(candidates.values(), key=lambda item: (item[1].target_table, item[1].target_field))]
    records = _reconcile_equivalent_fields(records, registry, metadata)
    records.extend(_derive_records(records, hidden, registry, metadata))
    return tuple(records)


def infer_table_unit(table: ExtractedTable, page_text: str = "") -> str | None:
    text = "\n".join(part for part in (table.section_title, table.markdown, page_text) if part)
    for match in re.finditer(r"单位\s*[:：]\s*([^\s，,；;。)）]+)", text):
        unit = _normalize_unit(match.group(1))
        if _is_known_unit(unit):
            return unit
    for unit in ("人民币亿元", "人民币万元", "人民币千元", "人民币元", "亿元", "万元", "千元", "元"):
        if unit in text:
            return _normalize_unit(unit)
    return _infer_unit_from_table_rows(table.raw_rows)


def parse_numeric_value(value: str) -> Decimal | None:
    text = str(value or "").strip()
    if normalize_text(text) in NULL_VALUES:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()（）")
    text = text.replace(",", "").replace("%", "").replace(" ", "")
    text = text.replace("−", "-").replace("－", "-")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return -number if negative and number > 0 else number


def standardize_value(value: Decimal, raw_unit: str | None, field: FieldDefinition) -> tuple[Decimal, str | None]:
    target_unit = _normalize_unit(field.unit)
    if not target_unit:
        return value, None
    if "%" in target_unit:
        return value, target_unit
    if "万元" not in target_unit:
        return value, target_unit
    if raw_unit is None:
        return value, None
    normalized_unit = _normalize_unit(raw_unit)
    if normalized_unit == "亿元":
        return value * Decimal("10000"), target_unit
    if normalized_unit == "万元":
        return value, target_unit
    if normalized_unit == "千元":
        return value / Decimal("10"), target_unit
    if normalized_unit == "元":
        return value / Decimal("10000"), target_unit
    return value, target_unit


def infer_period_scope(metadata: ReportMetadata, target_table: str) -> str:
    if target_table == "balance_sheet":
        return "point_in_time"
    period = (metadata.report_period or "").upper()
    if period == "FY":
        return "full_year"
    if period in {"HY", "Q3"}:
        return "year_to_date"
    if period == "Q1":
        return "single_period"
    return "unknown"


def _records_from_table(
    table: ExtractedTable,
    metadata: ReportMetadata,
    registry: SchemaRegistry,
    raw_unit: str | None,
    hidden: dict[str, Decimal],
) -> list[tuple[int, FinancialStagingRecord]]:
    rows = _merge_split_label_rows(_normalize_rows(table.raw_rows))
    if len(rows) < 2:
        return []
    header = _initial_header(rows)
    table_type = _infer_table_type(table, rows)
    if table_type == "unknown":
        return []
    if _is_non_statement_detail_table(table):
        return []

    records: list[tuple[int, FinancialStagingRecord]] = []
    for row in rows[1:]:
        source_label = _clean_label(_label_cell(row))
        if _is_period_header_row(source_label, row):
            header = row
            continue
        if not source_label or normalize_text(source_label) in {normalize_text(label) for label in HEADER_LABELS}:
            continue
        _capture_hidden_values(source_label, row, header, raw_unit, metadata, hidden)
        for rule in _matching_rules(source_label, table_type):
            record = _record_from_rule(row, header, table, metadata, registry, raw_unit, rule, source_label)
            if record is not None:
                records.append((_record_priority(rule, table_type, table), record))
    return records


def _record_from_rule(
    row: list[str],
    header: list[str],
    table: ExtractedTable,
    metadata: ReportMetadata,
    registry: SchemaRegistry,
    raw_unit: str | None,
    rule: FieldRule,
    source_label: str,
) -> FinancialStagingRecord | None:
    field = _get_field(registry, rule.target_table, rule.target_field)
    if field is None:
        return None
    value_cell = _select_numeric_cell(row, header, metadata, rule.value_kind, rule.target_table)
    if value_cell is None:
        return None
    raw_value, column_index = value_cell
    value = parse_numeric_value(raw_value)
    if value is None:
        return None
    effective_raw_unit = raw_unit or _infer_unit_from_row_label(row) or _default_statement_unit(rule.target_table)
    standard_value, standard_unit = standardize_value(value, effective_raw_unit, field)
    return FinancialStagingRecord(
        target_table=field.table_name,
        target_field=field.name,
        source_label=source_label,
        raw_value=raw_value,
        raw_unit=effective_raw_unit,
        standard_value=standard_value,
        standard_unit=standard_unit,
        period_scope=infer_period_scope(metadata, field.table_name),
        source_period_label=_source_period_label(header, column_index),
        page_number=table.page_number,
        table_index=table.table_index,
        confidence=Decimal("0.90") if effective_raw_unit else Decimal("0.76"),
    )


def _derive_records(
    records: list[FinancialStagingRecord],
    hidden: dict[str, Decimal],
    registry: SchemaRegistry,
    metadata: ReportMetadata,
) -> list[FinancialStagingRecord]:
    by_field = {(record.target_table, record.target_field): record for record in records}
    derived: list[FinancialStagingRecord] = []

    def value(table: str, field: str) -> Decimal | None:
        record = by_field.get((table, field))
        return record.standard_value if record else None

    def add(table: str, field: str, result: Decimal | None, formula: str, deps: tuple[tuple[str, str], ...], unit_override: str | None = None) -> None:
        if result is None or (table, field) in by_field:
            return
        source = next((by_field.get(dep) for dep in deps if by_field.get(dep)), None)
        if source is None:
            return
        unit = unit_override if unit_override is not None else _normalize_unit((_get_field(registry, table, field) or FieldDefinition(field, "", "", "", table)).unit)
        derived.append(
            FinancialStagingRecord(
                target_table=table,
                target_field=field,
                source_label=f"derived:{field}",
                raw_value=str(result),
                raw_unit=unit,
                standard_value=result,
                standard_unit=unit,
                period_scope=infer_period_scope(metadata, table),
                source_period_label="derived",
                page_number=source.page_number,
                table_index=source.table_index,
                confidence=Decimal("0.88"),
                is_derived=True,
                derivation_formula=formula,
            )
        )

    revenue = value("core_performance_indicators_sheet", "total_operating_revenue") or value("income_sheet", "total_operating_revenue")
    net_profit = value("core_performance_indicators_sheet", "net_profit_10k_yuan")
    cost = value("income_sheet", "operating_expense_cost_of_sales")
    if revenue and revenue != 0:
        add("core_performance_indicators_sheet", "gross_profit_margin", ((revenue - cost) / revenue * Decimal("100")) if cost is not None else None, "(营业总收入-营业成本)/营业总收入*100", (("core_performance_indicators_sheet", "total_operating_revenue"), ("income_sheet", "operating_expense_cost_of_sales")), "%")
        add("core_performance_indicators_sheet", "net_profit_margin", (net_profit / revenue * Decimal("100")) if net_profit is not None else None, "归母净利润/营业总收入*100", (("core_performance_indicators_sheet", "net_profit_10k_yuan"), ("core_performance_indicators_sheet", "total_operating_revenue")), "%")

    expense_keys = ("operating_expense_cost_of_sales", "operating_expense_selling_expenses", "operating_expense_administrative_expenses", "operating_expense_financial_expenses", "operating_expense_rnd_expenses", "operating_expense_taxes_and_surcharges")
    known_expenses = [value("income_sheet", key) for key in expense_keys if value("income_sheet", key) is not None]
    add("income_sheet", "total_operating_expenses", sum(known_expenses) if known_expenses else None, "营业成本+销售费用+管理费用+财务费用+研发费用+税金及附加", tuple(("income_sheet", key) for key in expense_keys), "万元")

    assets = value("balance_sheet", "asset_total_assets")
    liabilities = value("balance_sheet", "liability_total_liabilities")
    if assets and assets != 0:
        add("balance_sheet", "asset_liability_ratio", (liabilities / assets * Decimal("100")) if liabilities is not None else None, "总负债/总资产*100", (("balance_sheet", "liability_total_liabilities"), ("balance_sheet", "asset_total_assets")), "%")

    shares = hidden.get("total_shares")
    equity = value("balance_sheet", "equity_total_equity")
    operating_cf = value("cash_flow_sheet", "operating_cf_net_amount")
    if shares and shares != 0:
        add("core_performance_indicators_sheet", "net_asset_per_share", (equity / shares) if equity is not None else None, "归属于上市公司股东的净资产/总股本", (("balance_sheet", "equity_total_equity"),), "元")
        add("core_performance_indicators_sheet", "operating_cf_per_share", (operating_cf / shares) if operating_cf is not None else None, "经营活动产生的现金流量净额/总股本", (("cash_flow_sheet", "operating_cf_net_amount"),), "元")

    net_cash_flow = value("cash_flow_sheet", "net_cash_flow")
    net_cash_record = by_field.get(("cash_flow_sheet", "net_cash_flow"))
    net_cash_flow_wan = net_cash_flow / Decimal("10000") if net_cash_flow and net_cash_record and net_cash_record.standard_unit == "元" else net_cash_flow
    if net_cash_flow_wan and net_cash_flow_wan != 0:
        denominator = abs(net_cash_flow_wan)
        for target, source_field in (("operating_cf_ratio_of_net_cf", "operating_cf_net_amount"), ("investing_cf_ratio_of_net_cf", "investing_cf_net_amount"), ("financing_cf_ratio_of_net_cf", "financing_cf_net_amount")):
            source_value = value("cash_flow_sheet", source_field)
            add("cash_flow_sheet", target, (source_value / denominator * Decimal("100")) if source_value is not None else None, f"{source_field}/abs(net_cash_flow)*100", (("cash_flow_sheet", source_field), ("cash_flow_sheet", "net_cash_flow")), "%")
    return derived


def _reconcile_equivalent_fields(
    records: list[FinancialStagingRecord],
    registry: SchemaRegistry,
    metadata: ReportMetadata,
) -> list[FinancialStagingRecord]:
    by_field = {(record.target_table, record.target_field): record for record in records}
    replacements: dict[tuple[str, str], FinancialStagingRecord] = {}
    for primary_key, target_keys in EQUIVALENT_FIELD_RULES:
        primary = by_field.get(primary_key)
        if primary is None or primary.standard_value is None:
            continue
        for target_key in target_keys:
            existing = by_field.get(target_key)
            if existing is not None and existing.standard_value == primary.standard_value:
                continue
            replacement = _equivalent_record(primary, target_key, registry, metadata)
            if replacement is not None:
                replacements[target_key] = replacement
    for primary_key, target_keys in FALLBACK_EQUIVALENT_FIELD_RULES:
        primary = by_field.get(primary_key)
        if primary is None or primary.standard_value is None:
            continue
        for target_key in target_keys:
            if target_key in by_field or target_key in replacements:
                continue
            replacement = _equivalent_record(primary, target_key, registry, metadata)
            if replacement is not None:
                replacements[target_key] = replacement
    if not replacements:
        return records
    reconciled = [record for record in records if (record.target_table, record.target_field) not in replacements]
    reconciled.extend(replacements.values())
    return sorted(reconciled, key=lambda record: (record.target_table, record.target_field, record.is_derived))


def _equivalent_record(
    primary: FinancialStagingRecord,
    target_key: tuple[str, str],
    registry: SchemaRegistry,
    metadata: ReportMetadata,
) -> FinancialStagingRecord | None:
    target_table, target_field = target_key
    field = _get_field(registry, target_table, target_field)
    if field is None or primary.standard_value is None:
        return None
    standard_unit = _normalize_unit(field.unit) or primary.standard_unit
    return FinancialStagingRecord(
        target_table=target_table,
        target_field=target_field,
        source_label=f"equivalent:{primary.target_table}.{primary.target_field}",
        raw_value=str(primary.standard_value),
        raw_unit=primary.standard_unit,
        standard_value=primary.standard_value,
        standard_unit=standard_unit,
        period_scope=infer_period_scope(metadata, target_table),
        source_period_label=primary.source_period_label,
        page_number=primary.page_number,
        table_index=primary.table_index,
        confidence=Decimal("0.89"),
        is_derived=True,
        derivation_formula=f"同义字段对齐:{target_table}.{target_field}={primary.target_table}.{primary.target_field}",
    )


def _metadata_can_enter_staging(metadata: ReportMetadata, registry: SchemaRegistry) -> bool:
    return bool(metadata.stock_code and metadata.report_year is not None and metadata.report_period and metadata.stock_code in registry.companies)


def _get_field(registry: SchemaRegistry, table_name: str, field_name: str) -> FieldDefinition | None:
    return next((field for field in registry.tables[table_name].fields if field.name == field_name), None)


def _normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows if any(str(cell).strip() for cell in row)]


def _merge_split_label_rows(rows: list[list[str]]) -> list[list[str]]:
    merged: list[list[str]] = []
    index = 0
    while index < len(rows):
        row = list(rows[index])
        label_index = _label_index(row)
        if label_index is None or not _has_numeric_cell(row[label_index + 1 :]):
            merged.append(row)
            index += 1
            continue
        next_index = index + 1
        while next_index < len(rows):
            next_row = rows[next_index]
            next_label_index = _label_index(next_row)
            if next_label_index is None or _has_numeric_cell(next_row[next_label_index + 1 :]):
                break
            continuation = str(next_row[next_label_index] or "").strip()
            if not _looks_like_label_continuation(continuation):
                break
            row[label_index] = f"{row[label_index]}{continuation}"
            next_index += 1
        merged.append(row)
        index = next_index
    return merged


def _initial_header(rows: list[list[str]]) -> list[str]:
    for row in rows[:4]:
        if normalize_text(row[0]) in {normalize_text(label) for label in HEADER_LABELS} or any(_looks_like_period_header(cell) for cell in row[1:]):
            return row
    return rows[0]


def _infer_table_type(table: ExtractedTable, rows: list[list[str]]) -> str:
    text = normalize_text("\n".join([table.section_title or "", table.markdown[:800], *[" ".join(row[:3]) for row in rows[:10]]]))
    labels = {_clean_label(_label_cell(row)) for row in rows if row}
    income_hits = labels & {"营业总收入", "营业总成本", "营业利润", "利润总额", "净利润"}
    balance_hits = labels & {"流动资产", "非流动资产", "资产总计", "负债合计", "所有者权益合计"}
    cash_flow_hits = {label for label in labels if "现金流量" in label or "现金及现金等价物" in label}
    if sum(bool(hits) for hits in (income_hits, balance_hits, cash_flow_hits)) > 1:
        return "mixed_statement"
    if any(token in text for token in ("主要会计数据", "主要财务指标")) or labels & {"基本每股收益", "加权平均净资产收益率"}:
        return "core"
    if "资产负债表" in text or balance_hits:
        return "balance"
    if "利润表" in text or income_hits:
        return "income"
    if "现金流量表" in text or cash_flow_hits:
        return "cash_flow"
    return "unknown"


def _matching_rules(source_label: str, table_type: str) -> list[FieldRule]:
    normalized = normalize_text(source_label)
    return [
        rule
        for rule in FIELD_RULES
        if (table_type in rule.table_types or (table_type == "mixed_statement" and any(rule_type in {"balance", "income", "cash_flow"} for rule_type in rule.table_types)))
        and any(_label_matches(normalized, alias) for alias in rule.labels)
    ]


def _label_matches(normalized_label: str, alias: str) -> bool:
    normalized_alias = normalize_text(_clean_label(alias))
    return normalized_label == normalized_alias


def _record_priority(rule: FieldRule, table_type: str, table: ExtractedTable) -> int:
    priority = rule.priority
    title = normalize_text(table.section_title or "")
    if table_type in {"balance", "income", "cash_flow"}:
        priority += 10
    if "合并" in title:
        priority += 5
    if "母公司" in title:
        priority -= 20
    return priority


def _is_non_statement_detail_table(table: ExtractedTable) -> bool:
    text = normalize_text("\n".join(part for part in (table.section_title, table.markdown[:1200]) if part))
    if any(token in text for token in ("合并资产负债表", "合并利润表", "合并现金流量表", "主要会计数据", "主要财务指标")):
        return False
    return any(
        token in text
        for token in (
            "合营企业",
            "联营企业",
            "持股比例",
            "账面价值",
            "纳入评价范围",
            "内部控制",
            "缺陷认定",
            "关键审计事项",
        )
    )


def _default_statement_unit(target_table: str) -> str | None:
    if target_table in {"balance_sheet", "income_sheet", "cash_flow_sheet"}:
        return "元"
    return None


def _select_numeric_cell(row: list[str], header: list[str], metadata: ReportMetadata, value_kind: str, target_table: str) -> tuple[str, int] | None:
    scored: list[tuple[int, str, int]] = []
    for index, cell in enumerate(row[1:], 1):
        if parse_numeric_value(cell) is None:
            continue
        header_text = _header_text_for_index(header, index)
        if any(token in header_text for token in NOTE_TOKENS):
            continue
        score = _column_score(header_text, metadata, value_kind, target_table)
        if score <= -50:
            continue
        scored.append((score, str(cell).strip(), index))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    _, value, index = scored[0]
    return value, index


def _column_score(header_text: str, metadata: ReportMetadata, value_kind: str, target_table: str) -> int:
    year = str(metadata.report_year or "")
    previous_year = str((metadata.report_year or 0) - 1) if metadata.report_year else ""
    score = 0
    if value_kind == "growth":
        if any(token in header_text for token in ("增减", "增长", "同比", "%")):
            score += 100
        if year and year in header_text:
            score -= 10
        return score
    if any(token in header_text for token in ("增减", "增长", "同比")):
        return -100
    if header_text and not _looks_like_value_header(header_text, target_table):
        return -100
    if year and year in header_text:
        score += 100
    if any(token in header_text for token in CURRENT_TOKENS):
        score += 80
    if previous_year and previous_year in header_text:
        score -= 80
    if any(token in header_text for token in PRIOR_TOKENS):
        score -= 100
    if target_table == "balance_sheet" and any(token in header_text for token in ("12月31日", "期末", "年末")):
        score += 20
    if not header_text:
        score += 1
    return score


def _looks_like_value_header(header_text: str, target_table: str) -> bool:
    if re.search(r"20\d{2}年", header_text):
        return True
    if any(token in header_text for token in (*CURRENT_TOKENS, *PRIOR_TOKENS)):
        return True
    if target_table == "balance_sheet" and any(token in header_text for token in ("期末", "期初", "年末", "年初")):
        return True
    return False


def _is_period_header_row(source_label: str, row: list[str]) -> bool:
    return not source_label and any(_looks_like_period_header(cell) for cell in row[1:])


def _looks_like_period_header(value: str) -> bool:
    text = normalize_text(str(value or ""))
    return bool(re.search(r"20\d{2}年", text) or any(token in text for token in (*CURRENT_TOKENS, *PRIOR_TOKENS)))


def _capture_hidden_values(source_label: str, row: list[str], header: list[str], raw_unit: str | None, metadata: ReportMetadata, hidden: dict[str, Decimal]) -> None:
    normalized = normalize_text(source_label)
    if normalized not in {normalize_text("实收资本"), normalize_text("股本"), normalize_text("实收资本（或股本）"), normalize_text("实收资本(或股本)")}:
        return
    selected = _select_numeric_cell(row, header, metadata, "value", "balance_sheet")
    if selected is None:
        return
    value = parse_numeric_value(selected[0])
    if value is None:
        return
    hidden["total_shares"] = value / Decimal("10000") if _normalize_unit(raw_unit) == "元" else value


def _source_period_label(header: list[str], column_index: int) -> str | None:
    value = _raw_header_text_for_index(header, column_index)
    if not value or parse_numeric_value(value) is not None:
        return None
    return value


def _label_cell(row: list[str]) -> str:
    index = _label_index(row)
    return row[index] if index is not None else (row[0] if row else "")


def _infer_unit_from_row_label(row: list[str]) -> str | None:
    label = _label_cell(row)
    match = re.search(r"[（(]([^）)]+)[）)]", str(label or ""))
    return _normalize_unit(match.group(1)) if match else None


def _infer_unit_from_table_rows(rows: list[list[str]]) -> str | None:
    counts: dict[str, int] = {}
    for row in rows:
        for cell in row:
            for unit_text in re.findall(r"[（(]([^）)]+)[）)]", str(cell or "")):
                normalized = _normalize_unit(unit_text)
                if not _is_known_unit(normalized) or normalized == "%" or "股" in unit_text or "/" in unit_text:
                    continue
                counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _label_index(row: list[str]) -> int | None:
    for index, cell in enumerate(row):
        text = str(cell or "").strip()
        if not text or parse_numeric_value(text) is not None:
            continue
        normalized = normalize_text(_clean_label(text))
        if normalized and not _looks_like_period_header(text):
            return index
    return None


def _has_numeric_cell(cells: list[str]) -> bool:
    return any(parse_numeric_value(cell) is not None for cell in cells)


def _looks_like_label_continuation(value: str) -> bool:
    cleaned = _clean_label(value)
    if not cleaned:
        return False
    if normalize_text(cleaned) in {normalize_text(label) for label in HEADER_LABELS}:
        return False
    return bool(
        cleaned.startswith(("的", "及", "和", "或"))
        or cleaned in {"流量净额", "净利润", "填列", "列"}
        or cleaned.endswith(("流量净额", "净利润"))
    )


def _header_text_for_index(header: list[str], column_index: int) -> str:
    return normalize_text(_raw_header_text_for_index(header, column_index))


def _raw_header_text_for_index(header: list[str], column_index: int) -> str:
    if column_index < len(header) and str(header[column_index] or "").strip():
        return str(header[column_index] or "").strip()
    for offset in (1, 2):
        next_index = column_index + offset
        if next_index < len(header) and str(header[next_index] or "").strip():
            return str(header[next_index] or "").strip()
    for offset in (1, 2):
        previous_index = column_index - offset
        if previous_index >= 0 and str(header[previous_index] or "").strip():
            return str(header[previous_index] or "").strip()
    return ""


def _clean_label(value: str) -> str:
    label = str(value or "").strip()
    label = re.sub(r"^[一二三四五六七八九十]+[、.．]\s*", "", label)
    label = re.sub(r"^\d+[、.．)]\s*", "", label)
    label = re.sub(r"^(其中|加|减)\s*[:：]?\s*", "", label)
    label = label.replace("（", "(").replace("）", ")")
    label = re.sub(r"\([^)]*$", "", label)
    label = re.sub(r"\([^)]*\)", "", label)
    label = re.sub(r"[()（）]", "", label)
    return re.sub(r"\s+", "", label)


def _normalize_unit(unit: str | None) -> str | None:
    text = re.sub(r"\s+", "", str(unit or "").strip())
    if not text:
        return None
    if "%" in text or "百分" in text:
        return "%"
    if "亿元" in text:
        return "亿元"
    if "万元" in text:
        return "万元"
    if "千元" in text:
        return "千元"
    if "元" in text:
        return "元"
    return text


def _is_known_unit(unit: str | None) -> bool:
    return unit in {"亿元", "万元", "千元", "元", "%"}
