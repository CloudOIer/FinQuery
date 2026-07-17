from __future__ import annotations

import csv
import re
from pathlib import Path

from finquery_agent.config import get_settings
from finquery_agent.schema.models import Company, FieldDefinition, TableDefinition


TABLE_FILES: dict[str, tuple[str, str]] = {
    "core_performance_indicators_sheet": ("业绩指标表", "数据库信息-核心业绩指标表.csv"),
    "balance_sheet": ("资产负债表", "数据库信息-资产负债表.csv"),
    "income_sheet": ("利润表", "数据库信息-利润表.csv"),
    "cash_flow_sheet": ("现金流量表", "数据库信息-现金流量表.csv"),
}

DIMENSION_FIELDS = {"serial_number", "stock_code", "stock_abbr", "report_period", "report_year"}


class SchemaRegistry:
    def __init__(self, tables: dict[str, TableDefinition], companies: dict[str, Company]):
        self.tables = tables
        self.companies = companies
        self._field_index = self._build_field_index()

    def _build_field_index(self) -> dict[str, list[FieldDefinition]]:
        index: dict[str, list[FieldDefinition]] = {}
        for table in self.tables.values():
            for field in table.fields:
                keys = {field.name, field.chinese_name, *field.aliases}
                for key in keys:
                    normalized = normalize_text(key)
                    if normalized:
                        index.setdefault(normalized, []).append(field)
        return index

    def resolve_metric(self, metric: str) -> FieldDefinition | None:
        normalized = normalize_text(metric)
        candidates = [field for field in self._field_index.get(normalized, []) if not field.is_dimension]
        if candidates:
            return sorted(candidates, key=_field_priority)[0]

        partial_matches: list[FieldDefinition] = []
        for key, fields in self._field_index.items():
            if normalized and (normalized in key or key in normalized):
                partial_matches.extend(field for field in fields if not field.is_dimension)
        if partial_matches:
            return sorted(partial_matches, key=_field_priority)[0]
        return None

    def resolve_metric_candidates(self, metric: str) -> list[FieldDefinition]:
        normalized = normalize_text(metric)
        candidates = [field for field in self._field_index.get(normalized, []) if not field.is_dimension]
        if candidates:
            return sorted(candidates, key=_field_priority)

        partial_matches: list[FieldDefinition] = []
        for key, fields in self._field_index.items():
            if normalized and (normalized in key or key in normalized):
                partial_matches.extend(field for field in fields if not field.is_dimension)
        unique = {(field.table_name, field.name): field for field in partial_matches}
        return sorted(unique.values(), key=_field_priority)

    def resolve_company_code(self, value: str) -> str | None:
        code = normalize_stock_code(value)
        if code in self.companies:
            return code
        normalized = normalize_text(value)
        for company in self.companies.values():
            if normalized in {normalize_text(company.stock_abbr), normalize_text(company.company_name)}:
                return company.stock_code
        return None

    def get_table(self, table_name: str) -> TableDefinition:
        return self.tables[table_name]


def load_default_registry() -> SchemaRegistry:
    settings = get_settings()
    return load_registry(settings.data_root)


def load_registry(data_root: Path) -> SchemaRegistry:
    db_info_dir = data_root / "数据库信息"
    tables = {
        table_name: _load_table_definition(db_info_dir / file_name, table_name, chinese_name)
        for table_name, (chinese_name, file_name) in TABLE_FILES.items()
    }
    companies = _load_companies(data_root / "公司基本信息" / "公司基本信息.csv")
    return SchemaRegistry(tables=tables, companies=companies)


def _load_table_definition(path: Path, table_name: str, chinese_name: str) -> TableDefinition:
    fields: list[FieldDefinition] = []
    for row in _read_csv_dicts(path):
        field_name = row.get("字段名称", "").strip()
        chinese_field_name = row.get("中文名称", "").strip()
        data_type = _get_by_prefix(row, "字段类型").strip()
        description = row.get("字段说明", "").strip()
        fields.append(
            FieldDefinition(
                name=field_name,
                chinese_name=chinese_field_name,
                data_type=data_type,
                description=description,
                table_name=table_name,
                unit=_extract_unit(chinese_field_name),
                aliases=_build_aliases(field_name, chinese_field_name),
                is_dimension=field_name in DIMENSION_FIELDS,
            )
        )
    return TableDefinition(name=table_name, chinese_name=chinese_name, fields=tuple(fields))


def _load_companies(path: Path) -> dict[str, Company]:
    companies: dict[str, Company] = {}
    for row in _read_csv_dicts(path):
        stock_code = normalize_stock_code(row.get("股票代码", ""))
        if not stock_code:
            continue
        companies[stock_code] = Company(
            stock_code=stock_code,
            stock_abbr=row.get("A股简称", "").strip(),
            company_name=row.get("公司名称", "").strip(),
            exchange=row.get("上市交易所", "").strip() or None,
            industry=row.get("所属证监会行业", "").strip() or None,
        )
    return companies


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({(key or "").strip(): (value or "").strip() for key, value in row.items()})
        return rows


def _get_by_prefix(row: dict[str, str], prefix: str) -> str:
    for key, value in row.items():
        if key.strip().startswith(prefix):
            return value
    return ""


def _extract_unit(chinese_name: str) -> str | None:
    match = re.search(r"[（(]([^）)]+)[）)]", chinese_name)
    if not match:
        return None
    return match.group(1).strip()


def _build_aliases(field_name: str, chinese_name: str) -> tuple[str, ...]:
    aliases = {field_name, chinese_name}
    cleaned = re.sub(r"[（(][^）)]+[）)]", "", chinese_name)
    aliases.add(cleaned)
    if not any(keyword in cleaned for keyword in ("同比", "环比", "增长")):
        aliases.update(part for part in re.split(r"[-－—/、]", cleaned) if part)

    manual_aliases = {
        "total_operating_revenue": ("营收", "营业收入", "营业总收入", "收入"),
        "net_profit": ("利润表净利润", "净利润合计", "含少数股东损益净利润"),
        "net_profit_10k_yuan": ("净利润", "净利润万元", "归母净利润", "归属于上市公司股东的净利润"),
        "net_profit_excl_non_recurring": ("扣非净利润", "扣除非经常性损益的净利润", "归母扣非净利润"),
        "net_profit_yoy_growth": ("净利润同比", "净利润同比增长"),
        "total_profit": ("利润总额", "税前利润"),
        "operating_profit": ("营业利润",),
        "roe": ("ROE", "净资产收益率"),
        "gross_profit_margin": ("毛利率", "销售毛利率"),
        "net_profit_margin": ("净利率", "销售净利率"),
        "operating_expense_cost_of_sales": ("营业成本", "营业支出", "成本"),
        "operating_cf_net_amount": ("经营现金流", "经营性现金流", "经营性现金流净额", "经营性现金流量净额", "经营活动现金流净额", "经营活动现金流量净额"),
        "asset_total_assets": ("总资产",),
        "liability_total_liabilities": ("总负债",),
    }
    aliases.update(manual_aliases.get(field_name, ()))
    return tuple(sorted(alias for alias in aliases if alias))


def _field_priority(field: FieldDefinition) -> tuple[int, str]:
    table_priority = {
        "core_performance_indicators_sheet": 0,
        "income_sheet": 1,
        "balance_sheet": 2,
        "cash_flow_sheet": 3,
    }
    return (table_priority.get(field.table_name, 99), field.name)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def normalize_stock_code(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(6)
    return text
