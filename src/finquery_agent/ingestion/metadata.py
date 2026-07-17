from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from finquery_agent.ingestion.models import ReportMetadata
from finquery_agent.schema import load_default_registry
from finquery_agent.schema.registry import normalize_stock_code, normalize_text


FILENAME_RE = re.compile(r"(?P<code>\d{3,6})_(?P<date>\d{8})")


def infer_report_metadata(pdf_path: Path) -> ReportMetadata:
    match = FILENAME_RE.search(pdf_path.name)
    stock_code = infer_stock_code_from_text(pdf_path.stem)
    if not match:
        report_year, report_period = infer_period_from_text(pdf_path.stem)
        return ReportMetadata(source_path=pdf_path, stock_code=stock_code, report_year=report_year, report_period=report_period)

    stock_code = normalize_stock_code(match.group("code")) or stock_code
    raw_date = match.group("date")
    parsed_date = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8]))
    report_year, report_period = infer_period_from_date(parsed_date)
    return ReportMetadata(
        source_path=pdf_path,
        stock_code=stock_code,
        report_year=report_year,
        report_period=report_period,
        file_date=parsed_date,
    )


def refine_report_metadata_from_text(metadata: ReportMetadata, text: str) -> ReportMetadata:
    report_year, report_period = infer_period_from_text(text)
    stock_code = metadata.stock_code or infer_stock_code_from_text(text) or infer_stock_code_from_text(metadata.source_path.stem)
    if report_year is None or report_period is None:
        if stock_code == metadata.stock_code:
            return metadata
        return ReportMetadata(
            source_path=metadata.source_path,
            stock_code=stock_code,
            report_year=metadata.report_year,
            report_period=metadata.report_period,
            file_date=metadata.file_date,
        )
    if stock_code == metadata.stock_code and report_year == metadata.report_year and report_period == metadata.report_period:
        return metadata
    return ReportMetadata(
        source_path=metadata.source_path,
        stock_code=stock_code,
        report_year=report_year,
        report_period=report_period,
        file_date=metadata.file_date,
    )


def infer_stock_code_from_text(text: str) -> str | None:
    normalized_head = re.sub(r"\s+", "", (text or "")[:20000])
    explicit_code = re.search(r"证券代码[:：]?([0-9]{6})", normalized_head)
    if explicit_code:
        code = normalize_stock_code(explicit_code.group(1))
        return code or None

    try:
        registry = load_default_registry()
    except Exception:
        return None

    direct = registry.resolve_company_code(_filename_company_hint(text)) or registry.resolve_company_code(text)
    if direct:
        return direct

    normalized = normalize_text(text)
    candidates = sorted(registry.companies.values(), key=lambda company: len(company.company_name), reverse=True)
    for company in candidates:
        names = (company.company_name, company.stock_abbr)
        if any(name and normalize_text(name) in normalized for name in names):
            return company.stock_code
    return None


def _filename_company_hint(value: str) -> str:
    name = Path(str(value or "")).stem
    return re.split(r"[:：_\-—\s]", name, maxsplit=1)[0]


def infer_period_from_text(text: str) -> tuple[int | None, str | None]:
    head = re.sub(r"\s+", "", (text or "")[:20000])
    patterns = (
        (r"(20\d{2})年年度报告", "FY"),
        (r"(20\d{2})年第一季度报告", "Q1"),
        (r"(20\d{2})年半年度报告", "HY"),
        (r"(20\d{2})年第三季度报告", "Q3"),
    )
    for pattern, period in patterns:
        match = re.search(pattern, head)
        if match:
            return int(match.group(1)), period
    return None, None


def infer_period_from_date(value: date) -> tuple[int, str]:
    month_day = (value.month, value.day)
    if month_day <= (4, 30):
        return value.year - 1, "FY"
    if month_day <= (8, 31):
        return value.year, "HY"
    if month_day <= (10, 31):
        return value.year, "Q3"
    return value.year, "FY"
