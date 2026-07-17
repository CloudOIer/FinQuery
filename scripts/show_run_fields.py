from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from finquery_agent.db import create_database_engine
from finquery_agent.schema import load_default_registry
from finquery_agent.schema.models import FieldDefinition, TableDefinition

#===================================================================================================
# 这个脚本用于展示某次 extraction run（通过 run_id 指定）覆盖了哪些目标字段，字段值是什么，以及来源信息等。
# 主要功能包括：
# cd /data/lvyunfei/FinQuery
# conda activate finquery-agent

# 查看 run_id=14 的全部目标字段：covered + missing
# python scripts/show_run_fields.py 14

# 只看已覆盖字段
# python scripts/show_run_fields.py 14 --covered-only

# 只看某张表，比如利润表
# python scripts/show_run_fields.py 14 --table income_sheet

# 导出 Markdown 报告
# python scripts/show_run_fields.py 14 --format markdown --output data/evaluation/run_14_fields.md

# 导出 CSV，适合 Excel/表格筛选
# python scripts/show_run_fields.py 14 --format csv --output data/evaluation/run_14_fields.csv

# 导出 JSON，适合程序继续处理
# python scripts/show_run_fields.py 14 --format json --output data/evaluation/run_14_fields.json
#===================================================================================================

@dataclass(frozen=True)
class CoveredValue:
    target_table: str
    target_field: str
    raw_value: str | None
    raw_unit: str | None
    standard_value: Decimal | None
    standard_unit: str | None
    source_label: str | None
    source_period_label: str | None
    period_scope: str | None
    is_derived: bool
    derivation_formula: str | None
    page_number: int | None
    confidence: Decimal | None


def main() -> None:
    parser = argparse.ArgumentParser(description="Show field coverage and values for a financial_staging run_id.")
    parser.add_argument("run_id", type=int, help="Extraction run_id to inspect, for example 14.")
    parser.add_argument("--covered-only", action="store_true", help="Only print fields covered by this run.")
    parser.add_argument("--table", choices=("core_performance_indicators_sheet", "balance_sheet", "income_sheet", "cash_flow_sheet"), help="Limit output to one target table.")
    parser.add_argument("--format", choices=("text", "markdown", "csv", "json"), default="text", help="Output format. Default: text.")
    parser.add_argument("--output", help="Write output to this file instead of stdout.")
    args = parser.parse_args()

    registry = load_default_registry()
    engine = create_database_engine()
    with engine.connect() as connection:
        run_info = _load_run_info(connection, args.run_id)
        if run_info is None:
            raise SystemExit(f"run_id={args.run_id} not found in extraction_runs.")
        covered_values = _load_covered_values(connection, args.run_id)

    all_rows = _build_rows(registry.tables, covered_values, table_filter=args.table, covered_only=False)
    rows = [row for row in all_rows if row["status"] == "covered"] if args.covered_only else all_rows
    summary = _build_summary(args.run_id, run_info, all_rows)
    output = _render(summary, rows, args.format)

    if args.output:
        mode = "w"
        newline = "" if args.format == "csv" else None
        with open(args.output, mode, encoding="utf-8", newline=newline) as file:
            file.write(output)
        return
    sys.stdout.write(output)
    if output and not output.endswith("\n"):
        sys.stdout.write("\n")


def _load_run_info(connection, run_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        text(
            """
            SELECT
                er.run_id,
                er.tool_name,
                er.status,
                er.started_at,
                sd.stock_code,
                sd.report_year,
                sd.report_period,
                sd.source_path
            FROM extraction_runs er
            JOIN source_documents sd ON sd.document_id = er.document_id
            WHERE er.run_id = :run_id
            """
        ),
        {"run_id": run_id},
    ).mappings().first()
    return dict(row) if row else None


def _load_covered_values(connection, run_id: int) -> dict[tuple[str, str], CoveredValue]:
    rows = connection.execute(
        text(
            """
            SELECT
                target_table,
                target_field,
                raw_value,
                raw_unit,
                standard_value,
                standard_unit,
                source_label,
                source_period_label,
                period_scope,
                is_derived,
                derivation_formula,
                page_number,
                confidence
            FROM financial_staging
            WHERE run_id = :run_id
            ORDER BY target_table, target_field, is_derived DESC, confidence DESC NULLS LAST
            """
        ),
        {"run_id": run_id},
    ).mappings().all()

    values: dict[tuple[str, str], CoveredValue] = {}
    for row in rows:
        key = (row["target_table"], row["target_field"])
        values.setdefault(
            key,
            CoveredValue(
                target_table=row["target_table"],
                target_field=row["target_field"],
                raw_value=row["raw_value"],
                raw_unit=row["raw_unit"],
                standard_value=row["standard_value"],
                standard_unit=row["standard_unit"],
                source_label=row["source_label"],
                source_period_label=row["source_period_label"],
                period_scope=row["period_scope"],
                is_derived=bool(row["is_derived"]),
                derivation_formula=row["derivation_formula"],
                page_number=row["page_number"],
                confidence=row["confidence"],
            ),
        )
    return values


def _build_rows(
    tables: dict[str, TableDefinition],
    covered_values: dict[tuple[str, str], CoveredValue],
    table_filter: str | None,
    covered_only: bool,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for table in tables.values():
        if table_filter and table.name != table_filter:
            continue
        for field in table.fields:
            if field.is_dimension:
                continue
            covered = covered_values.get((table.name, field.name))
            if covered_only and covered is None:
                continue
            output_rows.append(_row_from_field(table, field, covered))
    return output_rows


def _row_from_field(table: TableDefinition, field: FieldDefinition, covered: CoveredValue | None) -> dict[str, Any]:
    value = _format_decimal(covered.standard_value) if covered and covered.standard_value is not None else ""
    raw_value = covered.raw_value if covered and covered.raw_value is not None else ""
    return {
        "status": "covered" if covered else "missing",
        "table": table.name,
        "table_cn": table.chinese_name,
        "field": field.name,
        "field_cn": field.chinese_name,
        "description": field.description,
        "value": value,
        "unit": covered.standard_unit if covered and covered.standard_unit else "",
        "raw_value": raw_value,
        "raw_unit": covered.raw_unit if covered and covered.raw_unit else "",
        "source_label": covered.source_label if covered and covered.source_label else "",
        "source_period_label": covered.source_period_label if covered and covered.source_period_label else "",
        "period_scope": covered.period_scope if covered and covered.period_scope else "",
        "is_derived": covered.is_derived if covered else False,
        "derivation_formula": covered.derivation_formula if covered and covered.derivation_formula else "",
        "page_number": covered.page_number if covered and covered.page_number is not None else "",
        "confidence": _format_decimal(covered.confidence) if covered and covered.confidence is not None else "",
    }


def _build_summary(run_id: int, run_info: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered_count = sum(1 for row in rows if row["status"] == "covered")
    total_count = len(rows)
    ratio = Decimal(covered_count) / Decimal(total_count) if total_count else Decimal("0")
    return {
        "run_id": run_id,
        "stock_code": run_info.get("stock_code"),
        "report_year": run_info.get("report_year"),
        "report_period": run_info.get("report_period"),
        "tool_name": run_info.get("tool_name"),
        "status": run_info.get("status"),
        "source_path": run_info.get("source_path"),
        "covered_count": covered_count,
        "total_count": total_count,
        "coverage_ratio": f"{ratio:.2%}",
    }


def _render(summary: dict[str, Any], rows: list[dict[str, Any]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps({"summary": summary, "fields": rows}, ensure_ascii=False, indent=2, default=str)
    if output_format == "csv":
        return _render_csv(rows)
    if output_format == "markdown":
        return _render_markdown(summary, rows)
    return _render_text(summary, rows)


def _render_text(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"run_id={summary['run_id']} stock_code={summary['stock_code']} {summary['report_year']}/{summary['report_period']} tool={summary['tool_name']} status={summary['status']}",
        f"field_coverage={summary['covered_count']}/{summary['total_count']} ({summary['coverage_ratio']})",
        "",
    ]
    for row in rows:
        value = row["value"] or "-"
        unit = row["unit"] or ""
        derived = " derived" if row["is_derived"] else ""
        source = f" source={row['source_label']}" if row["source_label"] else ""
        lines.append(
            f"[{row['status']}{derived}] {row['table']}.{row['field']} | {row['field_cn']} | {row['description']} | value={value}{unit}{source}"
        )
    return "\n".join(lines)


def _render_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# run_id={summary['run_id']} 字段覆盖情况",
        "",
        f"- 股票代码：{summary['stock_code']}",
        f"- 报告期：{summary['report_year']}/{summary['report_period']}",
        f"- 工具：{summary['tool_name']}",
        f"- 覆盖率：{summary['covered_count']}/{summary['total_count']} ({summary['coverage_ratio']})",
        "",
        "| 状态 | 表 | 字段名 | 中文含义 | 字段说明 | 字段值 | 单位 | 来源标签 | 期间标签 | 派生公式 |",
        "|---|---|---|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _escape_markdown(str(value))
                for value in (
                    row["status"],
                    row["table_cn"],
                    row["field"],
                    row["field_cn"],
                    row["description"],
                    row["value"],
                    row["unit"],
                    row["source_label"],
                    row["source_period_label"],
                    row["derivation_formula"],
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _render_csv(rows: list[dict[str, Any]]) -> str:
    fieldnames = [
        "status",
        "table",
        "table_cn",
        "field",
        "field_cn",
        "description",
        "value",
        "unit",
        "raw_value",
        "raw_unit",
        "source_label",
        "source_period_label",
        "period_scope",
        "is_derived",
        "derivation_formula",
        "page_number",
        "confidence",
    ]
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _format_decimal(value: Decimal | None) -> str:
    if value is None:
        return ""
    normalized = value.normalize()
    return format(normalized, "f")


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
