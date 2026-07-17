from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from finquery_agent.db import create_database_engine
from finquery_agent.nl2sql import RuleBasedIntentEngine, SQLBuilder
from finquery_agent.nl2sql.executor import QueryExecutor
from finquery_agent.nl2sql.planner import split_intent_by_table
from finquery_agent.schema import load_default_registry


DEFAULT_INPUT = Path("第一批数据/附件4：问题汇总.CSV")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task2 questions without golden answers by checking parse/build/execute status.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Question CSV path.")
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/task2_question_eval.csv"), help="Output report path.")
    parser.add_argument("--format", choices=("csv", "markdown", "json"), default="csv")
    parser.add_argument("--no-execute", action="store_true", help="Only parse/build SQL, do not execute.")
    parser.add_argument("--sample-rows", type=int, default=2, help="Number of result rows to include as sample.")
    args = parser.parse_args()

    registry = load_default_registry()
    db_engine = create_database_engine()
    intent_engine = RuleBasedIntentEngine(registry)
    sql_builder = SQLBuilder(registry)
    executor = QueryExecutor(db_engine, registry)

    rows = []
    for item in _load_questions(args.input):
        rows.append(_evaluate_question(item, intent_engine, sql_builder, executor, execute=not args.no_execute, sample_rows=args.sample_rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        args.output.write_text(json.dumps({"summary": _summary(rows), "items": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.format == "markdown":
        args.output.write_text(_render_markdown(rows), encoding="utf-8")
    else:
        _write_csv(args.output, rows)

    print(json.dumps(_summary(rows), ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


def _load_questions(path: Path) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            payload = json.loads(row["问题"])
            for index, item in enumerate(payload, 1):
                questions.append(
                    {
                        "id": row["编号"],
                        "question_type": row["问题类型"],
                        "question_index": str(index),
                        "question": item["Q"],
                    }
                )
    return questions


def _evaluate_question(
    item: dict[str, str],
    intent_engine: RuleBasedIntentEngine,
    sql_builder: SQLBuilder,
    executor: QueryExecutor,
    execute: bool,
    sample_rows: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {**item, "status": "unknown"}
    try:
        intent = intent_engine.parse(item["question"])
        result.update(
            {
                "intent_type": intent.intent_type,
                "needs_clarification": intent.needs_clarification,
                "clarification": intent.clarification.question if intent.clarification else "",
                "metrics": ";".join(intent.metrics),
                "company_codes": ";".join(intent.company_codes),
                "years": ";".join(str(year) for year in intent.years),
                "periods": ";".join(intent.periods),
                "query_count": 0,
                "tables": "",
                "sql": "",
                "row_counts": "",
                "sample_rows": "",
                "error": "",
            }
        )
        if intent.needs_clarification:
            result["status"] = "clarification"
            return result

        query_items = []
        table_names = []
        row_counts = []
        sample_payload = []
        for sub_intent in split_intent_by_table(intent, intent_engine.registry):
            query = sql_builder.build(sub_intent.to_dsl())
            table_names.append(query.table_name)
            query_items.append(query.sql)
            if execute:
                query_result = executor.execute(query)
                row_counts.append(query_result.row_count)
                sample_payload.append(list(query_result.rows[:sample_rows]))
        result["query_count"] = len(query_items)
        result["tables"] = ";".join(table_names)
        result["sql"] = "\n---\n".join(query_items)
        result["row_counts"] = ";".join(str(count) for count in row_counts)
        result["sample_rows"] = json.dumps(sample_payload, ensure_ascii=False)
        if not execute:
            result["status"] = "sql_built"
        elif any(count > 0 for count in row_counts):
            result["status"] = "ok_nonempty"
        else:
            result["status"] = "ok_empty"
        return result
    except Exception as exc:
        result.setdefault("intent_type", "")
        result.setdefault("needs_clarification", "")
        result.setdefault("clarification", "")
        result.setdefault("metrics", "")
        result.setdefault("company_codes", "")
        result.setdefault("years", "")
        result.setdefault("periods", "")
        result.setdefault("query_count", 0)
        result.setdefault("tables", "")
        result.setdefault("sql", "")
        result.setdefault("row_counts", "")
        result.setdefault("sample_rows", "")
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    by_type: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
        question_type = row["question_type"]
        by_type.setdefault(question_type, {})[status] = by_type.setdefault(question_type, {}).get(status, 0) + 1
    return {"total": len(rows), "status_counts": counts, "by_question_type": by_type}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "question_type",
        "question_index",
        "question",
        "status",
        "intent_type",
        "needs_clarification",
        "clarification",
        "metrics",
        "company_codes",
        "years",
        "periods",
        "query_count",
        "tables",
        "row_counts",
        "error",
        "sql",
        "sample_rows",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Task2 问题集 SQL 可执行性评估", "", "## 汇总", "", "```json", json.dumps(_summary(rows), ensure_ascii=False, indent=2), "```", "", "## 明细", ""]
    lines.append("| 编号 | 类型 | 状态 | 意图 | 指标 | 公司 | 年份 | 报告期 | 行数 | 问题 | 错误/澄清 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _escape_md(str(value))
                for value in (
                    row["id"],
                    row["question_type"],
                    row["status"],
                    row.get("intent_type", ""),
                    row.get("metrics", ""),
                    row.get("company_codes", ""),
                    row.get("years", ""),
                    row.get("periods", ""),
                    row.get("row_counts", ""),
                    row["question"],
                    row.get("error") or row.get("clarification", ""),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


if __name__ == "__main__":
    main()
