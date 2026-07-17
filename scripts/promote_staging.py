from __future__ import annotations

import argparse
import json
from decimal import Decimal

from finquery_agent.db import create_database_engine
from finquery_agent.ingestion.promotion import (
    DEFAULT_CORE_COVERAGE_THRESHOLD,
    DEFAULT_FIELD_COVERAGE_THRESHOLD,
    promote_run_to_formal_tables,
)


def progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0/0 0%"
    ratio = min(max(current / total, 0), 1)
    filled = round(width * ratio)
    percent = round(ratio * 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} {percent}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a financial_staging run into formal financial tables unless validation has FAIL issues.")
    parser.add_argument("run_id", nargs="?", type=int, help="Extraction run_id to promote.")
    parser.add_argument("--run-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive run id range to promote.")
    parser.add_argument("--core-threshold", type=Decimal, default=DEFAULT_CORE_COVERAGE_THRESHOLD)
    parser.add_argument("--field-threshold", type=Decimal, default=DEFAULT_FIELD_COVERAGE_THRESHOLD)
    parser.add_argument("--force", action="store_true", help="Deprecated compatibility flag; FAIL issues still block promotion.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    run_ids = []
    if args.run_id is not None:
        run_ids.append(args.run_id)
    if args.run_range:
        start, end = args.run_range
        if start > end:
            start, end = end, start
        run_ids.extend(range(start, end + 1))
    run_ids = sorted(dict.fromkeys(run_ids))
    if not run_ids:
        raise SystemExit("Provide run_id or --run-range.")

    engine = create_database_engine()
    results = []
    total = len(run_ids)
    for index, run_id in enumerate(run_ids, 1):
        result = promote_run_to_formal_tables(
            engine,
            run_id,
            core_threshold=args.core_threshold,
            field_threshold=args.field_threshold,
            force=args.force,
        )
        results.append(result)
        if not args.json and total > 1:
            print(
                f"{progress_bar(index, total)} run_id={result.run_id} status={result.status} "
                f"promoted={result.promoted} core={result.core_coverage_ratio:.2%} "
                f"fields={result.field_coverage_ratio:.2%}",
                flush=True,
            )

    if args.json:
        payload = [result.to_dict() for result in results] if len(results) > 1 else results[0].to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if total > 1:
        counts = {}
        promoted_count = 0
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
            promoted_count += 1 if result.promoted else 0
        print(f"summary status_counts={counts} promoted={promoted_count}/{total}")
        return

    result = results[0]
    print(
        f"run_id={result.run_id} status={result.status} promoted={result.promoted} "
        f"core={result.core_coverage_ratio:.2%}/{result.core_threshold:.2%} "
        f"fields={result.field_coverage_ratio:.2%}/{result.field_threshold:.2%}"
    )
    if result.promoted_tables:
        print("promoted_tables=" + ",".join(result.promoted_tables))
    if result.message:
        print(result.message)
    for issue in result.issues:
        print(f"- {issue}")


if __name__ == "__main__":
    main()
