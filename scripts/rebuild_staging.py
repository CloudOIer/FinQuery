from __future__ import annotations

import argparse

from finquery_agent.db import create_database_engine
from finquery_agent.ingestion.repository import rebuild_staging_for_run


def progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0/0 0%"
    ratio = min(max(current / total, 0), 1)
    filled = round(width * ratio)
    percent = round(ratio * 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} {percent}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild financial_staging from already extracted raw pages/tables.")
    parser.add_argument("--run-id", action="append", type=int, default=[], help="Run id to rebuild. Can be repeated.")
    parser.add_argument("--run-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive run id range to rebuild.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ids = list(args.run_id)
    if args.run_range:
        start, end = args.run_range
        if start > end:
            start, end = end, start
        run_ids.extend(range(start, end + 1))
    run_ids = sorted(dict.fromkeys(run_ids))
    if not run_ids:
        raise SystemExit("Provide --run-id or --run-range.")

    engine = create_database_engine()
    total = len(run_ids)
    print(f"Rebuilding staging for {total} run(s)")
    for index, run_id in enumerate(run_ids, 1):
        print(f"{progress_bar(index - 1, total)} rebuilding run_id={run_id}", flush=True)
        try:
            count = rebuild_staging_for_run(engine, run_id)
        except Exception as exc:
            print(f"{progress_bar(index, total)} failed run_id={run_id}: {exc}", flush=True)
            continue
        print(f"{progress_bar(index, total)} rebuilt run_id={run_id} staging_rows={count}", flush=True)


if __name__ == "__main__":
    main()
