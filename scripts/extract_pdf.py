from __future__ import annotations

import argparse
from pathlib import Path

from finquery_agent.db import create_database_engine
from finquery_agent.ingestion import run_pdf_ingestion
from finquery_agent.ingestion.mineru import MinerUOptions
from finquery_agent.ingestion.promotion import promote_run_to_formal_tables


def progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0/0 0%"
    ratio = min(max(current / total, 0), 1)
    filled = round(width * ratio)
    percent = round(ratio * 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total} {percent}%"


def iter_pdfs(paths: list[str], limit: int | None = None) -> list[Path]:
    results: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            results.extend(sorted(path.rglob("*.pdf")))
        elif path.suffix.lower() == ".pdf":
            results.append(path)
    return results[:limit] if limit else results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract financial report PDFs into raw ingestion tables")
    parser.add_argument("paths", nargs="+", help="PDF files or directories")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parser", choices=("pdfplumber", "mineru"), default="pdfplumber")
    parser.add_argument("--mineru-model-version", default="vlm")
    parser.add_argument("--mineru-poll-interval", type=int, default=5)
    parser.add_argument("--mineru-timeout", type=int, default=900)
    parser.add_argument("--mineru-max-pages", type=int, default=200)
    parser.add_argument("--promote", action="store_true", help="Promote each completed staging run into formal financial tables unless validation has FAIL issues.")
    args = parser.parse_args()

    pdfs = iter_pdfs(args.paths, limit=args.limit)
    if not pdfs:
        print("No PDF files found")
        return

    total = len(pdfs)
    print(f"Processing {total} PDF file(s) with parser={args.parser}")
    engine = create_database_engine() if args.promote else None
    for index, pdf in enumerate(pdfs, 1):
        print(f"{progress_bar(index - 1, total)} extracting {pdf}", flush=True)
        mineru_options = MinerUOptions(
            model_version=args.mineru_model_version,
            poll_interval_seconds=args.mineru_poll_interval,
            timeout_seconds=args.mineru_timeout,
            max_pages_per_file=args.mineru_max_pages,
        )
        try:
            run_id = run_pdf_ingestion(pdf, parser=args.parser, mineru_options=mineru_options)
        except Exception as exc:
            print(f"{progress_bar(index, total)} failed: {pdf} ({exc})", flush=True)
            continue
        if args.promote and engine is not None:
            promotion = promote_run_to_formal_tables(engine, run_id)
            print(
                f"{progress_bar(index, total)} stored extraction run_id={run_id} "
                f"promotion_status={promotion.status} promoted={promotion.promoted}",
                flush=True,
            )
            continue
        print(f"{progress_bar(index, total)} stored extraction run_id={run_id}", flush=True)


if __name__ == "__main__":
    main()
