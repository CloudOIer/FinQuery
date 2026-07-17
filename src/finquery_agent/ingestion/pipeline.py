from __future__ import annotations

from pathlib import Path

from finquery_agent.db import create_database_engine
from finquery_agent.ingestion.mineru import MinerUOptions, parse_pdf_with_mineru
from finquery_agent.ingestion.pdf_markdown import parse_pdf_to_markdown
from finquery_agent.ingestion.repository import store_parsed_document


def run_pdf_ingestion(
    pdf_path: str | Path,
    output_dir: Path | None = None,
    parser: str = "pdfplumber",
    mineru_options: MinerUOptions | None = None,
) -> int:
    if parser == "pdfplumber":
        parsed = parse_pdf_to_markdown(pdf_path, output_dir=output_dir)
    elif parser == "mineru":
        parsed = parse_pdf_with_mineru(pdf_path, output_dir=output_dir, options=mineru_options)
    else:
        raise ValueError(f"Unsupported PDF parser: {parser}")
    engine = create_database_engine()
    return store_parsed_document(engine, parsed, tool_name=parser)
