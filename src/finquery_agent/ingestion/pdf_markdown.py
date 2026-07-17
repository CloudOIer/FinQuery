from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

from finquery_agent.config import get_settings
from finquery_agent.ingestion.clean_markdown import clean_markdown
from finquery_agent.ingestion.metadata import infer_report_metadata, refine_report_metadata_from_text
from finquery_agent.ingestion.models import ExtractedPage, ExtractedTable, ParsedPdfDocument


def parse_pdf_to_markdown(pdf_path: str | Path, output_dir: Path | None = None) -> ParsedPdfDocument:
    try:
        pdfplumber = import_module("pdfplumber")
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required for local PDF parsing. Install the pdf extra dependencies.") from exc

    pdf_path = Path(pdf_path)
    settings = get_settings()
    base_output = output_dir or (settings.project_root / "data" / "extracted_markdown")
    markdown_dir = base_output / "raw"
    clean_dir = base_output / "clean"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    pages: list[ExtractedPage] = []
    document_parts: list[str] = []
    table_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            page_parts = [f"## Page {page_number}"]
            text_content = page.extract_text() or ""
            if text_content.strip():
                page_parts.append(text_content.strip())

            tables = []
            for raw_table in page.extract_tables() or []:
                table_counter += 1
                markdown = table_to_markdown(raw_table)
                if not markdown.strip():
                    continue
                table = ExtractedTable(
                    page_number=page_number,
                    table_index=table_counter,
                    markdown=markdown,
                    raw_rows=_normalize_table(raw_table),
                    section_title=_nearest_section_title(text_content),
                )
                tables.append(table)
                page_parts.append(f"### Table {page_number}-{table_counter}")
                page_parts.append(markdown)

            page_markdown = "\n\n".join(page_parts)
            document_parts.append(page_markdown)
            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text_content=text_content,
                    markdown_content=page_markdown,
                    tables=tuple(tables),
                )
            )

    markdown = "\n\n".join(document_parts)
    clean = clean_markdown(markdown)

    markdown_path = markdown_dir / f"{pdf_path.stem}.md"
    clean_path = clean_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    clean_path.write_text(clean, encoding="utf-8")

    return ParsedPdfDocument(
        metadata=refine_report_metadata_from_text(infer_report_metadata(pdf_path), markdown),
        markdown=markdown,
        clean_markdown=clean,
        pages=tuple(pages),
        markdown_path=markdown_path,
        clean_markdown_path=clean_path,
    )


def table_to_markdown(table: list[list[object]]) -> str:
    rows = _normalize_table(table)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    separator = ["---"] * width
    body = [header, separator, *padded[1:]]
    return "\n".join("| " + " | ".join(_escape_cell(cell) for cell in row) + " |" for row in body)


def table_to_json(table: ExtractedTable) -> str:
    return json.dumps(table.raw_rows, ensure_ascii=False)


def _normalize_table(table: list[list[object]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table or []:
        cleaned = ["" if cell is None else str(cell).replace("\n", " ").replace("\r", " ").strip() for cell in row]
        if any(cleaned):
            rows.append(cleaned)
    return rows


def _escape_cell(cell: str) -> str:
    return cell.replace("|", "\\|")


def _nearest_section_title(text_content: str) -> str | None:
    for line in reversed((text_content or "").splitlines()[:20]):
        stripped = line.strip()
        if any(keyword in stripped for keyword in ("资产负债表", "利润表", "现金流量表", "主要会计数据", "主要财务指标")):
            return stripped
    return None
