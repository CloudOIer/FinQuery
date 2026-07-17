from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class ReportMetadata:
    source_path: Path
    stock_code: str | None = None
    report_year: int | None = None
    report_period: str | None = None
    file_date: date | None = None


@dataclass(frozen=True)
class ExtractedTable:
    page_number: int
    table_index: int
    markdown: str
    raw_rows: list[list[str]] = field(default_factory=list)
    section_title: str | None = None


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text_content: str
    markdown_content: str
    tables: tuple[ExtractedTable, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParsedPdfDocument:
    metadata: ReportMetadata
    markdown: str
    clean_markdown: str
    pages: tuple[ExtractedPage, ...]
    markdown_path: Path
    clean_markdown_path: Path
