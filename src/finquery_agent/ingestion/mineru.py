from __future__ import annotations

import io
import os
import re
import time
import zipfile
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from finquery_agent.config import get_settings
from finquery_agent.ingestion.clean_markdown import clean_markdown
from finquery_agent.ingestion.metadata import infer_report_metadata, refine_report_metadata_from_text
from finquery_agent.ingestion.models import ExtractedPage, ExtractedTable, ParsedPdfDocument


MINERU_BATCH_URL = "https://mineru.net/api/v4/file-urls/batch"
MINERU_RESULT_URL = "https://mineru.net/api/v4/extract-results/batch/{batch_id}"


@dataclass(frozen=True)
class MinerUOptions:
    model_version: str = "vlm"
    poll_interval_seconds: int = 5
    timeout_seconds: int = 900
    max_pages_per_file: int = 200


@dataclass(frozen=True)
class PdfChunk:
    path: Path
    start_page: int
    end_page: int


def parse_pdf_with_mineru(
    pdf_path: str | Path,
    output_dir: Path | None = None,
    options: MinerUOptions | None = None,
) -> ParsedPdfDocument:
    token = os.getenv("MINERU_API_TOKEN")
    if not token:
        raise RuntimeError("MINERU_API_TOKEN is required to use the MinerU parser.")

    pdf_path = Path(pdf_path)
    options = options or MinerUOptions(model_version=os.getenv("MINERU_MODEL_VERSION", "vlm"))
    markdown = request_mineru_markdown_with_splitting(pdf_path, token=token, options=options)
    return parsed_document_from_markdown(pdf_path, markdown, output_dir=output_dir, parser_name="mineru")


def request_mineru_markdown_with_splitting(pdf_path: Path, token: str, options: MinerUOptions) -> str:
    page_count = count_pdf_pages(pdf_path)
    if page_count <= options.max_pages_per_file:
        return _request_mineru_markdown(pdf_path, token=token, options=options)

    with TemporaryDirectory(prefix="finquery_mineru_") as tmp_dir:
        chunks = split_pdf_into_chunks(pdf_path, Path(tmp_dir), max_pages_per_file=options.max_pages_per_file)
        markdown_chunks = []
        for chunk in chunks:
            chunk_markdown = _request_mineru_markdown(chunk.path, token=token, options=options)
            markdown_chunks.append((chunk, chunk_markdown))
        return merge_mineru_markdown_chunks(markdown_chunks)


def count_pdf_pages(pdf_path: str | Path) -> int:
    fitz = _import_fitz()
    with fitz.open(pdf_path) as document:
        return int(document.page_count)


def split_pdf_into_chunks(pdf_path: str | Path, output_dir: Path, max_pages_per_file: int = 200) -> tuple[PdfChunk, ...]:
    if max_pages_per_file < 1:
        raise ValueError("max_pages_per_file must be positive")

    fitz = _import_fitz()
    pdf_path = Path(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[PdfChunk] = []
    with fitz.open(pdf_path) as source:
        total_pages = int(source.page_count)
        if total_pages <= max_pages_per_file:
            return (PdfChunk(path=pdf_path, start_page=1, end_page=total_pages),)

        for start_index in range(0, total_pages, max_pages_per_file):
            end_index = min(start_index + max_pages_per_file, total_pages)
            chunk_number = len(chunks) + 1
            chunk_path = output_dir / f"{pdf_path.stem}.part{chunk_number:03d}.p{start_index + 1:04d}-{end_index:04d}.pdf"
            with fitz.open() as target:
                target.insert_pdf(source, from_page=start_index, to_page=end_index - 1)
                target.save(chunk_path)
            chunks.append(PdfChunk(path=chunk_path, start_page=start_index + 1, end_page=end_index))
    return tuple(chunks)


def merge_mineru_markdown_chunks(markdown_chunks: list[tuple[PdfChunk, str]]) -> str:
    merged_parts: list[str] = []
    for chunk, markdown in markdown_chunks:
        offset = chunk.start_page - 1
        adjusted = _offset_markdown_page_headings(markdown, offset)
        if not _has_page_headings(adjusted):
            adjusted = f"## Page {chunk.start_page}\n\n{adjusted.strip()}"
        merged_parts.append(f"<!-- MinerU chunk pages {chunk.start_page}-{chunk.end_page} -->\n\n{adjusted.strip()}")
    return "\n\n".join(part for part in merged_parts if part.strip())


def parsed_document_from_markdown(
    pdf_path: str | Path,
    markdown: str,
    output_dir: Path | None = None,
    parser_name: str = "mineru",
) -> ParsedPdfDocument:
    pdf_path = Path(pdf_path)
    settings = get_settings()
    base_output = output_dir or (settings.project_root / "data" / "extracted_markdown")
    markdown_dir = base_output / parser_name / "raw"
    clean_dir = base_output / parser_name / "clean"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    clean = clean_markdown(markdown)
    pages = tuple(_pages_from_markdown(markdown))

    markdown_path = markdown_dir / f"{pdf_path.stem}.md"
    clean_path = clean_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    clean_path.write_text(clean, encoding="utf-8")

    return ParsedPdfDocument(
        metadata=refine_report_metadata_from_text(infer_report_metadata(pdf_path), markdown),
        markdown=markdown,
        clean_markdown=clean,
        pages=pages,
        markdown_path=markdown_path,
        clean_markdown_path=clean_path,
    )


def _request_mineru_markdown(pdf_path: Path, token: str, options: MinerUOptions) -> str:
    try:
        requests = import_module("requests")
    except ImportError as exc:
        raise RuntimeError("requests is required to use the MinerU parser.") from exc

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    batch_response = requests.post(
        MINERU_BATCH_URL,
        headers=headers,
        json={
            "files": [{"name": pdf_path.name, "data_id": pdf_path.stem[:120]}],
            "model_version": options.model_version,
        },
        timeout=60,
    )
    batch_payload = _checked_mineru_json(batch_response, "申请 MinerU 上传链接失败")
    batch_id = batch_payload["data"]["batch_id"]
    upload_url = batch_payload["data"]["file_urls"][0]

    with pdf_path.open("rb") as file:
        upload_response = requests.put(upload_url, data=file, timeout=300)
    if upload_response.status_code not in (200, 201):
        raise RuntimeError(f"MinerU 文件上传失败: HTTP {upload_response.status_code}")

    deadline = time.monotonic() + options.timeout_seconds
    result_url = MINERU_RESULT_URL.format(batch_id=batch_id)
    while time.monotonic() < deadline:
        result_response = requests.get(result_url, headers=headers, timeout=60)
        result_payload = _checked_mineru_json(result_response, "查询 MinerU 解析结果失败")
        extract_result = result_payload["data"]["extract_result"][0]
        state = extract_result["state"]
        if state == "done":
            zip_url = extract_result.get("full_zip_url")
            if not zip_url:
                raise RuntimeError("MinerU 解析完成，但未返回 full_zip_url。")
            zip_response = requests.get(zip_url, timeout=300)
            if zip_response.status_code != 200:
                raise RuntimeError(f"下载 MinerU 结果 ZIP 失败: HTTP {zip_response.status_code}")
            return extract_markdown_from_zip_bytes(zip_response.content)
        if state == "failed":
            raise RuntimeError(f"MinerU 解析失败: {extract_result.get('err_msg', 'unknown error')}")
        time.sleep(options.poll_interval_seconds)

    raise TimeoutError(f"MinerU parsing timed out after {options.timeout_seconds} seconds for {pdf_path}")


def extract_markdown_from_zip_bytes(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        md_files = [name for name in archive.namelist() if name.lower().endswith(".md")]
        if not md_files:
            raise RuntimeError(f"MinerU ZIP 中未找到 Markdown 文件: {archive.namelist()}")
        target = _choose_markdown_file(md_files)
        with archive.open(target) as file:
            return file.read().decode("utf-8")


def _checked_mineru_json(response: Any, message: str) -> dict:
    if response.status_code != 200:
        raise RuntimeError(f"{message}: HTTP {response.status_code}, {response.text}")
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"{message}: {payload}")
    return payload


def _choose_markdown_file(md_files: list[str]) -> str:
    for name in md_files:
        lowered = name.lower()
        if "full.md" in lowered or "auto" in lowered:
            return name
    return md_files[0]


def _import_fitz():
    try:
        return import_module("fitz")
    except ImportError as exc:
        raise RuntimeError("pymupdf is required to split PDFs for MinerU parsing.") from exc


def _offset_markdown_page_headings(markdown: str, offset: int) -> str:
    if offset == 0:
        return markdown

    def replace(match: re.Match[str]) -> str:
        return f"{match.group('prefix')}{int(match.group('page')) + offset}{match.group('suffix')}"

    return re_page_heading().sub(replace, markdown)


def _has_page_headings(markdown: str) -> bool:
    return bool(re_page_heading().search(markdown))


def _pages_from_markdown(markdown: str) -> list[ExtractedPage]:
    sections = _split_markdown_pages(markdown)
    pages: list[ExtractedPage] = []
    table_counter = 0
    for page_number, content in sections:
        tables: list[ExtractedTable] = []
        for raw_table in _extract_tables(content):
            table_counter += 1
            tables.append(
                ExtractedTable(
                    page_number=page_number,
                    table_index=table_counter,
                    markdown=_table_rows_to_markdown(raw_table),
                    raw_rows=raw_table,
                    section_title=_nearest_section_title(content),
                )
            )
        text_content = _strip_tables(content)
        pages.append(
            ExtractedPage(
                page_number=page_number,
                text_content=text_content,
                markdown_content=content,
                tables=tuple(tables),
            )
        )
    return pages or [ExtractedPage(page_number=1, text_content=markdown, markdown_content=markdown)]


def _split_markdown_pages(markdown: str) -> list[tuple[int, str]]:
    page_heading = re_page_heading()
    matches = list(page_heading.finditer(markdown))
    if not matches:
        return [(1, markdown)]

    sections: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections.append((int(match.group("page")), markdown[start:end].strip()))
    return sections


def _extract_markdown_tables(markdown: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if _is_table_line(line):
            current.append(line)
            continue
        if current:
            rows = _parse_markdown_table(current)
            if rows:
                tables.append(rows)
            current = []
    if current:
        rows = _parse_markdown_table(current)
        if rows:
            tables.append(rows)
    return tables


def _extract_tables(markdown: str) -> list[list[list[str]]]:
    return [*_extract_markdown_tables(markdown), *_extract_html_tables(markdown)]


def _extract_html_tables(markdown: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    for match in re.finditer(r"<table\b.*?</table>", markdown, flags=re.IGNORECASE | re.DOTALL):
        rows = _parse_html_table(match.group(0))
        if rows:
            tables.append(rows)
    return tables


def _parse_html_table(html: str) -> list[list[str]]:
    parser = _HtmlTableParser()
    parser.feed(html)
    parser.close()
    return parser.rows


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._column_index = 0
        self._cell_parts: list[str] | None = None
        self._cell_colspan = 1
        self._cell_rowspan = 1
        self._rowspans: dict[int, tuple[str, int]] = {}
        self._table_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            return
        if self._table_depth < 1:
            return
        if tag == "tr":
            self._row = []
            self._column_index = 0
            return
        if tag in {"td", "th"} and self._row is not None:
            self._consume_rowspans()
            attributes = dict(attrs)
            self._cell_colspan = _positive_int(attributes.get("colspan"), default=1)
            self._cell_rowspan = _positive_int(attributes.get("rowspan"), default=1)
            self._cell_parts = []
            return
        if tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._row is not None and self._cell_parts is not None:
            value = _normalize_cell_text("".join(self._cell_parts))
            for offset in range(self._cell_colspan):
                cell_value = value if offset == 0 else ""
                self._row.append(cell_value)
                if self._cell_rowspan > 1:
                    self._rowspans[self._column_index] = (cell_value, self._cell_rowspan - 1)
                self._column_index += 1
            self._cell_parts = None
            return
        if tag == "tr" and self._row is not None:
            self._consume_rowspans()
            if any(cell.strip() for cell in self._row):
                self.rows.append(self._row)
            self._row = None
            return
        if tag == "table" and self._table_depth > 0:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def _consume_rowspans(self) -> None:
        if self._row is None:
            return
        while self._column_index in self._rowspans:
            value, remaining = self._rowspans[self._column_index]
            self._row.append(value)
            if remaining <= 1:
                del self._rowspans[self._column_index]
            else:
                self._rowspans[self._column_index] = (value, remaining - 1)
            self._column_index += 1


def _positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _normalize_cell_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
    rows = [_split_table_row(line) for line in lines if not _is_separator_line(line)]
    return [row for row in rows if any(cell.strip() for cell in row)]


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.replace("\\|", "|").strip() for cell in stripped.split("|")]


def _table_rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    return "\n".join(
        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
        for row in [padded[0], ["---"] * width, *padded[1:]]
    )


def _strip_markdown_tables(markdown: str) -> str:
    return "\n".join(line for line in markdown.splitlines() if not _is_table_line(line))


def _strip_tables(markdown: str) -> str:
    without_html = re.sub(r"<table\b.*?</table>", "", markdown, flags=re.IGNORECASE | re.DOTALL)
    return _strip_markdown_tables(without_html)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _is_separator_line(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)


def _nearest_section_title(content: str) -> str | None:
    for line in reversed(content.splitlines()[:40]):
        stripped = line.strip("# ").strip()
        if any(keyword in stripped for keyword in ("资产负债表", "利润表", "现金流量表", "主要会计数据", "主要财务指标")):
            return stripped
    return None


def re_page_heading():
    return re.compile(
        r"^(?P<prefix>#{1,3}\s*(?:Page|第)\s*)(?P<page>\d+)(?P<suffix>\s*(?:页)?\s*)$",
        re.IGNORECASE | re.MULTILINE,
    )