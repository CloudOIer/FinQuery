import io
import zipfile
from pathlib import Path

import pytest

from finquery_agent.ingestion.mineru import (
    PdfChunk,
    count_pdf_pages,
    extract_markdown_from_zip_bytes,
    merge_mineru_markdown_chunks,
    parse_pdf_with_mineru,
    parsed_document_from_markdown,
    split_pdf_into_chunks,
)


def test_extract_markdown_from_mineru_zip_prefers_full_md():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("other.md", "other")
        archive.writestr("full.md", "# full")

    assert extract_markdown_from_zip_bytes(buffer.getvalue()) == "# full"


def test_parsed_document_from_mineru_markdown_extracts_tables(tmp_path):
    markdown = """
# 2023 年年度报告

## Page 1

### 合并利润表

| 项目 | 2023年度 |
| --- | --- |
| 营业收入 | 123 |
| 净利润 | 45 |
"""

    parsed = parsed_document_from_markdown(Path("688222_20240425_TEST.pdf"), markdown, output_dir=tmp_path)

    assert parsed.metadata.report_year == 2023
    assert parsed.metadata.report_period == "FY"
    assert len(parsed.pages) == 1
    assert len(parsed.pages[0].tables) == 1
    assert parsed.pages[0].tables[0].raw_rows[1] == ["营业收入", "123"]
    assert parsed.markdown_path.exists()
    assert parsed.clean_markdown_path.exists()


def test_parsed_document_from_mineru_markdown_extracts_html_tables(tmp_path):
    markdown = """
## Page 1

### 合并利润表

<table><tr><td>项目</td><td>2023年度</td><td>2022年度</td></tr><tr><td>一、营业总收入</td><td>371,324,936.81</td><td>329,650,037.29</td></tr><tr><td>三、利润总额</td><td>45,489,063.23</td><td>20,571,701.75</td></tr></table>
"""

    parsed = parsed_document_from_markdown(Path("688222_20240425_TEST.pdf"), markdown, output_dir=tmp_path)

    assert len(parsed.pages) == 1
    assert len(parsed.pages[0].tables) == 1
    assert parsed.pages[0].tables[0].raw_rows == [
        ["项目", "2023年度", "2022年度"],
        ["一、营业总收入", "371,324,936.81", "329,650,037.29"],
        ["三、利润总额", "45,489,063.23", "20,571,701.75"],
    ]
    assert "<table>" not in parsed.pages[0].text_content


def test_mineru_parser_requires_token(monkeypatch):
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="MINERU_API_TOKEN"):
        parse_pdf_with_mineru("dummy.pdf")


def test_split_pdf_into_chunks_respects_page_limit(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as document:
        for _ in range(5):
            document.new_page()
        document.save(pdf_path)

    chunks = split_pdf_into_chunks(pdf_path, tmp_path / "chunks", max_pages_per_file=2)

    assert [(chunk.start_page, chunk.end_page) for chunk in chunks] == [(1, 2), (3, 4), (5, 5)]
    assert [count_pdf_pages(chunk.path) for chunk in chunks] == [2, 2, 1]


def test_merge_mineru_markdown_chunks_offsets_page_headings():
    markdown = merge_mineru_markdown_chunks(
        [
            (PdfChunk(Path("part1.pdf"), 1, 2), "## Page 1\n\nA\n\n## Page 2\n\nB"),
            (PdfChunk(Path("part2.pdf"), 3, 4), "## Page 1\n\nC\n\n## Page 2\n\nD"),
        ]
    )

    assert "## Page 1" in markdown
    assert "## Page 2" in markdown
    assert "## Page 3" in markdown
    assert "## Page 4" in markdown


def test_merge_mineru_markdown_chunks_adds_heading_when_missing():
    markdown = merge_mineru_markdown_chunks([(PdfChunk(Path("part2.pdf"), 201, 250), "content")])

    assert "## Page 201" in markdown