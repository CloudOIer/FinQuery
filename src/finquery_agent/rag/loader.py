from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable

from finquery_agent.config import RAGSettings
from finquery_agent.rag.models import ResearchChunk, ResearchDocument, path_as_posix


def load_research_documents(settings: RAGSettings) -> list[ResearchDocument]:
    stock_meta = _load_metadata(settings.stock_metadata_file, "stock") if settings.stock_metadata_file else {}
    industry_meta = _load_metadata(settings.industry_metadata_file, "industry") if settings.industry_metadata_file else {}
    metadata = {**industry_meta, **stock_meta}
    documents: list[ResearchDocument] = []
    for root in settings.data_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = _extract_title(text, path)
            meta = metadata.get(_normalize_title(title)) or metadata.get(_normalize_title(path.stem)) or {}
            report_type = str(meta.get("report_type") or _infer_report_type(meta, path))
            documents.append(
                ResearchDocument(
                    doc_id=_stable_id(path_as_posix(path), title),
                    title=title,
                    report_type=report_type,
                    source_path=path_as_posix(path),
                    text=text,
                    stock_name=str(meta.get("stock_name", "")),
                    stock_code=str(meta.get("stock_code", "")),
                    org_name=str(meta.get("org_name", "")),
                    org_sname=str(meta.get("org_sname", "")),
                    publish_date=str(meta.get("publish_date", ""))[:10],
                    industry_name=str(meta.get("industry_name", "")),
                    rating=str(meta.get("rating", "")),
                    researcher=str(meta.get("researcher", "")),
                    metadata=meta,
                )
            )
    return documents


def chunk_documents(documents: Iterable[ResearchDocument], chunk_size: int = 800, chunk_overlap: int = 150) -> list[ResearchChunk]:
    chunks: list[ResearchChunk] = []
    for document in documents:
        chunk_index = 0
        for section_title, section_text in _iter_sections(document.text, document.title):
            for piece in _split_text(section_text, chunk_size=chunk_size, overlap=chunk_overlap):
                if len(piece.strip()) < 40:
                    continue
                chunk_index += 1
                chunks.append(
                    ResearchChunk(
                        chunk_id=f"{document.doc_id}-{chunk_index:04d}",
                        doc_id=document.doc_id,
                        chunk_index=chunk_index,
                        title=document.title,
                        section_title=section_title,
                        text=piece,
                        report_type=document.report_type,
                        source_path=document.source_path,
                        stock_name=document.stock_name,
                        stock_code=document.stock_code,
                        org_name=document.org_name,
                        publish_date=document.publish_date,
                        industry_name=document.industry_name,
                        metadata={
                            "org_sname": document.org_sname,
                            "rating": document.rating,
                            "researcher": document.researcher,
                        },
                    )
                )
    return chunks


def _load_metadata(path: Path | None, report_type: str) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            meta = {
                "report_type": report_type,
                "stock_name": str(row.get("stockName", "")).strip(),
                "stock_code": str(row.get("stockCode", "")).strip().zfill(6) if str(row.get("stockCode", "")).strip().isdigit() else str(row.get("stockCode", "")).strip(),
                "org_name": str(row.get("orgName", "")).strip(),
                "org_sname": str(row.get("orgSName", "")).strip(),
                "publish_date": str(row.get("publishDate", "")).strip()[:10],
                "industry_name": str(row.get("industryName", row.get("indvInduName", ""))).strip(),
                "rating": str(row.get("emRatingName", row.get("sRatingName", ""))).strip(),
                "researcher": str(row.get("researcher", "")).strip(),
            }
            result[_normalize_title(title)] = meta
    return result


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines()[:30]:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title
    return path.stem


def _infer_report_type(meta: dict[str, str], path: Path) -> str:
    if meta.get("stock_name") or meta.get("stock_code"):
        return "stock"
    if meta.get("industry_name"):
        return "industry"
    return "industry" if "行业" in path.stem or "白皮书" in path.stem or "研究报告" in path.stem else "stock"


def _iter_sections(text: str, fallback_title: str):
    current_title = fallback_title
    buffer: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,4}\s+", line.strip()):
            if buffer:
                yield current_title, _clean_text("\n".join(buffer))
                buffer = []
            current_title = line.strip().lstrip("#").strip() or fallback_title
        else:
            buffer.append(line)
    if buffer:
        yield current_title, _clean_text("\n".join(buffer))


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n|(?<=[。！？])\s+", text) if item.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= chunk_size:
            current = f"{current}\n{paragraph}".strip()
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = _tail(current, overlap) + "\n" + paragraph if overlap and current else paragraph
        else:
            if current and overlap:
                chunks.append((_tail(current, overlap) + "\n" + paragraph[:chunk_size]).strip())
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start : start + chunk_size])
                start += max(1, chunk_size - overlap)
            current = ""
    if current:
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _tail(text: str, max_chars: int) -> str:
    compact = text.strip()
    return compact[-max_chars:] if max_chars > 0 else ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def _stable_id(*parts: str) -> str:
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return f"doc-{digest[:16]}"