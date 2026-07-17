from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchDocument:
    doc_id: str
    title: str
    report_type: str
    source_path: str
    text: str
    stock_name: str = ""
    stock_code: str = ""
    org_name: str = ""
    org_sname: str = ""
    publish_date: str = ""
    industry_name: str = ""
    rating: str = ""
    researcher: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchDocument":
        return cls(**payload)


@dataclass(frozen=True)
class ResearchChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    text: str
    section_title: str = ""
    report_type: str = ""
    source_path: str = ""
    stock_name: str = ""
    stock_code: str = ""
    org_name: str = ""
    publish_date: str = ""
    industry_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchChunk":
        return cls(**payload)


@dataclass(frozen=True)
class SearchResult:
    chunk: ResearchChunk
    score: float
    score_detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = self.chunk.to_dict()
        payload["score"] = self.score
        payload["score_detail"] = self.score_detail
        payload["snippet"] = make_snippet(self.chunk.text)
        return payload


def make_snippet(text: str, max_chars: int = 260) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= max_chars else compact[: max_chars - 1] + "…"


def path_as_posix(path: Path) -> str:
    return path.as_posix()