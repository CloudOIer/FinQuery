from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finquery_agent.config import LLMSettings, RAGSettings, load_llm_settings
from finquery_agent.llm import LLMClient
from finquery_agent.rag.index import build_rag_index, load_rag_index
from finquery_agent.rag.models import SearchResult, make_snippet
from finquery_agent.rag.retriever import HybridRetriever


@dataclass(frozen=True)
class RAGAnswer:
    answer_text: str
    answer_source: str
    sources: list[dict[str, Any]]
    llm_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "answer_source": self.answer_source,
            "llm_used": self.llm_used,
            "sources": self.sources,
        }


class RAGService:
    def __init__(self, settings: RAGSettings, llm_settings: LLMSettings | None = None):
        self.settings = settings
        self.llm_settings = llm_settings or load_llm_settings()
        self._llm_client = LLMClient(self.llm_settings)
        self.retriever = HybridRetriever(_load_or_build_index(settings), settings)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        use_vector: bool | None = None,
        use_reranker: bool | None = None,
        stock_codes: tuple[str, ...] = (),
        industries: tuple[str, ...] = (),
        report_type: str | None = None,
    ) -> list[SearchResult]:
        return self.retriever.search(
            question,
            top_k=top_k,
            use_vector=use_vector,
            use_reranker=use_reranker,
            stock_codes=stock_codes,
            industries=industries,
            report_type=report_type,
        )

    def answer(self, question: str, top_k: int | None = None, use_vector: bool | None = None, use_llm: bool = True) -> RAGAnswer:
        results = self.search(question, top_k=top_k, use_vector=use_vector)
        sources = [result.to_dict() for result in results]
        deterministic = _deterministic_answer(question, results)
        if not use_llm or not self._llm_available() or not results:
            return RAGAnswer(answer_text=deterministic, answer_source="deterministic", sources=sources, llm_used=False)
        llm_answer = self._compose_with_llm(question, results, deterministic)
        return RAGAnswer(
            answer_text=llm_answer or deterministic,
            answer_source="llm" if llm_answer else "deterministic",
            sources=sources,
            llm_used=bool(llm_answer),
        )

    def _llm_available(self) -> bool:
        return self._llm_client.is_available()

    def _compose_with_llm(self, question: str, results: list[SearchResult], fallback: str) -> str | None:
        evidence = [
            {
                "index": index + 1,
                "title": result.chunk.title,
                "report_type": result.chunk.report_type,
                "stock_name": result.chunk.stock_name,
                "industry_name": result.chunk.industry_name,
                "org_name": result.chunk.org_name,
                "publish_date": result.chunk.publish_date,
                "section_title": result.chunk.section_title,
                "text": make_snippet(result.chunk.text, max_chars=900),
            }
            for index, result in enumerate(results)
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "你是研报知识库问答助手。只能根据提供的研报证据回答。"
                    "不要编造未给出的事实；如果证据不足，请明确说明。"
                    "回答要中文、简洁、有分析结论，并在句末用[来源1]、[来源2]标注依据。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question, "evidence": evidence, "fallback_answer": fallback}, ensure_ascii=False),
            },
        ]
        return self._llm_client.chat(messages, temperature=0.2)


def _load_or_build_index(settings: RAGSettings):
    index_dir = settings.index_dir or Path("data/rag")
    if (index_dir / "chunks.jsonl").exists():
        return load_rag_index(index_dir)
    try:
        return build_rag_index(settings, use_vector=settings.use_vector)
    except Exception:
        if not settings.use_vector:
            raise
        return build_rag_index(settings, use_vector=False)


def _deterministic_answer(question: str, results: list[SearchResult]) -> str:
    if not results:
        return "未在当前研报知识库中检索到足够相关的内容。"
    lines = [f"围绕“{question}”，检索到 {len(results)} 条相关研报片段："]
    for index, result in enumerate(results[:5], start=1):
        chunk = result.chunk
        source = f"《{chunk.title}》"
        meta = "，".join(item for item in (chunk.org_name, chunk.publish_date, chunk.section_title) if item)
        lines.append(f"{index}. {source}{'（' + meta + '）' if meta else ''}：{make_snippet(chunk.text, max_chars=160)}")
    lines.append("可在启用 LLM 后基于这些证据生成更完整的分析回答。")
    return "\n".join(lines)