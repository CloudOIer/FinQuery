from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from finquery_agent.config import RAGSettings
from finquery_agent.rag.index import RAGIndex
from finquery_agent.rag.models import ResearchChunk, SearchResult
from finquery_agent.rag.reranker import CrossEncoderReranker


class HybridRetriever:
    def __init__(self, rag_index: RAGIndex, settings: RAGSettings):
        self.index = rag_index
        self.settings = settings
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in rag_index.chunks}
        self._bm25 = None
        self._tokenized_chunks: list[list[str]] | None = None
        self._embedding_model = None
        self._reranker = CrossEncoderReranker(settings)

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
        """两阶段检索:粗排融合 → 候选阶段文档配额 → (可选)cross-encoder 精排 → 文档配额 → top_k。

        stock_codes/industries/report_type 为元数据硬过滤(在粗排后、精排前应用):
        把"只看某公司/某行业的研报"从调用方的事后筛选下沉为检索能力,
        避免过滤后数量不足 top_k。过滤条件之间是 AND,条件内多值是 OR。
        """
        top_k = top_k or self.settings.final_top_k
        use_vector = self.settings.use_vector if use_vector is None else use_vector
        use_reranker = self.settings.use_reranker if use_reranker is None else use_reranker
        has_filter = bool(stock_codes or industries or report_type)

        # 精排/过滤都需要比最终 top_k 更多的粗排候选;有过滤时再放大一档,
        # 补偿被过滤掉的候选。
        candidate_k = max(top_k, self.settings.rerank_candidate_k if use_reranker else top_k)
        pool_k = max(candidate_k, self.settings.coarse_pool_k)
        coarse_k = pool_k * 3 if has_filter else pool_k

        results = self._coarse_search(question, coarse_k, use_vector)
        if has_filter:
            results = [r for r in results if _matches_filters(r.chunk, stock_codes, industries, report_type)]
        results = _apply_doc_quota(results, self.settings.candidate_max_chunks_per_doc)
        results = results[:candidate_k]
        rerank_scope = self.settings.rerank_scope
        if use_reranker and rerank_scope == "candidates" and len(results) > 1:
            results = self._reranker.rerank(question, results, top_k=len(results))
        results = _apply_doc_quota(results, self.settings.max_chunks_per_doc)[:top_k]
        if use_reranker and rerank_scope == "final" and len(results) > 1:
            results = self._reranker.rerank(question, results, top_k=len(results))
        return results

    def _coarse_search(self, question: str, top_k: int, use_vector: bool) -> list[SearchResult]:
        channels: dict[str, list[tuple[str, float]]] = {
            "bm25": self._bm25_search(question, max(top_k, self.settings.bm25_top_k)),
        }
        if use_vector and self.index.vector_index is not None and self.index.vector_chunk_ids:
            channels["vector"] = self._vector_search(question, max(top_k, self.settings.vector_top_k))
        results = []
        for chunk_id, score, detail in _fuse(channels, self.settings):
            chunk = self._chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(SearchResult(chunk=chunk, score=score, score_detail=detail))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def _bm25_search(self, question: str, top_k: int) -> list[tuple[str, float]]:
        self._ensure_bm25()
        if self._bm25 is None or not self.index.chunks:
            return []
        query_tokens = _tokenize(question)
        raw_scores = self._bm25.get_scores(query_tokens)
        if len(raw_scores) == 0:
            return []
        pairs = sorted(enumerate(raw_scores), key=lambda item: float(item[1]), reverse=True)[:top_k]
        if not pairs or max(float(score) for _, score in pairs) <= 0:
            return self._keyword_overlap_search(query_tokens, top_k)
        max_score = max(float(score) for _, score in pairs) or 1.0
        return [(self.index.chunks[index].chunk_id, max(float(score) / max_score, 0.0)) for index, score in pairs if float(score) > 0]

    def _keyword_overlap_search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        query_set = {token for token in query_tokens if len(token) > 1}
        if not query_set or self._tokenized_chunks is None:
            return []
        pairs = []
        for index, tokens in enumerate(self._tokenized_chunks):
            overlap = len(query_set.intersection(token for token in tokens if len(token) > 1))
            if overlap:
                pairs.append((index, float(overlap)))
        pairs.sort(key=lambda item: item[1], reverse=True)
        max_score = max((score for _, score in pairs[:top_k]), default=1.0)
        return [(self.index.chunks[index].chunk_id, score / max_score) for index, score in pairs[:top_k]]

    def _vector_search(self, question: str, top_k: int) -> list[tuple[str, float]]:
        import numpy as np

        model = self._get_embedding_model()
        query_embedding = model.encode([question], normalize_embeddings=True)
        matrix = np.asarray(query_embedding, dtype="float32")
        scores, indices = self.index.vector_index.search(matrix, min(top_k, len(self.index.vector_chunk_ids)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            chunk_id = self.index.vector_chunk_ids[int(index)]
            results.append((chunk_id, max(float(score), 0.0)))
        return results

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        self._tokenized_chunks = [_tokenize(_search_text(chunk)) for chunk in self.index.chunks]
        self._bm25 = BM25Okapi(self._tokenized_chunks)

    def _get_embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.settings.embedding_model)
        return self._embedding_model


def _fuse(
    channels: dict[str, list[tuple[str, float]]],
    settings: RAGSettings,
) -> list[tuple[str, float, dict[str, float]]]:
    """合并多路召回。

    rrf 仅用排名:两路分数量纲不同(BM25 无界、余弦有界),归一化后仍会因
    候选集内 max 缩放而失真,且单路命中的 chunk 会被"缺失通道记 0"隐式惩罚。
    weighted 保留原加权求和,仅用于消融对比。
    """
    weights = {"bm25": settings.bm25_weight, "vector": settings.vector_weight}
    use_rrf = settings.fusion_method == "rrf"
    detail: dict[str, dict[str, float]] = defaultdict(dict)
    combined: dict[str, float] = defaultdict(float)
    for name, pairs in channels.items():
        for rank, (chunk_id, score) in enumerate(pairs, start=1):
            detail[chunk_id][name] = score
            combined[chunk_id] += 1.0 / (settings.rrf_k + rank) if use_rrf else weights.get(name, 0.0) * score
    return [(chunk_id, score, dict(detail[chunk_id])) for chunk_id, score in combined.items() if score > 0]


def _matches_filters(
    chunk: ResearchChunk,
    stock_codes: tuple[str, ...],
    industries: tuple[str, ...],
    report_type: str | None,
) -> bool:
    if stock_codes and chunk.stock_code not in stock_codes:
        return False
    if industries and not any(industry and industry in (chunk.industry_name or "") for industry in industries):
        return False
    if report_type and chunk.report_type != report_type:
        return False
    return True


def _apply_doc_quota(results: list[SearchResult], max_per_doc: int) -> list[SearchResult]:
    """同文档 chunk 相邻内容高度重叠,不限制时一篇长文档会占满 top_k,
    答案只有单一信息源;配额强制结果覆盖多篇文档,提升证据多样性。"""
    if max_per_doc <= 0:
        return results
    counts: dict[str, int] = defaultdict(int)
    kept = []
    for result in results:
        if counts[result.chunk.doc_id] >= max_per_doc:
            continue
        counts[result.chunk.doc_id] += 1
        kept.append(result)
    return kept


def _search_text(chunk: ResearchChunk) -> str:
    return "\n".join(
        item
        for item in (chunk.title, chunk.section_title, chunk.stock_name, chunk.stock_code, chunk.industry_name, chunk.org_name, chunk.text)
        if item
    )


def _tokenize(text: str) -> list[str]:
    import jieba

    normalized = re.sub(r"\s+", " ", text.lower())
    tokens = [token.strip() for token in jieba.cut(normalized) if token.strip()]
    tokens.extend(re.findall(r"[a-zA-Z0-9]+", normalized))
    return tokens