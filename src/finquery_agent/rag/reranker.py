"""Cross-encoder 精排层。

粗排(BM25+向量融合)负责从 8000+ chunks 里快速召回候选,精排用 cross-encoder
对 query-chunk 对做逐对打分重排。bi-encoder 分别编码 query 和文档再算相似度,
无法建模两者间的词级交互;cross-encoder 把两段文本拼接后过完整注意力,精度
显著更高,但代价是每对都要一次前向,只能用于小候选集。

CPU 环境约束:模型 lazy load(首次调用才加载),predict 按 batch 分批,
文本截断到 max_length=512;候选数由 rerank_candidate_k 控制在 30 以内。
加载失败(模型文件缺失等)时降级返回粗排原序,不阻断检索服务。
"""

from __future__ import annotations

from dataclasses import replace

from finquery_agent.config import RAGSettings
from finquery_agent.rag.models import SearchResult


class CrossEncoderReranker:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self._model = None
        self._load_failed = False

    def available(self) -> bool:
        return not self._load_failed

    def rerank(self, question: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """对粗排结果重打分并按新分数排序;失败时原样返回粗排结果。

        score 替换为 cross-encoder 分数(0~1,已过 sigmoid),粗排融合分保留在
        score_detail["hybrid"] 中,便于评测与前端展示两阶段得分。
        """
        if len(results) <= 1:
            return results[:top_k]
        model = self._get_model()
        if model is None:
            return results[:top_k]
        pairs = [[question, _rerank_text(result)] for result in results]
        scores = model.predict(pairs, batch_size=self.settings.rerank_batch_size, show_progress_bar=False)
        reranked = [
            replace(
                result,
                score=float(score),
                score_detail={**result.score_detail, "hybrid": result.score, "rerank": float(score)},
            )
            for result, score in zip(results, scores)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]

    def _get_model(self):
        if self._model is None and not self._load_failed:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.settings.reranker_model, max_length=512)
            except Exception:
                # 模型缺失/损坏不应让整个检索服务不可用,降级为粗排结果。
                self._load_failed = True
        return self._model


def _rerank_text(result: SearchResult) -> str:
    """打分文本带上标题与元信息:chunk 正文可能不含公司名(在文档标题里),
    拼上下文才能让 cross-encoder 正确判断'这段话是否在讲问题里的公司'。"""
    chunk = result.chunk
    header = " ".join(item for item in (chunk.title, chunk.section_title, chunk.stock_name, chunk.industry_name) if item)
    return f"{header}\n{chunk.text}" if header else chunk.text
