"""Reranker、文档多样性配额与 metadata filter 测试。

CrossEncoder 用 stub 替代:单测验证重排/配额/过滤逻辑本身,
不加载真实模型(CPU 上加载需数秒且依赖模型文件存在)。
"""

from __future__ import annotations

from finquery_agent.config import RAGSettings
from finquery_agent.rag.models import ResearchChunk, SearchResult
from finquery_agent.rag.reranker import CrossEncoderReranker
from finquery_agent.rag.retriever import _apply_doc_quota, _matches_filters


def _chunk(chunk_id: str, doc_id: str, **overrides) -> ResearchChunk:
    payload = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_index": 0,
        "title": f"研报{doc_id}",
        "text": f"chunk {chunk_id} 的内容",
    }
    payload.update(overrides)
    return ResearchChunk(**payload)


def _result(chunk_id: str, doc_id: str, score: float, **overrides) -> SearchResult:
    return SearchResult(chunk=_chunk(chunk_id, doc_id, **overrides), score=score, score_detail={"bm25": score})


class _StubModel:
    """按文本长度给分的假 CrossEncoder:分数可控且与粗排顺序无关。"""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        return self._scores[: len(pairs)]


def test_reranker_reorders_by_cross_encoder_score():
    reranker = CrossEncoderReranker(RAGSettings())
    reranker._model = _StubModel([0.1, 0.9, 0.5])
    results = [_result("c1", "d1", 0.9), _result("c2", "d2", 0.8), _result("c3", "d3", 0.7)]

    reranked = reranker.rerank("问题", results, top_k=3)

    # 粗排第 2 名的 c2 拿到最高精排分,应升到第 1。
    assert [r.chunk.chunk_id for r in reranked] == ["c2", "c3", "c1"]
    assert reranked[0].score == 0.9
    # 两阶段分数都保留,可解释每一步的排序依据。
    assert reranked[0].score_detail["hybrid"] == 0.8
    assert reranked[0].score_detail["rerank"] == 0.9


def test_reranker_degrades_to_coarse_order_when_model_unavailable():
    reranker = CrossEncoderReranker(RAGSettings(reranker_model="/nonexistent/model"))
    results = [_result("c1", "d1", 0.9), _result("c2", "d2", 0.8)]

    reranked = reranker.rerank("问题", results, top_k=2)

    assert [r.chunk.chunk_id for r in reranked] == ["c1", "c2"]
    assert reranker.available() is False


def test_doc_quota_limits_chunks_per_document():
    results = [
        _result("c1", "d1", 0.9),
        _result("c2", "d1", 0.8),
        _result("c3", "d1", 0.7),
        _result("c4", "d2", 0.6),
    ]

    kept = _apply_doc_quota(results, max_per_doc=2)

    assert [r.chunk.chunk_id for r in kept] == ["c1", "c2", "c4"]


def test_doc_quota_zero_means_unlimited():
    results = [_result(f"c{i}", "d1", 1.0 - i * 0.1) for i in range(4)]

    assert len(_apply_doc_quota(results, max_per_doc=0)) == 4


def test_metadata_filters_match_stock_industry_and_type():
    stock_chunk = _chunk("c1", "d1", stock_code="600332", report_type="stock")
    industry_chunk = _chunk("c2", "d2", industry_name="医药生物", report_type="industry")

    assert _matches_filters(stock_chunk, stock_codes=("600332",), industries=(), report_type=None)
    assert not _matches_filters(stock_chunk, stock_codes=("603259",), industries=(), report_type=None)
    assert _matches_filters(industry_chunk, stock_codes=(), industries=("医药",), report_type=None)
    assert not _matches_filters(industry_chunk, stock_codes=(), industries=("食品",), report_type=None)
    assert _matches_filters(industry_chunk, stock_codes=(), industries=(), report_type="industry")
    assert not _matches_filters(industry_chunk, stock_codes=(), industries=(), report_type="stock")


def test_retriever_search_applies_filter_and_quota(tmp_path):
    from finquery_agent.rag.index import build_rag_index
    from finquery_agent.rag.retriever import HybridRetriever

    markdown_dir = tmp_path / "研报markdown"
    markdown_dir.mkdir()
    (markdown_dir / "白云山业绩点评.md").write_text(
        "# 白云山业绩点评\n\n## 业绩\n\n白云山2024年营业收入稳健增长,大商业板块收入保持两位数增速,盈利能力持续改善,现金流表现良好。\n\n"
        "## 展望\n\n白云山大健康板块持续发力,王老吉品牌矩阵扩张,新品放量带动营收结构优化,全年业绩确定性较强。\n",
        encoding="utf-8",
    )
    (markdown_dir / "医药行业周报.md").write_text(
        "# 医药行业周报\n\n## 行情\n\n医药板块本周整体上涨,创新药和中药板块领涨,市场对板块内公司营收预期普遍改善,配置价值提升。\n",
        encoding="utf-8",
    )
    stock_meta = tmp_path / "个股_研报信息.CSV"
    stock_meta.write_text(
        "title,stockName,stockCode,orgName,orgSName,publishDate,emRatingName,researcher\n"
        "白云山业绩点评,白云山,600332,测试证券,测试,2025-10-01 00:00:00.000,买入,研究员\n",
        encoding="utf-8",
    )
    settings = RAGSettings(
        data_roots=(markdown_dir,),
        stock_metadata_file=stock_meta,
        industry_metadata_file=None,
        index_dir=tmp_path / "rag-index",
        chunk_size=80,
        chunk_overlap=10,
        use_vector=False,
        use_reranker=False,
        final_top_k=5,
        max_chunks_per_doc=1,
    )
    retriever = HybridRetriever(build_rag_index(settings, use_vector=False), settings)

    filtered = retriever.search("白云山营收", use_vector=False, stock_codes=("600332",))
    assert filtered
    assert all(r.chunk.stock_code == "600332" for r in filtered)
    # max_chunks_per_doc=1:即使该文档有多个相关 chunk,也只保留 1 个。
    assert len({r.chunk.doc_id for r in filtered}) == len(filtered)

    unfiltered = retriever.search("营收", use_vector=False)
    assert {r.chunk.doc_id for r in unfiltered} >= {r.chunk.doc_id for r in filtered}
