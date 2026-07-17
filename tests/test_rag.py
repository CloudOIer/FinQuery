import pytest

from finquery_agent.config import LLMSettings, RAGSettings
from finquery_agent.rag.index import build_rag_index, load_rag_index
from finquery_agent.rag.loader import chunk_documents, load_research_documents
from finquery_agent.rag.models import ResearchChunk, SearchResult
from finquery_agent.rag.retriever import HybridRetriever
from finquery_agent.rag.service import RAGAnswer, RAGService


def _write_sample_rag_files(tmp_path):
    markdown_dir = tmp_path / "研报markdown"
    markdown_dir.mkdir()
    (markdown_dir / "CXO行业景气度观察.md").write_text(
        """# CXO行业景气度观察

## 行业观点

CXO行业订单边际改善，海外需求逐步恢复。医药外包企业在手订单和产能利用率是判断景气度的重要指标。

## 风险提示

地缘政治和投融资周期仍可能影响海外业务节奏。
""",
        encoding="utf-8",
    )
    stock_meta = tmp_path / "个股_研报信息.CSV"
    stock_meta.write_text(
        "title,stockName,stockCode,orgName,orgSName,publishDate,emRatingName,researcher\n"
        "CXO行业景气度观察,药明康德,603259,测试证券,测试,2025-10-01 00:00:00.000,买入,研究员\n",
        encoding="utf-8",
    )
    return markdown_dir, stock_meta


def _sample_settings(tmp_path):
    markdown_dir, stock_meta = _write_sample_rag_files(tmp_path)
    return RAGSettings(
        data_roots=(markdown_dir,),
        stock_metadata_file=stock_meta,
        industry_metadata_file=None,
        index_dir=tmp_path / "rag-index",
        chunk_size=160,
        chunk_overlap=20,
        use_vector=False,
        final_top_k=3,
        bm25_top_k=5,
    )


def test_load_research_documents_and_chunk_metadata(tmp_path):
    settings = _sample_settings(tmp_path)

    documents = load_research_documents(settings)
    chunks = chunk_documents(documents, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    assert len(documents) == 1
    assert documents[0].title == "CXO行业景气度观察"
    assert documents[0].stock_name == "药明康德"
    assert documents[0].stock_code == "603259"
    assert chunks
    assert chunks[0].title == "CXO行业景气度观察"
    assert chunks[0].stock_name == "药明康德"


def test_bm25_retriever_finds_relevant_research_chunk(tmp_path):
    settings = _sample_settings(tmp_path)
    rag_index = build_rag_index(settings, use_vector=False)
    loaded = load_rag_index(settings.index_dir)
    retriever = HybridRetriever(loaded, settings)

    results = retriever.search("CXO行业景气度和海外订单如何", use_vector=False)

    assert results
    assert "CXO" in results[0].chunk.title
    assert "海外" in results[0].chunk.text
    assert results[0].score > 0


def test_rag_service_generates_deterministic_answer(tmp_path):
    settings = _sample_settings(tmp_path)
    build_rag_index(settings, use_vector=False)
    service = RAGService(settings, LLMSettings())

    answer = service.answer("CXO行业景气度如何", use_vector=False, use_llm=False)

    assert answer.llm_used is False
    assert answer.answer_source == "deterministic"
    assert "检索到" in answer.answer_text
    assert answer.sources


def test_rag_api_search_and_ask(monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from finquery_agent.api import main as api_main
    from finquery_agent.api.main import app

    chunk = ResearchChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_index=1,
        title="CXO行业景气度观察",
        text="CXO行业订单边际改善，海外需求逐步恢复。",
        org_name="测试证券",
        publish_date="2025-10-01",
    )
    result = SearchResult(chunk=chunk, score=0.9, score_detail={"bm25": 1.0})

    class FakeRAGService:
        def search(self, question, top_k=None, use_vector=None, **kwargs):
            return [result]

        def answer(self, question, top_k=None, use_vector=None, use_llm=True):
            return RAGAnswer(answer_text="CXO行业景气度边际改善。", answer_source="deterministic", sources=[result.to_dict()], llm_used=False)

    monkeypatch.setattr(api_main, "get_rag_service", lambda: FakeRAGService())
    client = TestClient(app)

    search_response = client.post("/rag/search", json={"question": "CXO景气度", "top_k": 3, "use_vector": False})
    ask_response = client.post("/rag/ask", json={"question": "CXO景气度", "top_k": 3, "use_vector": False, "use_llm": False})

    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["title"] == "CXO行业景气度观察"
    assert ask_response.status_code == 200
    assert ask_response.json()["answer_text"] == "CXO行业景气度边际改善。"
    assert ask_response.json()["sources"]