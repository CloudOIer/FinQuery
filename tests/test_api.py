import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from finquery_agent.api import main as api_main
from finquery_agent.api.main import app
from finquery_agent.nl2sql.executor import QueryResult
from finquery_agent.rag.models import ResearchChunk, SearchResult


def test_health_and_schema_tables_api():
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    response = client.get("/schema/tables")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["tables"]) == 4


def test_web_ui_static_assets_are_served():
    client = TestClient(app)

    index = client.get("/")
    script = client.get("/static/app.js")

    assert index.status_code == 200
    assert "FinQuery 智能问数" in index.text
    assert "conversation-list" in index.text
    assert "新对话" in index.text
    assert "执行 SQL" not in index.text
    assert script.status_code == 200
    assert "submitQuestion" in script.text
    assert "renderChartImage" in script.text
    assert "renderBarChart" not in script.text


def test_generate_sql_api():
    client = TestClient(app)

    response = client.post(
        "/nl2sql/generate",
        json={
            "metrics": ["营收"],
            "company_names": ["白云山"],
            "years": [2024],
            "periods": ["FY"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["table_name"] == "core_performance_indicators_sheet"
    assert "total_operating_revenue" in payload["sql"]
    assert payload["params"]["stock_code_0"] == "600332"


def test_query_intent_api_returns_structured_intent():
    client = TestClient(app)

    response = client.post("/query/intent", json={"question": "白云山2024年年报营收是多少"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent_type"] == "metric_query"
    assert payload["company_codes"] == ["600332"]
    assert payload["periods"] == ["FY"]
    assert payload["needs_clarification"] is False


def test_query_ask_api_returns_clarification_when_slots_missing():
    client = TestClient(app)

    response = client.post("/query/ask", json={"question": "2024年年报营收是多少"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "clarification"
    assert payload["clarification"]["missing_slots"] == ["company"]


def test_query_ask_api_returns_sql_for_supported_query():
    client = TestClient(app)

    response = client.post("/query/ask", json={"question": "白云山2024年年报营收是多少"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answer"
    assert "total_operating_revenue" in payload["sql"]
    assert payload["params"]["stock_code_0"] == "600332"


def test_query_ask_api_can_execute_query(monkeypatch):
    class FakeExecutor:
        def execute(self, query):
            return QueryResult(
                columns=("stock_code", "total_operating_revenue"),
                rows=({"stock_code": "600332", "total_operating_revenue": 123.45},),
                units={"stock_code": None, "total_operating_revenue": "万元"},
                row_count=1,
            )

    monkeypatch.setattr(api_main, "query_executor", FakeExecutor())
    client = TestClient(app)

    response = client.post("/query/ask", json={"question": "白云山2024年年报营收是多少", "execute": True, "use_llm": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answer"
    assert payload["result"]["row_count"] == 1
    assert payload["result"]["units"]["total_operating_revenue"] == "万元"
    assert "answer_text" in payload
    assert payload["answer_source"] == "deterministic"


def test_query_ask_api_returns_backend_chart_image(monkeypatch):
    class FakeExecutor:
        def execute(self, query):
            return QueryResult(
                columns=("stock_code", "stock_abbr", "report_year", "report_period", "total_operating_revenue"),
                rows=(
                    {"stock_code": "600332", "stock_abbr": "", "report_year": 2022, "report_period": "FY", "total_operating_revenue": 100.0},
                    {"stock_code": "600332", "stock_abbr": "", "report_year": 2023, "report_period": "FY", "total_operating_revenue": 120.0},
                    {"stock_code": "600332", "stock_abbr": "", "report_year": 2024, "report_period": "FY", "total_operating_revenue": 150.0},
                ),
                units={"total_operating_revenue": "万元"},
                row_count=3,
            )

    monkeypatch.setattr(api_main, "query_executor", FakeExecutor())
    client = TestClient(app)

    response = client.post("/query/ask", json={"question": "白云山近三年营收趋势图", "execute": True, "use_llm": False})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answer"
    assert len(payload["chart_images"]) == 1
    assert payload["chart_image"]["chart_type"] == "line"
    assert payload["chart_image"]["image_data_url"].startswith("data:image/svg+xml;base64,")
    assert "白云山" in payload["chart_image"]["title"]
    assert "营业总收入" in payload["chart_image"]["title"]
    assert "营业总收入" in payload["chart_image"]["y_axis_label"]
    assert "total_operating_revenue" not in payload["chart_image"]["title"]


def test_query_ask_api_returns_one_chart_per_metric(monkeypatch):
    class FakeExecutor:
        def execute(self, query):
            return QueryResult(
                columns=("stock_code", "stock_abbr", "report_year", "report_period", "total_operating_revenue", "net_profit_10k_yuan"),
                rows=(
                    {"stock_code": "300347", "stock_abbr": "泰格医药", "report_year": 2022, "report_period": "Q3", "total_operating_revenue": 100.0, "net_profit_10k_yuan": 20.0},
                    {"stock_code": "300347", "stock_abbr": "泰格医药", "report_year": 2023, "report_period": "Q3", "total_operating_revenue": 120.0, "net_profit_10k_yuan": 22.0},
                    {"stock_code": "300347", "stock_abbr": "泰格医药", "report_year": 2024, "report_period": "Q3", "total_operating_revenue": 150.0, "net_profit_10k_yuan": 25.0},
                ),
                units={"total_operating_revenue": "万元", "net_profit_10k_yuan": "万元"},
                row_count=3,
            )

    monkeypatch.setattr(api_main, "query_executor", FakeExecutor())
    client = TestClient(app)

    response = client.post(
        "/query/ask",
        json={"question": "绘制泰格医药2022-2025年第三季度营业总收入和净利润的趋势折线图", "execute": True, "use_llm": False},
    )

    assert response.status_code == 200
    payload = response.json()
    titles = [chart["title"] for chart in payload["chart_images"]]
    assert len(payload["chart_images"]) == 2
    assert any("营业总收入" in title for title in titles)
    assert any("净利润" in title for title in titles)
    assert payload["chart_image"] == payload["chart_images"][0]


def test_query_ask_api_merges_clarification_reply_with_session():
    client = TestClient(app)
    session_id = "api-session-clarify"

    first = client.post("/query/ask", json={"question": "2024年年报营收是多少", "session_id": session_id})
    second = client.post("/query/ask", json={"question": "白云山", "session_id": session_id})

    assert first.json()["status"] == "clarification"
    payload = second.json()
    assert payload["status"] == "answer"
    assert payload["params"]["stock_code_0"] == "600332"
    assert payload["params"]["report_year_0"] == 2024


def test_query_ask_api_splits_cross_table_query():
    client = TestClient(app)

    response = client.post("/query/ask", json={"question": "白云山2024年年报营收和总资产"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answer"
    assert len(payload["queries"]) == 2
    assert {query["table_name"] for query in payload["queries"]} == {"core_performance_indicators_sheet", "balance_sheet"}


def test_analysis_ask_api_answers_rag_only_question(monkeypatch):
    result = _fake_rag_result()

    class FakeRAGService:
        def search(self, question, top_k=None, use_vector=None, **kwargs):
            return [result]

    monkeypatch.setattr(api_main, "_analysis_service", None)
    monkeypatch.setattr(api_main, "get_rag_service", lambda: FakeRAGService())
    client = TestClient(app)

    response = client.post(
        "/analysis/ask",
        json={"question": "当前CXO行业的景气度如何", "use_llm": False, "use_vector": False, "rag_top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answer"
    assert payload["financial"]["status"] == "skipped"
    assert payload["sources"][0]["title"] == "CXO行业景气度观察"
    assert "研报依据" in payload["answer_text"]


def test_analysis_ask_api_combines_financial_query_and_rag(monkeypatch):
    class FakeExecutor:
        def execute(self, query):
            return QueryResult(
                columns=("stock_code", "stock_abbr", "report_year", "report_period", "total_operating_revenue"),
                rows=({"stock_code": "603259", "stock_abbr": "药明康德", "report_year": 2025, "report_period": "Q3", "total_operating_revenue": 3285671.65},),
                units={"total_operating_revenue": "万元"},
                row_count=1,
            )

    result = _fake_rag_result()

    class FakeRAGService:
        def search(self, question, top_k=None, use_vector=None, **kwargs):
            return [result]

    monkeypatch.setattr(api_main, "query_executor", FakeExecutor())
    monkeypatch.setattr(api_main, "_analysis_service", None)
    monkeypatch.setattr(api_main, "get_rag_service", lambda: FakeRAGService())
    client = TestClient(app)

    response = client.post(
        "/analysis/ask",
        json={"question": "药明康德2025年第三季度营业总收入是多少，并结合研报分析", "use_llm": False, "use_vector": False, "rag_top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answer"
    assert payload["financial"]["status"] == "answer"
    assert payload["financial"]["queries"]
    assert payload["sources"]
    assert "财务数据结果" in payload["answer_text"]
    assert "研报依据" in payload["answer_text"]


def _fake_rag_result():
    chunk = ResearchChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        chunk_index=1,
        title="CXO行业景气度观察",
        text="CXO行业订单边际改善，海外需求逐步恢复。TIDES业务增长强劲。",
        section_title="行业观点",
        report_type="industry",
        org_name="测试证券",
        publish_date="2025-10-01",
    )
    return SearchResult(chunk=chunk, score=0.9, score_detail={"bm25": 1.0})
