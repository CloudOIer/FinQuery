import json

from finquery_agent.nl2sql.answer import AnswerComposer
from finquery_agent.nl2sql.intent import StructuredIntent
from finquery_agent.config import LLMSettings
from finquery_agent.schema import load_default_registry


def test_answer_composer_generates_single_metric_sentence_without_llm():
    composer = AnswerComposer(load_default_registry())
    intent = StructuredIntent(
        original_question="药明康德2025年第三季度营业总收入是多少",
        intent_type="metric_query",
        metrics=("营业总收入",),
        company_codes=("603259",),
        years=(2025,),
        periods=("Q3",),
    )
    queries = [
        {
            "table_name": "core_performance_indicators_sheet",
            "metric_columns": ("total_operating_revenue",),
            "result": {
                "columns": ["stock_code", "stock_abbr", "report_year", "report_period", "total_operating_revenue"],
                "rows": [{"stock_code": "603259", "stock_abbr": "药明康德", "report_year": 2025, "report_period": "Q3", "total_operating_revenue": 3285671.65}],
                "units": {"total_operating_revenue": "万元"},
                "row_count": 1,
            },
        }
    ]

    answer = composer.compose(intent.original_question, intent, queries, use_llm=False)

    assert answer.llm_used is False
    assert answer.answer_source == "deterministic"
    assert "药明康德2025年第三季度" in answer.answer_text
    assert "营业总收入" in answer.answer_text
    assert "3,285,671.65万元" in answer.answer_text


def test_answer_composer_reports_zero_rows_as_zero_count():
    composer = AnswerComposer(load_default_registry())
    intent = StructuredIntent(original_question="负数公司数量", intent_type="ranking_query", metrics=("净利润",), years=(2025,), periods=("Q3",))
    queries = [{"table_name": "core_performance_indicators_sheet", "metric_columns": ("net_profit_10k_yuan",), "result": {"columns": [], "rows": [], "units": {}, "row_count": 0}}]

    answer = composer.compose(intent.original_question, intent, queries, use_llm=False)

    assert "数量为 0" in answer.answer_text


def test_answer_composer_uses_registry_company_abbr_when_row_abbr_missing():
    composer = AnswerComposer(load_default_registry())
    intent = StructuredIntent(
        original_question="药明康德2025年第三季度营业总收入是多少",
        intent_type="metric_query",
        metrics=("营业总收入",),
        company_codes=("603259",),
        years=(2025,),
        periods=("Q3",),
    )
    queries = [
        {
            "table_name": "core_performance_indicators_sheet",
            "metric_columns": ("total_operating_revenue",),
            "result": {
                "columns": ["stock_code", "stock_abbr", "report_year", "report_period", "total_operating_revenue"],
                "rows": [{"stock_code": "603259", "stock_abbr": "", "report_year": 2025, "report_period": "Q3", "total_operating_revenue": 3285671.65}],
                "units": {"total_operating_revenue": "万元"},
                "row_count": 1,
            },
        }
    ]

    answer = composer.compose(intent.original_question, intent, queries, use_llm=False)

    assert "药明康德2025年第三季度" in answer.answer_text


def test_answer_composer_uses_llm_when_enabled_and_requested(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"\xe8\xbf\x99\xe6\x98\xaf LLM \xe7\x94\x9f\xe6\x88\x90\xe7\x9a\x84\xe7\xae\x80\xe6\xb4\x81\xe5\x9b\x9e\xe7\xad\x94\xe3\x80\x82"}}]}'

    calls = []

    def fake_urlopen(request, timeout):
        calls.append({"url": request.full_url, "headers": request.headers, "json": json.loads(request.data.decode("utf-8")), "timeout": timeout})
        return FakeResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    composer = AnswerComposer(
        load_default_registry(),
        LLMSettings(enabled=True, provider="deepseek", model="deepseek-v4-flash", api_key="test-key", base_url="https://api.deepseek.com", timeout_seconds=7),
    )
    intent = StructuredIntent(original_question="白云山2024年年报营收是多少", intent_type="metric_query", metrics=("营收",), company_codes=("600332",), years=(2024,), periods=("FY",))

    answer = composer.compose(intent.original_question, intent, [], use_llm=True)

    assert answer.llm_used is True
    assert answer.answer_source == "llm"
    assert answer.answer_text == "这是 LLM 生成的简洁回答。"
    assert calls[0]["url"] == "https://api.deepseek.com/chat/completions"
    assert calls[0]["json"]["model"] == "deepseek-v4-flash"
