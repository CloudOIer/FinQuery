"""LlmIntentEngine / HybridIntentEngine 测试。

全部用 mock 替代真实 LLM 调用:单测要验证的是"给定 LLM 输出,校验与降级
逻辑是否正确",LLM 输出本身的质量由评测脚本(compare_intent_engines.py)负责。
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from finquery_agent.config import LLMSettings
from finquery_agent.nl2sql import HybridIntentEngine, LlmIntentEngine, LlmIntentError, RuleBasedIntentEngine
from finquery_agent.schema import load_default_registry

REGISTRY = load_default_registry()
ENABLED_SETTINGS = LLMSettings(
    enabled=True,
    provider="deepseek",
    model="deepseek-v4-flash",
    api_key="test-key",
    base_url="https://api.deepseek.com",
    intent_enabled=True,
)


def _llm_engine(reply: str | None) -> LlmIntentEngine:
    engine = LlmIntentEngine(REGISTRY, ENABLED_SETTINGS, reference_date=date(2026, 6, 18))
    engine._client.chat = lambda *args, **kwargs: reply  # type: ignore[method-assign]
    return engine


def _reply(**slots) -> str:
    payload = {
        "intent_type": "metric_query",
        "metrics": [],
        "companies": [],
        "years": [],
        "periods": [],
        "allow_all_periods": False,
        "limit": 100,
        "order_by_metric": None,
        "sort_direction": "desc",
        "metric_filters": [],
        "chart": None,
    }
    payload.update(slots)
    return json.dumps(payload, ensure_ascii=False)


def test_llm_engine_parses_valid_slots():
    engine = _llm_engine(_reply(metrics=["营业总收入"], companies=["600332"], years=[2024], periods=["FY"]))

    intent = engine.parse("白云山2024年年报营收是多少")

    assert intent.intent_source == "llm"
    assert intent.needs_clarification is False
    assert intent.metrics == ("营业总收入",)
    assert intent.company_codes == ("600332",)
    assert intent.company_names == ("白云山",)
    assert intent.years == (2024,)
    assert intent.periods == ("FY",)


def test_llm_engine_drops_hallucinated_metric_and_company():
    engine = _llm_engine(_reply(metrics=["不存在的指标"], companies=["999999", "特斯拉"], years=[2024], periods=["FY"]))

    intent = engine.parse("特斯拉2024年 XX 指标")

    # 幻觉槽位全部被白名单校验丢弃 → 缺指标 → 澄清而不是错误查询。
    assert intent.metrics == ()
    assert intent.company_codes == ()
    assert intent.needs_clarification is True
    assert intent.clarification is not None
    assert "metric" in intent.clarification.missing_slots


def test_llm_engine_validates_periods_years_and_filters():
    engine = _llm_engine(
        _reply(
            intent_type="ranking_query",
            metrics=["净利润"],
            years=[2024, 1800, "abc"],
            periods=["FY", "Q9"],
            limit=99999,
            metric_filters=[
                {"metric": "净利润", "operator": ">", "value": 20000},
                {"metric": "净利润", "operator": "DROP TABLE", "value": 1},
                {"metric": "未校验指标", "operator": ">", "value": 1},
                {"metric": "净利润", "operator": "<", "value": "not-a-number"},
            ],
        )
    )

    intent = engine.parse("2024年净利润超过2亿的公司")

    assert intent.years == (2024,)  # 1800 与 "abc" 被丢弃
    assert intent.periods == ("FY",)  # Q9 被丢弃
    assert intent.limit == 500  # 超大 limit 被钳制
    assert len(intent.metric_filters) == 1  # 只保留合法 filter
    assert intent.metric_filters[0].operator == ">"
    assert intent.order_by_metric == "净利润"  # ranking 缺 order_by 时回填首个指标


def test_llm_engine_accepts_fenced_json():
    engine = _llm_engine("```json\n" + _reply(metrics=["净利润"], companies=["白云山"], years=[2023], periods=["FY"]) + "\n```")

    intent = engine.parse("白云山2023年净利润")

    assert intent.metrics == ("净利润",)
    assert intent.company_codes == ("600332",)


@pytest.mark.parametrize("bad_reply", [None, "这不是JSON", "[1,2,3]"])
def test_llm_engine_raises_on_invalid_output(bad_reply):
    engine = _llm_engine(bad_reply)

    with pytest.raises(LlmIntentError):
        engine.parse("白云山2024年营收")


def test_hybrid_uses_llm_result_when_available():
    llm = _llm_engine(_reply(metrics=["营业总收入"], companies=["600332"], years=[2024], periods=["FY"]))
    hybrid = HybridIntentEngine(RuleBasedIntentEngine(REGISTRY), llm)

    intent = hybrid.parse("白云山2024年年报营收是多少")

    assert intent.intent_source == "llm"


def test_hybrid_falls_back_to_rule_on_llm_failure():
    llm = _llm_engine("不是JSON")
    hybrid = HybridIntentEngine(RuleBasedIntentEngine(REGISTRY), llm)

    intent = hybrid.parse("白云山2024年年报营收是多少")

    # 规则引擎兜底,槽位仍然解析成功,来源被标记为降级。
    assert intent.intent_source == "rule_fallback"
    assert intent.company_codes == ("600332",)
    assert intent.years == (2024,)


def test_hybrid_skips_llm_when_intent_disabled():
    settings = LLMSettings(
        enabled=True,
        provider="deepseek",
        model="deepseek-v4-flash",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        intent_enabled=False,  # 主开关开、意图开关关 → 不应发起 LLM 调用
    )
    llm = LlmIntentEngine(REGISTRY, settings, reference_date=date(2026, 6, 18))

    def _boom(*args, **kwargs):
        raise AssertionError("intent_enabled=False 时不应调用 LLM")

    llm._client.chat = _boom  # type: ignore[method-assign]
    hybrid = HybridIntentEngine(RuleBasedIntentEngine(REGISTRY), llm)

    intent = hybrid.parse("白云山2024年年报营收是多少")

    assert intent.intent_source == "rule"
    assert intent.company_codes == ("600332",)
