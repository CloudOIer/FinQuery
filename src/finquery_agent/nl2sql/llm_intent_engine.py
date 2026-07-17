"""LLM 意图识别引擎与混合引擎。

LlmIntentEngine.parse() 与 RuleBasedIntentEngine.parse() 接口一致,输出同样的
StructuredIntent:LLM 只负责"自然语言 → 槽位 JSON"的翻译,SQL 仍由
DSL/SQLBuilder 参数化生成,安全边界不变。

选择槽位 JSON 而非直接生成 SQL 的原因:
1. 可校验:指标/公司槽位逐个对照 registry 白名单,非法值直接丢弃;
2. 可降级:槽位缺失时走澄清问句,而非报错;
3. 可评测:结构化槽位可与规则引擎逐槽位 diff。

校验策略:
- metrics 逐个过 resolve_metric_with_policy,companies 逐个过
  registry.resolve_company_code,失败的丢弃;
- periods/operator/sort_direction 走枚举白名单,years 做范围检查;
- 校验后槽位不足以执行时,复用与规则引擎相同的 build_clarification,
  两条链路对外行为一致。
"""

from __future__ import annotations

import json
import re
from datetime import date

from finquery_agent.config import LLMSettings
from finquery_agent.llm import LLMClient
from finquery_agent.nl2sql.dsl import MetricFilter
from finquery_agent.nl2sql.intent import ChartSpec, StructuredIntent
from finquery_agent.nl2sql.intent_engine import RuleBasedIntentEngine, build_clarification, metric_warnings
from finquery_agent.schema.metrics import resolve_metric_with_policy
from finquery_agent.schema.registry import SchemaRegistry

VALID_INTENT_TYPES = {"metric_query", "trend_query", "comparison_query", "ranking_query", "unsupported"}
VALID_PERIODS = {"FY", "Q1", "HY", "Q3"}
VALID_OPERATORS = {">", ">=", "<", "<=", "="}


class LlmIntentEngine:
    """用 LLM 抽取查询槽位;所有输出经 registry 白名单校验。"""

    def __init__(self, registry: SchemaRegistry, llm_settings: LLMSettings, reference_date: date | None = None):
        self.registry = registry
        self.llm_settings = llm_settings
        self.reference_date = reference_date or date.today()
        self._client = LLMClient(llm_settings)
        # prompt 只依赖 registry(进程内不变),缓存避免每次 parse 重复拼接 60 指标 + 76 公司。
        self._system_prompt = self._build_system_prompt()

    def available(self) -> bool:
        return self._client.is_available() and self.llm_settings.intent_enabled

    def parse(self, question: str) -> StructuredIntent:
        """解析失败(网络/JSON/校验不通过)时抛 LlmIntentError,由 Hybrid 层决定降级。

        为什么抛异常而不是返回 None:parse 的返回类型须与规则引擎一致
        (StructuredIntent),用异常区分"LLM 解析了但槽位为空(合法,走澄清)"
        与"LLM 本身失败(应降级到规则)"两种情况,避免语义混叠。
        """
        content = self._client.chat(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        if content is None:
            raise LlmIntentError("LLM 请求失败或未启用。")
        payload = _extract_json(content)
        if payload is None:
            raise LlmIntentError(f"LLM 输出不是合法 JSON:{content[:200]}")
        return self._validated_intent(question, payload)

    # ------------------------------------------------------------------
    # prompt 构建
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        # 指标词表按表分组注入。带别名是为了让 LLM 把口语表述(如"挣了多少钱")
        # 映射到词表中的规范名,输出规范名后校验环节就能稳定命中 registry。
        metric_lines: list[str] = []
        for table in self.registry.tables.values():
            names = []
            for field in table.fields:
                if field.is_dimension:
                    continue
                aliases = [a for a in field.aliases if a != field.chinese_name and not a.isascii()]
                suffix = f"(别名:{'、'.join(aliases[:4])})" if aliases else ""
                names.append(f"{field.chinese_name}{suffix}")
            metric_lines.append(f"[{table.chinese_name}] {'；'.join(names)}")
        company_lines = [f"{c.stock_code} {c.stock_abbr}" for c in self.registry.companies.values()]
        return f"""你是财报数据库查询意图解析器。把用户的中文问题解析成槽位 JSON,不要生成 SQL,不要回答问题本身。
今天是 {self.reference_date.isoformat()},"去年"等相对时间要换算成绝对年份。

输出 JSON schema(只输出一个 JSON 对象,无其他文字):
{{
  "intent_type": "metric_query|trend_query|comparison_query|ranking_query|unsupported",
  "metrics": ["指标规范名(必须从下方指标词表选,最多4个)"],
  "companies": ["股票代码(6位,优先)或公司简称"],
  "years": [2024],
  "periods": ["FY|Q1|HY|Q3"],
  "allow_all_periods": false,
  "limit": 10,
  "order_by_metric": "排名依据指标(仅 ranking_query 填)",
  "sort_direction": "desc|asc",
  "metric_filters": [{{"metric": "指标名", "operator": ">|>=|<|<=|=", "value": 100.0}}],
  "chart": {{"chart_type": "line|bar"}} 或 null
}}

规则:
1. intent_type:单公司单点取数=metric_query;多年份看走势/变化=trend_query;多公司或"对比/相比"=comparison_query;全市场排名/筛选/统计("哪些公司""最高""前N""超过X""中位数""总额""占比")=ranking_query;完全与财务数据无关=unsupported。
2. metrics:只能输出指标词表中的规范名。问题里没有可映射指标时输出 []。"利润"默认映射"净利润","营收/收入"映射"营业总收入"。需要计算/统计(复合增长率、比值、中位数、差值等)的问题不是 unsupported:输出计算所需的基础指标(可多个),intent_type 按上条判断。
3. periods:年报/年度/全年=FY,一季报=Q1,半年报/中报=HY,三季报=Q3;"各报告期/所有报告期"时 periods=[] 且 allow_all_periods=true;说了年份但没说报告期时默认 ["FY"]。
4. 金额过滤条件单位统一为万元:用户说"2亿"→ value=20000,"500万"→ value=500;百分比指标保持原数值。
5. 排名类问题默认 limit=10,"前5"→ limit=5;"最低/最少/倒数"→ sort_direction="asc"。
6. companies 只填问题中明确提到的公司。筛选/排名/统计类问题("哪些公司满足条件")的答案由数据库筛选得出,companies 必须输出 [],绝对不要自己枚举公司。提到的公司必须在下方公司列表中,输出对应股票代码;不在列表中的公司不要输出。
7. 缺失的槽位输出空数组/null,不要瞎猜。年份缺失就是 [],不要用今天的年份填充。

指标词表:
{chr(10).join(metric_lines)}

公司列表(代码 简称):
{chr(10).join(company_lines)}"""

    # ------------------------------------------------------------------
    # 输出校验
    # ------------------------------------------------------------------

    def _validated_intent(self, question: str, payload: dict) -> StructuredIntent:
        intent_type = str(payload.get("intent_type") or "unsupported")
        if intent_type not in VALID_INTENT_TYPES:
            intent_type = "unsupported"

        # 指标:必须能解析到具体表字段,否则丢弃 —— 宁可触发澄清也不让
        # 未知指标流入 SQLBuilder(那里会抛错,用户体验更差)。
        metrics: list[str] = []
        seen_fields: set[str] = set()
        for raw in _as_str_list(payload.get("metrics"))[:4]:
            resolution = resolve_metric_with_policy(self.registry, raw)
            if resolution.default_field is None:
                continue
            key = f"{resolution.default_field.table_name}.{resolution.default_field.name}"
            if key in seen_fields:
                continue
            seen_fields.add(key)
            metrics.append(raw)

        # 公司:LLM 幻觉出的代码/名称在 registry 查不到就丢弃。
        company_codes: list[str] = []
        company_names: list[str] = []
        for raw in _as_str_list(payload.get("companies")):
            code = self.registry.resolve_company_code(raw)
            if code is None or code in company_codes:
                continue
            company_codes.append(code)
            company_names.append(self.registry.companies[code].stock_abbr)

        years = tuple(sorted({int(y) for y in payload.get("years") or [] if _is_valid_year(y)}))
        periods = tuple(dict.fromkeys(p for p in _as_str_list(payload.get("periods")) if p in VALID_PERIODS))
        allow_all_periods = bool(payload.get("allow_all_periods", False))

        metric_filters: list[MetricFilter] = []
        for item in payload.get("metric_filters") or []:
            if not isinstance(item, dict):
                continue
            metric = str(item.get("metric") or "")
            operator = str(item.get("operator") or "")
            # filter 的 metric 必须是已校验通过的指标,防止过滤条件引用未知字段。
            if metric not in metrics or operator not in VALID_OPERATORS:
                continue
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            metric_filters.append(MetricFilter(metric=metric, operator=operator, value=value))

        sort_direction = payload.get("sort_direction") if payload.get("sort_direction") in {"asc", "desc"} else "desc"
        limit = _clamp_limit(payload.get("limit"), intent_type)
        order_by_metric = payload.get("order_by_metric") if payload.get("order_by_metric") in metrics else None
        if intent_type == "ranking_query" and order_by_metric is None and metrics:
            order_by_metric = metrics[0]

        chart = None
        chart_payload = payload.get("chart")
        if isinstance(chart_payload, dict) and chart_payload.get("chart_type") in {"line", "bar"}:
            chart = ChartSpec(
                chart_type=str(chart_payload["chart_type"]),
                x="report_year",
                y=metrics[0] if metrics else None,
                title=None,
            )

        if not metrics:
            intent_type = "unsupported"

        clarification = build_clarification(
            intent_type,
            has_companies=bool(company_codes),
            has_metrics=bool(metrics),
            years=years,
            periods=periods,
            allow_all_periods=allow_all_periods,
        )
        return StructuredIntent(
            original_question=question,
            intent_type=intent_type,
            metrics=tuple(metrics),
            company_codes=tuple(company_codes),
            company_names=tuple(company_names),
            years=years,
            periods=periods,
            limit=limit,
            order_by_metric=order_by_metric,
            sort_direction=sort_direction,
            metric_filters=tuple(metric_filters),
            allow_all_periods=allow_all_periods,
            needs_clarification=clarification is not None,
            clarification=clarification,
            warnings=tuple(metric_warnings(self.registry, metrics)),
            chart=chart,
            intent_source="llm",
        )


class LlmIntentError(RuntimeError):
    """LLM 意图解析失败(区别于'解析成功但槽位缺失')。"""


class HybridIntentEngine:
    """LLM 优先、规则兜底的意图引擎。

    降级触发条件(任一):intent_enabled 关闭、LLM 请求失败/超时、
    输出非 JSON。降级产物标记 intent_source="rule_fallback" 以便追踪。

    为什么不做"LLM 结果差就换规则结果"的择优:两个引擎对同一题可能给出
    不同但都合法的解析,运行时没有 ground truth 无从判断谁对;择优逻辑
    只会引入隐蔽的不确定性。评测阶段用 compare_intent_engines.py 离线对比。
    """

    def __init__(self, rule_engine: RuleBasedIntentEngine, llm_engine: LlmIntentEngine):
        self.rule_engine = rule_engine
        self.llm_engine = llm_engine

    def parse(self, question: str) -> StructuredIntent:
        if not self.llm_engine.available():
            return self.rule_engine.parse(question)
        try:
            return self.llm_engine.parse(question)
        except LlmIntentError:
            intent = self.rule_engine.parse(question)
            return _with_source(intent, "rule_fallback")


def _with_source(intent: StructuredIntent, source: str) -> StructuredIntent:
    from dataclasses import replace

    return replace(intent, intent_source=source)


def _extract_json(content: str) -> dict | None:
    """容错解析:response_format 通常保证纯 JSON,但部分模型仍会包 ```json 围栏。"""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_valid_year(value: object) -> bool:
    try:
        return 1990 <= int(value) <= 2100
    except (TypeError, ValueError):
        return False


def _clamp_limit(value: object, intent_type: str) -> int:
    default = 10 if intent_type == "ranking_query" else 100
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return default
    # 上限与 SQLBuilder 的保护一致,防止 LLM 输出异常大的 limit 拖垮查询。
    return max(1, min(limit, 500))
