"""对比规则意图引擎与 LLM 意图引擎的槽位解析质量。

两部分评测,对应两类数据的不同可信度:
1. 附件4 的 49 题(规范表述):规则引擎在这批题上已被 evaluate_task2_questions
   验证过(48 ok_nonempty + 1 ok_empty),其输出可当 silver label;
   LLM 输出与之逐槽位 diff,一致率高说明 LLM 没有在基本盘上退化。
2. 变体题(口语化/换表述/错别字,data/evaluation/intent_variant_questions.jsonl):
   题目为自造,带人工确认的期望槽位(golden),两个引擎分别打分,
   预期 LLM 在这批题上显著领先 —— 这是引入 LLM 的核心收益。

槽位对比不做字符串精确匹配而是先 canonicalize:
规则引擎输出用户表面词("营收"),LLM 输出词表规范名("营业总收入(万元)"),
两者字符串不同但都解析到 core_performance_indicators_sheet.total_operating_revenue,
应视为一致。指标统一映射到 table.field 键后再比较集合。

用法:
    python scripts/compare_intent_engines.py                 # 全量(需要 DeepSeek 可用)
    python scripts/compare_intent_engines.py --limit 5       # 试跑
    python scripts/compare_intent_engines.py --skip-official # 只跑变体题
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

from finquery_agent.config import load_llm_settings
from finquery_agent.nl2sql import LlmIntentEngine, LlmIntentError, RuleBasedIntentEngine, StructuredIntent
from finquery_agent.schema import load_default_registry
from finquery_agent.schema.metrics import resolve_metric_with_policy
from finquery_agent.schema.registry import SchemaRegistry

DEFAULT_OFFICIAL = Path("第一批数据/附件4：问题汇总.CSV")
DEFAULT_VARIANTS = Path("data/evaluation/intent_variant_questions.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/intent_engine_comparison.md")

# 参与对比的槽位。chart/warnings 不比:chart 是表现层增强,warnings 是提示文案,
# 都不影响 SQL 查询结果的正确性。
COMPARED_SLOTS = ("intent_type", "metrics", "company_codes", "years", "periods", "needs_clarification", "metric_filters")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rule-based vs LLM intent engines slot by slot.")
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N questions of each part (0 = all).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent LLM requests.")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-variants", action="store_true")
    args = parser.parse_args()

    registry = load_default_registry()
    # 固定参考日期保证"近三年"等相对时间在两个引擎、多次运行间可复现。
    reference_date = date.today()
    rule_engine = RuleBasedIntentEngine(registry, reference_date=reference_date)
    llm_settings = load_llm_settings()
    llm_engine = LlmIntentEngine(registry, llm_settings, reference_date=reference_date)
    if not llm_engine.available():
        raise SystemExit("LLM 意图识别不可用:请检查 config/llm.json 的 enabled/intent_enabled/api_key。")

    sections: list[str] = [
        "# 意图引擎对比评测(规则 vs LLM)",
        "",
        f"- 运行日期:{date.today().isoformat()}(相对时间参考日)",
        f"- LLM:{llm_settings.provider} {llm_settings.model}",
        f"- 对比槽位:{', '.join(COMPARED_SLOTS)};指标先归一化到 table.field 再比较",
        "",
    ]

    if not args.skip_official:
        official = _load_official_questions(args.official)
        if args.limit:
            official = official[: args.limit]
        print(f"[official] {len(official)} questions, silver label = rule engine output")
        rows = _run_pairs(official, rule_engine, llm_engine, registry, args.workers)
        sections.extend(_render_official(rows))

    if not args.skip_variants:
        variants = _load_variants(args.variants)
        if args.limit:
            variants = variants[: args.limit]
        print(f"[variants] {len(variants)} questions, golden = human-confirmed expected slots")
        rows = _run_variants(variants, rule_engine, llm_engine, registry, args.workers)
        sections.extend(_render_variants(rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------

def _load_official_questions(path: Path) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            payload = json.loads(row["问题"])
            for index, item in enumerate(payload, 1):
                questions.append({"id": f"{row['编号']}-{index}", "question": item["Q"]})
    return questions


def _load_variants(path: Path) -> list[dict[str, Any]]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ----------------------------------------------------------------------
# 槽位归一化与对比
# ----------------------------------------------------------------------

def _canonical_metrics(registry: SchemaRegistry, metrics: tuple[str, ...] | list[str]) -> frozenset[str]:
    keys = set()
    for metric in metrics:
        resolution = resolve_metric_with_policy(registry, metric)
        field = resolution.default_field
        keys.add(f"{field.table_name}.{field.name}" if field else f"unresolved:{metric}")
    return frozenset(keys)


def _slots(registry: SchemaRegistry, intent: StructuredIntent) -> dict[str, Any]:
    return {
        "intent_type": intent.intent_type,
        "metrics": _canonical_metrics(registry, intent.metrics),
        "company_codes": frozenset(intent.company_codes),
        "years": frozenset(intent.years),
        "periods": frozenset(intent.periods),
        "needs_clarification": intent.needs_clarification,
        "metric_filters": frozenset(
            (next(iter(_canonical_metrics(registry, [f.metric]))), f.operator, f.value) for f in intent.metric_filters
        ),
    }


def _expected_slots(registry: SchemaRegistry, expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent_type": expected["intent_type"],
        "metrics": _canonical_metrics(registry, expected.get("metrics", [])),
        "company_codes": frozenset(expected.get("company_codes", [])),
        "years": frozenset(expected.get("years", [])),
        "periods": frozenset(expected.get("periods", [])),
        # 变体题都设计为槽位完整,期望不触发澄清。
        "needs_clarification": False,
        "metric_filters": frozenset(
            (next(iter(_canonical_metrics(registry, [f["metric"]]))), f["operator"], float(f["value"]))
            for f in expected.get("metric_filters", [])
        ),
    }


def _diff(lhs: dict[str, Any], rhs: dict[str, Any]) -> list[str]:
    return [slot for slot in COMPARED_SLOTS if lhs[slot] != rhs[slot]]


def _parse_llm(llm_engine: LlmIntentEngine, question: str) -> StructuredIntent | Exception:
    try:
        return llm_engine.parse(question)
    except LlmIntentError as exc:
        return exc


# ----------------------------------------------------------------------
# Part A:附件4,规则输出为 silver label
# ----------------------------------------------------------------------

def _run_pairs(
    questions: list[dict[str, str]],
    rule_engine: RuleBasedIntentEngine,
    llm_engine: LlmIntentEngine,
    registry: SchemaRegistry,
    workers: int,
) -> list[dict[str, Any]]:
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        llm_results = list(pool.map(lambda item: _parse_llm(llm_engine, item["question"]), questions))
    print(f"[official] LLM parsing finished in {time.time() - started:.1f}s")

    rows = []
    for item, llm_result in zip(questions, llm_results):
        rule_intent = rule_engine.parse(item["question"])
        row: dict[str, Any] = {"id": item["id"], "question": item["question"]}
        if isinstance(llm_result, Exception):
            row.update({"status": "llm_error", "error": str(llm_result), "diff_slots": list(COMPARED_SLOTS)})
        else:
            rule_slots = _slots(registry, rule_intent)
            llm_slots = _slots(registry, llm_result)
            diff = _diff(rule_slots, llm_slots)
            row.update({
                "status": "match" if not diff else "diff",
                "diff_slots": diff,
                "rule": {k: _show(v) for k, v in rule_slots.items() if k in diff},
                "llm": {k: _show(v) for k, v in llm_slots.items() if k in diff},
            })
        rows.append(row)
    return rows


def _render_official(rows: list[dict[str, Any]]) -> list[str]:
    total = len(rows)
    match = sum(1 for r in rows if r["status"] == "match")
    errors = sum(1 for r in rows if r["status"] == "llm_error")
    slot_diff_counts = {slot: sum(1 for r in rows if slot in r["diff_slots"]) for slot in COMPARED_SLOTS}
    lines = [
        "## Part A:附件4 官方 49 题(silver label = 规则引擎)",
        "",
        f"- 整题槽位全一致:{match}/{total}({match / total:.1%})",
        f"- LLM 解析失败:{errors}",
        "",
        "| 槽位 | 不一致题数 | 槽位一致率 |",
        "| --- | --- | --- |",
    ]
    for slot, count in slot_diff_counts.items():
        lines.append(f"| {slot} | {count} | {(total - count) / total:.1%} |")
    diff_rows = [r for r in rows if r["status"] != "match"]
    if diff_rows:
        lines += ["", "### 差异明细(不一致 ≠ LLM 错误,需人工判读哪边更合理)", ""]
        for row in diff_rows:
            lines.append(f"- **{row['id']}** {row['question']}")
            if row["status"] == "llm_error":
                lines.append(f"  - LLM 失败:{row['error']}")
            else:
                for slot in row["diff_slots"]:
                    lines.append(f"  - {slot}:规则={row['rule'][slot]} | LLM={row['llm'][slot]}")
    lines.append("")
    return lines


# ----------------------------------------------------------------------
# Part B:变体题,人工确认的 expected 为 golden
# ----------------------------------------------------------------------

def _run_variants(
    variants: list[dict[str, Any]],
    rule_engine: RuleBasedIntentEngine,
    llm_engine: LlmIntentEngine,
    registry: SchemaRegistry,
    workers: int,
) -> list[dict[str, Any]]:
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        llm_results = list(pool.map(lambda item: _parse_llm(llm_engine, item["question"]), variants))
    print(f"[variants] LLM parsing finished in {time.time() - started:.1f}s")

    rows = []
    for item, llm_result in zip(variants, llm_results):
        expected = _expected_slots(registry, item["expected"])
        rule_slots = _slots(registry, rule_engine.parse(item["question"]))
        row: dict[str, Any] = {
            "id": item["id"],
            "variant_type": item["variant_type"],
            "question": item["question"],
            "rule_diff": _diff(expected, rule_slots),
        }
        if isinstance(llm_result, Exception):
            row["llm_diff"] = list(COMPARED_SLOTS)
            row["llm_error"] = str(llm_result)
        else:
            llm_slots = _slots(registry, llm_result)
            row["llm_diff"] = _diff(expected, llm_slots)
            row["detail"] = {
                slot: {"expected": _show(expected[slot]), "rule": _show(rule_slots[slot]), "llm": _show(llm_slots[slot])}
                for slot in set(row["rule_diff"]) | set(row["llm_diff"])
            }
        rows.append(row)
    return rows


def _render_variants(rows: list[dict[str, Any]]) -> list[str]:
    total = len(rows)
    lines = ["## Part B:变体题(golden = 人工确认期望槽位)", ""]
    variant_types = sorted({r["variant_type"] for r in rows})
    lines += ["| 变体类型 | 题数 | 规则引擎全对 | LLM 引擎全对 |", "| --- | --- | --- | --- |"]
    for vt in [*variant_types, "ALL"]:
        subset = rows if vt == "ALL" else [r for r in rows if r["variant_type"] == vt]
        rule_ok = sum(1 for r in subset if not r["rule_diff"])
        llm_ok = sum(1 for r in subset if not r["llm_diff"])
        lines.append(f"| {vt} | {len(subset)} | {rule_ok}/{len(subset)}({rule_ok / len(subset):.0%}) | {llm_ok}/{len(subset)}({llm_ok / len(subset):.0%}) |")
    lines += ["", "### 错误明细", ""]
    for row in rows:
        if not row["rule_diff"] and not row["llm_diff"]:
            continue
        lines.append(f"- **{row['id']}**({row['variant_type']}){row['question']}")
        if row.get("llm_error"):
            lines.append(f"  - LLM 失败:{row['llm_error']}")
        lines.append(f"  - 规则错误槽位:{row['rule_diff'] or '无'};LLM 错误槽位:{row['llm_diff'] or '无'}")
        for slot, detail in (row.get("detail") or {}).items():
            lines.append(f"  - {slot}:期望={detail['expected']} | 规则={detail['rule']} | LLM={detail['llm']}")
    lines.append("")
    return lines


def _show(value: Any) -> str:
    if isinstance(value, frozenset):
        return "{" + ", ".join(str(v) for v in sorted(value, key=str)) + "}"
    return str(value)


if __name__ == "__main__":
    main()
