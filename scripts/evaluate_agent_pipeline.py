"""Agent 管道 vs 单轮固定管道:LLM-as-judge pairwise 盲评。

评测对象是附件6 的复杂题(多意图/归因分析):这正是 Agent 化的目标场景,
简单取数题两条链路应当等价,不纳入对比。

盲评协议(控制评审偏差):
- judge 看到的是"回答A/回答B",不知道来自哪条链路;
- 每题随机决定 A/B 与新旧链路的对应关系(防位置偏差:LLM judge 已知偏好首位);
- judge 输出 winner(A/B/tie)+ 简短理由;
- judge 与被评 LLM 同为 DeepSeek,存在 self-preference,但对两条链路对称,
  不影响相对比较的有效性(报告中如实标注)。

用法:
    python scripts/evaluate_agent_pipeline.py --limit 3   # 试跑
    python scripts/evaluate_agent_pipeline.py             # 全量(耗时较长)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import date
from pathlib import Path
from typing import Any

from finquery_agent.agent import AgentService
from finquery_agent.analysis import AnalysisService
from finquery_agent.config import load_llm_settings, load_rag_settings
from finquery_agent.db import create_database_engine
from finquery_agent.llm import LLMClient
from finquery_agent.nl2sql import QueryExecutor, RuleBasedIntentEngine, SQLBuilder
from finquery_agent.nl2sql.answer import AnswerComposer
from finquery_agent.nl2sql.charting import ChartRenderer
from finquery_agent.rag.service import RAGService
from finquery_agent.schema import load_default_registry

DEFAULT_QUESTIONS = Path("第一批数据/附件6：问题汇总.CSV")
DEFAULT_OUTPUT = Path("data/evaluation/agent_vs_pipeline.md")
TARGET_TYPES = ("多意图", "归因分析")

_JUDGE_SYSTEM = """你是财务问答质量评审。对同一问题的两个回答,从以下维度比较:
1. 完整性:是否回答了问题的全部子问题;
2. 数据支撑:数字是否具体、计算是否给出;
3. 证据引用:是否区分财务数据与研报观点、标注来源;
4. 清晰度:结构与表达。
输出 JSON:{"winner": "A"|"B"|"tie", "reason": "一句话理由"}。
只根据回答文本评判,不要臆测系统实现。"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise blind evaluation: agent vs fixed pipeline.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="A/B 随机换位的种子,保证可复现。")
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    print(f"{len(questions)} questions (types: {TARGET_TYPES})")

    analysis, agent = _build_services()
    judge = LLMClient(load_llm_settings())
    if not judge.is_available():
        raise SystemExit("judge LLM 不可用:检查 config/llm.json。")

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {item['id']} {item['question'][:40]} ...")
        row = _evaluate_question(item, analysis, agent, judge, rng)
        rows.append(row)
        print(f"    pipeline {row['pipeline_seconds']:.0f}s | agent {row['agent_seconds']:.0f}s ({row['agent_steps']}步) | winner: {row['winner']}")

    _write_report(rows, args.output, args.seed)
    print(f"wrote {args.output}")


def _build_services() -> tuple[AnalysisService, AgentService]:
    registry = load_default_registry()
    llm = load_llm_settings()
    analysis = AnalysisService(
        registry=registry,
        intent_engine=RuleBasedIntentEngine(registry),
        sql_builder=SQLBuilder(registry),
        query_executor=QueryExecutor(create_database_engine(), registry),
        answer_composer=AnswerComposer(registry, llm),
        chart_renderer=ChartRenderer(registry),
        rag_service=RAGService(load_rag_settings(), llm),
        llm_settings=llm,
    )
    return analysis, AgentService(analysis, llm)


def _load_questions(path: Path) -> list[dict[str, str]]:
    questions = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            # 问题类型可能是"归因分析；多意图"复合标注,包含目标类型即入选。
            if not any(t in row["问题类型"] for t in TARGET_TYPES):
                continue
            for item in json.loads(row["问题"]):
                questions.append({"id": row["编号"], "type": row["问题类型"], "question": item["Q"]})
    return questions


def _evaluate_question(
    item: dict[str, str],
    analysis: AnalysisService,
    agent: AgentService,
    judge: LLMClient,
    rng: random.Random,
) -> dict[str, Any]:
    started = time.time()
    pipeline_result = analysis.ask(item["question"])
    pipeline_seconds = time.time() - started

    started = time.time()
    agent_result = agent.ask(item["question"])
    agent_seconds = time.time() - started

    pipeline_text = pipeline_result.answer_text or "(无回答)"
    agent_text = agent_result.answer_text or "(无回答)"
    agent_degraded = agent_result.answer_source != "agent_llm"

    # 随机换位盲评。
    agent_is_a = rng.random() < 0.5
    text_a, text_b = (agent_text, pipeline_text) if agent_is_a else (pipeline_text, agent_text)
    verdict = _judge(judge, item["question"], text_a, text_b)
    winner_raw = verdict.get("winner", "tie")
    if winner_raw == "tie":
        winner = "tie"
    elif (winner_raw == "A") == agent_is_a:
        winner = "agent"
    else:
        winner = "pipeline"

    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "winner": winner,
        "reason": verdict.get("reason", ""),
        "agent_is_a": agent_is_a,
        "agent_degraded": agent_degraded,
        "agent_steps": len(agent_result.execution_trace or []),
        "pipeline_seconds": pipeline_seconds,
        "agent_seconds": agent_seconds,
        "pipeline_answer": pipeline_text,
        "agent_answer": agent_text,
    }


def _judge(client: LLMClient, question: str, answer_a: str, answer_b: str) -> dict[str, str]:
    content = client.chat(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "answer_A": answer_a[:2500], "answer_B": answer_b[:2500]},
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    if not content:
        return {"winner": "tie", "reason": "judge 请求失败"}
    try:
        data = json.loads(content)
        if data.get("winner") in {"A", "B", "tie"}:
            return data
    except json.JSONDecodeError:
        pass
    return {"winner": "tie", "reason": "judge 输出无法解析"}


def _write_report(rows: list[dict[str, Any]], output: Path, seed: int) -> None:
    total = len(rows)
    wins = {"agent": 0, "pipeline": 0, "tie": 0}
    for row in rows:
        wins[row["winner"]] += 1
    degraded = sum(1 for row in rows if row["agent_degraded"])
    avg_pipeline = sum(row["pipeline_seconds"] for row in rows) / total
    avg_agent = sum(row["agent_seconds"] for row in rows) / total
    avg_steps = sum(row["agent_steps"] for row in rows) / total

    lines = [
        "# Agent 管道 vs 单轮固定管道(LLM-as-judge 盲评)",
        "",
        f"- 运行日期:{date.today().isoformat()};judge=DeepSeek(与被评模型相同,self-preference 对称);随机种子 {seed}",
        f"- 题目:附件6 {'/'.join(TARGET_TYPES)} 共 {total} 题;A/B 每题随机换位",
        "",
        "| 结果 | 数量 | 占比 |",
        "| --- | --- | --- |",
        f"| Agent 胜 | {wins['agent']} | {wins['agent'] / total:.0%} |",
        f"| 固定管道胜 | {wins['pipeline']} | {wins['pipeline'] / total:.0%} |",
        f"| 平局 | {wins['tie']} | {wins['tie'] / total:.0%} |",
        "",
        f"- Agent 降级次数:{degraded};平均步数:{avg_steps:.1f}",
        f"- 平均耗时:固定管道 {avg_pipeline:.1f}s vs Agent {avg_agent:.1f}s",
        "",
        "## 每题明细",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['id']}({row['type']})winner={row['winner']}")
        lines.append("")
        lines.append(f"- 问题:{row['question']}")
        lines.append(f"- judge 理由:{row['reason']}")
        lines.append(f"- Agent:{row['agent_steps']}步/{row['agent_seconds']:.0f}s{'(降级)' if row['agent_degraded'] else ''};固定管道:{row['pipeline_seconds']:.0f}s")
        lines.append("")
        lines.append(f"<details><summary>Agent 回答</summary>\n\n{row['agent_answer'][:1200]}\n\n</details>")
        lines.append("")
        lines.append(f"<details><summary>固定管道回答</summary>\n\n{row['pipeline_answer'][:1200]}\n\n</details>")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
