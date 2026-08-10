"""RAG 检索质量评测:四种检索配置的消融对比。

评测口径是文档级:golden set 标注的是"哪些研报文档与问题相关"(文档级标注
人工可核对,chunk 级标注成本过高且边界模糊),预测取 top_k chunks 去重后的
doc_id 集合参与计算。

指标:
- hit_rate@k:至少命中一篇相关文档的问题占比(问答场景的底线指标——
  有一篇对的文档,答案就有依据);
- recall@k:命中的相关文档数 / 标注相关文档数,再对问题取平均;
- MRR:第一篇相关文档出现位置的倒数,衡量"最相关的排多前"。

底下两项回答"指标差异是否可信"与"该优化哪一阶段":
- bootstrap 95% 置信区间:两十几题的规模下单题翻转就能带来几个百分点波动,
  点估计无法支撑"某配置更好"的结论;
- 粗排候选集召回诊断:精排只能重排候选,粗排没召回的文档后续无法找回,
  这个数字是整条链路的硬上限。

消融配置:单路(bm25/vector) × 融合方式(weighted/rrf) × 是否精排。
加权求和需要先把两路分数归一化到同一量纲,RRF 只用排名,对比二者可以
判定融合环节是否是当前瓶颈。cross-encoder 在 CPU 上逐题打分较慢,脚本带进度输出。

用法:
    python scripts/evaluate_rag_retrieval.py                # 全量
    python scripts/evaluate_rag_retrieval.py --limit 3      # 试跑
    python scripts/evaluate_rag_retrieval.py --only-confirmed  # 只用已人工确认的题
    python scripts/evaluate_rag_retrieval.py --skip-coarse-diagnosis
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

from finquery_agent.config import load_rag_settings
from finquery_agent.rag.index import load_rag_index
from finquery_agent.rag.retriever import HybridRetriever

DEFAULT_GOLDEN = Path("data/evaluation/rag_golden_set.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/rag_retrieval_ablation.md")

# (名称, 检索参数, settings 覆盖项)
ABLATIONS: tuple[tuple[str, dict, dict], ...] = (
    ("bm25", {"use_vector": False, "use_reranker": False}, {}),
    ("vector", {"use_vector": True, "use_reranker": False, "bm25_off": True}, {}),
    ("hybrid(weighted)", {"use_vector": True, "use_reranker": False}, {"fusion_method": "weighted"}),
    ("hybrid(rrf)", {"use_vector": True, "use_reranker": False}, {"fusion_method": "rrf"}),
    (
        "hybrid(rrf)+rerank(candidates)",
        {"use_vector": True, "use_reranker": True},
        {"fusion_method": "rrf", "rerank_scope": "candidates"},
    ),
    (
        "hybrid(rrf)+rerank(final)",
        {"use_vector": True, "use_reranker": True},
        {"fusion_method": "rrf", "rerank_scope": "final"},
    ),
    (
        "基线:weighted+rerank无候选配额",
        {"use_vector": True, "use_reranker": True},
        {"fusion_method": "weighted", "rerank_scope": "candidates", "candidate_max_chunks_per_doc": 0},
    ),
)

COARSE_CANDIDATE_KS = (30, 50, 100)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study of RAG retrieval configurations.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-confirmed", action="store_true", help="Only use questions with confirmed=true.")
    parser.add_argument("--skip-coarse-diagnosis", action="store_true")
    args = parser.parse_args()

    questions = _load_golden(args.golden, args.only_confirmed)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("golden set 为空(或没有 confirmed=true 的题)。")

    settings = load_rag_settings()
    retriever = HybridRetriever(load_rag_index(settings.index_dir), settings)
    relevant_counts = [len(q["relevant"]) for q in questions]

    sections = [
        "# RAG 检索消融评测",
        "",
        f"- 运行日期:{date.today().isoformat()}",
        f"- 题目数:{len(questions)}(文档级标注,confirmed 过滤:{args.only_confirmed})",
        f"- 每题标注相关文档数:均值 {sum(relevant_counts) / len(relevant_counts):.2f},最大 {max(relevant_counts)}",
        f"- top_k={args.top_k};reranker={settings.reranker_model},candidate_k={settings.rerank_candidate_k}",
        f"- 粗排池 coarse_pool_k={settings.coarse_pool_k};候选配额={settings.candidate_max_chunks_per_doc}",
        f"- 多样性配额 max_chunks_per_doc={settings.max_chunks_per_doc};rrf_k={settings.rrf_k}",
        f"- recall 天花板(top_k={args.top_k} 与生效配额共同决定):{_recall_ceiling(relevant_counts, args.top_k, _effective_quota(settings)):.1%}",
        "",
        "区间为 bootstrap 95% 置信区间;区间重叠说明差异尚不显著。",
        "",
        f"| 配置 | hit_rate@{args.top_k} | recall@{args.top_k} | MRR | 平均耗时/题 |",
        "| --- | --- | --- | --- | --- |",
    ]
    per_question_rows: dict[str, dict[str, str]] = {q["id"]: {} for q in questions}

    for name, options, overrides in ABLATIONS:
        print(f"[{name}] evaluating {len(questions)} questions ...")
        retriever.settings = replace(settings, **overrides)
        metrics, elapsed = _evaluate(retriever, questions, args.top_k, options, name)
        hit_lo, hit_hi = _bootstrap_ci(metrics["hit_values"])
        rec_lo, rec_hi = _bootstrap_ci(metrics["recall_values"])
        sections.append(
            f"| {name} | {metrics['hit_rate']:.1%} [{hit_lo:.1%}, {hit_hi:.1%}] "
            f"| {metrics['recall']:.1%} [{rec_lo:.1%}, {rec_hi:.1%}] "
            f"| {metrics['mrr']:.3f} | {elapsed / len(questions):.2f}s |"
        )
        for qid, hit in metrics["hits"].items():
            per_question_rows[qid][name] = hit
    retriever.settings = settings

    if not args.skip_coarse_diagnosis:
        sections += [
            "",
            "## 粗排候选集召回诊断",
            "",
            "精排只能重排候选。这里的 recall 是整条链路的硬上限:它偏低说明该投入召回环节",
            "(融合方式、候选数、查询改写),它已足够高则瓶颈在排序环节(精排、配额)。",
            "",
            "| 融合方式 | candidate_k | 候选集 recall |",
            "| --- | --- | --- |",
        ]
        for fusion, candidate_k, recall in _diagnose_coarse(retriever, questions, settings, COARSE_CANDIDATE_KS):
            sections.append(f"| {fusion} | {candidate_k} | {recall:.1%} |")
        retriever.settings = settings

    sections += ["", "## 每题命中明细(✓=至少命中一篇相关文档)", ""]
    config_names = [name for name, _, _ in ABLATIONS]
    sections.append("| 题号 | " + " | ".join(config_names) + " |")
    sections.append("| --- | " + " | ".join(["---"] * len(config_names)) + " |")
    for q in questions:
        row = per_question_rows[q["id"]]
        sections.append(f"| {q['id']} | " + " | ".join(row.get(name, "") for name in config_names) + " |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


def _load_golden(path: Path, only_confirmed: bool) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if only_confirmed and not item.get("confirmed"):
            continue
        relevant = {doc["doc_id"] for doc in item.get("relevant_docs", [])}
        if relevant:
            questions.append({"id": item["id"], "question": item["question"], "relevant": relevant})
    return questions


def _evaluate(retriever: HybridRetriever, questions: list[dict], top_k: int, options: dict, name: str):
    hit_values: list[float] = []
    recall_values: list[float] = []
    mrr_sum = 0.0
    hits: dict[str, str] = {}
    started = time.time()
    for index, item in enumerate(questions, 1):
        results = _search(retriever, item["question"], top_k, options)
        doc_ids = list(dict.fromkeys(result.chunk.doc_id for result in results))
        relevant = item["relevant"]
        matched = [doc_id for doc_id in doc_ids if doc_id in relevant]
        hits[item["id"]] = "✓" if matched else "✗"
        hit_values.append(1.0 if matched else 0.0)
        recall_values.append(len(set(matched)) / len(relevant))
        if matched:
            first_rank = next(i for i, doc_id in enumerate(doc_ids, 1) if doc_id in relevant)
            mrr_sum += 1.0 / first_rank
        if index % 5 == 0 or index == len(questions):
            print(f"  [{name}] {index}/{len(questions)}")
    elapsed = time.time() - started
    total = len(questions)
    return (
        {
            "hit_rate": sum(hit_values) / total,
            "recall": sum(recall_values) / total,
            "mrr": mrr_sum / total,
            "hits": hits,
            "hit_values": hit_values,
            "recall_values": recall_values,
        },
        elapsed,
    )


def _bootstrap_ci(values: list[float], iterations: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """固定种子保证多次运行结果可复现,否则区间本身会每次报告都变。"""
    rng = random.Random(20260807)
    size = len(values)
    means = sorted(sum(values[rng.randrange(size)] for _ in range(size)) / size for _ in range(iterations))
    return means[int(iterations * alpha / 2)], means[int(iterations * (1 - alpha / 2)) - 1]


def _recall_ceiling(relevant_counts: list[int], top_k: int, max_per_doc: int) -> float:
    """配额限制下 top_k 能容纳的文档数就是 recall 的上限——一道题标注 12 篇而只有
    8 个 slot 时,即使篇篇命中也拿不到满分。拿实测值对标 100% 会高估实际差距。"""
    max_docs = top_k // max_per_doc if max_per_doc > 0 else 1
    return sum(min(max_docs, count) / count for count in relevant_counts) / len(relevant_counts)


def _effective_quota(settings) -> int:
    """候选阶段配额一旦生效,最终配额就不再是约束——每篇文档进入精排的片段本就更少。"""
    candidate_quota = settings.candidate_max_chunks_per_doc
    if candidate_quota <= 0:
        return settings.max_chunks_per_doc
    if settings.max_chunks_per_doc <= 0:
        return candidate_quota
    return min(candidate_quota, settings.max_chunks_per_doc)


def _diagnose_coarse(
    retriever: HybridRetriever,
    questions: list[dict],
    base_settings,
    candidate_ks: tuple[int, ...],
) -> list[tuple[str, int, float]]:
    rows = []
    for fusion in ("weighted", "rrf"):
        retriever.settings = replace(base_settings, fusion_method=fusion)
        for candidate_k in candidate_ks:
            print(f"  [coarse:{fusion}] candidate_k={candidate_k}")
            recalls = []
            for item in questions:
                results = retriever._coarse_search(item["question"], candidate_k, use_vector=True)
                doc_ids = {result.chunk.doc_id for result in results}
                recalls.append(len(doc_ids & item["relevant"]) / len(item["relevant"]))
            rows.append((fusion, candidate_k, sum(recalls) / len(recalls)))
    return rows


def _search(retriever: HybridRetriever, question: str, top_k: int, options: dict):
    # "纯向量"配置:BM25 权重无法在 search() 关闭,直接用内部向量通道 + 相同的
    # 文档配额,保证消融只改变"召回来源"这一个变量。
    if options.get("bm25_off"):
        from finquery_agent.rag.models import SearchResult
        from finquery_agent.rag.retriever import _apply_doc_quota

        pairs = retriever._vector_search(question, max(top_k, retriever.settings.vector_top_k))
        results = [
            SearchResult(chunk=retriever._chunk_by_id[chunk_id], score=score, score_detail={"vector": score})
            for chunk_id, score in pairs
            if chunk_id in retriever._chunk_by_id
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        return _apply_doc_quota(results, retriever.settings.max_chunks_per_doc)[:top_k]
    return retriever.search(
        question,
        top_k=top_k,
        use_vector=options.get("use_vector", True),
        use_reranker=options.get("use_reranker", False),
    )


if __name__ == "__main__":
    main()
