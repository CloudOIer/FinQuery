"""RAG 检索质量评测:四种检索配置的消融对比。

评测口径是文档级:golden set 标注的是"哪些研报文档与问题相关"(文档级标注
人工可核对,chunk 级标注成本过高且边界模糊),预测取 top_k chunks 去重后的
doc_id 集合参与计算。

指标:
- hit_rate@k:至少命中一篇相关文档的问题占比(问答场景的底线指标——
  有一篇对的文档,答案就有依据);
- recall@k:命中的相关文档数 / 标注相关文档数,再对问题取平均;
- MRR:第一篇相关文档出现位置的倒数,衡量"最相关的排多前"。

消融配置:bm25 / vector / hybrid / hybrid+rerank,同一 golden set 上跑四遍,
输出 markdown 对比表。cross-encoder 在 CPU 上逐题打分较慢,脚本带进度输出。

用法:
    python scripts/evaluate_rag_retrieval.py                # 全量
    python scripts/evaluate_rag_retrieval.py --limit 3      # 试跑
    python scripts/evaluate_rag_retrieval.py --only-confirmed  # 只用已人工确认的题
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from finquery_agent.config import load_rag_settings
from finquery_agent.rag.index import load_rag_index
from finquery_agent.rag.retriever import HybridRetriever

DEFAULT_GOLDEN = Path("data/evaluation/rag_golden_set.jsonl")
DEFAULT_OUTPUT = Path("data/evaluation/rag_retrieval_ablation.md")

ABLATIONS: tuple[tuple[str, dict], ...] = (
    ("bm25", {"use_vector": False, "use_reranker": False}),
    ("vector", {"use_vector": True, "use_reranker": False, "bm25_off": True}),
    ("hybrid", {"use_vector": True, "use_reranker": False}),
    ("hybrid+rerank", {"use_vector": True, "use_reranker": True}),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study of RAG retrieval configurations.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-confirmed", action="store_true", help="Only use questions with confirmed=true.")
    args = parser.parse_args()

    questions = _load_golden(args.golden, args.only_confirmed)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("golden set 为空(或没有 confirmed=true 的题)。")

    settings = load_rag_settings()
    retriever = HybridRetriever(load_rag_index(settings.index_dir), settings)

    sections = [
        "# RAG 检索消融评测",
        "",
        f"- 运行日期:{date.today().isoformat()}",
        f"- 题目数:{len(questions)}(文档级标注,confirmed 过滤:{args.only_confirmed})",
        f"- top_k={args.top_k};reranker={settings.reranker_model},candidate_k={settings.rerank_candidate_k}",
        f"- 多样性配额 max_chunks_per_doc={settings.max_chunks_per_doc}",
        "",
        f"| 配置 | hit_rate@{args.top_k} | recall@{args.top_k} | MRR | 平均耗时/题 |",
        "| --- | --- | --- | --- | --- |",
    ]
    per_question_rows: dict[str, dict[str, str]] = {q["id"]: {} for q in questions}

    for name, options in ABLATIONS:
        print(f"[{name}] evaluating {len(questions)} questions ...")
        metrics, elapsed = _evaluate(retriever, questions, args.top_k, options, name)
        sections.append(
            f"| {name} | {metrics['hit_rate']:.1%} | {metrics['recall']:.1%} | {metrics['mrr']:.3f} | {elapsed / len(questions):.2f}s |"
        )
        for qid, hit in metrics["hits"].items():
            per_question_rows[qid][name] = hit

    sections += ["", "## 每题命中明细(✓=至少命中一篇相关文档)", ""]
    config_names = [name for name, _ in ABLATIONS]
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
    hit_count = 0
    recall_sum = 0.0
    mrr_sum = 0.0
    hits: dict[str, str] = {}
    started = time.time()
    for index, item in enumerate(questions, 1):
        results = _search(retriever, item["question"], top_k, options)
        doc_ids = list(dict.fromkeys(result.chunk.doc_id for result in results))
        relevant = item["relevant"]
        matched = [doc_id for doc_id in doc_ids if doc_id in relevant]
        hits[item["id"]] = "✓" if matched else "✗"
        if matched:
            hit_count += 1
            recall_sum += len(set(matched)) / len(relevant)
            first_rank = next(i for i, doc_id in enumerate(doc_ids, 1) if doc_id in relevant)
            mrr_sum += 1.0 / first_rank
        if index % 5 == 0 or index == len(questions):
            print(f"  [{name}] {index}/{len(questions)}")
    elapsed = time.time() - started
    total = len(questions)
    return (
        {"hit_rate": hit_count / total, "recall": recall_sum / total, "mrr": mrr_sum / total, "hits": hits},
        elapsed,
    )


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
