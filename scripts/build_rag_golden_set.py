"""RAG golden set 预填:候选召回 + LLM 相关性判定,产出待人工确认的标注。

标注口径是文档级——"哪几篇研报能为回答该问题提供实质证据"。文档级人工可逐篇
核对,chunk 级成本过高且边界模糊。

**避免自证循环**是这个脚本最关键的设计约束。若候选池取自被评测的检索配置,
标注就会偏向当前实现,消融结论随之失效(所有配置都会"恰好"命中自己召回的文档)。
因此候选池由三个与排序无关的来源取并集:

- BM25 单路大 k 召回(不参与融合);
- 向量单路大 k 召回(不参与融合);
- 元数据直配:问题中提及的公司,其名下全部研报无条件进入候选。

相关性由 LLM 逐篇阅读内容判定,不使用任何检索分数或排名。

残留偏差需在报告中说明:两路都召不回、且未被元数据命中的研报无法进入候选,
因此 recall 绝对值仍可能被高估;但各配置面对同一标注集,相对比较不受影响。

标注标准必须全集统一。早期标注的候选池取自当时的检索 top-k,范围远窄于现在的
双路并集,漏标明显;若与新题混用,新题的 recall 会因分母更大而系统性偏低。
因此扩充时应用 --relabel-all 重标全集,而非只补新题。

用法:
    python scripts/build_rag_golden_set.py --dry-run      # 只列出将新增的题目
    python scripts/build_rag_golden_set.py --relabel-all --workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from threading import Lock
from typing import Any

from finquery_agent.config import load_llm_settings, load_rag_settings
from finquery_agent.llm import LLMClient
from finquery_agent.rag.index import load_rag_index
from finquery_agent.rag.retriever import HybridRetriever
from finquery_agent.schema import load_default_registry

DEFAULT_QUESTIONS = Path("第一批数据/附件6：问题汇总.CSV")
DEFAULT_GOLDEN = Path("data/evaluation/rag_golden_set.jsonl")

# 持续跑大 prompt 会撞上 DeepSeek 的 TPM 限流,而限流窗口按分钟重置;
# LLMClient 内置的 2s/4s 退避跨不过去,批次级再加一层更长的重试。
_BATCH_RETRY_WAITS = (15.0, 45.0, 90.0)

# 研报每页都带页脚声明,整段只有这些内容的 chunk 无法体现研报主题。
_BOILERPLATE_MARKERS = (
    "免责声明",
    "分析师声明",
    "证券投资咨询执业资格",
    "请务必阅读",
    "评级说明",
    "投资评级标准",
)

# 这些章节存在于每篇研报且内容雷同,无法区分研报主题,不适合作为判断依据。
_LOW_VALUE_SECTION_MARKERS = (
    "风险",
    "免责",
    "声明",
    "简介",
    "团队",
    "评级",
    "附录",
    "图表",
)

# 数据校验类问题只需查库比对,不需要研报观点;其余类型都可能引用研报。
EXCLUDED_TYPES = {"数据校验"}
# 多意图题里既有纯计算也有归因,靠这些词判断是否需要定性证据。
EVIDENCE_KEYWORDS = ("分析", "原因", "研报", "解释", "评估", "结合", "驱动", "影响", "判断", "前景")

_JUDGE_SYSTEM_PROMPT = """你在为研报检索系统构建评测标注集。给定一个问题和若干篇研报,判断每篇研报能否为回答该问题提供**实质证据**。

判为相关的标准(满足其一):
- 研报直接讨论问题所问的公司、行业、业务板块或财务现象;
- 研报提供了回答该问题所需的观点、驱动因素、行业判断或风险提示。

判为不相关的情形:
- 仅在标题或正文顺带提及关键词,没有实质论述;
- 讨论的是其他公司或其他业务,与问题无关;
- 只有财务数字罗列,没有可用于该问题的分析结论。

宁严勿宽:不确定时判为不相关。只返回 JSON:{"relevant": [编号, ...]}"""


class JudgeError(RuntimeError):
    """判定失败必须中断而非降级:标注集缺条目不会报错,但会让后续所有指标失真。"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-fill the RAG golden set with LLM-judged relevance labels.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--pool-per-channel", type=int, default=120, help="每路召回的 chunk 数,越大候选覆盖越全")
    parser.add_argument("--batch-size", type=int, default=8, help="每次 LLM 调用判定的研报数")
    parser.add_argument("--workers", type=int, default=2, help="并发数;DeepSeek 大 prompt 并发过高会触发 TPM 限流")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--relabel-all",
        action="store_true",
        help="连已有题目一并重新判定。早期标注的候选池更窄,与新题混用会使两部分题目的 recall 不可比",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="以已人工确认的标注为基准度量判定器本身的召回率,先验证再使用",
    )
    parser.add_argument("--export-review", type=Path, help="导出人工确认用的勾选清单")
    parser.add_argument("--import-review", type=Path, help="把勾选完的清单回收为 golden set")
    parser.add_argument("--review-top-n", type=int, default=12, help="清单中每题列出的候选数")
    args = parser.parse_args()

    if args.import_review:
        _import_review(args.import_review, args.golden)
        return

    existing = _load_existing(args.golden)
    questions = _load_questions(args.questions)
    checkpoint_path = args.golden.with_suffix(".checkpoint.jsonl")
    done = _load_existing(checkpoint_path)
    pending = questions if args.relabel_all else [item for item in questions if item["id"] not in existing]
    pending = [item for item in pending if item["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"已有标注 {len(existing)} 题;检查点已完成 {len(done)} 题;待预填 {len(pending)} 题")
    for item in pending:
        print(f"  {item['id']} [{item['type']}] {item['question'][:60]}")
    if args.dry_run:
        return

    llm_settings = load_llm_settings()
    client = LLMClient(llm_settings)
    if not client.is_available() and not args.export_review:
        raise SystemExit("LLM 不可用:请检查 config/llm.json。相关性判定必须由 LLM 完成。")

    rag_settings = load_rag_settings()
    retriever = HybridRetriever(load_rag_index(rag_settings.index_dir), rag_settings)
    registry = load_default_registry()
    checkpoint_lock = Lock()

    if args.export_review:
        _export_review(args.export_review, questions, retriever, registry, existing, args)
        return

    if args.validate:
        _validate_judge(client, retriever, registry, existing, args)
        return

    def build(item: dict[str, Any]) -> str | None:
        candidates = _candidate_docs(retriever, registry, item["question"], args.pool_per_channel)
        try:
            relevant = _judge(client, item["question"], candidates, args.batch_size)
        except JudgeError as exc:
            print(f"  [{item['id']}] 判定失败:{exc}")
            return item["id"]
        record = {
            "id": item["id"],
            "question": item["question"],
            "relevant_docs": relevant,
            "confirmed": False,
        }
        # 逐题落盘:限流导致的中途失败不应让已完成的判定作废。
        with checkpoint_lock:
            with checkpoint_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  [{item['id']}] 候选 {len(candidates)} 篇 → 判定相关 {len(relevant)} 篇")
        return None

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        failed = [qid for qid in pool.map(build, pending) if qid is not None]

    if failed:
        print(f"\n仍有 {len(failed)} 题未完成:{failed}")
        print(f"已完成的判定保存在 {checkpoint_path},重跑同一命令会自动续跑。")
        print("在全部题目完成前不覆盖 golden set,避免用残缺标注替换现有标注。")
        return

    _write_golden(args.golden, checkpoint_path, existing, args.relabel_all)


def _validate_judge(
    client: LLMClient,
    retriever: HybridRetriever,
    registry: Any,
    reference: dict[str, dict[str, Any]],
    args: Any,
) -> None:
    """判定器先验证再使用:它产出的是评测基准本身,不可靠的基准会让后续所有消融失真。

    以人工确认的标注为基准:命中率过低说明判定器漏判严重,不能用它替代人工。
    """
    confirmed = {qid: item for qid, item in reference.items() if item.get("confirmed")}
    if not confirmed:
        raise SystemExit("没有已人工确认的标注可用于验证。")

    covered = total = extra = 0
    missing_pool = 0
    print(f"以 {len(confirmed)} 道人工确认题验证判定器\n")
    for qid, item in sorted(confirmed.items()):
        candidates = _candidate_docs(retriever, registry, item["question"], args.pool_per_channel)
        candidate_ids = {doc["doc_id"] for doc in candidates}
        human = {doc["doc_id"] for doc in item["relevant_docs"]}
        missing_pool += len(human - candidate_ids)
        try:
            judged = {doc["doc_id"] for doc in _judge(client, item["question"], candidates, args.batch_size)}
        except JudgeError as exc:
            print(f"  {qid} 判定失败:{exc}")
            continue
        hit = len(human & judged)
        covered += hit
        total += len(human)
        extra += len(judged - human)
        print(f"  {qid} 人工 {len(human):3d} 判定 {len(judged):3d} 命中 {hit:3d} 额外 {len(judged - human):3d}")

    if total:
        print(f"\n判定器对人工标注的覆盖率:{covered}/{total} = {covered / total:.1%}")
    print(f"人工标注中未进入候选池的:{missing_pool} 条(候选召回问题,与判定无关)")
    print("覆盖率偏低时不应用判定结果替换人工标注,应改进文档表示或判定提示词。")


def _export_review(
    path: Path,
    questions: list[dict[str, str]],
    retriever: HybridRetriever,
    registry: Any,
    existing: dict[str, dict[str, Any]],
    args: Any,
) -> None:
    """导出勾选清单。已确认的标注预先勾上并置顶,避免重复审阅已做过的判断。"""
    lines = [
        "# RAG golden set 人工确认清单",
        "",
        "标注口径:该研报能否为回答这个问题提供**实质证据**(观点、驱动因素、行业判断、风险提示)。",
        "仅顺带提及关键词、或只有数字罗列而无分析结论的,判为不相关。",
        "",
        "宽泛问题(如行业景气度)请保持内部一致的尺度:要么只收行业级研报,要么一并收录",
        "能佐证行业趋势的个股研报——尺度不一致会直接体现为 recall 指标的噪声。",
        "",
        "勾选方式:相关的把 `- [ ]` 改成 `- [x]`。已勾选的是你此前确认过的标注。",
        "完成后运行:",
        "",
        "```bash",
        f"python scripts/build_rag_golden_set.py --import-review {path}",
        "```",
        "",
    ]

    for item in questions:
        confirmed = {doc["doc_id"]: doc for doc in existing.get(item["id"], {}).get("relevant_docs", [])}
        candidates = _candidate_docs(retriever, registry, item["question"], args.pool_per_channel)
        by_id = {doc["doc_id"]: doc for doc in candidates}
        shortlist = [doc for doc in candidates if doc["doc_id"] in confirmed]
        shortlist += [doc for doc in candidates if doc["doc_id"] not in confirmed][: args.review_top_n]
        # 既有标注可能落在候选池外,仍需列出,否则回收时会被静默丢弃。
        for doc_id, doc in confirmed.items():
            if doc_id not in by_id:
                shortlist.append({**doc, "org": "", "date": "", "snippet": "(不在当前候选池中)"})

        lines += [f"## {item['id']} [{item['type']}]", "", f"**问题**:{item['question']}", ""]
        for doc in shortlist:
            mark = "x" if doc["doc_id"] in confirmed else " "
            meta = " · ".join(part for part in (doc.get("org"), doc.get("date")) if part)
            snippet = re.sub(r"\s+", " ", doc["snippet"])[:180]
            lines.append(f"- [{mark}] `{doc['doc_id']}` {doc['title']}")
            lines.append(f"    <sub>{meta} — {snippet}</sub>")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(1 for line in lines if line.startswith("- ["))
    print(f"已导出 {path}:{len(questions)} 题,共 {total} 条待勾选(其中已预勾选为你此前确认的标注)")


def _import_review(path: Path, golden_path: Path) -> None:
    question_id = None
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^##\s+(\S+)", line)
        if header:
            question_id = header.group(1)
            records[question_id] = {"id": question_id, "question": "", "relevant_docs": [], "confirmed": True}
            continue
        if question_id is None:
            continue
        text = re.match(r"^\*\*问题\*\*[:：](.*)$", line)
        if text:
            records[question_id]["question"] = text.group(1).strip()
            continue
        entry = re.match(r"^- \[([ xX])\]\s+`(doc-[0-9a-f]+)`\s+(.*)$", line)
        if entry and entry.group(1).lower() == "x":
            records[question_id]["relevant_docs"].append({"doc_id": entry.group(2), "title": entry.group(3).strip()})

    kept = {qid: item for qid, item in records.items() if item["relevant_docs"]}
    empty = sorted(set(records) - set(kept))
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", encoding="utf-8") as file:
        for qid in sorted(kept):
            file.write(json.dumps(kept[qid], ensure_ascii=False) + "\n")

    counts = [len(item["relevant_docs"]) for item in kept.values()]
    print(f"写入 {golden_path}:{len(kept)} 题,相关文档合计 {sum(counts)} 条,每题均值 {sum(counts) / len(counts):.2f}")
    if empty:
        print(f"未勾选任何文档因而排除的题目:{empty}")


def _write_golden(
    golden_path: Path,
    checkpoint_path: Path,
    existing: dict[str, dict[str, Any]],
    relabel_all: bool,
) -> None:
    prefilled = _load_existing(checkpoint_path)
    kept = existing if not relabel_all else {}
    merged = {**kept, **{qid: item for qid, item in prefilled.items() if item["relevant_docs"]}}
    dropped = [qid for qid, item in prefilled.items() if not item["relevant_docs"]]

    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", encoding="utf-8") as file:
        for qid in sorted(merged):
            file.write(json.dumps(merged[qid], ensure_ascii=False) + "\n")
    checkpoint_path.unlink(missing_ok=True)

    print(f"\n写入 {golden_path}:共 {len(merged)} 题")
    if dropped:
        print(f"判定为无需研报证据而丢弃:{dropped}")
    print("下一步:人工核对条目,确认无误后把 confirmed 改为 true,再重跑消融。")


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {item["id"]: item for item in items}


def _load_questions(path: Path) -> list[dict[str, str]]:
    questions = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            question_type = str(row.get("问题类型", "")).strip()
            if question_type in EXCLUDED_TYPES:
                continue
            text = _extract_question(row.get("问题", ""))
            if not text:
                continue
            if "多意图" in question_type and not any(word in text for word in EVIDENCE_KEYWORDS):
                continue
            questions.append({"id": str(row.get("编号", "")).strip(), "type": question_type, "question": text})
    return questions


def _extract_question(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return str(raw).strip()
    if isinstance(payload, list) and payload:
        return str(payload[0].get("Q", "")).strip()
    return ""


def _candidate_docs(
    retriever: HybridRetriever,
    registry: Any,
    question: str,
    pool_per_channel: int,
    chunks_per_doc: int = 3,
    snippet_chars: int = 1200,
) -> list[dict[str, str]]:
    """候选池刻意由单路召回与元数据直配取并集,不经过任何融合或精排。

    每篇研报用多个片段拼接作为表示:单个 chunk 很可能正好落在免责声明、
    参考文献列表或图注上,据此判定会把实际相关的研报误判为不相关。
    """
    ranked_bm25 = [chunk_id for chunk_id, _ in retriever._bm25_search(question, pool_per_channel)]
    ranked_vector: list[str] = []
    if retriever.index.vector_index is not None and retriever.index.vector_chunk_ids:
        ranked_vector = [chunk_id for chunk_id, _ in retriever._vector_search(question, pool_per_channel)]
    ranked = _interleave(ranked_bm25, ranked_vector)

    per_doc: dict[str, list[Any]] = defaultdict(list)
    seen: set[str] = set()
    for chunk_id in ranked:
        chunk = retriever._chunk_by_id.get(chunk_id)
        if chunk is None or chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        per_doc[chunk.doc_id].append(chunk)

    for code in _mentioned_company_codes(registry, question):
        for chunk in retriever.index.chunks:
            if chunk.stock_code == code and chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                per_doc[chunk.doc_id].append(chunk)

    docs = []
    for doc_id, bucket in per_doc.items():
        # 优先展示实质分析段落:页脚声明、风险提示模板、纯数据表格都无法体现研报主题。
        ordered = sorted(bucket, key=_chunk_rank_key)[:chunks_per_doc]
        if not ordered:
            continue
        parts = []
        for chunk in ordered:
            prefix = f"[{chunk.section_title}] " if chunk.section_title and chunk.section_title != chunk.title else ""
            parts.append(prefix + chunk.text)
        head = ordered[0]
        docs.append(
            {
                "doc_id": doc_id,
                "title": head.title,
                "org": head.org_name,
                "date": head.publish_date,
                "report_type": head.report_type,
                "snippet": "\n---\n".join(parts)[:snippet_chars],
            }
        )
    return docs


def _interleave(*channels: list[str]) -> list[str]:
    """两路交替取:候选顺序若偏向某一路,人工审阅时看到的就是那一路的结果,
    标注会反过来偏祖该通道——而召回方式正是被评测对象。"""
    merged: list[str] = []
    seen: set[str] = set()
    for group in zip_longest(*channels):
        for chunk_id in group:
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                merged.append(chunk_id)
    return merged


def _is_boilerplate(text: str) -> bool:
    return any(marker in text for marker in _BOILERPLATE_MARKERS)


def _chunk_rank_key(chunk: Any) -> tuple[bool, bool, bool]:
    section = chunk.section_title or ""
    table_density = chunk.text.count("|") / max(len(chunk.text), 1)
    return (
        _is_boilerplate(chunk.text),
        any(marker in section for marker in _LOW_VALUE_SECTION_MARKERS),
        table_density > 0.05,
    )


def _mentioned_company_codes(registry: Any, question: str) -> set[str]:
    codes = set()
    for token in re.findall(r"[\u4e00-\u9fa5A-Za-z]{2,10}", question):
        code = registry.resolve_company_code(token)
        if code:
            codes.add(code)
    return codes


def _judge(client: LLMClient, question: str, candidates: list[dict[str, str]], batch_size: int) -> list[dict[str, str]]:
    """任一批次判定失败即抛错。

    LLMClient 失败返回 None,若把它当成"本批无相关文档"处理,失败批次会被静默
    丢弃且与真实的否定判定无法区分——标注集会缺失整批文档,而指标看不出异常。
    """
    relevant: list[dict[str, str]] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        listing = [
            {"编号": index, "标题": item["title"], "内容片段": item["snippet"]}
            for index, item in enumerate(batch)
        ]
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"问题": question, "候选研报": listing}, ensure_ascii=False)},
        ]
        indices = None
        for attempt, wait in enumerate((0.0, *_BATCH_RETRY_WAITS)):
            if wait:
                time.sleep(wait)
            content = client.chat(messages, temperature=0.0, response_format={"type": "json_object"})
            if content is not None:
                indices = _parse_indices(content, len(batch))
                if indices is not None:
                    break
        if indices is None:
            raise JudgeError(f"第 {start // batch_size + 1} 批重试 {len(_BATCH_RETRY_WAITS)} 次仍失败")
        for index in indices:
            relevant.append({"doc_id": batch[index]["doc_id"], "title": batch[index]["title"]})
    return relevant


def _parse_indices(content: str, size: int) -> list[int] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    values = payload.get("relevant") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        return None
    return [int(value) for value in values if isinstance(value, (int, float)) and 0 <= int(value) < size]


if __name__ == "__main__":
    main()
