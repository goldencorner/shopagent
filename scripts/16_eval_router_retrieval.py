"""对比「中文 query 的英文改写方式」对检索效果的影响。

四个对照组（同一批中文 query、同一套 ASIN 标签、同一种检索模式）：
    raw_zh        中文 query 直接检索（下界）
    planner_llm   DeepSeek 现场翻译（老链路）
    lora_router   本地 LoRA 小模型改写（新链路）
    english_gold  原始英文 query（上界，仅用于标定）

指标：Hit@3 / MRR / nDCG@3，外加每种方式的改写延迟和失败数。

用法:
    python scripts/16_eval_router_retrieval.py
    python scripts/16_eval_router_retrieval.py --store memory --limit 5
    python scripts/16_eval_router_retrieval.py --modes planner_llm,lora_router

前置：先跑 scripts/15_make_zh_retrieval.py 生成 eval/retrieval_eval_zh50.jsonl。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent import llm  # noqa: E402
from shopagent.local_router import LocalRouter, router_status  # noqa: E402
from shopagent.retrieval import Retriever  # noqa: E402
from shopagent.retrieval.backend import open_backend  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EVAL_SET = ROOT / "eval" / "retrieval_eval_zh50.jsonl"
METHODS = ["raw_zh", "planner_llm", "lora_router", "english_gold"]

TRANSLATE_PROMPT = """你是电商导购 Agent 的检索规划器。把用户的中文购物需求改写成英文检索关键词。

要求：
1. 保留品类、属性、使用场景、尺寸、预算等所有约束；
2. 数字和币种保持不变；
3. 不要补充用户没说的条件，不要解释。

只返回 JSON：{{"en_query": "..."}}

用户：{query}"""


def load_items(path: Path, limit: int | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到 {path}，先跑 scripts/15_make_zh_retrieval.py")
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        items.append(json.loads(line))
        if limit and len(items) >= limit:
            break
    return items


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, asin in enumerate(retrieved, start=1):
        if asin in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, asin in enumerate(retrieved[:k], start=1)
        if asin in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def metric_block(rows: list[dict], hit_k: int) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "hit_k": hit_k,
        "hit": round(sum(row["hit"] for row in rows) / n, 4),
        "mrr": round(sum(row["mrr"] for row in rows) / n, 4),
        "ndcg": round(sum(row["ndcg"] for row in rows) / n, 4),
        "failed_rewrite": sum(1 for row in rows if not row["en_query"]),
        "p50_rewrite_ms": round(
            statistics.median([row["rewrite_ms"] for row in rows]), 1
        )
        if rows
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=EVAL_SET)
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--retrieval-mode", default="hybrid_rerank")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--hit-k", type=int, default=3, dest="hit_k")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--local-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        if method not in METHODS:
            raise SystemExit(f"未知 method：{method}")

    items = load_items(args.eval, args.limit)
    print(f"评测集 {len(items)} 条，检索模式 {args.retrieval_mode}，命中口径 top-{args.hit_k}")

    local = None
    if "lora_router" in methods:
        if args.local_model:
            os.environ["SHOPAGENT_LOCAL_ROUTER_MODEL"] = args.local_model
        local = LocalRouter(model=args.local_model)
        status = router_status(force=True)
        print(f"本地 router：{local.model}  {status['detail']}  可用={status['available']}")
        if not status["available"]:
            raise SystemExit("本地 router 不可用，先启动 LM Studio 并加载模型。")

    details: dict[str, list[dict]] = {method: [] for method in methods}
    with open_backend(args.store) as backend:
        retriever = Retriever(backend, mode=args.retrieval_mode)
        for idx, item in enumerate(items, start=1):
            zh = item["query"]
            en_gold = item.get("query_en") or zh
            rewrites: dict[str, tuple[str, float]] = {"raw_zh": (zh, 0.0)}

            if "english_gold" in methods:
                rewrites["english_gold"] = (en_gold, 0.0)

            if "planner_llm" in methods:
                started = time.perf_counter()
                try:
                    with llm.use_model("deepseek"):
                        obj = llm.chat_json(
                            [{"role": "user", "content": TRANSLATE_PROMPT.format(query=zh)}],
                            temperature=0,
                        )
                    en = str(obj.get("en_query") or "").strip()
                except Exception as exc:
                    print(f"  [{idx}] planner 翻译失败：{type(exc).__name__}")
                    en = ""
                rewrites["planner_llm"] = (en, (time.perf_counter() - started) * 1000)

            if "lora_router" in methods:
                started = time.perf_counter()
                try:
                    result = local.route(zh)
                    en = result["en_query"]
                except Exception as exc:
                    print(f"  [{idx}] lora 改写失败：{type(exc).__name__}")
                    en = ""
                rewrites["lora_router"] = (en, (time.perf_counter() - started) * 1000)

            relevant = set(item["relevant_asins"])
            for method in methods:
                en, rewrite_ms = rewrites[method]
                retrieved = [hit["asin"] for hit in retriever.search(en, top_k=args.k)] if en else []
                window = retrieved[: args.hit_k]
                rank = next(
                    (i + 1 for i, asin in enumerate(retrieved) if asin in relevant), None
                )
                details[method].append(
                    {
                        "id": item.get("id"),
                        "query_zh": zh,
                        "en_query": en,
                        "rewrite_ms": round(rewrite_ms, 1),
                        "rank": rank,
                        "hit": 1 if rank and rank <= args.hit_k else 0,
                        "mrr": mrr(window, relevant),
                        "ndcg": ndcg_at_k(window, relevant, args.hit_k),
                    }
                )
            print(f"  [{idx}/{len(items)}] {zh[:40]}")

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "eval": str(args.eval),
        "store": args.store,
        "retrieval_mode": args.retrieval_mode,
        "k": args.k,
        "hit_k": args.hit_k,
        "metrics": {},
        "details": details,
    }

    print(f"\n{'method':<14}{'Hit@'+str(args.hit_k):>9}{'MRR':>9}{'nDCG':>9}"
          f"{'fail':>7}{'p50-ms':>9}")
    for method in methods:
        block = metric_block(details[method], args.hit_k)
        report["metrics"][method] = block
        print(
            f"{method:<14}{block['hit']:>9.3f}{block['mrr']:>9.3f}{block['ndcg']:>9.3f}"
            f"{block['failed_rewrite']:>7}{block['p50_rewrite_ms']:>9.1f}"
        )

    out = (
        Path(args.out)
        if args.out
        else ROOT / "eval" / f"router_retrieval_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
