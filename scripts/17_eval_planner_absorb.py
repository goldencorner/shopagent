"""衡量 LoRA router 的改写差距有多少被 planner 兜住（最终商品命中率）。

16 号脚本里的 lora_router 是**绕过 planner**、直接拿 en_query 去检索，测的是改写本身。
线上不是这样：en_query 只是 planner prompt 的第 7 条提示，planner 有权改写它。
这个脚本按 nodes.py::planner 的真实 prompt 跑一遍，比较：

    gold         原始英文 query 直接走 search_products（上界）
    lora_direct  en_query 直接走 search_products（=16 号脚本的 lora_router）
    planner_llm  老链路：planner 自己翻，没有提示
    planner_lora 新链路：planner 收到 LoRA 的 en_query 作为提示

指标：search_products 返回 top-5 里的 Hit@3 / MRR / nDCG@3，外加 planner 延迟、
planner 是否真的采纳了提示。

用法:
    python scripts/17_eval_planner_absorb.py --local-model router-lora
    python scripts/17_eval_planner_absorb.py --store memory --limit 5

前置：eval/retrieval_eval_zh50.jsonl（scripts/15_make_zh_retrieval.py 生成）。
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

from shopagent import llm, memory  # noqa: E402
from shopagent.agent.nodes import MAX_STEPS, PLANNER_PROMPT  # noqa: E402
from shopagent.local_router import LocalRouter, router_status  # noqa: E402
from shopagent.tools import TOOL_SPECS  # noqa: E402
from shopagent.tools import shop  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EVAL_SET = ROOT / "eval" / "retrieval_eval_zh50.jsonl"
METHODS = ["gold", "lora_direct", "planner_llm", "planner_lora", "planner_lora_soft"]


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


def hint_block(rewritten: str, style: str = "hard") -> str:
    """hard 与 shopagent/agent/nodes.py::planner 完全一致；soft 是"可修正的初稿"。"""
    if not rewritten:
        return ""
    if style == "soft":
        return (
            f"\n7. 小模型给了一个英文检索初稿：{rewritten}。"
            "它可能漏掉或翻错约束，请对照用户原话核对、修正后，再决定 search_products 的 query。"
        )
    return (
        f"\n7. 小模型已把本轮需求改写成英文检索词：{rewritten}。"
        "调用 search_products 时优先使用它；用户原话里更精确的条件仍然以用户为准。"
    )


def planner_messages(zh: str, rewritten: str, style: str = "hard") -> list[dict]:
    return [
        {
            "role": "system",
            "content": PLANNER_PROMPT.format(
                max_steps=MAX_STEPS,
                profile=memory.profile_summary({}),
                rewritten_hint=hint_block(rewritten, style),
            ),
        },
        {"role": "user", "content": zh},
    ]


def call_planner(zh: str, rewritten: str, style: str = "hard") -> dict:
    """跑一次 planner，返回它选定的 search_products 参数。"""
    started = time.perf_counter()
    message = llm.chat_tools(
        planner_messages(zh, rewritten, style), TOOL_SPECS, temperature=0
    )
    ms = (time.perf_counter() - started) * 1000

    calls: list[dict] = []
    for call in message.tool_calls or []:
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": call.function.name, "args": args})

    chosen = next((c for c in calls if c["name"] == "search_products"), None)
    return {
        "ms": round(ms, 1),
        "tools": [c["name"] for c in calls],
        "query": (chosen or {}).get("args", {}).get("query", ""),
        "max_price": (chosen or {}).get("args", {}).get("max_price"),
        "no_search": chosen is None,
    }


def jaccard(a: str, b: str) -> float:
    ta = {w.strip(".,!?;:\"'").lower() for w in str(a).split()} - {""}
    tb = {w.strip(".,!?;:\"'").lower() for w in str(b).split()} - {""}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def search_asins(query: str, max_price, top_k: int) -> list[str]:
    if not query:
        return []
    hits = shop.search_products(query, max_price=max_price, top_k=top_k)
    return [hit["asin"] for hit in hits]


def mrr(window: list[str], relevant: set[str]) -> float:
    for rank, asin in enumerate(window, start=1):
        if asin in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(window: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, asin in enumerate(window[:k], start=1)
        if asin in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def metric_block(rows: list[dict], hit_k: int) -> dict:
    n = max(len(rows), 1)
    lat = [row["query_ms"] for row in rows if row.get("query_ms")]
    return {
        "n": len(rows),
        "hit_k": hit_k,
        "hit": round(sum(row["hit"] for row in rows) / n, 4),
        "mrr": round(sum(row["mrr"] for row in rows) / n, 4),
        "ndcg": round(sum(row["ndcg"] for row in rows) / n, 4),
        "no_search": sum(1 for row in rows if row.get("no_search")),
        "p50_query_ms": round(statistics.median(lat), 1) if lat else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=EVAL_SET)
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--retrieval-mode", default="hybrid_rerank")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
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
    if any(m in methods for m in ("lora_direct", "planner_lora", "planner_lora_soft")):
        if args.local_model:
            os.environ["SHOPAGENT_LOCAL_ROUTER_MODEL"] = args.local_model
        local = LocalRouter(model=args.local_model)
        status = router_status(force=True)
        print(f"本地 router：{local.model}  可用={status['available']}  {status['detail']}")
        if not status["available"]:
            raise SystemExit("本地 router 不可用，先启动 LM Studio 并加载模型。")

    shop.configure(args.store, args.retrieval_mode)
    details: dict[str, list[dict]] = {method: [] for method in methods}

    try:
        for idx, item in enumerate(items, start=1):
            zh = item["query"]
            en_gold = item.get("query_en") or zh
            relevant = set(item["relevant_asins"])

            lora_en = ""
            intent = ""
            lora_ms = 0.0
            if local is not None:
                started = time.perf_counter()
                try:
                    result = local.route(zh)
                    lora_en = (result.get("en_query") or "").strip()
                    intent = result.get("intent") or ""
                except Exception as exc:
                    print(f"  [{idx}] lora 改写失败：{type(exc).__name__}")
                lora_ms = round((time.perf_counter() - started) * 1000, 1)

            results: dict[str, dict] = {}
            if "gold" in methods:
                results["gold"] = {"query": en_gold, "max_price": None, "ms": 0.0,
                                   "no_search": False, "tools": []}
            if "lora_direct" in methods:
                results["lora_direct"] = {"query": lora_en, "max_price": None, "ms": 0.0,
                                          "no_search": not lora_en, "tools": []}
            if "planner_llm" in methods:
                with llm.use_model("deepseek"):
                    results["planner_llm"] = call_planner(zh, "")
            if "planner_lora" in methods:
                with llm.use_model("deepseek"):
                    results["planner_lora"] = call_planner(zh, lora_en, "hard")
            if "planner_lora_soft" in methods:
                with llm.use_model("deepseek"):
                    results["planner_lora_soft"] = call_planner(zh, lora_en, "soft")

            for method in methods:
                res = results[method]
                asins = search_asins(res["query"], res.get("max_price"), args.top_k)
                window = asins[: args.hit_k]
                rank = next(
                    (i + 1 for i, asin in enumerate(asins) if asin in relevant), None
                )
                row = {
                    "id": item.get("id"),
                    "query_zh": zh,
                    "query_en_gold": en_gold,
                    "planner_query": res["query"],
                    "max_price": res.get("max_price"),
                    "asins": asins,
                    "rank": rank,
                    "hit": 1 if rank and rank <= args.hit_k else 0,
                    "mrr": mrr(window, relevant),
                    "ndcg": ndcg_at_k(window, relevant, args.hit_k),
                    "query_ms": res.get("ms", 0.0),
                    "no_search": bool(res.get("no_search")),
                    "tools": res.get("tools", []),
                }
                details[method].append(row)

            # 有 lora 参与的几种方式，记录提示与采纳情况
            for method in ("lora_direct", "planner_lora", "planner_lora_soft"):
                if method in details:
                    details[method][-1]["lora_en"] = lora_en
                    details[method][-1]["lora_intent"] = intent
                    details[method][-1]["lora_ms"] = lora_ms
                    if method != "lora_direct":
                        details[method][-1]["jaccard_with_hint"] = round(
                            jaccard(lora_en, results[method]["query"]), 3
                        )

            print(f"  [{idx}/{len(items)}] {zh[:36]}  gold={results['gold']['query'][:24] if 'gold' in results else ''}")
    finally:
        shop.close()

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "eval": str(args.eval),
        "store": args.store,
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "hit_k": args.hit_k,
        "metrics": {},
        "details": details,
    }

    print(f"\n{'method':<14}{'Hit@'+str(args.hit_k):>9}{'MRR':>9}{'nDCG':>9}"
          f"{'no-search':>11}{'p50-ms':>9}")
    for method in methods:
        block = metric_block(details[method], args.hit_k)
        report["metrics"][method] = block
        print(
            f"{method:<14}{block['hit']:>9.3f}{block['mrr']:>9.3f}{block['ndcg']:>9.3f}"
            f"{block['no_search']:>11}{block['p50_query_ms']:>9.1f}"
        )

    variants = [m for m in ("planner_lora", "planner_lora_soft") if m in methods]
    if "planner_llm" in methods and variants:
        report["hit_diffs"] = {}
        report["summary"] = {}
        for variant in variants:
            diffs = []
            for a, b in zip(details["planner_llm"], details[variant]):
                if a["hit"] != b["hit"]:
                    diffs.append(
                        {
                            "query": a["query_zh"],
                            "planner_llm": a["planner_query"],
                            variant: b["planner_query"],
                            "lora_en": b.get("lora_en", ""),
                            "delta": b["hit"] - a["hit"],
                        }
                    )
            report["hit_diffs"][variant] = diffs
            gained = sum(d["delta"] for d in diffs if d["delta"] > 0)
            lost = -sum(d["delta"] for d in diffs if d["delta"] < 0)
            adopted = [r.get("jaccard_with_hint", 0) for r in details[variant]]
            report["summary"][variant] = {
                "gained": gained,
                "lost": lost,
                "adopt_mean_jaccard": round(sum(adopted) / max(len(adopted), 1), 3),
                "adopt_ge_0_6": sum(1 for j in adopted if j >= 0.6),
            }
            print(
                f"\nplanner_llm → {variant}：命中 +{gained} / -{lost}，"
                f"提示采纳（Jaccard≥0.6）{report['summary'][variant]['adopt_ge_0_6']}/{len(adopted)}"
            )

    out = (
        Path(args.out)
        if args.out
        else ROOT / "eval" / f"planner_absorb_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
