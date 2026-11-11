"""消融实验：RRF 融合参数 / router 方案 / generate 约束。

用法:
    python scripts/11_ablation.py rrf                      # RRF k 与两路权重扫描（不调 LLM）
    python scripts/11_ablation.py rrf --no-rerank-final    # 只跑融合层，不确认精排
    python scripts/11_ablation.py agent                    # router 与 generate 消融（调 LLM）
    python scripts/11_ablation.py agent --limit 3          # 先小样本冒烟

结果写入 eval/ablation_*.json，并在终端打印对比表。
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.embeddings import embed_texts  # noqa: E402
from shopagent.retrieval import RERANK_CANDIDATES, Retriever  # noqa: E402
from shopagent.retrieval.backend import open_backend  # noqa: E402
from shopagent.retrieval.fusion import dedupe_by_asin, rrf  # noqa: E402
from shopagent.retrieval.rerank import rerank  # noqa: E402

EVAL_SETS = {
    "handwritten50": ROOT / "eval" / "retrieval_eval_handwritten50.jsonl",
    "generated50": ROOT / "eval" / "retrieval_eval_generated50.jsonl",
}
K = 5
HIT_K = 3
CANDIDATE_K = 50


# ---------------- 指标 ----------------


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
    dcg = 0.0
    for rank, asin in enumerate(retrieved[:k], start=1):
        if asin in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def load_eval_set(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(json.loads(line))
    return items


def metrics_for(rankings: list[list[str]], items: list[dict]) -> dict:
    recalls, mrrs, ndcgs = [], [], []
    for retrieved, item in zip(rankings, items):
        relevant = set(item["relevant_asins"])
        window = retrieved[:HIT_K]
        recalls.append(recall_at_k(window, list(relevant), HIT_K))
        mrrs.append(mrr(window, relevant))
        ndcgs.append(ndcg_at_k(window, relevant, HIT_K))
    n = max(len(items), 1)
    return {
        "n_queries": len(items),
        "hit_k": HIT_K,
        "recall": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "ndcg": sum(ndcgs) / n,
    }


# ---------------- RRF 扫描 ----------------


def _pool(backend, items: list[dict]) -> list[tuple[list[dict], list[dict]]]:
    """每条 query 只算一次向量/BM25 召回，之后换参数只重排名次。"""
    bm25 = backend.bm25_index()
    pools = []
    for i, item in enumerate(items, start=1):
        query = item["query"]
        emb = embed_texts([query])[0].tolist()
        vec_hits = backend.vector_search(emb, top_k=CANDIDATE_K)
        bm25_hits = bm25.search(query, top_k=CANDIDATE_K)
        pools.append((vec_hits, bm25_hits))
        if i % 20 == 0:
            print(f"    召回池 {i}/{len(items)}")
    return pools


def _rank_asins(fused: list[dict], top_k: int = K) -> list[str]:
    return [h["asin"] for h in dedupe_by_asin(fused, limit=top_k)]


def run_rrf(store: str, rerank_final: bool) -> None:
    backend_ctx = open_backend(store)
    backend = backend_ctx.__enter__()

    combos: list[tuple[int, float, float]] = [(k, 1.0, 1.0) for k in (10, 20, 30, 60, 100)]
    combos += [(60, wv, wb) for wv, wb in ((2.0, 1.0), (3.0, 1.0), (1.0, 2.0), (1.0, 3.0))]

    report: dict = {"store": store, "k": K, "hit_k": HIT_K, "sets": {}}
    print(f"RRF 扫描：后端 {store}，{len(combos)} 组参数\n")

    for name, path in EVAL_SETS.items():
        items = load_eval_set(path)
        print(f"[{name}] {len(items)} 条 query，构建召回池…")
        pools = _pool(backend, items)

        table = {}
        for k, wv, wb in combos:
            rankings = []
            for vec_hits, bm25_hits in pools:
                fused = rrf([vec_hits, bm25_hits], k=k, weights=[wv, wb])
                rankings.append(_rank_asins(fused))
            key = f"k={k},w={wv:g}:{wb:g}"
            table[key] = metrics_for(rankings, items)
            print(f"  {key:<14} Hit@{HIT_K}={table[key]['recall']:.4f}  "
                  f"MRR={table[key]['mrr']:.4f}  nDCG={table[key]['ndcg']:.4f}")

        best_key, best = max(table.items(), key=lambda kv: kv[1]["recall"])
        report["sets"][name] = {"grid": table, "best_hybrid": best_key}

        if rerank_final:
            # 只确认「默认」和「融合层最优」两组，精排很贵
            for label, key in (("default", "k=60,w=1:1"), ("best", best_key)):
                k, w = key.split(",w=")
                k = int(k.split("=")[1])
                wv, wb = (float(x) for x in w.split(":"))
                rankings = []
                for vec_hits, bm25_hits in pools:
                    fused = rrf([vec_hits, bm25_hits], k=k, weights=[wv, wb])
                    candidates = dedupe_by_asin(fused, limit=RERANK_CANDIDATES)
                    hits = rerank(items[len(rankings)]["query"], candidates, top_k=K)
                    rankings.append([h["asin"] for h in hits])
                m = metrics_for(rankings, items)
                report["sets"][name].setdefault("rerank", {})[label] = {"params": key, **m}
                print(f"  +rerank[{label}] {key:<14} Hit@{HIT_K}={m['recall']:.4f}  "
                      f"MRR={m['mrr']:.4f}")

    backend_ctx.__exit__(None, None, None)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "eval" / f"ablation_rrf_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 -> {out}")


# ---------------- Agent 消融 ----------------


def _load_agent_eval():
    spec = importlib.util.spec_from_file_location(
        "agent_eval", ROOT / "scripts" / "09_eval_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_agent(store: str, mode: str, limit: int | None) -> None:
    ev = _load_agent_eval()
    from shopagent import llm, memory
    from shopagent.agent.graph import run_agent as run_graph
    from shopagent.config import settings
    from shopagent.tools.shop import close, configure

    tasks = ev.load_tasks(ROOT / "eval" / "agent_tasks.jsonl")
    if limit:
        tasks = tasks[:limit]

    tmp_profile = settings.data_dir / "_eval_profile.json"
    memory.PROFILE_PATH = tmp_profile
    configure(store, mode)

    configs = [
        ("baseline", {"SHOPAGENT_ROUTER": "llm", "SHOPAGENT_GENERATE_GUARD": "1"}),
        ("router=keyword", {"SHOPAGENT_ROUTER": "keyword", "SHOPAGENT_GENERATE_GUARD": "1"}),
        ("router=shop", {"SHOPAGENT_ROUTER": "shop", "SHOPAGENT_GENERATE_GUARD": "1"}),
        ("guard=off", {"SHOPAGENT_ROUTER": "llm", "SHOPAGENT_GENERATE_GUARD": "0"}),
    ]

    report: dict = {"store": store, "mode": mode, "n_tasks": len(tasks), "configs": {}}
    for label, env in configs:
        os.environ.update(env)
        print(f"\n===== {label}  {env} =====")
        records = []
        for task in tasks:
            tmp_profile.write_text(
                json.dumps(task.get("profile") or {}, ensure_ascii=False), encoding="utf-8"
            )
            llm.reset_calls()
            t0 = time.perf_counter()
            error = None
            try:
                state = run_graph(task["query"], history=task.get("history") or [])
                checks = ev.run_checks(task, state)
            except Exception as exc:  # 单条失败不中断整轮
                error = f"{type(exc).__name__}: {exc}"
                state, checks = {}, []
            elapsed = time.perf_counter() - t0
            passed = error is None and all(c["ok"] for c in checks) and bool(checks)
            records.append({
                "id": task["id"],
                "passed": passed,
                "error": error,
                "checks": checks,
                "steps": state.get("steps", 0),
                "llm_calls": llm.call_count(),
                "latency_s": round(elapsed, 2),
            })
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {task['id']:<24} {elapsed:5.1f}s {llm.call_count()} 次 LLM")

        n = len(records)
        by_check: dict[str, list[bool]] = {}
        for r in records:
            for c in r["checks"]:
                by_check.setdefault(c["name"], []).append(c["ok"])
        latencies = sorted(r["latency_s"] for r in records)
        summary = {
            "task_pass_rate": sum(r["passed"] for r in records) / n,
            "by_check": {k: sum(v) / len(v) for k, v in by_check.items()},
            "avg_steps": statistics.mean(r["steps"] for r in records),
            "avg_llm_calls": statistics.mean(r["llm_calls"] for r in records),
            "latency_p50_s": statistics.median(latencies),
            "latency_p95_s": latencies[min(n - 1, int(n * 0.95))],
        }
        report["configs"][label] = {"env": env, "summary": summary, "tasks": records}
        print(f"  通过率 {summary['task_pass_rate']:.1%} | "
              f"意图 {summary['by_check'].get('intent', float('nan')):.1%} | "
              f"价格无幻觉 {summary['by_check'].get('grounded_prices', float('nan')):.1%} | "
              f"LLM {summary['avg_llm_calls']:.2f} 次 | P50 {summary['latency_p50_s']:.1f}s")

    for key in ("SHOPAGENT_ROUTER", "SHOPAGENT_GENERATE_GUARD"):
        os.environ.pop(key, None)
    close()
    tmp_profile.unlink(missing_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "eval" / f"ablation_agent_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)

    p_rrf = sub.add_parser("rrf", help="RRF k 与权重扫描")
    p_rrf.add_argument("--store", choices=["pg", "memory"], default="pg")
    p_rrf.add_argument("--no-rerank-final", action="store_true",
                       help="跳过精排确认，只跑融合层")

    p_agent = sub.add_parser("agent", help="router / generate 消融")
    p_agent.add_argument("--store", choices=["pg", "memory"], default="pg")
    p_agent.add_argument("--mode", default="hybrid_rerank")
    p_agent.add_argument("--limit", type=int, help="只跑前 N 条任务（冒烟用）")

    args = parser.parse_args()
    if args.kind == "rrf":
        run_rrf(args.store, not args.no_rerank_final)
    else:
        run_agent(args.store, args.mode, args.limit)


if __name__ == "__main__":
    main()
