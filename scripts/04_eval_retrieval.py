"""对比四种检索模式，按「目标进入 top-3 就算命中」的口径产出指标。

命中口径：目标 ASIN 排在返回结果的前 --hit-k 名（默认 3）即命中。
Hit@3 = 命中的 query 占比；MRR / nDCG 也只在前 --hit-k 名里计算，口径与命中一致。
--k 只决定向检索器要回多少条，方便看目标掉在 4/5 名的情况。

用法:
    python scripts/04_eval_retrieval.py --k 5
    python scripts/04_eval_retrieval.py --k 5 --hit-k 3 --modes vector hybrid_rerank
    python scripts/04_eval_retrieval.py --k 5 --store memory   # 不用 Docker
    python scripts/04_eval_retrieval.py --k 5 --dump-ranks eval/ranks.json

评测集格式（一行一个 JSON）:
    {"query": "...", "relevant_asins": ["B00XXXX", "B00YYYY"]}
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.retrieval import (  # noqa: E402
    MODES, Retriever, current_rerank_candidates, set_rerank_candidates,
)
from shopagent.retrieval.backend import open_backend  # noqa: E402


def load_eval_set(path: Path) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if not obj.get("query") or not obj.get("relevant_asins"):
                print(f"  跳过第 {lineno} 行：缺少 query 或 relevant_asins")
                continue
            items.append(obj)
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
    dcg = 0.0
    for rank, asin in enumerate(retrieved[:k], start=1):
        if asin in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(retriever: Retriever, items: list[dict], k: int, hit_k: int) -> tuple[dict, list[dict]]:
    recalls, mrrs, ndcgs, details = [], [], [], []
    for item in items:
        hits = retriever.search(item["query"], top_k=k)
        retrieved = [h["asin"] for h in hits]
        relevant = set(item["relevant_asins"])
        window = retrieved[:hit_k]  # 只有进前 hit_k 名才算命中
        recalls.append(recall_at_k(window, list(relevant), hit_k))
        mrrs.append(mrr(window, relevant))
        ndcgs.append(ndcg_at_k(window, relevant, hit_k))
        details.append({
            "id": item.get("id"),
            "query": item["query"],
            "ranks": {a: (retrieved.index(a) + 1 if a in retrieved else None)
                      for a in item["relevant_asins"]},
        })

    n = max(len(items), 1)
    metrics = {
        "n_queries": len(items),
        "hit_k": hit_k,
        "recall": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "ndcg": sum(ndcgs) / n,
    }
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=ROOT / "eval" / "retrieval_eval.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--hit-k", type=int, default=3, dest="hit_k",
                        help="目标排在多少名以内算命中，默认 3")
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--dump-ranks", type=Path, default=None,
                        help="把每条目标在各模式下的排名存成 JSON")
    parser.add_argument("--rerank-candidates", type=int, default=None,
                        dest="rerank_candidates",
                        help="本次评测使用的精排候选数（不改服务端配置）")
    args = parser.parse_args()

    if args.rerank_candidates is not None:
        set_rerank_candidates(args.rerank_candidates)
    print(f"精排候选数 = {current_rerank_candidates()}")

    if args.hit_k < 1:
        raise SystemExit("--hit-k 必须 >= 1")
    if args.hit_k > args.k:
        print(f"--hit-k={args.hit_k} 大于 --k={args.k}，按 {args.k} 计算")
        args.hit_k = args.k

    if not args.eval.exists():
        raise SystemExit(f"找不到评测集 {args.eval}，先跑 scripts/05_make_eval_set.py")

    items = load_eval_set(args.eval)
    if not items:
        raise SystemExit("评测集为空。")

    print(f"评测集 {len(items)} 条，检索深度 K={args.k}，命中口径 top-{args.hit_k}\n")

    results: dict[str, dict] = {}
    details: dict[str, list[dict]] = {}
    with open_backend(args.store) as backend:
        for mode in args.modes:
            retriever = Retriever(backend, mode=mode)
            metrics, mode_details = evaluate(retriever, items, args.k, args.hit_k)
            results[mode] = metrics
            details[mode] = mode_details
            print(f"{mode:<15} Hit@{args.hit_k}={metrics['recall']:.4f}  "
                  f"MRR@{args.hit_k}={metrics['mrr']:.4f}  "
                  f"NDCG@{args.hit_k}={metrics['ndcg']:.4f}")

    head = f"{'mode':<15}{'Hit@' + str(args.hit_k):>12}{'MRR@' + str(args.hit_k):>10}{'nDCG@' + str(args.hit_k):>12}"
    print(f"\n{head}")
    print("-" * 50)
    for mode, m in results.items():
        print(f"{mode:<15}{m['recall']:>12.4f}{m['mrr']:>10.4f}{m['ndcg']:>12.4f}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "eval" / f"results_{stamp}.json"
    out.write_text(
        json.dumps({"k": args.k, "hit_k": args.hit_k, "n_queries": len(items),
                    "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已保存 -> {out}")

    if args.dump_ranks:
        rows = []
        for i, item in enumerate(items):
            rows.append({
                "id": item.get("id"),
                "query": item["query"],
                "relevant_asins": item["relevant_asins"],
                "ranks": {mode: details[mode][i]["ranks"] for mode in results},
            })
        args.dump_ranks.parent.mkdir(parents=True, exist_ok=True)
        args.dump_ranks.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"逐条排名 -> {args.dump_ranks}")


if __name__ == "__main__":
    main()
