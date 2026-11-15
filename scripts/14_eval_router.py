"""对比 router 方案：关键词规则 / DeepSeek / 本地 LoRA 小模型。

评测集：
    data/router/test.jsonl   教师（DeepSeek）生成的留出集，标签是教师标签，不是人工金标；
                             适合看「学生和教师差多少」，绝对准确率会偏乐观。
    eval/router_hard.jsonl   人工标注的 39 条边界样例（闲聊+商品词、隐含对比、否定、指代），
                             这才是判断能不能上线的集合。

指标：intent 准确率 / macro-F1 / 混淆矩阵 / 非法输出数 / P50、P95 延迟，
以及 shop|compare 样本上 en_query 与参考改写的完全一致率和 token Jaccard。

用法:
    python scripts/14_eval_router.py --sets hard --modes keyword,deepseek
    python scripts/14_eval_router.py --sets hard,test --modes keyword,deepseek,local \\
        --local-model qwen2.5-1.5b-router
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent import llm  # noqa: E402
from shopagent.agent.nodes import _keyword_intent  # noqa: E402
from shopagent.local_router import LocalRouter, router_messages, router_status  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LABELS = ["shop", "compare", "chitchat"]


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"找不到评测集：{path}")
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def macro_f1(gold: list[str], pred: list[str | None]) -> dict:
    per_class = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for g in gold if g == label),
        }
    present = [v["f1"] for v in per_class.values() if v["support"]]
    return {
        "macro_f1": round(sum(present) / len(present), 4) if present else 0.0,
        "per_class": per_class,
    }


def evaluate(rows: list[dict], mode: str, local: LocalRouter | None) -> dict:
    gold = [row["intent"] for row in rows]
    pred: list[str | None] = []
    en_preds: list[str] = []
    latencies: list[float] = []
    errors: list[str] = []
    details: list[dict] = []

    for row in rows:
        query = row["query"]
        started = time.perf_counter()
        intent: str | None = None
        en_query = ""
        error = ""
        try:
            if mode == "keyword":
                intent = _keyword_intent(query)
            elif mode == "deepseek":
                obj = llm.chat_json(router_messages(query), temperature=0)
                intent = str(obj.get("intent", "")).strip().lower()
                en_query = str(obj.get("en_query") or "").strip()
            elif mode == "local":
                if local is None:
                    raise RuntimeError("local 模式需要 LocalRouter 实例")
                result = local.route(query)
                intent = result["intent"]
                en_query = result["en_query"]
            else:
                raise ValueError(f"未知 mode：{mode}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        if intent not in LABELS:
            intent = None
        pred.append(intent)
        en_preds.append(en_query)
        details.append(
            {
                "query": query,
                "gold": row["intent"],
                "pred": intent,
                "gold_en": row.get("en_query", ""),
                "pred_en": en_query,
                "ms": round(elapsed_ms, 1),
                "error": error,
            }
        )

    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0
    metrics = macro_f1(gold, pred)

    rewrite_rows = [
        (row.get("en_query", ""), pred_en)
        for row, pred_en in zip(rows, en_preds)
        if row["intent"] in {"shop", "compare"} and row.get("en_query")
    ]
    rewrite_exact = (
        sum(1 for g, p in rewrite_rows if tokens(g) == tokens(p)) / len(rewrite_rows)
        if rewrite_rows
        else 0.0
    )
    rewrite_jaccard = (
        statistics.mean(jaccard(g, p) for g, p in rewrite_rows) if rewrite_rows else 0.0
    )

    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        if p is not None:
            confusion[g][p] += 1

    return {
        "mode": mode,
        "n": len(rows),
        "accuracy": round(accuracy, 4),
        "invalid": sum(1 for p in pred if p is None),
        **metrics,
        "confusion": confusion,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
        },
        "rewrite": {
            "n": len(rewrite_rows),
            "exact": round(rewrite_exact, 4),
            "jaccard": round(rewrite_jaccard, 4),
        },
        "errors": errors[:20],
        "rows": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="hard,test")
    parser.add_argument("--modes", default="keyword,deepseek,local")
    parser.add_argument("--local-base-url", default=None)
    parser.add_argument("--local-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_paths = {
        "hard": ROOT / "eval" / "router_hard.jsonl",
        "test": ROOT / "data" / "router" / "test.jsonl",
    }
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    local = None
    if "local" in modes:
        # 探测函数读的是环境变量，先把命令行参数写进去，避免探测和实际调用用的不是同一个模型
        if args.local_model:
            os.environ["SHOPAGENT_LOCAL_ROUTER_MODEL"] = args.local_model
        if args.local_base_url:
            os.environ["SHOPAGENT_LOCAL_ROUTER_BASE_URL"] = args.local_base_url
        local = LocalRouter(base_url=args.local_base_url, model=args.local_model)
        status = router_status(force=True)
        print(f"本地 router：model={local.model}  状态={status['detail']}  可用={status['available']}")

    report: dict = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "local_model": local.model if local else None,
        "sets": {},
    }

    for set_name in [s.strip() for s in args.sets.split(",") if s.strip()]:
        rows = load_jsonl(set_paths[set_name], limit=args.limit)
        print(f"\n== {set_name}（n={len(rows)}）==")
        print(
            f"{'mode':<10}{'acc':>8}{'macroF1':>9}{'p50ms':>9}{'p95ms':>9}"
            f"{'invalid':>9}{'rw_exact':>10}{'rw_jac':>9}"
        )
        report["sets"][set_name] = {"rows": len(rows), "results": {}}
        for mode in modes:
            if mode == "deepseek":
                with llm.use_model("deepseek"):
                    result = evaluate(rows, mode, local)
            else:
                result = evaluate(rows, mode, local)
            report["sets"][set_name]["results"][mode] = result
            print(
                f"{mode:<10}{result['accuracy']:>8.3f}{result['macro_f1']:>9.3f}"
                f"{result['latency_ms']['p50']:>9.0f}{result['latency_ms']['p95']:>9.0f}"
                f"{result['invalid']:>9}{result['rewrite']['exact']:>10.3f}"
                f"{result['rewrite']['jaccard']:>9.3f}"
            )
            if set_name == "hard":
                for row in result["rows"]:
                    if row["pred"] != row["gold"]:
                        print(
                            f"    DIFF gold={row['gold']:<8} pred={row['pred']!s:<8} {row['query']}"
                        )

    out = (
        Path(args.out)
        if args.out
        else ROOT / "eval" / f"router_eval_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果写入 {out}")


if __name__ == "__main__":
    main()
