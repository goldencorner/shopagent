"""Agent 级评测：工具选择、检索命中、预算合规、价格幻觉、拒答、步数与延迟。

和 04 的检索评测不同，这里跑的是完整 Agent（router→recall→planner→tool→generate），
每条任务都是中文用户原话，模型自己决定调用哪个工具。

用法:
    python scripts/09_eval_agent.py                      # 默认 memory 后端，全部任务
    python scripts/09_eval_agent.py --store pg
    python scripts/09_eval_agent.py --ids search_yoga_mat refuse_cat_food
    python scripts/09_eval_agent.py --limit 5

任务集 eval/agent_tasks.jsonl 字段见文件本身；checks 支持：
    intent / required_tools / forbidden_tools / must_retrieve_any /
    must_mention_any / must_not_mention / expect_no_match / budget /
    grounded_prices / max_answer_chars
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent import llm, memory  # noqa: E402
from shopagent.agent.graph import run_agent  # noqa: E402
from shopagent.config import settings  # noqa: E402
from shopagent.tools.shop import configure, close  # noqa: E402

PRICE_RE = re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)")
REFUSAL_RE = re.compile(r"没有|未找到|找不到|暂无|无法|抱歉|不提供|不在售|没有相关|未收录")


# ---------------- 从 state 里抽事实 ----------------


def tool_names(state: dict) -> list[str]:
    return [r.get("tool", "") for r in (state.get("tool_results") or [])]


def _rows(state: dict) -> list[dict]:
    rows: list[dict] = []
    for item in state.get("tool_results") or []:
        result = (item.get("output") or {}).get("result")
        if result is None:
            continue
        for row in result if isinstance(result, list) else [result]:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def retrieved_asins(state: dict) -> set[str]:
    return {r["asin"] for r in _rows(state) if r.get("asin")}


def retrieved_prices(state: dict) -> set[float]:
    return {round(float(r["price"]), 2) for r in _rows(state) if r.get("price") is not None}


def mentioned_prices(answer: str) -> list[float]:
    return [float(x) for x in PRICE_RE.findall(answer or "")]


def _approx_in(value: float, pool: set[float], tol: float = 0.011) -> bool:
    return any(abs(value - p) <= tol for p in pool)


# ---------------- 单条任务的检查 ----------------


def run_checks(task: dict, state: dict) -> list[dict]:
    checks = task.get("checks") or {}
    answer = state.get("answer") or ""
    tools = tool_names(state)
    asins = retrieved_asins(state)
    prices = mentioned_prices(answer)
    out: list[dict] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        out.append({"name": name, "ok": bool(ok), "detail": detail})

    if "intent" in checks:
        ok = state.get("intent") == checks["intent"]
        add("intent", ok, f"got={state.get('intent')} want={checks['intent']}")

    if checks.get("required_tools"):
        missing = [t for t in checks["required_tools"] if t not in tools]
        add("required_tools", not missing, f"missing={missing} called={tools}")

    if checks.get("forbidden_tools"):
        hit = [t for t in checks["forbidden_tools"] if t in tools]
        add("forbidden_tools", not hit, f"called={hit}")

    if checks.get("must_retrieve_any"):
        hit = asins & set(checks["must_retrieve_any"])
        add("must_retrieve_any", bool(hit), f"hit={sorted(hit)}")

    if checks.get("must_retrieve_title_regex"):
        # 不绑定具体 ASIN：同一需求可以接受多款相关商品，只看召回的标题里
        # 有没有品类词。比“必须命中我预先挑的那几个 ASIN”更公平。
        pattern = re.compile(checks["must_retrieve_title_regex"], re.I)
        titles = [r.get("title") or "" for r in _rows(state)]
        hit = [t for t in titles if pattern.search(t)]
        add("must_retrieve_title_regex", bool(hit),
            f"hit={[t[:40] for t in hit[:2]]}")

    if checks.get("must_mention_any"):
        hit = [k for k in checks["must_mention_any"] if k.lower() in answer.lower()]
        add("must_mention_any", bool(hit), f"hit={hit}")

    if checks.get("must_not_mention"):
        hit = [k for k in checks["must_not_mention"] if k.lower() in answer.lower()]
        add("must_not_mention", not hit, f"hit={hit}")

    if checks.get("expect_no_match"):
        ok = bool(REFUSAL_RE.search(answer))
        add("expect_no_match", ok, f"answer={answer[:60]!r}")

    if "budget" in checks:
        budget = float(checks["budget"])
        over = [p for p in prices if p > budget + 0.011]
        add("budget", not over, f"budget={budget} over={over}")

    if checks.get("grounded_prices", True):
        allowed = set(retrieved_prices(state))
        if "budget" in checks:
            allowed.add(round(float(checks["budget"]), 2))
        profile_budget = (task.get("profile") or {}).get("budget_max")
        if profile_budget:
            allowed.add(round(float(profile_budget), 2))
        ungrounded = [p for p in prices if not _approx_in(p, allowed)]
        add("grounded_prices", not ungrounded, f"ungrounded={ungrounded}")

    if "max_answer_chars" in checks:
        ok = len(answer) <= int(checks["max_answer_chars"])
        add("max_answer_chars", ok, f"len={len(answer)}")

    return out


# ---------------- 主流程 ----------------


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path} 第 {lineno} 行不是合法 JSON: {exc}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=ROOT / "eval" / "agent_tasks.jsonl")
    parser.add_argument("--store", choices=["pg", "memory"], default="memory")
    parser.add_argument("--mode", default="hybrid_rerank")
    parser.add_argument("--ids", nargs="*", help="只跑指定 id")
    parser.add_argument("--limit", type=int, help="只跑前 N 条")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if args.ids:
        tasks = [t for t in tasks if t.get("id") in set(args.ids)]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("没有可跑的任务。")

    # 评测不能污染真实画像：写临时文件，每条任务前重置
    tmp_profile = settings.data_dir / "_eval_profile.json"
    memory.PROFILE_PATH = tmp_profile

    configure(args.store, args.mode)
    print(f"Agent 评测：{len(tasks)} 条任务，后端 {args.store}，模式 {args.mode}\n")

    results = []
    for task in tasks:
        tmp_profile.write_text(
            json.dumps(task.get("profile") or {}, ensure_ascii=False), encoding="utf-8"
        )
        llm.reset_calls()
        t0 = time.perf_counter()
        error = None
        try:
            state = run_agent(task["query"], history=task.get("history") or [])
            checks = run_checks(task, state)
            answer = state.get("answer") or ""
        except Exception as exc:  # 单条失败不该中断整轮评测
            error = f"{type(exc).__name__}: {exc}"
            state, checks, answer = {}, [], ""
        elapsed = time.perf_counter() - t0

        passed = error is None and all(c["ok"] for c in checks) and bool(checks)
        record = {
            "id": task["id"],
            "query": task["query"],
            "passed": passed,
            "error": error,
            "checks": checks,
            "steps": state.get("steps", 0),
            "llm_calls": llm.call_count(),
            "latency_s": round(elapsed, 2),
            "answer": answer,
        }
        results.append(record)

        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {task['id']:<24} {elapsed:5.1f}s  "
              f"{record['llm_calls']} 次 LLM  {record['steps']} 步")
        for c in checks:
            if not c["ok"]:
                print(f"        ✗ {c['name']}: {c['detail']}")
        if error:
            print(f"        ✗ error: {error}")

    # ---------- 汇总 ----------
    n = len(results)
    task_pass = sum(r["passed"] for r in results) / n

    by_check: dict[str, list[bool]] = {}
    for r in results:
        for c in r["checks"]:
            by_check.setdefault(c["name"], []).append(c["ok"])

    def rate(*names: str) -> str:
        vals = [v for name in names for v in by_check.get(name, [])]
        if not vals:
            return "—"
        return f"{sum(vals)}/{len(vals)} ({sum(vals) / len(vals):.1%})"

    latencies = sorted(r["latency_s"] for r in results)
    p50 = statistics.median(latencies)
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]

    print("\n" + "=" * 58)
    print(f"任务通过率        {sum(r['passed'] for r in results)}/{n} ({task_pass:.1%})")
    print(f"意图识别          {rate('intent')}")
    print(f"工具选择          {rate('required_tools')}")
    print(f"检索命中          {rate('must_retrieve_any', 'must_retrieve_title_regex')}")
    print(f"预算合规          {rate('budget')}")
    print(f"价格无幻觉        {rate('grounded_prices')}")
    print(f"拒答正确          {rate('expect_no_match')}")
    print(f"平均步数          {statistics.mean(r['steps'] for r in results):.2f}")
    print(f"平均 LLM 调用     {statistics.mean(r['llm_calls'] for r in results):.2f}")
    print(f"延迟 P50 / P95    {p50:.1f}s / {p95:.1f}s")
    print("=" * 58)

    summary = {
        "n_tasks": n,
        "task_pass_rate": task_pass,
        "by_check": {k: sum(v) / len(v) for k, v in by_check.items()},
        "avg_steps": statistics.mean(r["steps"] for r in results),
        "avg_llm_calls": statistics.mean(r["llm_calls"] for r in results),
        "latency_p50_s": p50,
        "latency_p95_s": p95,
    }
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "eval" / f"agent_results_{stamp}.json"
    out.write_text(
        json.dumps({"summary": summary, "tasks": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n已保存 -> {out}")

    close()
    tmp_profile.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
