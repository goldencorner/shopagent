"""把现有英文检索评测集翻成中文，得到 50 条中文 query（保留原 ASIN 标签）。

用途：量化「小模型改写 vs planner 现场翻译」对检索的影响。
原始英文 query 直接检索是上界；中文 query 直接检索是下界；两种翻译方式在中间。

用法:
    python scripts/15_make_zh_retrieval.py
    python scripts/15_make_zh_retrieval.py --limit 5      # 冒烟

输出: eval/retrieval_eval_zh50.jsonl
    query        中文 query
    query_en     原始英文 query
    relevant_asins 等字段原样保留
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent import llm  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SRC = ROOT / "eval" / "retrieval_eval_handwritten50.jsonl"
OUT = ROOT / "eval" / "retrieval_eval_zh50.jsonl"

PROMPT = """你是电商搜索的查询翻译器。把下面的英文商品需求翻译成中文用户会说的搜索词。

要求：
1. 保留品类、属性、使用场景、尺寸、预算等所有约束；
2. 数字和币种保持不变；
3. 不要补充原文没有的条件，不要解释；
4. 品牌名、型号可以保留英文。

只返回 JSON：{{"query_zh": "..."}}

英文：{query}"""


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = [
        json.loads(line)
        for line in args.src.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if args.limit:
        items = items[: args.limit]

    done: set[str] = set()
    rows: list[dict] = []
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add(str(row["id"]))
                rows.append(row)

    todo = [item for item in items if str(item["id"]) not in done]
    print(f"源 {len(items)} 条，已完成 {len(done)} 条，待翻译 {len(todo)} 条")

    for idx, item in enumerate(todo, start=1):
        query_en = item["query"]
        with llm.use_model("deepseek"):
            obj = llm.chat_json(
                [{"role": "user", "content": PROMPT.format(query=query_en)}],
                temperature=0,
            )
        query_zh = str(obj.get("query_zh") or "").strip()
        if not query_zh or not has_chinese(query_zh):
            print(f"  [{idx}/{len(todo)}] 跳过（翻译为空或无中文）：{query_en[:60]}")
            continue
        row = dict(item)
        row["id"] = f"zh_{item['id']}"
        row["query_en"] = query_en
        row["query"] = query_zh
        rows.append(row)
        print(f"  [{idx}/{len(todo)}] {query_zh}")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写入 {args.out}（{len(rows)} 条）")


if __name__ == "__main__":
    main()
