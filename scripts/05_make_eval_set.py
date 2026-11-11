"""用 LLM 从商品库里生成检索评测集草稿。

直接读 data/products.jsonl，不依赖数据库。

用法:
    python scripts/05_make_eval_set.py --n 40
    python scripts/05_make_eval_set.py --n 40 --seed 42 --out eval/retrieval_eval.jsonl

生成的 query 是英文（数据本身是英文，BM25 才能正常工作）。
输出后务必人工过一遍：删掉语义不清、答案不唯一的条目。
**评测集质量决定你简历上数字的可信度，不能全信 LLM。**
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.config import settings  # noqa: E402
from shopagent.ingest import product_to_text  # noqa: E402
from shopagent.llm import chat_json  # noqa: E402

PROMPT = """You are building an evaluation set for an e-commerce product search system.

Product:
{product}

Write ONE realistic English shopping query that this product would be a good answer to.

Rules:
- Do NOT mention the exact product title, brand, or ASIN.
- Describe the shopper's need (use case, constraint, feature), 5-20 words.
- Specific enough that this product is clearly relevant, but not so specific that nothing else could match.

Return JSON: {{"query": "..."}}"""


def load_products(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_products(products: list[dict], n: int, seed: float) -> list[dict]:
    # 描述太短的条目不适合当评测目标
    pool = [p for p in products if len(p.get("description") or "") > 200]
    if not pool:
        raise SystemExit("没有描述超过 200 字符的商品，先扩充数据。")
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--input", type=Path, default=settings.data_dir / "products.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "retrieval_eval.jsonl")
    args = parser.parse_args()

    products = sample_products(load_products(args.input), args.n, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, product in enumerate(products, start=1):
            prompt = PROMPT.format(product=product_to_text(product)[:2500])
            try:
                obj = chat_json([{"role": "user", "content": prompt}])
                query = (obj.get("query") or "").strip()
            except Exception as exc:  # 单条失败不影响整体
                print(f"[{i}/{len(products)}] 失败: {exc}")
                continue

            if not query:
                print(f"[{i}/{len(products)}] 空结果，跳过")
                continue

            record = {
                "id": f"g_{i:02d}",
                "query": query,
                "relevant_asins": [product["asin"]],
                "product_title": product["title"],
                "brand": product.get("brand"),
                "price": product.get("price"),
                "category": product.get("category"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            print(f"[{i}/{len(products)}] {query}")
            time.sleep(0.2)

    print(f"\n写入 {written} 条 -> {args.out}")
    print("下一步：人工 review 这份文件，删掉不合理的条目，再跑 04_eval_retrieval.py")


if __name__ == "__main__":
    main()
