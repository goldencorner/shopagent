"""下载并清洗商品数据 -> data/products.jsonl

直接流式读取 HuggingFace 上的 JSONL，不落全量原始文件。
（比用 datasets 加载脚本更稳，也不受 datasets 版本影响）

用法:
    python scripts/01_load_data.py --sample 100000 --seed 42   # 全量随机抽样
    python scripts/01_load_data.py --limit 3000                # 取前 3000，快速试跑
    python scripts/01_load_data.py --category raw_meta_Sports_and_Outdoors --limit 1000

输出 JSONL 每行字段（自定义数据也照这个格式）:
    asin, title, brand, category, price, description, features
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.config import settings  # noqa: E402  先导入以设置 HF_ENDPOINT

_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _join(value) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(str(v).strip() for v in value if v)
    return str(value).strip()


def parse_price(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _PRICE_RE.search(str(value).replace(",", ""))
    return float(match.group(1)) if match else None


def normalize(row: dict) -> dict | None:
    asin = (row.get("parent_asin") or row.get("asin") or "").strip()
    title = (row.get("title") or "").strip()
    if not asin or not title:
        return None

    description = _join(row.get("description"))
    features = _join(row.get("features"))
    # 文本太少的商品检索价值低，直接丢弃
    if len(description) + len(features) < 80:
        return None

    categories = row.get("categories") or []
    category = categories[0] if categories else row.get("main_category")

    return {
        "asin": asin,
        "title": title[:500],
        "brand": (row.get("store") or row.get("brand") or "").strip() or None,
        "category": category,
        "price": parse_price(row.get("price")),
        "description": description[:4000],
        "features": features[:2000],
    }


META_BASE = "/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories"


def _file_url(category: str) -> str:
    endpoint = (_get_env("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    name = category.removeprefix("raw_meta_")
    return f"{endpoint}{META_BASE}/meta_{name}.jsonl"


def _get_env(name: str) -> str:
    import os

    return os.getenv(name, "").strip()


def _iter_amazon(category: str):
    """流式产出清洗后的商品，边下边解析，不落全量原始文件。"""
    import requests

    url = _file_url(category)
    print(f"流式下载: {url}")

    scanned = 0
    kept = 0
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(chunk_size=1 << 20, decode_unicode=False):
            if not raw_line:
                continue
            scanned += 1
            if scanned % 100000 == 0:
                print(f"  已扫描 {scanned}，有效 {kept}")
            try:
                row = json.loads(raw_line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            product = normalize(row)
            if product is not None:
                kept += 1
                yield product
    print(f"扫描 {scanned} 条，有效 {kept} 条")


def _iter_local(path: Path):
    # 原始元数据里个别行有坏字节，replace 掉继续，不让整轮抽样挂掉
    with open(path, encoding="utf-8", errors="replace") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            product = normalize(row)
            if product is not None:
                yield product


def _take(rows, limit: int, sample: int | None, seed: int) -> list[dict]:
    """取前 limit 条，或从全量里做蓄水池抽样。

    随机抽样必须读全量，不能取前 N 条：Amazon 元数据的顺序和 ASIN/类目相关，
    前 1 万条不是随机样本（前 3000 条恰好是 10w 那条文件的前缀，就是证据）。
    """
    if sample is None:
        out: list[dict] = []
        for product in rows:
            out.append(product)
            if len(out) >= limit:
                break
        return out

    import random

    rng = random.Random(seed)
    reservoir: list[dict] = []
    asins: set[str] = set()
    seen = 0
    for product in rows:
        asin = product["asin"]
        if asin in asins:
            continue
        seen += 1
        if len(reservoir) < sample:
            reservoir.append(product)
            asins.add(asin)
            continue
        j = rng.randrange(seen)
        if j < sample:
            old = reservoir[j]
            asins.discard(old["asin"])
            reservoir[j] = product
            asins.add(asin)

    rng.shuffle(reservoir)
    print(f"有效商品 {seen} 条，随机抽样 {len(reservoir)} 条（seed={seed}）")
    return reservoir


def _write(rows: list[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for product in rows:
            f.write(json.dumps(product, ensure_ascii=False) + "\n")
    print(f"保留 {len(rows)} 条 -> {out_path}")
    return len(rows)


def from_amazon(category: str, limit: int, out_path: Path,
                sample: int | None = None, seed: int = 42) -> int:
    return _write(_take(_iter_amazon(category), limit, sample, seed), out_path)


def from_local(path: Path, limit: int, out_path: Path,
               sample: int | None = None, seed: int = 42) -> int:
    return _write(_take(_iter_local(path), limit, sample, seed), out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["amazon", "local"], default="amazon")
    parser.add_argument("--category", default="raw_meta_Sports_and_Outdoors")
    parser.add_argument("--input", type=Path, help="--source local 时的输入 JSONL")
    parser.add_argument("--limit", type=int, default=3000,
                        help="取前 N 条（快速试跑用）")
    parser.add_argument("--sample", type=int, default=None,
                        help="从全量里随机抽 N 条；会读完整个类目文件，适合正式语料")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样种子")
    parser.add_argument("--out", type=Path, default=settings.data_dir / "products.jsonl")
    args = parser.parse_args()

    if args.source == "local":
        if not args.input:
            parser.error("--source local 需要 --input")
        n = from_local(args.input, args.limit, args.out, args.sample, args.seed)
    else:
        n = from_amazon(args.category, args.limit, args.out, args.sample, args.seed)

    if n == 0:
        print("没有可用数据，检查类别或输入格式。")
        return

    print("\n样例（检查字段是否正常）:")
    with open(args.out, encoding="utf-8") as f:
        for _ in range(2):
            print(json.dumps(json.loads(f.readline()), ensure_ascii=False, indent=2)[:500])


if __name__ == "__main__":
    main()
