"""商品明细表：asin -> 商品完整记录（按需读盘）。

索引里存的是 chunk（一段描述），工具层经常需要整条商品
（拿全 description / features / 价格做对比），所以单独从
data/products.jsonl 读明细。

10 万条全字段放进 Python dict 实测约 290MB——文件只有 107MB，膨胀来自
50 万个字符串对象头和 dict 开销。检索链路不依赖这份明细（chunk 结果里
已经带了 title/price），只有详情/对比/商品库页签会用到，因此不做全量常驻：

- asin -> 文件偏移 的索引常驻（约 11MB），首次访问时建一次；
- 单条记录 seek 读一行（约 0.07ms），最近 3000 条走 LRU；
- 商品库页签按「筛选条件 + 排序」缓存排好序的偏移（每份约 3.6MB，
  最多 4 份），翻页时只读当页的十几条记录。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from statistics import median

from .config import settings

# 单条记录缓存：工具只会对刚讨论过的商品反复取详情，3000 条足够覆盖一轮会话。
RECORD_CACHE_SIZE = 3000
# 商品库页签的筛选结果缓存：翻页/换排序靠它，避免每次都扫全库。
QUERY_CACHE_SIZE = 4

# 商品库页签用的价格分桶：左闭右开，最后一桶不设上限。
PRICE_BINS: list[tuple[float, float | None, str]] = [
    (0, 10, "≤ $10"),
    (10, 25, "$10–25"),
    (25, 50, "$25–50"),
    (50, 100, "$50–100"),
    (100, 200, "$100–200"),
    (200, None, "$200+"),
]


def _path() -> Path:
    return settings.data_dir / "products.jsonl"


def _check(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}\n先运行: python scripts/01_load_data.py --sample 100000 --seed 42"
        )


@lru_cache(maxsize=1)
def _offsets(path_str: str) -> dict[str, int]:
    """asin -> 行首字节偏移。10 万条约 11MB，换掉 290MB 的全量 dict。"""
    path = Path(path_str)
    _check(path)
    offsets: dict[str, int] = {}
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            if line.strip():
                offsets[json.loads(line)["asin"]] = pos
            pos += len(line)
    return offsets


def _iter_rows(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _read_at(path: Path, offset: int) -> dict:
    # 每条单独开文件：FastAPI 会在线程池里并发调用，共享句柄的 seek 不原子
    with open(path, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


@lru_cache(maxsize=RECORD_CACHE_SIZE)
def _record(path_str: str, asin: str) -> dict | None:
    offset = _offsets(path_str).get(asin)
    return None if offset is None else _read_at(Path(path_str), offset)


def count() -> int:
    return len(_offsets(str(_path())))


def get_product(asin: str) -> dict | None:
    return _record(str(_path()), asin)


def brief(product: dict) -> dict:
    """给 LLM 看的精简版，避免把 4000 字描述整个塞进 prompt。"""
    desc = (product.get("description") or "").strip()
    return {
        "asin": product["asin"],
        "title": product.get("title") or "",
        "price": product.get("price"),
        "brand": product.get("brand"),
        "description": desc[:600],
    }


# ---------------- 商品库页签 ----------------


def _hay(row: dict) -> str:
    """搜索要扫全库描述；这里现算现丢，不再按 asin 常驻（10 万条约 146MB）。"""
    return " ".join(
        [
            row.get("title") or "",
            row.get("brand") or "",
            row.get("features") or "",
            row.get("description") or "",
        ]
    ).lower()


def _matches(
    row: dict,
    tokens: list[str],
    price_min: float | None,
    price_max: float | None,
    priced: str | None,
) -> bool:
    price = row.get("price")
    if priced == "yes" and price is None:
        return False
    if priced == "no" and price is not None:
        return False
    if price_min is not None and (price is None or float(price) < price_min):
        return False
    if price_max is not None and (price is None or float(price) > price_max):
        return False
    if tokens and not all(t in _hay(row) for t in tokens):
        return False
    return True


def _sort_key(row: dict, index: int, sort: str) -> tuple:
    price = row.get("price")
    if sort == "price_asc":
        return (price is None, price or 0, row.get("title") or "")
    if sort == "price_desc":
        return (price is None, -(price or 0), row.get("title") or "")
    if sort == "brand":
        return ((row.get("brand") or "").lower(), row.get("title") or "")
    return (index,)


@lru_cache(maxsize=QUERY_CACHE_SIZE)
def _ordered_offsets(
    path_str: str,
    query: str,
    price_min: float | None,
    price_max: float | None,
    priced: str | None,
    sort: str,
) -> tuple[int, ...]:
    """扫全库，返回按当前排序排好的行偏移。

    只留偏移（10 万条约 3.6MB），不留整条记录；翻页时再按偏移读那十几条。
    """
    path = Path(path_str)
    _check(path)
    tokens = [t for t in (query or "").lower().split() if t]

    matched: list[tuple[tuple, int]] = []
    with open(path, "rb") as f:
        pos = 0
        index = 0
        for line in f:
            offset = pos
            pos += len(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if _matches(row, tokens, price_min, price_max, priced):
                matched.append((_sort_key(row, index, sort), offset))
            index += 1

    if sort != "default":
        matched.sort(key=lambda kv: kv[0])
    return tuple(offset for _, offset in matched)


def stats() -> dict:
    """概览数字只算一次：商品文件在进程生命周期内不变。

    结果只有几 KB，值得常驻；扫全库的过程本身不保留任何记录。
    """
    return _stats(str(_path()))


@lru_cache(maxsize=1)
def _stats(path_str: str) -> dict:
    path = Path(path_str)
    _check(path)
    total = 0
    prices: list[float] = []
    brands: Counter[str] = Counter()

    for row in _iter_rows(path):
        total += 1
        price = row.get("price")
        if price is not None:
            prices.append(float(price))
        brands[row.get("brand") or "未知品牌"] += 1

    histogram = []
    for low, high, label in PRICE_BINS:
        count = sum(1 for p in prices if p >= low and (high is None or p < high))
        histogram.append({"label": label, "min": low, "max": high, "count": count})

    return {
        "total": total,
        "priced": len(prices),
        "unpriced": total - len(prices),
        "brands": len(brands),
        "median_price": round(median(prices), 2) if prices else None,
        "price_min": round(min(prices), 2) if prices else None,
        "price_max": round(max(prices), 2) if prices else None,
        "histogram": histogram,
        "top_brands": [
            {"brand": brand, "count": count} for brand, count in brands.most_common(8)
        ],
    }


def _snippet(row: dict, tokens: list[str]) -> str:
    """命中点不在标题/品牌里时，截一段包含关键词的原文。

    否则会出现「搜 yoga，卡片上看不到 yoga」的情况——匹配藏在被截断的
    description 或 features 里，用户无法判断为什么这条被返回。
    """
    hay = " ".join(
        [
            row.get("title") or "",
            row.get("brand") or "",
            row.get("features") or "",
            row.get("description") or "",
        ]
    )
    lowered = hay.lower()
    positions = [lowered.find(t) for t in tokens if lowered.find(t) >= 0]
    if not positions:
        return ""

    pos = min(positions)
    start = lowered.rfind(" ", max(0, pos - 60), pos)
    start = start + 1 if start != -1 else max(0, pos - 60)
    end = hay.find(" ", start + 220)
    end = end if end != -1 else len(hay)
    text = " ".join(hay[start:end].split())
    return f"{'…' if start else ''}{text}{'…' if end < len(hay) else ''}"


def _card(row: dict, tokens: list[str]) -> dict:
    """列表卡片只需要一小段描述，不把整条 4000 字记录发给浏览器。"""
    title = row.get("title") or ""
    brand = row.get("brand") or ""
    snippet = ""
    if tokens and not all(t in f"{title} {brand}".lower() for t in tokens):
        snippet = _snippet(row, tokens)

    return {
        "asin": row["asin"],
        "title": title,
        "brand": brand,
        "price": row.get("price"),
        "description": (row.get("description") or "")[:260],
        "features": (row.get("features") or "")[:180],
        "snippet": snippet,
    }


def browse(
    query: str = "",
    price_min: float | None = None,
    price_max: float | None = None,
    priced: str | None = None,  # "yes" / "no" / None（不限）
    page: int = 1,
    page_size: int = 12,
    sort: str = "default",
) -> dict:
    """商品库列表：搜索 + 价格筛选 + 排序 + 分页。

    搜索是「所有关键词都命中」的 AND 语义，和 BM25 的模糊相关性检索不同；
    这里的目的只是让人快速定位商品，不参与 Agent 的回答链路。
    """
    path = _path()
    offsets = _ordered_offsets(
        str(path), query or "", price_min, price_max, priced, sort
    )
    tokens = [t for t in (query or "").lower().split() if t]

    page_size = max(1, min(int(page_size), 60))
    total = len(offsets)
    pages = math.ceil(total / page_size) if total else 0
    page = max(1, min(int(page), pages or 1))
    start = (page - 1) * page_size
    items = [_card(_read_at(path, offset), tokens) for offset in offsets[start : start + page_size]]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "items": items,
    }
