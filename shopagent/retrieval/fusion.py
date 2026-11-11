from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[dict]], k: int = 60, weights: list[float] | None = None) -> list[dict]:
    """Reciprocal Rank Fusion。

    为什么用 RRF 而不是加权分数：BM25 分数无上界、余弦相似度在 [-1,1]，
    量纲不同，直接加权不可比。RRF 只用排名，天然免疫量纲问题。
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = defaultdict(float)
    docs: dict[int, dict] = {}

    for ranking, weight in zip(rankings, weights):
        for rank, item in enumerate(ranking, start=1):
            key = item["chunk_id"]
            docs[key] = item
            scores[key] += weight / (k + rank)

    fused = []
    for key, score in sorted(scores.items(), key=lambda x: -x[1]):
        doc = dict(docs[key])
        doc["score"] = score
        fused.append(doc)
    return fused


def dedupe_by_asin(items: list[dict], limit: int | None = None) -> list[dict]:
    """同一商品只保留排名最高的 chunk，避免一个 SKU 占满结果。"""
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        asin = item.get("asin")
        if asin in seen:
            continue
        seen.add(asin)
        out.append(item)
        if limit and len(out) >= limit:
            break
    return out
