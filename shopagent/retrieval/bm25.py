from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ..db import rows_to_dicts

# 英文按词、中文按字切分。商品数据是英文时走前半段。
_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """内存版 BM25。几千条 chunk 够用；10w 级别改用 fts.FTSIndex。"""

    def __init__(self, docs: list[dict]):
        self.docs = docs
        corpus = [tokenize(d["content"]) for d in docs]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    @classmethod
    def from_db(cls, conn) -> "BM25Index":
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id AS chunk_id, c.product_id, c.content,
                       p.asin, p.title, p.price
                FROM chunks c
                JOIN products p ON p.id = c.product_id
                """
            )
            return cls(rows_to_dicts(cur))

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = scores.argsort()[::-1][:top_k]
        results: list[dict] = []
        for i in order:
            if scores[i] <= 0:
                continue
            doc = dict(self.docs[int(i)])
            doc["score"] = float(scores[int(i)])
            results.append(doc)
        return results
