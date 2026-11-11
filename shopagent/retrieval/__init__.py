from __future__ import annotations

from ..config import settings
from ..embeddings import embed_texts
from .bm25 import BM25Index
from .fusion import dedupe_by_asin, rrf
from .rerank import rerank

MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]

# 精排候选数，改 SHOPAGENT_RERANK_CANDIDATES 即可（默认 20）。40 条评测里
# 50→20 指标不降反升，再降到 15 召回开始掉（见 README §4.3）。
# 召回和融合仍用 candidate_k=50。
#
# 页面上也能热改（POST /config/rerank_candidates）：只改这个模块级变量，
# 正在跑的请求读到的还是旧值，新请求立刻生效；进程重启后回到环境变量默认值。
MIN_RERANK_CANDIDATES = 1
MAX_RERANK_CANDIDATES = 50
RERANK_CANDIDATES = settings.rerank_candidates


def current_rerank_candidates() -> int:
    return RERANK_CANDIDATES


def set_rerank_candidates(value: int) -> int:
    global RERANK_CANDIDATES
    RERANK_CANDIDATES = max(MIN_RERANK_CANDIDATES, min(MAX_RERANK_CANDIDATES, int(value)))
    return RERANK_CANDIDATES


class Retriever:
    """四种模式的统一入口，后端可以是 PgBackend 或 MemoryBackend。

    - bm25:          仅关键词召回
    - vector:        仅向量召回
    - hybrid:        BM25 + 向量，RRF 融合
    - hybrid_rerank: 融合后再用 cross-encoder 精排
    """

    def __init__(self, backend, mode: str = "hybrid_rerank"):
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode!r}")
        self.backend = backend
        self.mode = mode
        self._bm25: BM25Index | None = None

    def _get_bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = self.backend.bm25_index()
        return self._bm25

    def search(self, query: str, top_k: int = 5, candidate_k: int = 50) -> list[dict]:
        if self.mode == "bm25":
            hits = self._get_bm25().search(query, top_k=candidate_k)
            return dedupe_by_asin(hits, limit=top_k)

        query_emb = embed_texts([query])[0].tolist()

        if self.mode == "vector":
            hits = self.backend.vector_search(query_emb, top_k=candidate_k)
            return dedupe_by_asin(hits, limit=top_k)

        bm25_hits = self._get_bm25().search(query, top_k=candidate_k)
        vec_hits = self.backend.vector_search(query_emb, top_k=candidate_k)
        fused = dedupe_by_asin(rrf([vec_hits, bm25_hits]), limit=candidate_k)

        if self.mode == "hybrid":
            return fused[:top_k]

        return rerank(query, fused[:RERANK_CANDIDATES], top_k=top_k)
