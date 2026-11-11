"""检索后端：Postgres(pgvector) 和纯内存两种实现。

内存后端用于 Docker/WSL2 不可用时先跑通链路；
切回 pgvector 时上层代码（Retriever / 脚本）不用改。
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import psycopg

from ..config import settings
from .bm25 import BM25Index
from .fts import FTSIndex

MEMORY_INDEX_PATH = settings.data_dir / "memory_index"


class PgBackend:
    """pgvector 实现。"""

    def __init__(self, conn):
        self.conn = conn
        self._bm25 = None

    def vector_search(self, query_embedding: list[float], top_k: int = 50) -> list[dict]:
        from .vector import vector_search

        return vector_search(self.conn, query_embedding, top_k)

    def bm25_index(self):
        if self._bm25 is None:
            # 大语料优先读磁盘版 FTS5：内存版 166k chunk 实测 2.6GB
            fts = FTSIndex()
            self._bm25 = fts if fts.available() else BM25Index.from_db(self.conn)
        return self._bm25


class MemoryBackend:
    """numpy 矩阵 + rank_bm25，零外部依赖。

    向量已归一化，所以内积即余弦相似度。
    """

    def __init__(self, docs: list[dict], embeddings):
        if len(docs) != len(embeddings):
            raise ValueError(f"docs({len(docs)}) 与 embeddings({len(embeddings)}) 数量不一致")
        self.docs = docs
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self._bm25: BM25Index | None = None

    def vector_search(self, query_embedding: list[float], top_k: int = 50) -> list[dict]:
        query = np.asarray(query_embedding, dtype=np.float32)
        scores = self.embeddings @ query
        order = np.argsort(-scores)[:top_k]

        results: list[dict] = []
        for i in order:
            doc = dict(self.docs[int(i)])
            doc["score"] = float(scores[int(i)])
            results.append(doc)
        return results

    def bm25_index(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.docs)
        return self._bm25

    # ---------- 持久化 ----------

    def save(self, path: Path = MEMORY_INDEX_PATH) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "docs.jsonl", "w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        np.save(path / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, path: Path = MEMORY_INDEX_PATH) -> "MemoryBackend":
        path = Path(path)
        docs_file = path / "docs.jsonl"
        emb_file = path / "embeddings.npy"
        if not docs_file.exists() or not emb_file.exists():
            raise FileNotFoundError(
                f"内存索引不存在: {path}\n"
                "先运行: python scripts/02_build_index.py --store memory"
            )
        with open(docs_file, encoding="utf-8") as f:
            docs = [json.loads(line) for line in f if line.strip()]
        return cls(docs, np.load(emb_file))


@contextmanager
def open_backend(store: str = "pg"):
    """统一入口：store="pg" 或 "memory"。"""
    if store == "memory":
        yield MemoryBackend.load()
        return

    from ..db import get_conn

    try:
        with get_conn() as conn:
            yield PgBackend(conn)
    except psycopg.OperationalError as exc:
        raise SystemExit(
            "连不上 Postgres（Docker 没启动？）\n"
            "  方案 A: docker compose up -d\n"
            "  方案 B: 改用内存索引，命令里加 --store memory\n"
            f"原始错误: {exc}"
        ) from exc
