"""磁盘版 BM25：SQLite FTS5。

rank_bm25 把语料全部 tokenize 进 Python 对象，166k chunk 实测占 2.6GB，
4G 的部署机上和模型一起必然 OOM。FTS5 的 bm25() 是同一套 BM25 公式，
倒排索引放在磁盘上，进程内存只占几 MB，查询也是毫秒级。

小语料（几千条）继续用 bm25.BM25Index，省掉建文件这一步；
backend 检测到 FTS 文件就优先用它。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..config import settings

FTS_INDEX_PATH = settings.data_dir / "fts_index.db"

_CREATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    product_id UNINDEXED,
    asin UNINDEXED,
    title UNINDEXED,
    price UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
)
"""

_INSERT = """
INSERT INTO chunks_fts (rowid, content, product_id, asin, title, price)
VALUES (?, ?, ?, ?, ?, ?)
"""

_QUERY = """
SELECT rowid, product_id, asin, title, price, content, bm25(chunks_fts) AS rank
FROM chunks_fts
WHERE chunks_fts MATCH ?
ORDER BY rank
LIMIT ?
"""


class FTSIndex:
    """与 BM25Index.search 同接口，Retriever 不需要区分两者。"""

    def __init__(self, path: Path = FTS_INDEX_PATH):
        self.path = Path(path)

    def available(self) -> bool:
        return self.path.exists()

    @classmethod
    def build(cls, conn, path: Path = FTS_INDEX_PATH) -> "FTSIndex":
        """从 pg 的 chunks 表建 FTS5 索引，先写临时文件再原子替换。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".building.db")
        tmp.unlink(missing_ok=True)

        db = sqlite3.connect(tmp)
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute(_CREATE)

            # 服务端游标：16w 行不一次性拉进内存
            with conn.cursor(name="fts_build") as cur:
                cur.itersize = 5000
                cur.execute(
                    """
                    SELECT c.id, c.product_id, c.content, p.asin, p.title, p.price
                    FROM chunks c JOIN products p ON p.id = c.product_id
                    """
                )
                batch = []
                for chunk_id, product_id, content, asin, title, price in cur:
                    batch.append(
                        (
                            int(chunk_id),
                            content,
                            int(product_id),
                            asin,
                            title,
                            None if price is None else float(price),
                        )
                    )
                    if len(batch) >= 5000:
                        db.executemany(_INSERT, batch)
                        batch.clear()
                if batch:
                    db.executemany(_INSERT, batch)

            # 合并段，避免查询时扫一堆小段
            db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            db.commit()
        finally:
            db.close()

        tmp.replace(path)
        return cls(path)

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        from .bm25 import tokenize

        tokens = tokenize(query)
        if not tokens:
            return []
        # 每个词加引号，避免用户输入里的 AND/OR/NEAR/* 被当成 FTS5 语法
        match = " OR ".join(f'"{t}"' for t in tokens)

        # 每次查询单独开连接：FastAPI 会在线程池里并发调用，sqlite 连接不跨线程共享
        try:
            db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        except sqlite3.Error:
            return []
        try:
            rows = db.execute(_QUERY, (match, top_k)).fetchall()
        except sqlite3.Error:
            return []
        finally:
            db.close()

        results: list[dict] = []
        for chunk_id, product_id, asin, title, price, content, rank in rows:
            results.append(
                {
                    "chunk_id": int(chunk_id),
                    "product_id": int(product_id),
                    "asin": asin,
                    "title": title,
                    "price": None if price is None else float(price),
                    "content": content,
                    # FTS5 的 bm25() 越小越相关，翻成和 rank_bm25 一致的「越大越好」
                    "score": -float(rank),
                }
            )
        return results
