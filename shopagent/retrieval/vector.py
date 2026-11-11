from __future__ import annotations

from pgvector import Vector

from ..db import rows_to_dicts


def vector_search(conn, query_embedding: list[float], top_k: int = 50) -> list[dict]:
    """pgvector HNSW 余弦召回。分数越大越相似（1 - cosine_distance）。

    参数必须包成 Vector：pgvector 的 psycopg 适配器只认 ndarray / Vector，
    传裸 list 会被当成 numeric[]，SQL 直接报类型错误。
    """
    vec = Vector(query_embedding)
    sql = """
        SELECT c.id           AS chunk_id,
               c.product_id   AS product_id,
               c.content      AS content,
               p.asin         AS asin,
               p.title        AS title,
               p.price        AS price,
               1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c
        JOIN products p ON p.id = c.product_id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (vec, vec, top_k))
        return rows_to_dicts(cur)
