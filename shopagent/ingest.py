from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .embeddings import embed_texts


# ---------- 读取 ----------

def load_products(path: str | Path) -> list[dict]:
    products: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    return products


# ---------- 组装文本 ----------

def product_to_text(p: dict) -> str:
    """把一个商品拍平成一段可检索文本。字段顺序会影响检索效果，别乱调。"""
    parts = [
        f"Title: {p.get('title') or ''}",
        f"Brand: {p.get('brand') or 'Unknown'}",
        f"Category: {p.get('category') or 'Unknown'}",
    ]
    if p.get("price") is not None:
        parts.append(f"Price: ${float(p['price']):.2f}")
    if p.get("features"):
        parts.append(f"Features: {p['features']}")
    if p.get("description"):
        parts.append(f"Description: {p['description']}")
    return "\n".join(parts)


def split_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(products: Iterable[dict]) -> list[dict]:
    chunks: list[dict] = []
    for p in products:
        text = product_to_text(p)
        for i, piece in enumerate(split_text(text)):
            chunks.append({"asin": p["asin"], "chunk_index": i, "content": piece})
    return chunks


def build_chunk_docs(products: list[dict], chunks: list[dict]) -> list[dict]:
    """内存后端用：把商品元数据平铺到每个 chunk 上，字段与 DB 查询结果对齐。"""
    by_asin = {p["asin"]: p for p in products}
    docs: list[dict] = []
    for i, chunk in enumerate(chunks):
        product = by_asin[chunk["asin"]]
        docs.append(
            {
                "chunk_id": i,
                "asin": chunk["asin"],
                "title": product["title"],
                "price": product["price"],
                "content": chunk["content"],
            }
        )
    return docs


# ---------- 入库 ----------

def write_index(
    conn,
    products: list[dict],
    chunks: list[dict],
    embeddings,
    batch_size: int = 500,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks({len(chunks)}) 与 embeddings({len(embeddings)}) 数量不一致")

    # pgvector 只给 ndarray / Vector 注册了 dumper，裸 list 会被当成 numeric[]
    from pgvector import Vector

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO products (asin, title, brand, category, price, description, features)
            VALUES (%(asin)s, %(title)s, %(brand)s, %(category)s, %(price)s,
                    %(description)s, %(features)s)
            ON CONFLICT (asin) DO UPDATE SET
                title = EXCLUDED.title,
                brand = EXCLUDED.brand,
                category = EXCLUDED.category,
                price = EXCLUDED.price,
                description = EXCLUDED.description,
                features = EXCLUDED.features
            """,
            products,
        )
        asins = [p["asin"] for p in products]
        cur.execute("SELECT asin, id FROM products WHERE asin = ANY(%s)", (asins,))
        id_map = {asin: pid for asin, pid in cur.fetchall()}

    with conn.cursor() as cur:
        # 按批生成行，不一次性建 rows 列表：16w chunk 时每行 384 个 Python float
        # 会让 rows 本身吃掉 1.5GB 内存，而 executemany 本来就是一个批一个批发。
        for i in range(0, len(chunks), batch_size):
            batch = [
                (
                    id_map[chunk["asin"]],
                    chunk["chunk_index"],
                    chunk["content"],
                    Vector([float(x) for x in embeddings[j]]),
                )
                for j, chunk in enumerate(chunks[i : i + batch_size], start=i)
            ]
            cur.executemany(
                """
                INSERT INTO chunks (product_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                batch,
            )


def rebuild_index(products_path: str | Path, batch_size: int = 32) -> tuple[int, int]:
    """products.jsonl -> chunks -> embeddings -> DB。返回 (商品数, chunk 数)。"""
    from .db import get_conn

    products = load_products(products_path)
    chunks = build_chunks(products)
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts, batch_size=batch_size)

    with get_conn() as conn:
        write_index(conn, products, chunks, embeddings)

    return len(products), len(chunks)
