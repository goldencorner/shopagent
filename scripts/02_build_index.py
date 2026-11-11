"""products.jsonl -> 分块 -> 向量化 -> 写入 pgvector 或内存索引

用法:
    python scripts/02_build_index.py                        # 写入 pgvector
    python scripts/02_build_index.py --reset                # 先清空表
    python scripts/02_build_index.py --recreate             # 按 schema.sql 重建表
    python scripts/02_build_index.py --store memory         # 不依赖 Docker，存到 data/memory_index/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.config import settings  # noqa: E402
from shopagent.embeddings import embed_texts  # noqa: E402
from shopagent.ingest import (  # noqa: E402
    build_chunk_docs,
    build_chunks,
    load_products,
    write_index,
)
from shopagent.retrieval.backend import MEMORY_INDEX_PATH, MemoryBackend  # noqa: E402


def memory_from_pg() -> None:
    """把 pg 里已经算好的向量搬进内存索引。

    大规模重建时 embedding 是最贵的一步（本机 CPU 约 31 chunk/s）。
    先建 pg 再复制向量，比再嵌一遍省掉一整轮。
    """
    import numpy as np

    from shopagent.db import get_conn

    t0 = time.time()
    print("从 pg 读取 chunks + embeddings ...")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.content, p.asin, p.title, p.price, c.embedding
            FROM chunks c JOIN products p ON p.id = c.product_id
            ORDER BY c.id
            """
        )
        rows = cur.fetchall()

    docs = [
        {
            "chunk_id": r[0],
            "content": r[1],
            "asin": r[2],
            "title": r[3],
            "price": None if r[4] is None else float(r[4]),
        }
        for r in rows
    ]
    # psycopg 的 pgvector 适配器返回 Vector 对象，np.asarray(Vector) 会报错，
    # 要用它自己的 to_numpy()。
    embeddings = np.vstack([r[5].to_numpy().astype(np.float32) for r in rows])
    MemoryBackend(docs, embeddings).save()
    print(f"内存索引已保存 -> {MEMORY_INDEX_PATH}")
    print(f"完成：{len(docs)} chunks / 向量 {embeddings.shape}，用时 {time.time() - t0:.1f}s")


def _embed_with_cache(
    input_path: Path, texts: list[str], batch_size: int, block_size: int = 25000
):
    """向量化结果按输入文件缓存到 data/_embeddings_*.npy。

    本机 CPU 约 25-31 chunk/s，16w chunk 要 90 分钟；写到一半失败（锁冲突、
    磁盘满、Ctrl-C）不应该再来一遍。缓存文件名带上输入文件大小和 mtime，
    换数据自动失效。

    超过 block_size 时分块落盘（*.partN.npy），中断后重跑只补未完成的块。
    """
    import numpy as np

    st = input_path.stat()
    cache = settings.data_dir / f"_embeddings_{st.st_size}_{int(st.st_mtime)}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(texts):
            print(f"复用向量缓存 {cache.name}，shape={arr.shape}")
            return arr
        print(f"缓存条数不匹配（{arr.shape[0]} vs {len(texts)}），重新 embedding")

    if len(texts) <= block_size:
        arr = embed_texts(texts, batch_size=batch_size)
        np.save(cache, arr)
        print(f"向量已缓存 -> {cache.name}，shape={arr.shape}")
        return arr

    parts: list[np.ndarray] = []
    n_blocks = (len(texts) + block_size - 1) // block_size
    for i in range(n_blocks):
        block = texts[i * block_size : (i + 1) * block_size]
        part = cache.with_name(f"{cache.stem}.part{i}.npy")
        if part.exists():
            arr = np.load(part)
            if arr.shape[0] == len(block):
                print(f"复用分块 {i + 1}/{n_blocks}: {part.name}")
                parts.append(arr)
                continue
        print(f"embedding 分块 {i + 1}/{n_blocks}（{len(block)} 条）...")
        arr = embed_texts(block, batch_size=batch_size)
        np.save(part, arr)
        parts.append(arr)
        print(f"分块 {i + 1}/{n_blocks} 完成，shape={arr.shape}")

    arr = np.vstack(parts)
    np.save(cache, arr)
    for i in range(n_blocks):
        cache.with_name(f"{cache.stem}.part{i}.npy").unlink(missing_ok=True)
    print(f"向量已缓存 -> {cache.name}，shape={arr.shape}")
    return arr


def run_sql_file(conn, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                cur.execute(statement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=settings.data_dir / "products.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--block-size",
        type=int,
        default=25000,
        help="向量化分块落盘大小，中断后只补未完成的块",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="只算向量并写缓存，不建索引（大语料先在快机器上跑完再搬）",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="直接读入已算好的 .npy（服务器复用本机向量，不重新 embedding）",
    )
    parser.add_argument(
        "--skip-fts",
        action="store_true",
        help="不建磁盘版 BM25（FTS5）；默认建，大语料下内存版会 OOM",
    )
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--reset", action="store_true", help="清空两张表后重新写入")
    parser.add_argument("--recreate", action="store_true", help="按 schema.sql 重建表")
    parser.add_argument(
        "--from-pg",
        action="store_true",
        help="--store memory 专用：直接从 pg 里已有的向量生成内存索引，不重新 embedding",
    )
    args = parser.parse_args()

    if args.from_pg:
        if args.store != "memory":
            parser.error("--from-pg 只能配合 --store memory")
        memory_from_pg()
        return

    if not args.input.exists():
        raise SystemExit(f"找不到 {args.input}，先跑 scripts/01_load_data.py")

    t0 = time.time()
    print(f"读取 {args.input} ...")
    products = load_products(args.input)
    chunks = build_chunks(products)
    print(f"商品 {len(products)} 条，chunk {len(chunks)} 个")

    if args.embeddings is not None:
        import numpy as np

        embeddings = np.load(args.embeddings)
        if embeddings.shape[0] != len(chunks):
            raise SystemExit(
                f"向量条数 {embeddings.shape[0]} 与 chunk 数 {len(chunks)} 不一致："
                f"{args.embeddings}"
            )
        print(f"读入预计算向量 {args.embeddings}，shape={embeddings.shape}")
    else:
        print(f"向量化（模型 {settings.embed_model}，首次会下载）...")
        embeddings = _embed_with_cache(
            args.input,
            [c["content"] for c in chunks],
            args.batch_size,
            args.block_size,
        )
    print(f"向量维度 {embeddings.shape[1]}，耗时 {time.time() - t0:.1f}s")

    if args.embed_only:
        print("--embed-only：向量已写缓存，跳过建索引")
        return

    if embeddings.shape[1] != settings.embed_dim:
        raise SystemExit(
            f"向量维度 {embeddings.shape[1]} 与 EMBED_DIM={settings.embed_dim} 不一致。\n"
            "改 .env 里的 EMBED_DIM，同时把 sql/schema.sql 的 vector(...) 改成同样的维度，"
            "再用 --recreate 重建。"
        )

    if args.store == "memory":
        docs = build_chunk_docs(products, chunks)
        MemoryBackend(docs, embeddings).save()
        print(f"\n内存索引已保存 -> {MEMORY_INDEX_PATH}")
        print(f"完成：{len(products)} 商品 / {len(chunks)} chunks，用时 {time.time() - t0:.1f}s")
        return

    from shopagent.db import get_conn

    # 准备阶段单独提交：TRUNCATE 要拿 ACCESS EXCLUSIVE 锁，如果和后面的长写入
    # 共用一个事务，别的进程一条普通 SELECT 就可能和它死锁（实测踩过，整轮重建
    # 在写完 embedding 后被回滚）。这里让锁只持有几毫秒。
    with get_conn() as conn:
        if args.recreate:
            print("按 sql/schema.sql 重建表 ...")
            run_sql_file(conn, settings.root_dir / "sql" / "schema.sql")
        elif args.reset:
            print("清空表 ...")
            with conn.cursor() as cur:
                cur.execute("TRUNCATE chunks, products RESTART IDENTITY CASCADE")
        else:
            # products 有 upsert，chunks 没有：不加 --reset 重跑会把每个 chunk 插两遍
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM chunks")
                existing = cur.fetchone()[0]
            if existing:
                raise SystemExit(
                    f"chunks 表里已有 {existing} 条记录，直接写入会重复。\n"
                    "  重跑请加 --reset（清表后重写）或 --recreate（按 schema.sql 重建）"
                )

    with get_conn() as conn:
        # 大批量写入时索引要最后建：边插边维护 HNSW 会把写入拖成几倍，
        # 而且 16w 条规模的建索引本身比逐条插入便宜得多。
        print("写入数据库（先删 HNSW，写完再重建）...")
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw")
        t_write = time.time()
        write_index(conn, products, chunks, embeddings)
        print(f"写入完成，{time.time() - t_write:.1f}s；重建 HNSW ...")

        t_idx = time.time()
        with conn.cursor() as cur:
            # 默认 maintenance_work_mem 只有 64MB，16w 条建 HNSW 会一直落盘。
            # 512MB 在 4G 机器上够用（建索引时 API 是停的），本地更快。
            cur.execute("SET maintenance_work_mem = '512MB'")
            # 并行建索引走 /dev/shm，Docker 默认只给 64MB，16w 条会报
            # "could not resize shared memory segment ... No space left on device"。
            # 单进程建 HNSW 把内存放在后端私有内存里，慢一点但不会撞这个限制。
            cur.execute("SET max_parallel_maintenance_workers = 0")
            cur.execute(
                """
                CREATE INDEX chunks_embedding_hnsw ON chunks
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """
            )
        print(f"HNSW 建好，{time.time() - t_idx:.1f}s")

        if not args.skip_fts:
            from shopagent.retrieval.fts import FTS_INDEX_PATH, FTSIndex

            print("建磁盘版 BM25（SQLite FTS5）...")
            t_fts = time.time()
            FTSIndex.build(conn)
            size_mb = FTS_INDEX_PATH.stat().st_size / 2**20
            print(f"FTS5 建好，{time.time() - t_fts:.1f}s，{size_mb:.0f} MB -> {FTS_INDEX_PATH}")

    print(f"\n完成：{len(products)} 商品 / {len(chunks)} chunks，用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
