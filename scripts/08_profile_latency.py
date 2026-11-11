"""检索链路各阶段耗时分解。用于回答「瓶颈在哪」「要不要上 GPU」。

用法:
    python scripts/08_profile_latency.py                # pgvector
    python scripts/08_profile_latency.py --store memory # 内存后端
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.embeddings import embed_texts  # noqa: E402
from shopagent.retrieval import RERANK_CANDIDATES  # noqa: E402
from shopagent.retrieval.backend import open_backend  # noqa: E402
from shopagent.retrieval.fusion import dedupe_by_asin, rrf  # noqa: E402
from shopagent.retrieval.rerank import rerank  # noqa: E402

K = 50  # candidate pool


def chunk_count(backend) -> int:
    if hasattr(backend, "docs"):
        return len(backend.docs)
    with backend.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks")
        return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--eval", type=Path, default=ROOT / "eval" / "retrieval_eval.jsonl")
    args = parser.parse_args()

    queries = []
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line)["query"])

    with open_backend(args.store) as backend:
        bm25 = backend.bm25_index()

        # 预热：首次调用要加载模型 / 建 BM25
        embed_texts(["warmup"])
        bm25.search("warmup", K)
        _emb = embed_texts(["warmup"])[0].tolist()
        backend.vector_search(_emb, K)
        rerank("warmup", backend.vector_search(_emb, K)[:RERANK_CANDIDATES], top_k=5)

        t = {"embed": [], "bm25": [], "vector": [], "rrf": [], "rerank": []}
        for q in queries:
            s = time.perf_counter()
            emb = embed_texts([q])[0].tolist()
            t["embed"].append((time.perf_counter() - s) * 1000)

            s = time.perf_counter()
            bm = bm25.search(q, K)
            t["bm25"].append((time.perf_counter() - s) * 1000)

            s = time.perf_counter()
            vec = backend.vector_search(emb, K)
            t["vector"].append((time.perf_counter() - s) * 1000)

            s = time.perf_counter()
            fused = dedupe_by_asin(rrf([vec, bm]), limit=K)
            t["rrf"].append((time.perf_counter() - s) * 1000)

            s = time.perf_counter()
            rerank(q, fused[:RERANK_CANDIDATES], top_k=5)
            t["rerank"].append((time.perf_counter() - s) * 1000)

        print(f"backend={args.store}  queries={len(queries)}  chunks={chunk_count(backend)}")
        print(f"{'stage':10s}{'mean ms':>10s}{'p95 ms':>10s}")
        total = 0.0
        for k, v in t.items():
            v_sorted = sorted(v)
            p95 = v_sorted[int(len(v_sorted) * 0.95) - 1]
            total += statistics.mean(v)
            print(f"{k:10s}{statistics.mean(v):>10.2f}{p95:>10.2f}")
        print(f"{'合计':10s}{total:>10.2f}")


if __name__ == "__main__":
    main()
