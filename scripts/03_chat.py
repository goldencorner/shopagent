"""命令行 RAG 问答，用于快速验证检索链路。

用法:
    python scripts/03_chat.py --mode vector
    python scripts/03_chat.py --mode hybrid_rerank --topk 5
    python scripts/03_chat.py --store memory --mode hybrid_rerank   # 不用 Docker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.rag import answer  # noqa: E402
from shopagent.retrieval import MODES, Retriever  # noqa: E402
from shopagent.retrieval.backend import open_backend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--no-sources", action="store_true")
    args = parser.parse_args()

    with open_backend(args.store) as backend:
        retriever = Retriever(backend, mode=args.mode)
        print(f"后端: {args.store} | 模式: {args.mode} | 输入 exit 退出\n")

        while True:
            try:
                query = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                continue

            docs = retriever.search(query, top_k=args.topk)
            if not docs:
                print("助手> 没有检索到相关商品。\n")
                continue

            if not args.no_sources:
                print("\n召回:")
                for i, doc in enumerate(docs, start=1):
                    score = doc.get("rerank_score", doc.get("score", 0.0))
                    title = (doc.get("title") or "")[:70]
                    print(f"  [{i}] {title}  (score={score:.4f})")
                print()

            print("助手>", answer(query, docs), "\n")


if __name__ == "__main__":
    main()
