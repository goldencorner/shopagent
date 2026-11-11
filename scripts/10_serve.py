"""启动 HTTP 服务。

用法:
    python scripts/10_serve.py                     # 默认 127.0.0.1:8000，pg 后端
    python scripts/10_serve.py --store memory
    python scripts/10_serve.py --port 8001

文档: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402

from shopagent.service import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--mode", default="hybrid_rerank")
    args = parser.parse_args()

    app = create_app(store=args.store, mode=args.mode)
    print(f"ShopAgent API -> http://{args.host}:{args.port}  (store={args.store}, mode={args.mode})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
