"""把导购工具暴露成 MCP server（stdio），可直接挂到 Claude Desktop / Cursor。

用法（Claude Desktop 的 claude_desktop_config.json）:

    "mcpServers": {
      "shopagent": {
        "command": "C:\\\\Users\\\\asf\\\\Desktop\\\\agent-mvp\\\\.venv313\\\\Scripts\\\\python.exe",
        "args": ["C:\\\\Users\\\\asf\\\\Desktop\\\\agent-mvp\\\\scripts\\\\07_mcp_server.py"],
        "env": { "SHOPAGENT_STORE": "memory", "SHOPAGENT_MODE": "hybrid_rerank" }
      }
    }

注意：MCP 用 stdout 传协议，这个文件里不能有任何 print。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from shopagent.tools import shop  # noqa: E402

mcp = FastMCP("shopagent")

STORE = os.getenv("SHOPAGENT_STORE", "memory")
MODE = os.getenv("SHOPAGENT_MODE", "hybrid_rerank")


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


@mcp.tool()
def search_products(query: str, max_price: float = 0.0, top_k: int = 5) -> str:
    """按需求检索商品。query 用英文关键词效果最好（商品库是英文）。

    Args:
        query: 检索词，例如 "2 person camping tent"
        max_price: 价格上限（美元），0 表示不限
        top_k: 返回条数，默认 5
    """
    shop.configure(store=STORE, mode=MODE)
    limit = max_price if max_price and max_price > 0 else None
    return _dumps(shop.search_products(query, max_price=limit, top_k=top_k))


@mcp.tool()
def get_product_detail(asin: str) -> str:
    """查单个商品的完整信息（描述、卖点、价格）。需要判断某商品是否合适时用。"""
    shop.configure(store=STORE, mode=MODE)
    return _dumps(shop.get_product_detail(asin))


@mcp.tool()
def find_similar(asin: str, max_price: float = 0.0, top_k: int = 5) -> str:
    """找某个商品的同类替代品，用于「有没有更便宜的」「还有别的牌子吗」。"""
    shop.configure(store=STORE, mode=MODE)
    limit = max_price if max_price and max_price > 0 else None
    return _dumps(shop.find_similar(asin, max_price=limit, top_k=top_k))


@mcp.tool()
def compare_products(asins: list[str]) -> str:
    """并排对比 2-4 个商品（价格、品牌、卖点）。"""
    shop.configure(store=STORE, mode=MODE)
    return _dumps(shop.compare_products(asins))


if __name__ == "__main__":
    mcp.run()  # 默认 stdio
