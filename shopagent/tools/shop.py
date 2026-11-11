"""导购工具的底层实现。工具层只依赖 Retriever + catalog，不关心 Agent 怎么编排。

供两处复用：
- shopagent/agent/ 节点里的 tool_exec
- scripts/07_mcp_server.py（MCP 复用同一批函数）
"""

from __future__ import annotations

import contextlib

from ..catalog import brief, get_product
from ..retrieval import Retriever
from ..retrieval.backend import open_backend

MAX_TOP_K = 10

_STATE: dict = {"stack": None, "retriever": None, "store": None, "mode": None}


def configure(store: str = "memory", mode: str = "hybrid_rerank") -> Retriever:
    """打开后端并常驻。同一进程内重复调用是幂等的。"""
    if _STATE["retriever"] is not None and _STATE["store"] == store and _STATE["mode"] == mode:
        return _STATE["retriever"]

    close()
    stack = contextlib.ExitStack()
    backend = stack.enter_context(open_backend(store))
    _STATE.update(
        stack=stack,
        retriever=Retriever(backend, mode=mode),
        store=store,
        mode=mode,
    )
    return _STATE["retriever"]


def close() -> None:
    if _STATE["stack"] is not None:
        _STATE["stack"].close()
    _STATE.update(stack=None, retriever=None)


def _retriever() -> Retriever:
    if _STATE["retriever"] is None:
        configure()
    return _STATE["retriever"]


def search_products(query: str, max_price: float | None = None, top_k: int = 5) -> list[dict]:
    """语义+关键词混合检索，可选价格上限。

    预算只对**已知价格**生效：Amazon 元数据里约 46.5% 的商品没标价（3000 条中 1394 条），把 price=None
    一起丢掉，预算查询就只剩几条不相关结果（真实案例：预算 $20 的瑜伽垫只剩
    背带、瑜伽包、护膝这类配件）。未标价的照常返回，打 price_unknown，由生成层说明需自行确认。
    给了预算时，预算内且价格已知的排在前面。
    """
    top_k = max(1, min(int(top_k or 5), MAX_TOP_K))
    # 有价格过滤时多召回一些，避免过滤后不够数
    candidate_k = top_k * 4 if max_price is not None else top_k
    hits = _retriever().search(query, top_k=candidate_k)

    kept: list[dict] = []
    priced: list[dict] = []
    unpriced: list[dict] = []
    for hit in hits:
        price = hit.get("price")
        if max_price is not None and price is not None and float(price) > float(max_price):
            continue
        item = {
            "asin": hit["asin"],
            "title": hit.get("title") or "",
            "price": price,
            "price_unknown": price is None,
            "snippet": (hit.get("content") or "")[:300],
            "score": round(float(hit.get("rerank_score", hit.get("score", 0.0))), 4),
        }
        kept.append(item)
        (unpriced if price is None else priced).append(item)

    if max_price is None:
        return kept[:top_k]

    # 用户报了预算：先把确定买得起的给他，未标价的放后面并标记
    return (priced + unpriced)[:top_k]


def get_product_detail(asin: str) -> dict:
    product = get_product(asin)
    if product is None:
        return {"error": f"没有这个 ASIN: {asin}"}
    return {
        "asin": product["asin"],
        "title": product.get("title") or "",
        "brand": product.get("brand"),
        "price": product.get("price"),
        "description": (product.get("description") or "")[:1500],
        "features": product.get("features") or "",
    }


def compare_products(asins: list[str]) -> list[dict]:
    """并排对比 2-4 个商品。"""
    if not asins:
        return []
    out: list[dict] = []
    for asin in asins[:4]:
        product = get_product(asin)
        if product is None:
            out.append({"asin": asin, "error": "不存在"})
            continue
        item = brief(product)
        item["features"] = (product.get("features") or "")[:400]
        out.append(item)
    return out


def find_similar(asin: str, max_price: float | None = None, top_k: int = 5) -> list[dict]:
    """找同类替代品：拿商品自身标题+描述当查询。"""
    product = get_product(asin)
    if product is None:
        return [{"error": f"没有这个 ASIN: {asin}"}]

    query = f"{product.get('title') or ''} {(product.get('description') or '')[:300]}".strip()
    hits = search_products(query, max_price=max_price, top_k=top_k + 1)
    return [h for h in hits if h["asin"] != asin][:top_k]
