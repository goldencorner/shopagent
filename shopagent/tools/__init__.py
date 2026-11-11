"""工具注册表：一份 schema 同时喂给 DeepSeek function calling 和 MCP server。"""

from __future__ import annotations

import json

from . import shop

_TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "按用户需求检索商品，返回 ASIN、标题、价格和描述片段。"
                "适合『我想买 / 有没有 / 推荐』这类找货需求。"
                "结果里 price_unknown=true 表示该商品没有标价，回答时要注明「需自行确认价格」。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索词。商品库是英文，请把中文需求翻译成英文关键词，例如 '2 person camping tent'。",
                    },
                    "max_price": {
                        "type": "number",
                        "description": "价格上限（美元）。用户给了预算才传。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回条数，默认 5，最多 10。",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "查单个商品的完整信息（成分、用法、features）。需要判断某个商品是否合适时用。",
            "parameters": {
                "type": "object",
                "properties": {"asin": {"type": "string", "description": "商品 ASIN"}},
                "required": ["asin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar",
            "description": "找某个商品的同类替代品，常用于『有没有更便宜的』『还有别的牌子吗』。",
            "parameters": {
                "type": "object",
                "properties": {
                    "asin": {"type": "string", "description": "作为参照的商品 ASIN"},
                    "max_price": {"type": "number", "description": "预算上限（美元），可不传"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["asin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "把 2-4 个商品并排对比（价格、品牌、卖点）。用户问『哪个好 / 有什么区别』时用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "asins": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的 ASIN 列表，2-4 个",
                    }
                },
                "required": ["asins"],
            },
        },
    },
]


TOOL_FUNCS = {
    "search_products": shop.search_products,
    "get_product_detail": shop.get_product_detail,
    "find_similar": shop.find_similar,
    "compare_products": shop.compare_products,
}

# 给 OpenAI / DeepSeek 用
TOOL_SPECS = _TOOL_SPECS
TOOL_NAMES = list(TOOL_FUNCS)


def run_tool(name: str, arguments) -> dict:
    """执行工具，永远返回 dict，异常转成 {"error": ...} 交给模型自己处理。"""
    func = TOOL_FUNCS.get(name)
    if func is None:
        return {"error": f"未知工具: {name}", "available": TOOL_NAMES}

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"error": f"参数不是合法 JSON: {arguments!r}"}
    if not isinstance(arguments, dict):
        return {"error": "参数必须是对象"}

    try:
        result = func(**arguments)
    except TypeError as exc:
        return {"error": f"参数不匹配: {exc}"}
    except Exception as exc:  # 工具失败不该让整个 Agent 崩掉
        return {"error": f"{type(exc).__name__}: {exc}"}

    return {"ok": True, "result": result}
