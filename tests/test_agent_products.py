"""generate 节点要带出商品引用，前端才能把答案里的标题链到商品库。"""

from __future__ import annotations

import shopagent.agent.nodes as nodes


def test_generate_returns_product_refs(monkeypatch):
    monkeypatch.setattr(
        nodes, "_chat_or_stream", lambda messages, max_tokens: "根据资料推荐这款。"
    )

    state = {
        "query": "推荐一款商品",
        "profile": {},
        "tool_results": [
            {
                "tool": "get_product_detail",
                "args": {"asin": "B093GNLXFN"},
                "output": {
                    "result": {
                        "asin": "B093GNLXFN",
                        "title": "Weanas Folding Heavy Bag Stand",
                        "price": None,
                        "content": "商品描述",
                        "features": "卖点",
                    }
                },
            }
        ],
    }

    out = nodes.generate(state)
    assert out["answer"] == "根据资料推荐这款。"
    assert out["products"] == [
        {
            "asin": "B093GNLXFN",
            "title": "Weanas Folding Heavy Bag Stand",
            "price": None,
        }
    ]
