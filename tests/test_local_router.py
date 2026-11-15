"""小模型 router 的解析与回退逻辑（不依赖真实本地服务）。"""

from __future__ import annotations

import pytest

from shopagent.agent import nodes
from shopagent.local_router import parse_output


@pytest.mark.parametrize(
    "text",
    [
        '{"intent": "shop", "en_query": "camping tent"}',
        '说明文字\n{"intent": "compare", "en_query": "a vs b"}\n后面还有话',
    ],
)
def test_parse_output_accepts_json_with_noise(text):
    result = parse_output(text)
    assert result["intent"] in {"shop", "compare"}
    assert result["en_query"]


def test_parse_output_rejects_unknown_intent():
    with pytest.raises(ValueError):
        parse_output('{"intent": "buy", "en_query": "x"}')


def test_router_local_falls_back_when_model_unavailable(monkeypatch):
    monkeypatch.setenv("SHOPAGENT_ROUTER", "local")
    monkeypatch.setattr(
        nodes, "router_status", lambda: {"available": False, "detail": "模型未加载"}
    )
    monkeypatch.setattr(nodes, "chat_json", lambda messages: {"intent": "compare"})
    result = nodes.router({"query": "这两个哪个好"})
    assert result["intent"] == "compare"
    assert result["rewritten_query"] == ""
    assert "回退 llm" in result["trace"][0]


def test_router_local_falls_back_on_low_confidence(monkeypatch):
    monkeypatch.setenv("SHOPAGENT_ROUTER", "local")
    monkeypatch.setattr(nodes, "router_status", lambda: {"available": True, "detail": "ok"})
    monkeypatch.setattr(
        nodes,
        "route_local",
        lambda query: {"intent": "shop", "en_query": "yoga mat", "confidence": 0.2},
    )
    monkeypatch.setattr(nodes, "chat_json", lambda messages: {"intent": "chitchat"})
    result = nodes.router({"query": "谢谢，我先看看瑜伽垫"})
    assert result["intent"] == "chitchat"
    assert result["rewritten_query"] == ""
    assert "置信度" in result["trace"][0]


def test_router_local_uses_rewritten_query(monkeypatch):
    monkeypatch.setenv("SHOPAGENT_ROUTER", "local")
    monkeypatch.setattr(nodes, "router_status", lambda: {"available": True, "detail": "ok"})
    monkeypatch.setattr(
        nodes,
        "route_local",
        lambda query: {"intent": "shop", "en_query": "2 person camping tent", "confidence": 0.93},
    )
    result = nodes.router({"query": "推荐一款适合两个人用的露营帐篷"})
    assert result["intent"] == "shop"
    assert result["rewritten_query"] == "2 person camping tent"
    assert any("router 改写" in line for line in result["trace"])
