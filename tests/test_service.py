"""HTTP 接口的回归用例。

不调真实 LLM：把 service.run_agent 换成一个固定返回值，只测接口层
（会话落盘、历史读取、删除、画像接口）。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import shopagent.memory as memory
import shopagent.service as service
from shopagent.config import settings


def _fake_agent(query, history=None, model=None):
    return {
        "answer": f"回显：{query}",
        "intent": "shop",
        "steps": 1,
        "trace": ["router → shop", "generate → 基于 1 条商品资料"],
        "profile": {},
        "products": [
            {
                "asin": "B093GNLXFN",
                "title": "Weanas Folding Heavy Bag Stand",
                "price": None,
            }
        ],
    }


def _fake_stream(query, history=None, model=None, **kwargs):
    yield ("step", "router → shop")
    yield ("token", "回显：")
    yield ("token", query)
    yield (
        "done",
        {
            "answer": f"回显：{query}",
            "intent": "shop",
            "steps": 1,
            "trace": ["router → shop"],
            "profile": {},
            "products": [
                {
                    "asin": "B093GNLXFN",
                    "title": "Weanas Folding Heavy Bag Stand",
                    "price": None,
                }
            ],
        },
    )


def _events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        event, data = "message", None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if data is not None:
            events.append((event, data))
    return events


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(memory, "PROFILE_PATH", tmp_path / "profile.json")
    monkeypatch.setattr(service, "run_agent", _fake_agent)

    app = service.create_app(store="memory", mode="bm25")
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["products"] > 0
    assert body["rerank_candidates"] == settings.rerank_candidates


def test_models_endpoint_lists_availability(client, monkeypatch):
    fake = [
        {"id": "deepseek", "label": "DeepSeek", "model": "deepseek-chat",
         "available": True, "detail": "云端 API 已配置", "latency_ms": None},
        {"id": "qwen", "label": "Qwen 本地", "model": "qwen3.8-27b",
         "available": False, "detail": "连接失败：ConnectError", "latency_ms": None},
    ]
    monkeypatch.setattr(service.llm, "model_statuses", lambda force=False: fake)

    body = client.get("/models").json()
    assert body["default"] == "deepseek"
    assert [item["id"] for item in body["models"]] == ["deepseek", "qwen"]
    assert body["models"][1]["available"] is False


def test_chat_passes_selected_model(client, monkeypatch):
    seen: list[str | None] = []

    def spy(query, history=None, model=None):
        seen.append(model)
        return _fake_agent(query, history)

    monkeypatch.setattr(service, "run_agent", spy)
    monkeypatch.setattr(
        service.llm, "model_status",
        lambda name, force=False: {"available": True, "detail": "已连接"},
    )

    body = client.post(
        "/chat", json={"message": "推荐瑜伽垫", "session_id": "m1", "model": "qwen"}
    ).json()
    assert seen == ["qwen"]
    assert body["model"] == "qwen"

    # 不传 model 时仍走默认的 DeepSeek
    client.post("/chat", json={"message": "再推荐一款", "session_id": "m1"})
    assert seen[-1] == "deepseek"


def test_qwen_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        service.llm, "model_status",
        lambda name, force=False: {"available": False, "detail": "连接失败：ConnectError"},
    )
    resp = client.post(
        "/chat", json={"message": "推荐瑜伽垫", "session_id": "m2", "model": "qwen"}
    )
    assert resp.status_code == 503
    assert "本地 Qwen 不可用" in resp.json()["detail"]


def test_rerank_candidates_hot_update(client):
    from shopagent.retrieval import current_rerank_candidates, set_rerank_candidates

    original = current_rerank_candidates()
    try:
        resp = client.post("/config/rerank_candidates", json={"value": 30})
        assert resp.status_code == 200
        assert resp.json()["rerank_candidates"] == 30
        assert client.get("/health").json()["rerank_candidates"] == 30

        # 越界值被拒绝，运行时的值不变
        assert client.post("/config/rerank_candidates", json={"value": 0}).status_code == 422
        assert client.post("/config/rerank_candidates", json={"value": 51}).status_code == 422
        assert current_rerank_candidates() == 30
    finally:
        set_rerank_candidates(original)


def test_chat_persists_session(client):
    resp = client.post("/chat", json={"message": "推荐瑜伽垫", "session_id": "t1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "回显：推荐瑜伽垫"
    assert body["session_id"] == "t1"
    assert body["steps"] == 1
    assert body["products"][0]["asin"] == "B093GNLXFN"

    history = client.get("/sessions/t1").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == "推荐瑜伽垫"
    assert history[1]["meta"]["products"][0]["asin"] == "B093GNLXFN"


def test_chat_second_turn_includes_history(client, monkeypatch):
    seen: list[list[dict]] = []

    def spy(query, history=None, model=None):
        seen.append(list(history or []))
        return _fake_agent(query, history)

    monkeypatch.setattr(service, "run_agent", spy)

    client.post("/chat", json={"message": "第一句", "session_id": "t2"})
    client.post("/chat", json={"message": "第二句", "session_id": "t2"})

    assert [len(messages) for messages in seen] == [0, 2]  # 第二轮拿到上一轮两条
    # meta 只给前端，不能混进给 LLM 的历史
    assert all("meta" not in message for messages in seen for message in messages)
    assert len(client.get("/sessions/t2").json()["messages"]) == 4


def test_use_history_false(client):
    client.post("/chat", json={"message": "第一句", "session_id": "t3"})
    # 关掉历史后提问，虽然会话文件里已有两条，但接口不应把它传给 agent；
    # 这里只验证接口不报错、会话继续追加
    resp = client.post(
        "/chat", json={"message": "第二句", "session_id": "t3", "use_history": False}
    )
    assert resp.status_code == 200
    assert len(client.get("/sessions/t3").json()["messages"]) == 4


def test_list_sessions(client):
    assert client.get("/sessions").json()["sessions"] == []

    client.post("/chat", json={"message": "推荐瑜伽垫", "session_id": "l1"})
    client.post("/chat", json={"message": "第一句", "session_id": "l2"})
    client.post("/chat", json={"message": "第二句", "session_id": "l2"})

    sessions = client.get("/sessions").json()["sessions"]
    by_id = {item["session_id"]: item for item in sessions}
    assert set(by_id) == {"l1", "l2"}
    # 标题取第一条用户消息，计数是落盘的消息数
    assert by_id["l1"]["title"] == "推荐瑜伽垫"
    assert by_id["l1"]["message_count"] == 2
    assert by_id["l2"]["message_count"] == 4
    assert by_id["l2"]["updated_at"]
    # 最近更新的排前面（同一秒内也不应该抛错，只校验集合）
    assert len(sessions) == 2


def test_delete_session(client):
    client.post("/chat", json={"message": "hi", "session_id": "t4"})
    assert client.delete("/sessions/t4").status_code == 200
    assert client.delete("/sessions/t4").status_code == 404


def test_profile_endpoints(client):
    assert client.get("/profile").json()["profile"] == {}
    assert client.delete("/profile").status_code == 200
    assert client.get("/profile").json()["profile"] == {}


def test_empty_message_rejected(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_stream_emits_events_and_persists(client, monkeypatch):
    monkeypatch.setattr(service, "stream_agent", _fake_stream)

    resp = client.post(
        "/chat/stream", json={"message": "推荐瑜伽垫", "session_id": "s1"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _events(resp.text)
    assert events[0] == ("step", {"line": "router → shop"})
    assert "".join(d["text"] for e, d in events if e == "token") == "回显：推荐瑜伽垫"

    done = [d for e, d in events if e == "done"][0]
    assert done["answer"] == "回显：推荐瑜伽垫"
    assert done["steps"] == 1
    assert done["products"][0]["asin"] == "B093GNLXFN"

    history = client.get("/sessions/s1").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["content"] == "回显：推荐瑜伽垫"
    assert history[1]["meta"]["products"][0]["asin"] == "B093GNLXFN"


def test_chat_stream_error_does_not_persist(client, monkeypatch):
    def failing(query, history=None, model=None, **kwargs):
        yield ("step", "router → shop")
        yield ("error", "模型超时")

    monkeypatch.setattr(service, "stream_agent", failing)

    resp = client.post(
        "/chat/stream", json={"message": "推荐瑜伽垫", "session_id": "s2"}
    )
    assert ("error", {"message": "模型超时"}) in _events(resp.text)
    # 半截答案不落盘，避免下一轮历史里出现一条失败的问题
    assert client.get("/sessions/s2").json()["messages"] == []


def test_frontend_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "ShopAgent" in resp.text
    assert "/static/app.js" in resp.text
    # 侧边栏的检索配置漏斗（精排候选数由 /health 填充）
    assert 'id="config-funnel"' in resp.text
    # 生成中可随时停止
    assert 'id="stop"' in resp.text and "结束生成" in resp.text


def test_static_assets_served(client):
    for path in ("/static/styles.css", "/static/app.js", "/static/favicon.svg"):
        resp = client.get(path)
        assert resp.status_code == 200
        # 部署后不能出现「HTML 新、CSS/JS 旧」的半新半旧页面
        assert resp.headers["cache-control"] == "no-cache"


def test_catalog_stats(client):
    body = client.get("/catalog/stats").json()
    assert body["total"] > 0
    assert body["priced"] + body["unpriced"] == body["total"]
    assert sum(bin["count"] for bin in body["histogram"]) == body["priced"]
    assert len(body["top_brands"]) <= 8
    assert body["median_price"] is not None


def test_catalog_products_pagination_filters_and_sort(client):
    page = client.get("/catalog/products", params={"page": 2, "page_size": 5}).json()
    assert page["page"] == 2
    assert page["page_size"] == 5
    assert len(page["items"]) == 5
    assert page["pages"] == (page["total"] + 4) // 5

    priced = client.get(
        "/catalog/products", params={"priced": "yes", "page_size": 60, "sort": "price_asc"}
    ).json()
    prices = [item["price"] for item in priced["items"]]
    assert all(p is not None for p in prices)
    assert prices == sorted(prices)

    unpriced = client.get("/catalog/products", params={"priced": "no"}).json()
    assert unpriced["total"] > 0
    assert all(item["price"] is None for item in unpriced["items"])

    ranged = client.get(
        "/catalog/products", params={"price_min": 25, "price_max": 50, "page_size": 60}
    ).json()
    assert ranged["total"] > 0
    assert all(25 <= item["price"] <= 50 for item in ranged["items"])


def test_catalog_products_search(client):
    hits = client.get("/catalog/products", params={"q": "yoga", "page_size": 12}).json()
    assert hits["total"] > 0
    assert hits["items"]
    # 命中藏在 features / description 时，卡片要带上含关键词的上下文
    for item in hits["items"]:
        visible = " ".join([item["title"], item["brand"], item["snippet"]]).lower()
        assert "yoga" in visible

    miss = client.get("/catalog/products", params={"q": "zzzz-not-a-product"}).json()
    assert miss["total"] == 0
    assert miss["pages"] == 0
    assert miss["items"] == []


def test_catalog_product_detail(client):
    # 商品库会换（当前是 10w 随机抽样），不能写死某个 ASIN
    from shopagent.catalog import get_product

    asin = client.get("/catalog/products", params={"page_size": 1}).json()["items"][0]["asin"]
    detail = client.get(f"/catalog/products/{asin}")
    row = get_product(asin)
    assert row is not None
    assert detail.status_code == 200
    body = detail.json()
    assert body["asin"] == asin
    assert body["title"] == row["title"]
    assert body["brand"] == row["brand"]
    assert body["price"] == row["price"]
    assert body["category"] == row["category"]

    assert client.get("/catalog/products/NOT-A-REAL-ASIN").status_code == 404


def test_catalog_invalid_args(client):
    assert client.get("/catalog/products", params={"page_size": 100}).status_code == 422
    assert (
        client.get(
            "/catalog/products", params={"price_min": 50, "price_max": 10}
        ).status_code
        == 400
    )
