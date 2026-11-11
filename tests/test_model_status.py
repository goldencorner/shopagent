"""本地 Qwen 可用性探测的回归用例。

状态分两段：先问 qwen-proxy 反向 SSH 隧道是否连通，再读 LM Studio 原生
/api/v0/models 判断模型是否加载。LM Studio 的 /v1/models 会把已下载但未
加载的模型也列出来，所以第二段必须读 state 字段。探测只发 GET，不触发
模型加载。
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import shopagent.llm as llm


def _stub_settings(**overrides):
    values = dict(
        deepseek_api_key="sk-test",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        qwen_base_url="http://127.0.0.1:1234/v1",
        qwen_api_key="lm-studio",
        qwen_model="",
        qwen_timeout=30.0,
        qwen_health_url="http://127.0.0.1:1235/health",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _response(payload, status_code=200, url="http://test"):
    return httpx.Response(
        status_code, json=payload, request=httpx.Request("GET", url)
    )


def _health(tunnel=True, status_code=200):
    return _response(
        {"tunnel": tunnel, "latency_ms": 1}, status_code, "http://health"
    )


def _models(payload, status_code=200):
    return _response(payload, status_code, "http://models")


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    llm._STATUS_CACHE.clear()
    llm._QWEN_MODEL_CACHE.update({"id": "", "at": 0.0})
    monkeypatch.setattr(llm, "settings", _stub_settings())


def test_loaded_model_is_available(monkeypatch):
    urls: list[str] = []

    def fake_get(url, **kwargs):
        urls.append(url)
        if url.endswith("/health"):
            return _health()
        return _models(
            {
                "data": [
                    {"id": "qwen3.8-27b", "state": "loaded"},
                    {"id": "qwen-small", "state": "not-loaded"},
                ]
            }
        )

    monkeypatch.setattr(llm.httpx, "get", fake_get)
    llm.reset_calls()

    status = llm.model_status("qwen", force=True)

    assert status["available"] is True
    assert status["model"] == "qwen3.8-27b"
    assert status["detail"] == "隧道已连通，模型已加载"
    assert status["failed_stage"] is None
    assert [check["id"] for check in status["checks"]] == ["tunnel", "model"]
    assert all(check["ok"] is True for check in status["checks"])
    assert urls == [
        "http://127.0.0.1:1235/health",
        "http://127.0.0.1:1234/api/v0/models",
    ]
    # 只读查询，不调 completion；LM Studio 不会因此加载模型
    assert llm.call_count() == 0


def test_tunnel_down_skips_model_check(monkeypatch):
    urls: list[str] = []

    def fake_get(url, **kwargs):
        urls.append(url)
        if url.endswith("/health"):
            return _health(False, 503)
        raise AssertionError("隧道断了就不该再请求模型接口")

    monkeypatch.setattr(llm.httpx, "get", fake_get)

    status = llm.model_status("qwen", force=True)

    assert status["available"] is False
    assert status["failed_stage"] == "tunnel"
    assert "SSH 隧道未连通" in status["detail"]
    assert urls == ["http://127.0.0.1:1235/health"]
    tunnel, model = status["checks"]
    assert tunnel["ok"] is False
    assert model["ok"] is None
    assert "未检测" in model["detail"]


def test_tunnel_ok_but_model_not_loaded(monkeypatch):
    def fake_get(url, **kwargs):
        if url.endswith("/health"):
            return _health()
        return _models({"data": [{"id": "qwen3.8-27b", "state": "not-loaded"}]})

    monkeypatch.setattr(llm.httpx, "get", fake_get)

    status = llm.model_status("qwen", force=True)

    assert status["available"] is False
    assert status["failed_stage"] == "model"
    assert "隧道已连通，但没有已加载的 Qwen 模型" in status["detail"]
    assert status["checks"][1]["ok"] is False


def test_health_endpoint_unreachable_falls_back(monkeypatch):
    """本地开发没起 qwen-proxy：健康检查不可达不应直接把 Qwen 判死。"""

    def fake_get(url, **kwargs):
        if url.endswith("/health"):
            raise httpx.ConnectError("connection refused")
        return _models({"data": [{"id": "qwen3.8-27b", "state": "loaded"}]})

    monkeypatch.setattr(llm.httpx, "get", fake_get)

    status = llm.model_status("qwen", force=True)

    assert status["available"] is True
    assert status["failed_stage"] is None
    assert status["checks"][0]["ok"] is None
    assert "健康检查不可达" in status["checks"][0]["detail"]
    assert status["checks"][1]["ok"] is True


def test_configured_model_must_be_loaded(monkeypatch):
    monkeypatch.setattr(llm, "settings", _stub_settings(qwen_model="qwen3.8-27b"))

    def fake_get(url, **kwargs):
        if url.endswith("/health"):
            return _health()
        return _models({"data": [{"id": "qwen3.8-27b", "state": "not-loaded"}]})

    monkeypatch.setattr(llm.httpx, "get", fake_get)

    status = llm.model_status("qwen", force=True)

    assert status["available"] is False
    assert "模型 qwen3.8-27b 未加载" in status["detail"]


def test_openai_only_server_falls_back_to_model_list(monkeypatch):
    """vLLM / Ollama 没有 /api/v0/models，退回 /v1/models 并注明状态未确认。"""

    def fake_get(url, **kwargs):
        if url.endswith("/health"):
            return _health()
        if url.endswith("/api/v0/models"):
            return _models({"error": "not found"}, 404)
        return _models({"data": [{"id": "qwen2.5-7b"}]})

    monkeypatch.setattr(llm.httpx, "get", fake_get)

    status = llm.model_status("qwen", force=True)

    assert status["available"] is True
    assert status["model"] == "qwen2.5-7b"
    assert "未返回加载状态" in status["detail"]


def test_connection_error_reports_unavailable(monkeypatch):
    def boom(url, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.httpx, "get", boom)

    status = llm.model_status("qwen", force=True)

    assert status["available"] is False
    assert status["detail"] == "连接失败：ConnectError"
    assert status["failed_stage"] == "model"
    assert status["checks"][0]["ok"] is None
    assert status["checks"][1]["ok"] is False
