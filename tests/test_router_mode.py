"""router 开关：.env 默认值 + 运行时覆盖。"""

from __future__ import annotations

import pytest

import shopagent.router_mode as router_mode
from shopagent.agent import nodes


@pytest.fixture(autouse=True)
def _clear_override():
    router_mode.set_mode(None)
    yield
    router_mode.set_mode(None)


def test_env_default_is_used_without_override(monkeypatch):
    monkeypatch.setenv("SHOPAGENT_ROUTER", "keyword")
    assert router_mode.env_default() == "keyword"
    assert router_mode.current() == "keyword"
    assert router_mode.override() is None


def test_override_beats_env(monkeypatch):
    monkeypatch.setenv("SHOPAGENT_ROUTER", "keyword")
    assert router_mode.set_mode("local") == "local"
    assert router_mode.current() == "local"
    # 节点里读的是同一个开关
    assert nodes._router_mode() == "local"


def test_reset_to_default(monkeypatch):
    monkeypatch.setenv("SHOPAGENT_ROUTER", "shop")
    router_mode.set_mode("local")
    assert router_mode.set_mode("default") == "shop"
    assert router_mode.override() is None


@pytest.mark.parametrize("value", ["", "gpt", "LLM2"])
def test_set_mode_rejects_unknown(value):
    if value == "":
        # 空字符串视为恢复默认
        router_mode.set_mode("local")
        assert router_mode.set_mode(value) == "llm"
        return
    with pytest.raises(ValueError):
        router_mode.set_mode(value)


def test_invalid_env_value_falls_back(monkeypatch):
    monkeypatch.setenv("SHOPAGENT_ROUTER", "typo")
    assert router_mode.env_default() == "llm"
