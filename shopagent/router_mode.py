"""router 运行模式：默认读 .env，页面开关可热切换（只改内存，重启回到默认）。

四种模式：
    llm      云端 DeepSeek（默认）
    local    本地 LoRA 1.5B（不可用 / 超时 / 置信度低时自动回退 llm）
    keyword  关键词规则，0 延迟，不调模型
    shop     跳过 router，全部按 shop 处理

和 retrieval.set_rerank_candidates 一样：页面上改的是进程内变量，新请求立刻生效，
进程重启后回到 SHOPAGENT_ROUTER 的值。这样演示时切换不用重建容器。
"""

from __future__ import annotations

import os

MODES = ("llm", "local", "keyword", "shop")
DEFAULT = "llm"

# None 表示跟随 .env；非 None 是页面开关设置的值
_OVERRIDE: str | None = None


def env_default() -> str:
    mode = os.getenv("SHOPAGENT_ROUTER", DEFAULT).strip().lower()
    return mode if mode in MODES else DEFAULT


def current() -> str:
    return _OVERRIDE or env_default()


def override() -> str | None:
    return _OVERRIDE


def set_mode(value: str | None) -> str:
    """切到指定模式；传 None / "default" 表示恢复 .env 默认值。"""
    global _OVERRIDE
    mode = str(value or "").strip().lower()
    if mode in {"", "default", "env"}:
        _OVERRIDE = None
        return current()
    if mode not in MODES:
        raise ValueError(f"未知 router 模式：{value!r}，可选 {', '.join(MODES)}")
    _OVERRIDE = mode
    return mode
