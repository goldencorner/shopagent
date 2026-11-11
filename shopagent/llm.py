"""LLM 调用：DeepSeek（云端）与本地 Qwen（OpenAI 兼容协议）。

一轮对话里的 router / planner / generate 都走同一个「当前模型」，
由 service 用 use_model() 在请求入口设置，避免把模型名一层层传进节点。
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Callable

import httpx
from openai import APIConnectionError, OpenAI

from .config import settings

DEFAULT_MODEL = "deepseek"
MODEL_LABELS = {"deepseek": "DeepSeek", "qwen": "Qwen 本地"}

_ACTIVE_MODEL: ContextVar[str] = ContextVar("shopagent_model", default=DEFAULT_MODEL)


def current_model() -> str:
    return _ACTIVE_MODEL.get()


@contextmanager
def use_model(name: str):
    if name not in MODEL_LABELS:
        raise ValueError(f"未知模型：{name}")
    token = _ACTIVE_MODEL.set(name)
    try:
        yield
    finally:
        _ACTIVE_MODEL.reset(token)


@lru_cache(maxsize=2)
def get_client(name: str = DEFAULT_MODEL) -> OpenAI:
    if name == "qwen":
        if not settings.qwen_base_url:
            raise RuntimeError("QWEN_BASE_URL 未配置，无法连接本地 Qwen。")
        return OpenAI(
            api_key=settings.qwen_api_key or "lm-studio",
            base_url=settings.qwen_base_url,
            # 本地 27B 在消费级显卡上首 token 可能十几秒，超时给宽一点
            timeout=settings.qwen_timeout,
            max_retries=1,
        )

    if not settings.deepseek_api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请把 .env.example 复制为 .env 并填写。"
        )
    # 公网到 api.deepseek.com 会偶发连接被重置（穿代理时尤其明显）。
    # SDK 默认只重试 2 次、单次超时 10 分钟——一次抖动就能把 /chat 的全局锁
    # 占住十几分钟。这里收紧到 60s 超时、多给一次重试；流式调用另见 chat_stream。
    return OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=60.0,
        max_retries=3,
    )


class GenerationCancelled(RuntimeError):
    """客户端点了停止生成，用来尽快退出 Agent 图。"""


_CANCEL: ContextVar[threading.Event | None] = ContextVar(
    "shopagent_cancel", default=None
)


def _check_cancel() -> None:
    event = _CANCEL.get()
    if event is not None and event.is_set():
        raise GenerationCancelled("生成已停止")


@contextmanager
def cancel_token(event: threading.Event):
    token = _CANCEL.set(event)
    try:
        yield
    finally:
        _CANCEL.reset(token)


def _client() -> OpenAI:
    return get_client(current_model())


def _model_id() -> str:
    if current_model() == "qwen":
        return qwen_model_id()
    return settings.deepseek_model


# ---------------- 本地 Qwen 发现与可用性 ----------------


def _qwen_models(timeout: float = 2.0) -> list[str]:
    base = settings.qwen_base_url.rstrip("/")
    resp = httpx.get(
        f"{base}/models",
        headers={"Authorization": f"Bearer {settings.qwen_api_key or 'lm-studio'}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or []
    return [
        str(item.get("id"))
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]


def _qwen_state_models(timeout: float = 2.5) -> list[dict]:
    """LM Studio 原生接口：每个模型带 state=loaded / not-loaded。

    只读列表，不会触发模型加载；比 OpenAI 兼容的 /v1/models 多一个关键区别——
    后者会把「已下载但未加载」的模型也算在列表里，不能用来判断可用性。
    """
    base = settings.qwen_base_url.rstrip("/")
    root = base[:-3] if base.endswith("/v1") else base
    resp = httpx.get(
        f"{root}/api/v0/models",
        headers={"Authorization": f"Bearer {settings.qwen_api_key or 'lm-studio'}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json().get("data") or []
    return [
        item
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]


_QWEN_MODEL_CACHE = {"id": "", "at": 0.0}


def qwen_model_id() -> str:
    """显式配置优先；否则从 /v1/models 里挑第一个 qwen 模型并缓存 60s。"""
    if settings.qwen_model:
        return settings.qwen_model
    if _QWEN_MODEL_CACHE["id"] and time.time() - _QWEN_MODEL_CACHE["at"] < 60:
        return _QWEN_MODEL_CACHE["id"]

    ids = _qwen_models()
    qwen = [item for item in ids if "qwen" in item.lower()]
    if not qwen:
        raise RuntimeError(
            f"本地服务已连接，但没有找到 Qwen 模型（当前：{', '.join(ids) or '无'}）。"
        )
    _QWEN_MODEL_CACHE.update({"id": qwen[0], "at": time.time()})
    return qwen[0]


_STATUS_CACHE: dict[str, tuple[float, dict]] = {}
_STATUS_TTL = 15.0


def _qwen_tunnel_health(timeout: float = 2.0) -> dict | None:
    """问 qwen-proxy：服务器回环 1234（反向 SSH 隧道）是否可连接。

    返回 None 表示没有配置健康检查地址；返回 ok=None 表示健康检查端口本身
    连不上（本地开发没起 proxy），这两种情况都退回单段模型检查，不能把
    Qwen 直接判死。只有代理明确回答 tunnel=false 才算隧道断了。
    """
    url = (getattr(settings, "qwen_health_url", "") or "").strip()
    if not url:
        return None

    started = time.perf_counter()
    try:
        resp = httpx.get(url, timeout=timeout)
    except Exception as exc:
        return {
            "ok": None,
            "detail": f"健康检查不可达（{type(exc).__name__}）",
            "latency_ms": None,
        }

    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        data = resp.json()
    except ValueError:
        return {
            "ok": None,
            "detail": f"健康检查返回非 JSON（HTTP {resp.status_code}）",
            "latency_ms": latency_ms,
        }
    if not isinstance(data, dict) or "tunnel" not in data:
        return {
            "ok": None,
            "detail": f"健康检查响应缺少 tunnel 字段（HTTP {resp.status_code}）",
            "latency_ms": latency_ms,
        }

    ok = bool(data.get("tunnel"))
    if ok:
        detail = "已连通"
    else:
        reason = data.get("error") or f"HTTP {resp.status_code}"
        detail = f"未连通（{reason}）"
    return {"ok": ok, "detail": detail, "latency_ms": latency_ms}


def model_status(name: str, force: bool = False) -> dict:
    """给前端用的可用性探测。DeepSeek 只看是否配了 Key，Qwen 真去连一下。"""
    now = time.time()
    if not force:
        cached = _STATUS_CACHE.get(name)
        if cached and now - cached[0] < _STATUS_TTL:
            return cached[1]

    label = MODEL_LABELS.get(name, name)
    if name == "deepseek":
        available = bool(settings.deepseek_api_key)
        status = {
            "id": "deepseek",
            "label": label,
            "model": settings.deepseek_model,
            "available": available,
            "detail": "云端 API 已配置" if available else "未配置 DEEPSEEK_API_KEY",
            "latency_ms": None,
        }
    elif name == "qwen":
        status = {
            "id": "qwen",
            "label": label,
            "model": settings.qwen_model or "",
            "available": False,
            "detail": "",
            "latency_ms": None,
            # 两段式检测：先看反向 SSH 隧道，再看模型服务是否就绪。
            "checks": [],
            "failed_stage": None,
        }
        checks: list[dict] = status["checks"]

        tunnel = _qwen_tunnel_health()
        if tunnel is None:
            checks.append(
                {
                    "id": "tunnel",
                    "label": "SSH 隧道",
                    "ok": None,
                    "detail": "未配置健康检查",
                    "latency_ms": None,
                }
            )
        else:
            checks.append(
                {
                    "id": "tunnel",
                    "label": "SSH 隧道",
                    "ok": tunnel["ok"],
                    "detail": tunnel["detail"],
                    "latency_ms": tunnel["latency_ms"],
                }
            )

        if not settings.qwen_base_url:
            status["detail"] = "未配置 QWEN_BASE_URL"
            status["failed_stage"] = "tunnel"
            checks.append(
                {
                    "id": "model",
                    "label": "模型服务",
                    "ok": None,
                    "detail": "未检测（未配置 QWEN_BASE_URL）",
                    "latency_ms": None,
                }
            )
        elif tunnel is not None and tunnel["ok"] is False:
            status["detail"] = f"SSH 隧道未连通：{tunnel['detail']}"
            status["latency_ms"] = tunnel["latency_ms"]
            status["failed_stage"] = "tunnel"
            checks.append(
                {
                    "id": "model",
                    "label": "模型服务",
                    "ok": None,
                    "detail": "未检测（隧道未连通）",
                    "latency_ms": None,
                }
            )
        else:
            tunnel_ok = tunnel is not None and tunnel["ok"] is True
            prefix = "隧道已连通，" if tunnel_ok else ""
            started = time.perf_counter()
            model_ok = False
            model_detail = ""
            model_latency = 0
            try:
                try:
                    models = _qwen_state_models(timeout=2.5)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 404:
                        raise
                    # vLLM / Ollama 等没有加载状态接口，退回 OpenAI 兼容列表。
                    # 它们的列表本来就代表可服务模型，这里标注状态未经确认。
                    ids = _qwen_models(timeout=2.5)
                    model_latency = int((time.perf_counter() - started) * 1000)
                    candidates = (
                        [settings.qwen_model]
                        if settings.qwen_model
                        else [item for item in ids if "qwen" in item.lower()]
                    )
                    found = [item for item in candidates if item in ids]
                    if found:
                        model_ok = True
                        status.update(
                            available=True,
                            model=found[0],
                        )
                        model_detail = "模型服务已连接（未返回加载状态）"
                    else:
                        model_detail = (
                            f"没有 Qwen 模型（当前：{', '.join(ids) or '无'}）"
                        )
                else:
                    model_latency = int((time.perf_counter() - started) * 1000)
                    loaded = [
                        str(item["id"])
                        for item in models
                        if item.get("state") == "loaded"
                    ]
                    if settings.qwen_model:
                        if settings.qwen_model in loaded:
                            model_ok = True
                            status.update(
                                available=True,
                                model=settings.qwen_model,
                            )
                            model_detail = "模型已加载"
                        else:
                            model_detail = (
                                f"模型 {settings.qwen_model} 未加载"
                                f"（已加载：{', '.join(loaded) or '无'}）"
                            )
                    else:
                        qwen = [item for item in loaded if "qwen" in item.lower()]
                        if qwen:
                            model_ok = True
                            status.update(
                                available=True,
                                model=qwen[0],
                            )
                            model_detail = "模型已加载"
                        else:
                            model_detail = (
                                "没有已加载的 Qwen 模型"
                                f"（已加载：{', '.join(loaded) or '无'}）"
                            )
            except httpx.HTTPStatusError as exc:
                model_latency = int((time.perf_counter() - started) * 1000)
                model_detail = f"HTTP {exc.response.status_code}"
            except httpx.TimeoutException:
                model_latency = int((time.perf_counter() - started) * 1000)
                model_detail = "连接超时"
            except Exception as exc:  # 连接拒绝、DNS 失败等
                model_latency = int((time.perf_counter() - started) * 1000)
                model_detail = f"连接失败：{type(exc).__name__}"

            if tunnel_ok and model_ok:
                status["detail"] = f"{prefix}{model_detail}"
            elif tunnel_ok:
                status["detail"] = f"{prefix}但{model_detail}"
            else:
                status["detail"] = model_detail
            status["latency_ms"] = (tunnel["latency_ms"] or 0 if tunnel else 0) + model_latency
            status["failed_stage"] = None if model_ok else "model"
            checks.append(
                {
                    "id": "model",
                    "label": "模型服务",
                    "ok": model_ok,
                    "detail": model_detail,
                    "latency_ms": model_latency,
                }
            )
    else:
        status = {
            "id": name,
            "label": label,
            "model": "",
            "available": False,
            "detail": "未知模型",
            "latency_ms": None,
        }

    _STATUS_CACHE[name] = (now, status)
    return status


def model_statuses(force: bool = False) -> list[dict]:
    return [model_status(name, force=force) for name in MODEL_LABELS]


# 评测要统计「一轮对话花了几次 LLM 调用」，直接在出口计数最省事。
_CALLS = {"n": 0}


def reset_calls() -> None:
    _CALLS["n"] = 0


def call_count() -> int:
    return _CALLS["n"]


def _count() -> None:
    _CALLS["n"] += 1


def chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    _check_cancel()
    _count()
    resp = _client().chat.completions.create(
        model=_model_id(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_stream(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """流式版本：逐段 yield 文本增量，调用方负责拼完整答案。

    SDK 不会重试已经开始输出的流。连接在第一个 token 之前断掉时自己补一次重试，
    已经吐过字就不再重试——否则用户会看到重复的半截答案。
    """
    _check_cancel()
    _count()
    attempts = 3
    for attempt in range(1, attempts + 1):
        emitted = False
        try:
            stream = _client().chat.completions.create(
                model=_model_id(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                cancel_event = _CANCEL.get()
                if cancel_event is not None and cancel_event.is_set():
                    # 断开与模型的连接，别让已经没人看的生成继续烧 GPU
                    try:
                        stream.close()
                    except Exception:
                        pass
                    raise GenerationCancelled("生成已停止")
                # 有的兼容实现会在末尾补一个 choices=[] 的 usage chunk
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content
                if text:
                    emitted = True
                    yield text
            return
        except APIConnectionError:
            if emitted or attempt == attempts:
                raise
            time.sleep(2 ** (attempt - 1))


# generate 节点的流式出口。用 contextvar 而不是塞进 LangGraph state：
# state 会被节点合并、可能参与序列化，回调函数不适合放进去。
_TOKEN_SINK: ContextVar = ContextVar("shopagent_token_sink", default=None)


def current_token_sink():
    return _TOKEN_SINK.get()


@contextmanager
def token_sink(sink: Callable[[str], None]):
    token = _TOKEN_SINK.set(sink)
    try:
        yield
    finally:
        _TOKEN_SINK.reset(token)


def chat_tools(
    messages: list[dict],
    tools: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 800,
):
    """带工具调用的一轮对话，返回原始 message（可能带 tool_calls）。

    返回对象而不是字符串，因为调用方既要读 content，也要读 tool_calls。
    """
    _check_cancel()
    _count()
    resp = _client().chat.completions.create(
        model=_model_id(),
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message


def chat_json(messages: list[dict], temperature: float = 0.0) -> dict:
    """要求模型返回 JSON。解析失败时兜底截取第一个 {...}。"""
    _check_cancel()
    _count()
    kwargs: dict = {
        "model": _model_id(),
        "messages": messages,
        "temperature": temperature,
    }
    # DeepSeek 原生支持 json_object；LM Studio 等本地实现只认 json_schema/text，
    # 传 json_object 会直接 400，所以 Qwen 走 prompt 约束 + 解析兜底。
    if current_model() == "deepseek":
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 模型偶尔在 JSON 后面多写一段（如 {"query": "..."}\nNote: ...）。
        # raw_decode 只解析第一个完整对象，后面的内容直接忽略。
        start = text.find("{")
        if start == -1:
            return {}
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}
