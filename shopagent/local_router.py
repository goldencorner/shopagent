"""本地小模型 router：一次前向同时完成「意图分类 + 中文需求→英文检索词」。

模型是 Qwen2.5-1.5B-Instruct 的 LoRA 微调版（训练见 scripts/13_train_router_lora.py），
通过 OpenAI 兼容接口调用（LM Studio / vLLM 都可以）。

接入原则和本地 27B 一样：先探测可用性，可用才走本地；未加载、超时、输出不合法或
置信度低，一律回退原来的 DeepSeek router，绝不让小模型把整轮对话带挂。

训练脚本直接复用这里的 LOCAL_ROUTER_PROMPT，训练和推理的输入格式必须一致。
"""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache

import httpx
from openai import OpenAI

from .config import settings
from .router_prompt import LOCAL_ROUTER_PROMPT

INTENTS = {"shop", "compare", "chitchat"}


def router_messages(query: str) -> list[dict]:
    return [{"role": "user", "content": LOCAL_ROUTER_PROMPT.format(query=(query or "")[:500])}]


def parse_output(text: str) -> dict:
    """从模型输出里抠出 JSON；不合法就抛 ValueError，交给上层回退。"""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise ValueError(f"router 输出不是 JSON：{text[:80]}")
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            raise ValueError(f"router 输出 JSON 解析失败：{text[:80]}") from exc
    if not isinstance(obj, dict):
        raise ValueError("router 输出不是对象")

    intent = str(obj.get("intent", "")).strip().lower()
    if intent not in INTENTS:
        raise ValueError(f"router 输出非法 intent：{intent!r}")
    en_query = obj.get("en_query") or ""
    return {"intent": intent, "en_query": str(en_query).strip()[:300]}


# ---------------- 可用性探测 ----------------

_BASE = lambda: (os.getenv("SHOPAGENT_LOCAL_ROUTER_BASE_URL") or settings.qwen_base_url).rstrip("/")
_MODEL = lambda: (
    os.getenv("SHOPAGENT_LOCAL_ROUTER_MODEL")
    or settings.qwen_model
    or "qwen2.5-1.5b-instruct"
)

_STATUS_CACHE: dict = {"at": 0.0, "value": None}
_TRUE_TTL = 30.0
_FALSE_TTL = 5.0


def router_status(force: bool = False) -> dict:
    """本地 router 是否可用。缓存：可用 30s，不可用 5s，避免每次请求都探测。"""
    now = time.time()
    cached = _STATUS_CACHE["value"]
    if not force and cached:
        ttl = _TRUE_TTL if cached.get("available") else _FALSE_TTL
        if now - _STATUS_CACHE["at"] < ttl:
            return cached

    value = _probe()
    _STATUS_CACHE.update({"at": now, "value": value})
    return value


def _probe() -> dict:
    base = _BASE()
    started = time.perf_counter()
    model = _MODEL()

    # 1) 反向 SSH 隧道健康检查：只在显式配置时做。
    #    本地开发没有 qwen-proxy，若默认去连 1235 会白等 1.5s 才探测模型。
    health_url = os.getenv("SHOPAGENT_LOCAL_ROUTER_HEALTH_URL") or ""
    if health_url:
        try:
            resp = httpx.get(health_url, timeout=1.5)
            data = resp.json()
            if isinstance(data, dict) and data.get("tunnel") is False:
                reason = data.get("error") or f"HTTP {resp.status_code}"
                return _unavailable(f"SSH 隧道未连通（{reason}）", model, started)
        except Exception:
            # 健康检查端口没起不代表模型服务不可用（本地开发常见），继续探模型
            pass

    # 2) 模型是否已加载。LM Studio 原生接口带 state；vLLM 只有 /v1/models。
    #    原生接口挂在根路径（base 去掉 /v1），不少版本对错误路径仍返回 200 + error JSON，
    #    所以不能只看状态码，拿不到 data 就退回 OpenAI 兼容列表。
    root = base[:-3] if base.endswith("/v1") else base
    ids: list[str] = []
    try:
        data = _get_json(f"{root}/api/v0/models", timeout=1.5)
        entries = [item for item in (data.get("data") or []) if isinstance(item, dict)]
        ids = [str(item.get("id")) for item in entries]
        match = [item for item in entries if str(item.get("id")) == model]
        if not match:
            # 允许配置里只写模型短名，实际 id 带 publisher 前缀
            match = [item for item in entries if model in str(item.get("id", ""))]
        if match:
            entry = match[0]
            resolved = str(entry.get("id") or model)
            if entry.get("state") != "loaded":
                return _unavailable(f"模型 {resolved} 未加载", resolved, started)
            return _available(f"模型 {resolved} 已加载", resolved, started)
    except Exception:
        pass

    try:
        data = _get_json(f"{base}/models", timeout=1.5)
        ids = [str(item.get("id")) for item in (data.get("data") or []) if isinstance(item, dict)]
        match = [item for item in ids if item == model]
        if not match:
            match = [item for item in ids if model in item]
        if match:
            return _available(f"模型 {match[0]} 已连接（服务未返回加载状态）", match[0], started)
        return _unavailable(f"模型 {model} 不在列表中（当前：{', '.join(ids) or '无'}）", model, started)
    except Exception as exc:
        return _unavailable(f"本地服务不可达（{type(exc).__name__}）", model, started)


def _get_json(url: str, timeout: float) -> dict:
    resp = httpx.get(
        url,
        headers={"Authorization": f"Bearer {os.getenv('SHOPAGENT_LOCAL_ROUTER_API_KEY') or settings.qwen_api_key or 'lm-studio'}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _available(detail: str, model: str, started: float) -> dict:
    return {
        "available": True,
        "detail": detail,
        "model": model,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def _unavailable(detail: str, model: str, started: float) -> dict:
    return {
        "available": False,
        "detail": detail,
        "model": model,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


# ---------------- 调用 ----------------

class LocalRouter:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or _BASE()).rstrip("/")
        self.model = model or _MODEL()
        self.timeout = float(
            timeout if timeout is not None else os.getenv("SHOPAGENT_LOCAL_ROUTER_TIMEOUT", "8")
        )
        self._client = OpenAI(
            api_key=os.getenv("SHOPAGENT_LOCAL_ROUTER_API_KEY") or settings.qwen_api_key or "lm-studio",
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,
            # 本机 LM Studio 不走系统代理；Windows 上不关经常连不上 127.0.0.1
            http_client=httpx.Client(trust_env=False),
        )
        # LM Studio 各版本对 logprobs 支持不一致；第一次被 400 后就不再带
        self._logprobs_ok: bool | None = None

    def route(self, query: str) -> dict:
        started = time.perf_counter()
        resp = self._complete(query, with_logprobs=self._logprobs_ok is not False)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        text = resp.choices[0].message.content or ""
        result = parse_output(text)
        result["confidence"] = _confidence(resp, result["intent"])
        result["latency_ms"] = elapsed_ms
        result["raw"] = text[:300]
        return result

    def _complete(self, query: str, with_logprobs: bool):
        kwargs: dict = {
            "model": self.model,
            "messages": router_messages(query),
            "temperature": 0,
            "max_tokens": 64,
        }
        if with_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        try:
            resp = self._client.chat.completions.create(**kwargs)
            if with_logprobs:
                self._logprobs_ok = True
            return resp
        except Exception as exc:
            if with_logprobs and getattr(exc, "status_code", None) == 400:
                self._logprobs_ok = False
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                return self._client.chat.completions.create(**kwargs)
            raise


def _confidence(resp, intent: str) -> float:
    """intent 前后若干 token 的平均概率；拿不到 logprobs 时给 0.6。"""
    try:
        content = resp.choices[0].logprobs.content or []
    except Exception:
        return 0.6
    if not content:
        return 0.6

    probs: list[float] = []
    joined = ""
    for item in content:
        joined += item.token
        try:
            probs.append(math.exp(float(item.logprob)))
        except Exception:
            continue
        if intent in joined:
            break
    if not probs:
        return 0.6
    window = probs[-8:]
    return round(sum(window) / len(window), 4)


@lru_cache(maxsize=1)
def get_local_router() -> LocalRouter:
    return LocalRouter()


def route_local(query: str) -> dict:
    """给 nodes.router 用的入口；失败会抛异常，由调用方回退 DeepSeek。"""
    return get_local_router().route(query)
