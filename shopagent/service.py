"""FastAPI 服务：把 Agent 包成 HTTP 接口。

设计取舍：
- 一个全局锁串行化 /chat 和 /chat/stream。Retriever 的 BM25 是懒加载的，多线程
  首次并发会重复建索引；这个规模下串行比引入连接池和并发安全更划算。要横向扩展
  就把锁换成「每会话锁 + 后端进程池」，先说清楚边界。
- 会话与画像复用 Agent 用的同一套 JSON 存储，CLI 里聊到一半，HTTP 接上继续。
- /chat 同时返回 trace，前端/调试可以直接看到 router→planner→tool 的路径。
- /catalog/* 是只读的商品库浏览接口，直接读内存明细，不走检索、不调 LLM，
  因此不需要拿上面那把锁。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import llm, memory
from .agent import run_agent, stream_agent
from .catalog import browse, count, get_product, stats
from .config import settings
from .retrieval import (
    MAX_RERANK_CANDIDATES,
    MIN_RERANK_CANDIDATES,
    current_rerank_candidates,
    set_rerank_candidates,
)
from .tools import shop

WEB_DIR = Path(__file__).resolve().parent / "web"

logger = logging.getLogger(__name__)


class _NoCacheStaticFiles(StaticFiles):
    """静态资源每次回源校验（没变就 304，很便宜）。

    Starlette 默认只给 Last-Modified/ETag，浏览器会按启发式规则直接吃缓存，
    部署后经常出现「HTML 是新的、CSS/JS 还是旧的」的半新半旧页面。
    """

    async def get_response(self, path: str, scope: dict):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field("default", max_length=64)
    use_history: bool = True
    # deepseek 是默认；qwen 指向本地 OpenAI 兼容服务
    model: Literal["deepseek", "qwen"] = "deepseek"


class RerankCandidatesRequest(BaseModel):
    value: int = Field(..., ge=MIN_RERANK_CANDIDATES, le=MAX_RERANK_CANDIDATES)


def _next_or_none(iterator):
    """在线程里取下一项：StopIteration 不能穿过 anyio 的线程边界。"""
    try:
        return next(iterator)
    except StopIteration:
        return None


def create_app(store: str | None = None, mode: str | None = None) -> FastAPI:
    store = store or os.getenv("SHOPAGENT_STORE", "pg")
    mode = mode or os.getenv("SHOPAGENT_MODE", "hybrid_rerank")
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        shop.configure(store, mode)
        yield
        shop.close()

    app = FastAPI(
        title="ShopAgent API",
        version="0.3.0",
        description="电商导购 Agent：检索 + 工具调用 + 两级记忆",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 前端是零依赖静态页，和 API 同源部署：不需要 npm、不需要另配 Nginx。
    # /static 放资源，/ 返回单页；接口路径都在上面，不会被静态挂载吞掉。
    app.mount("/static", _NoCacheStaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "store": store,
            "mode": mode,
            "products": count(),
            "rerank_candidates": current_rerank_candidates(),
        }

    @app.get("/models")
    def models(refresh: bool = False) -> dict:
        """模型列表 + 可用性。

        Qwen 每次最多探测 2.5s，结果缓存 15s；前端手动刷新时带 refresh=1。
        """
        return {
            "default": llm.DEFAULT_MODEL,
            "models": llm.model_statuses(force=refresh),
        }

    def _require_model(name: str) -> None:
        """本地模型不可用时提前返回 503，别把 60s 超时留给用户等。"""
        if name != "qwen":
            return
        status = llm.model_status("qwen")
        if not status["available"]:
            raise HTTPException(
                status_code=503,
                detail=f"本地 Qwen 不可用：{status['detail']}",
            )

    @app.post("/config/rerank_candidates")
    def update_rerank_candidates(req: RerankCandidatesRequest) -> dict:
        """页面上热改精排候选数：只改内存，新请求立刻生效，重启后回到默认值。"""
        logger.info("rerank_candidates -> %s", req.value)
        return {"rerank_candidates": set_rerank_candidates(req.value)}

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict:
        _require_model(req.model)
        t0 = time.perf_counter()
        with lock:
            session = memory.Session.load(req.session_id)
            history = session.history() if req.use_history else []
            state = run_agent(req.message, history=history, model=req.model)
            answer = state.get("answer") or ""
            product_refs = state.get("products") or []

            session.append("user", req.message)
            session.append(
                "assistant", answer, {"products": product_refs} if product_refs else None
            )
            session.save()

        return {
            "answer": answer,
            "session_id": session.session_id,
            "intent": state.get("intent"),
            "steps": state.get("steps", 0),
            "trace": state.get("trace", []),
            "products": product_refs,
            "model": req.model,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "profile": memory.profile_summary(memory.load_profile()),
        }

    def _sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
        """和 /chat 同一套 Agent，只是把步骤和 token 边跑边推。

        事件：step（trace 一行）→ token（文本增量）→ done（元信息 + 完整答案）；
        失败时发 error。会话仍只在完整回答后落盘，避免把半截答案写进历史。

        全局锁交给 stream_agent 的 worker 负责，HTTP 这层只负责在客户端断开时
        set cancel：Starlette 丢弃同步生成器时 finally 不保证执行，锁放这里会泄漏。
        """
        # 可用性探测是同步 HTTP，放线程里，别卡住事件循环
        await run_in_threadpool(_require_model, req.model)

        async def event_stream():
            t0 = time.perf_counter()
            cancel = threading.Event()
            agent_iter = None
            try:
                session = memory.Session.load(req.session_id)
                history = session.history() if req.use_history else []

                final_state: dict | None = None
                pieces: list[str] = []
                agent_iter = stream_agent(
                    req.message,
                    history=history,
                    model=req.model,
                    cancel=cancel,
                    lock=lock,
                )

                while True:
                    if await request.is_disconnected():
                        cancel.set()
                        break
                    item = await run_in_threadpool(_next_or_none, agent_iter)
                    if item is None:
                        break
                    kind, payload = item
                    if kind == "token":
                        pieces.append(str(payload))
                        yield _sse("token", {"text": payload})
                    elif kind == "step":
                        yield _sse("step", {"line": payload})
                    elif kind == "done":
                        final_state = payload  # type: ignore[assignment]
                    elif kind == "error":
                        yield _sse("error", {"message": payload})
                        return

                if final_state is None:
                    if not cancel.is_set():
                        yield _sse("error", {"message": "没有收到完整回答，请重试"})
                    return

                answer = final_state.get("answer") or "".join(pieces)
                product_refs = final_state.get("products") or []
                session.append("user", req.message)
                session.append(
                    "assistant",
                    answer,
                    {"products": product_refs} if product_refs else None,
                )
                session.save()

                yield _sse(
                    "done",
                    {
                        "answer": answer,
                        "session_id": session.session_id,
                        "intent": final_state.get("intent"),
                        "steps": final_state.get("steps", 0),
                        "trace": final_state.get("trace", []),
                        "products": product_refs,
                        "model": req.model,
                        "latency_ms": int((time.perf_counter() - t0) * 1000),
                        "profile": memory.profile_summary(memory.load_profile()),
                    },
                )
            except Exception as exc:  # 头已经发出去了，只能在流里报错
                # 前端只看到 "Connection error."，堆栈要留在服务日志里才能定位
                logger.exception("chat/stream 失败 session=%s", req.session_id)
                yield _sse("error", {"message": str(exc)})
            finally:
                cancel.set()
                if agent_iter is not None:
                    # close 会等 worker 退出，别在事件循环里阻塞
                    await run_in_threadpool(agent_iter.close)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # 让 nginx 不要缓冲 SSE
            },
        )

    @app.get("/sessions")
    def get_sessions() -> dict:
        return {"sessions": memory.list_sessions()}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict:
        session = memory.Session.load(session_id)
        return {
            "session_id": session.session_id,
            "messages": session.messages_with_meta(),
        }

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict:
        removed = memory.clear_session(session_id)
        if not removed:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"deleted": session_id}

    # 商品库页签：概览数字 + 可搜索/筛选/分页的列表。
    # 走内存 catalog，不经过检索链路，所以不占全局锁，也不会加载 embedding 模型。
    @app.get("/catalog/stats")
    def catalog_stats() -> dict:
        return stats()

    @app.get("/catalog/products")
    def catalog_products(
        q: str = Query("", max_length=200),
        price_min: float | None = Query(None, ge=0),
        price_max: float | None = Query(None, ge=0),
        priced: str | None = Query(None, pattern="^(yes|no)$"),
        page: int = Query(1, ge=1),
        page_size: int = Query(12, ge=1, le=60),
        sort: str = Query("default", pattern="^(default|price_asc|price_desc|brand)$"),
    ) -> dict:
        if price_min is not None and price_max is not None and price_min > price_max:
            raise HTTPException(status_code=400, detail="price_min 不能大于 price_max")
        return browse(
            query=q,
            price_min=price_min,
            price_max=price_max,
            priced=priced,
            page=page,
            page_size=page_size,
            sort=sort,
        )

    @app.get("/catalog/products/{asin}")
    def catalog_product(asin: str) -> dict:
        product = get_product(asin)
        if product is None:
            raise HTTPException(status_code=404, detail="商品不存在")
        return product

    @app.get("/profile")
    def get_profile() -> dict:
        profile = memory.load_profile()
        return {"profile": profile, "summary": memory.profile_summary(profile)}

    @app.delete("/profile")
    def delete_profile() -> dict:
        memory.save_profile({})
        return {"deleted": True}

    return app
