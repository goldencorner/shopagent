"""把六个节点连成状态机并编译。

    START
      ↓
    router ──chitchat──────────────┐
      ↓                            │
    recall（更新画像）              │
      ↓ shop / compare             │
    planner ──无工具调用────────────┤
      │ 有 tool_calls              │
      ↓                            │
    tool_exec                      │
      ↓                            ↓
    reflect ──还有步数→ planner   generate → END
      │ 步数用尽/全失败 ────────────┘
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from langgraph.graph import END, START, StateGraph

from .. import llm, memory
from . import nodes
from .state import AgentState


def _after_router(state: dict) -> str:
    return "generate" if state.get("intent") == "chitchat" else "planner"


def _after_planner(state: dict) -> str:
    return "tool_exec" if state.get("pending_tool_calls") else "generate"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", nodes.router)
    graph.add_node("recall", nodes.recall)
    graph.add_node("planner", nodes.planner)
    graph.add_node("tool_exec", nodes.tool_exec)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "router")
    graph.add_edge("router", "recall")
    graph.add_conditional_edges(
        "recall", _after_router, {"planner": "planner", "generate": "generate"}
    )
    graph.add_conditional_edges(
        "planner", _after_planner, {"tool_exec": "tool_exec", "generate": "generate"}
    )
    graph.add_edge("tool_exec", "reflect")
    graph.add_conditional_edges(
        "reflect", nodes.route_after_reflect, {"planner": "planner", "generate": "generate"}
    )
    graph.add_edge("generate", END)

    return graph.compile()


GRAPH = build_graph()


def run_agent(
    query: str,
    history: list[dict] | None = None,
    profile: dict | None = None,
    model: str | None = None,
) -> dict:
    """跑一轮对话，返回最终 state（answer / trace / profile 都在里面）。"""
    if profile is None:
        profile = memory.load_profile()

    state: dict = {
        "query": query,
        "history": history or [],
        "profile": profile,
        "messages": [],
        "tool_results": [],
        "trace": [],
        "pending_tool_calls": [],
        "steps": 0,
    }
    with llm.use_model(model or llm.DEFAULT_MODEL):
        return GRAPH.invoke(state)


def stream_agent(
    query: str,
    history: list[dict] | None = None,
    profile: dict | None = None,
    model: str | None = None,
    *,
    cancel: threading.Event | None = None,
    lock: threading.Lock | None = None,
) -> Iterator[tuple[str, object]]:
    """流式跑一轮，依次 yield：

        ("step", "planner → search_products")
        ("token", "推荐")
        ("done", 最终 state)
        ("error", "错误信息")

    为什么用后台线程 + 队列：LangGraph 的 generate 节点在 GRAPH.stream 内部
    同步执行，主生成器在它返回前拿不到控制权；token 要实时出去，只能让
    generate 通过 contextvar 把增量写进队列，再由主生成器边读边 yield。

    cancel / lock 由调用方传入：客户端断开时 HTTP 层负责 set cancel；
    全局锁必须在 worker 里拿、worker 里放——同步生成器被 Starlette 丢弃时
    它的 finally 不保证执行，锁放在生成器里会永久泄漏。
    """
    if profile is None:
        profile = memory.load_profile()

    state: dict = {
        "query": query,
        "history": history or [],
        "profile": profile,
        "messages": [],
        "tool_results": [],
        "trace": [],
        "pending_tool_calls": [],
        "steps": 0,
    }

    events: queue.Queue = queue.Queue()
    cancel = cancel or threading.Event()

    def sink(text: str) -> None:
        if text and not cancel.is_set():
            events.put(("token", text))

    def worker() -> None:
        last = state
        seen_trace = 0
        locked = False
        try:
            # 锁归 worker 所有：即使 HTTP 生成器被丢弃，这里也会释放
            if lock is not None:
                lock.acquire()
                locked = True
            if cancel.is_set():
                return
            # ContextVar 不会自动传进新线程，模型选择和取消令牌都要在 worker 里重设
            with (
                llm.use_model(model or llm.DEFAULT_MODEL),
                llm.cancel_token(cancel),
                llm.token_sink(sink),
            ):
                for snapshot in GRAPH.stream(state, stream_mode="values"):
                    if cancel.is_set():
                        return
                    last = snapshot
                    trace = snapshot.get("trace") or []
                    for line in trace[seen_trace:]:
                        events.put(("step", line))
                    seen_trace = len(trace)
            events.put(("done", last))
        except llm.GenerationCancelled:
            pass  # 用户点了停止，不需要再往已断开的流里写错误
        except Exception as exc:  # 工具/模型异常转成事件，前端能展示
            if not cancel.is_set():
                events.put(("error", str(exc)))
        finally:
            if locked:
                lock.release()
            events.put(None)

    thread = threading.Thread(target=worker, name="shopagent-stream", daemon=True)
    thread.start()
    try:
        while True:
            item = events.get()
            if item is None:
                break
            yield item
    finally:
        # 客户端断开时尽快停掉 sink，并等 worker 退出后再让上层释放全局锁，
        # 避免上一轮还在生成时下一轮就进来。
        cancel.set()
        thread.join(timeout=60)
