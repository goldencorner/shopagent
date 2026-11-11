"""Agent 命令行：多轮对话 + 工具调用 + 两级记忆。

用法:
    python scripts/06_agent_chat.py --store memory
    python scripts/06_agent_chat.py --store memory --show-trace
    python scripts/06_agent_chat.py --store memory --session alice
    python scripts/06_agent_chat.py --reset          # 清空当前会话和用户画像
    python scripts/06_agent_chat.py --no-history     # 启动时不回显历史

命令:
    exit / q   退出
    /trace     切换是否打印执行轨迹
    /profile   查看已记住的用户画像
    /history   重新打印本会话的历史
    /new       开一个新会话（继续保留画像）
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent.agent import run_agent  # noqa: E402
from shopagent.memory import Session, clear_all, load_profile, profile_summary  # noqa: E402
from shopagent.tools import shop  # noqa: E402

PREVIEW_CHARS = 180
RULE = "-" * 64


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """压成一行并截断：只用于终端回显，完整内容仍在 data/sessions/<id>.json。"""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def print_history(session: Session) -> None:
    """启动时把上次的对话回显出来，让用户知道这次是接着聊的。"""
    messages = session.history()
    if not messages:
        print(f"[会话 {session.session_id}] 没有历史记录，这是一个新会话\n")
        return

    turns = (len(messages) + 1) // 2
    print(f"[会话 {session.session_id}] 已加载 {turns} 轮历史对话：")
    print(RULE)
    for message in messages:
        who = "你" if message.get("role") == "user" else "助手"
        print(f"{who}> {_preview(message.get('content'))}")
    print(RULE)
    print("接着上次继续聊；输入 /new 开新会话，/history 可重新查看\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=["pg", "memory"], default="memory")
    parser.add_argument("--mode", default="hybrid_rerank",
                        choices=["bm25", "vector", "hybrid", "hybrid_rerank"])
    parser.add_argument("--session", default="default")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--no-history", action="store_true", help="启动时不回显历史对话")
    parser.add_argument("--reset", action="store_true", help="清空该会话和用户画像后退出")
    args = parser.parse_args()

    if args.reset:
        clear_all(args.session)
        print(f"已清空会话 {args.session} 和用户画像。")
        return

    shop.configure(store=args.store, mode=args.mode)
    session = Session.load(args.session)
    show_trace = args.show_trace

    print(f"ShopAgent | 后端 {args.store} | 检索 {args.mode} | 会话 {args.session}")
    print(f"已记住: {profile_summary(load_profile())}\n")
    if not args.no_history:
        print_history(session)
    print("输入 exit 退出，/trace 切换轨迹，/profile 看画像，/history 看历史，/new 新会话\n")

    while True:
        try:
            query = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        if query == "/trace":
            show_trace = not show_trace
            print(f"[轨迹显示: {'开' if show_trace else '关'}]\n")
            continue
        if query == "/profile":
            print(f"[画像] {profile_summary(load_profile())}\n")
            continue
        if query == "/history":
            print()
            print_history(session)
            continue
        if query == "/new":
            session = Session(uuid.uuid4().hex[:8])
            print(f"[新会话 {session.session_id}，画像保留]\n")
            print(f"[会话 {session.session_id}] 没有历史记录，这是一个新会话\n")
            continue

        state = run_agent(query, history=session.history())

        if show_trace:
            print("\n轨迹:")
            for line in state.get("trace", []):
                print(f"  · {line}")
            print()

        answer = state.get("answer") or "(没有产生回答)"
        print(f"\n助手> {answer}\n")

        session.append("user", query)
        session.append("assistant", answer)
        session.save()

    shop.close()


if __name__ == "__main__":
    main()
