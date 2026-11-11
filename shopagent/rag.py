from __future__ import annotations

from collections.abc import Iterator

from .llm import chat, chat_stream

SYSTEM_PROMPT = """你是一名电商导购助手。
规则：
1. 只依据「商品资料」回答，不要编造不存在的商品、价格或参数。
2. 资料不足以回答时，直接说明，并建议用户换个问法，不要硬答。
3. 用中文回答，简洁分点，引用商品标题，方便用户核对。
"""


def format_context(docs: list[dict]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        price = f"${float(d['price']):.2f}" if d.get("price") is not None else "未知"
        blocks.append(
            f"[{i}] {d['title']} | 价格: {price} | ASIN: {d['asin']}\n{d['content']}"
        )
    return "\n\n".join(blocks)


def build_messages(
    query: str,
    docs: list[dict],
    history: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"商品资料：\n{format_context(docs)}\n\n用户问题：{query}",
        }
    )
    return messages


def answer(
    query: str,
    docs: list[dict],
    history: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    return chat(
        build_messages(query, docs, history, system_prompt),
        temperature=0.2,
        max_tokens=800,
    )


def answer_stream(
    query: str,
    docs: list[dict],
    history: list[dict] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Iterator[str]:
    return chat_stream(
        build_messages(query, docs, history, system_prompt),
        temperature=0.2,
        max_tokens=800,
    )
