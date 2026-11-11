"""图的六个节点：router / planner / tool_exec / reflect / generate / update_memory。

设计取舍（面试会问）：
- router 单独成节点，是为了让「闲聊」不过工具链，省一次 planner 的 prompt 长度；
- planner 只决定调什么工具，不负责写答案，生成的答案统一在 generate 里做，
  这样「依据哪些资料回答」这件事只有一个出口，不容易编造；
- reflect 用规则判断（步数/结果是否全失败），没有额外走 LLM：
  这个规模下 LLM 自我反思的收益抵不过一次调用的延迟；
- 每次工具调用都写进 trace，出问题能直接看到是检索错了还是模型选错了工具。
"""

from __future__ import annotations

import json
import os
import re

from .. import memory, rag
from ..llm import (
    GenerationCancelled,
    chat,
    chat_json,
    chat_stream,
    chat_tools,
    current_token_sink,
)
from ..tools import TOOL_SPECS, run_tool

MAX_STEPS = 4
MAX_DOCS = 8

ROUTER_PROMPT = """判断用户这句话属于哪一类，只返回 JSON：

- shop：想买、想找商品、问推荐、问价格/功效
- compare：要求在多个商品之间选一个、问区别
- chitchat：打招呼、闲聊、问你是谁、与购物无关

用户：{query}

返回：{{"intent": "shop"}}"""

PLANNER_PROMPT = """你是一个电商导购 Agent 的规划器。你的唯一职责是决定下一步调哪个工具，
**不要直接回答用户**。信息够了就不要再调工具，直接结束。

规则：
1. 商品库是英文的，调用 search_products 时 query 必须翻成英文关键词。
   例：「有没有适合两个人用的露营帐篷」→ "2 person camping tent"
2. 用户给了预算，就传 max_price。
3. 要对比商品，先 search_products 拿到 ASIN，再调 compare_products。不要编造 ASIN。
   用户说「对比 A 和 B」时，对 A、B 各 search 一次拿到各自的 ASIN，然后**必须**调
   一次 compare_products 传这几个 ASIN；不要用多次搜索或 get_product_detail 代替对比。
   对比类问题的完成条件是：tool_results 里已经出现过 compare_products。没出现过就
   不算查完，不要直接结束。
4. 一次只调一个最有必要的工具，最多 {max_steps} 步。
5. 用户问「有没有更便宜的」，用 find_similar 或 search_products 加 max_price。
6. 历史对话只用来理解指代（「那款」「它」「再便宜点」）。当前问题和历史话题无关时，
   按当前问题检索，不要沿用上一轮的品类关键词。

已知用户偏好：{profile}"""

GENERATE_PROMPT = """你是一名电商导购助手。
规则：
1. 只依据给出的商品资料回答，不要编造商品、价格、成分。
2. 资料不足就直说，并给出下一步建议（换关键词、放宽预算）。
3. 用中文回答，分点，带上商品标题和价格，方便用户核对。
4. 用户画像里的预算上限是硬约束：**已知价格**超过预算的商品直接剔除，不要列出来，
   也不要写成"预算超出"却给出低于预算的价格——判断前先核对数字。
5. 搜索结果里 price_unknown=true 表示该商品没有标价。可以列出，但必须注明
   「未标价，需自行确认」，不要猜价格；未标价的商品不要排在已知预算内商品前面。
6. 用户画像里的预算/品牌/偏好要在首句说明是否已按此筛选；但与当前品类无关的
   画像字段直接忽略，不要提它的名字，也不要解释为什么不筛选——不要写
   "未按 X 筛选""X 与本轮品类无关"这类句子。
7. 历史对话只用来理解指代（"那款""它""再便宜点"）和延续正在进行的挑选。
   用户当前问什么就答什么，不要提历史里已经结束的话题或品类。
   例：上一轮问帐篷、这一轮问瑜伽垫，就只回答瑜伽垫，不要写"没有帐篷"。
8. 预算/偏好这类信息，优先级是：本轮用户明确说的 > 用户画像 > 更早的历史。
   历史里出现过的旧数字不要直接沿用；回答里用哪个预算，就说哪个。
9. 回答里出现的每一个"$数字"都必须来自商品资料或用户给的预算。建议部分也不要
   自己编价位（不要写"再找找 $25 左右的"）；想建议放宽预算就说"如果愿意适当
   提高预算"，数字交给用户定。
   对比两个商品时，不要写两个价格相减得到的差价（比如"贵约 $43"）——这个数字
   不在资料里。要比差价就用比例（"贵一倍多""价格约是它的 2.3 倍"），或者直接
   让用户看两个标价。"""

CHITCHAT_PROMPT = """你是一名电商导购助手，名叫 ShopAgent。
用中文简短回答，一两句话即可，并自然地把话题引回「你想找什么商品」。
不要编造商品信息。"""

# 消融用：只保留 1-3 条基础要求，去掉预算硬约束/未标价/历史隔离/价格引用等约束。
# 通过 SHOPAGENT_GENERATE_GUARD=0 启用（见 README §4.7）。
GENERATE_PROMPT_BASE = """你是一名电商导购助手。
规则：
1. 只依据给出的商品资料回答，不要编造商品、价格、成分。
2. 资料不足就直说，并给出下一步建议（换关键词、放宽预算）。
3. 用中文回答，分点，带上商品标题和价格，方便用户核对。"""


def _generate_prompt() -> str:
    guard = os.getenv("SHOPAGENT_GENERATE_GUARD", "1").strip().lower()
    return GENERATE_PROMPT if guard not in {"0", "false", "off", "no"} else GENERATE_PROMPT_BASE


# ---------------- router ----------------

_CHITCHAT_RE = re.compile(
    r"^(你好|您好|hi|hello|hey|你是谁|你叫什么|在吗|谢谢|再见|早上好|晚上好)", re.I
)
_COMPARE_RE = re.compile(r"对比|区别|哪个好|选哪个|比较|vs\.?", re.I)


def _router_mode() -> str:
    """SHOPAGENT_ROUTER=llm（默认）/ shop（全部走导购）/ keyword（规则路由）。"""
    return os.getenv("SHOPAGENT_ROUTER", "llm").strip().lower()


def _keyword_intent(query: str) -> str:
    text = (query or "").strip()
    if _COMPARE_RE.search(text):
        return "compare"
    if _CHITCHAT_RE.search(text):
        return "chitchat"
    return "shop"


def router(state: dict) -> dict:
    query = state.get("query", "")
    mode = _router_mode()

    if mode == "shop":
        intent = "shop"
    elif mode == "keyword":
        intent = _keyword_intent(query)
    else:
        try:
            obj = chat_json([{"role": "user", "content": ROUTER_PROMPT.format(query=query[:500])}])
            intent = str(obj.get("intent", "shop")).strip().lower()
        except GenerationCancelled:
            raise
        except Exception:
            intent = "shop"

    if intent not in {"shop", "compare", "chitchat"}:
        intent = "shop"
    return {"intent": intent, "trace": [f"router → {intent}（{mode}）"]}


# ---------------- planner ----------------


def planner(state: dict) -> dict:
    messages: list[dict] = [
        {
            "role": "system",
            "content": PLANNER_PROMPT.format(
                max_steps=MAX_STEPS,
                profile=memory.profile_summary(state.get("profile") or {}),
            ),
        }
    ]
    messages.extend(state.get("history") or [])
    messages.append({"role": "user", "content": state.get("query", "")})
    messages.extend(state.get("messages") or [])

    message = chat_tools(messages, TOOL_SPECS)

    assistant: dict = {"role": "assistant", "content": message.content or ""}
    calls: list[dict] = []
    if message.tool_calls:
        for call in message.tool_calls:
            calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
        assistant["tool_calls"] = calls

    names = [c["function"]["name"] for c in calls]
    trace = [f"planner → {', '.join(names)}"] if names else ["planner → 信息够了"]
    return {"messages": [assistant], "pending_tool_calls": calls, "trace": trace}


# ---------------- tool_exec ----------------


def tool_exec(state: dict) -> dict:
    results: list[dict] = []
    tool_messages: list[dict] = []
    trace: list[str] = []

    for call in state.get("pending_tool_calls") or []:
        name = call["function"]["name"]
        raw_args = call["function"]["arguments"]
        output = run_tool(name, raw_args)
        results.append({"tool": name, "args": raw_args, "output": output})

        ok = "error" not in output
        trace.append(f"tool  {name}({_short(raw_args)}) → {'ok' if ok else '失败'}")

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(output, ensure_ascii=False)[:6000],
            }
        )

    return {
        "messages": tool_messages,
        "tool_results": results,
        "pending_tool_calls": [],
        "steps": int(state.get("steps", 0)) + 1,
        "trace": trace,
    }


def _short(text: str, limit: int = 60) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------- reflect ----------------


def _reflect_decision(state: dict) -> str:
    results = state.get("tool_results") or []
    last_batch = results[-1:]
    all_failed = bool(last_batch) and all("error" in r["output"] for r in last_batch)

    if all_failed:
        return "generate"
    if int(state.get("steps", 0)) >= MAX_STEPS:
        return "generate"
    return "planner"


def reflect(state: dict) -> dict:
    """节点本身只负责把决策写进 trace；真正的分支由 route_after_reflect 决定。"""
    decision = _reflect_decision(state)
    note = "继续查" if decision == "planner" else "收尾"
    return {"trace": [f"reflect → {note}（已用 {state.get('steps', 0)}/{MAX_STEPS} 步）"]}


def route_after_reflect(state: dict) -> str:
    """条件边：规则判断，不调 LLM。"""
    return _reflect_decision(state)


# ---------------- generate ----------------


def _chat_or_stream(messages: list[dict], max_tokens: int) -> str:
    """有流式出口就边生成边推送，否则退回普通 chat。

    两条路的 prompt、参数完全一致，保证「流式」只改变传输方式，不改变答案。
    """
    sink = current_token_sink()
    if sink is None:
        return chat(messages, max_tokens=max_tokens)

    parts: list[str] = []
    for chunk in chat_stream(messages, max_tokens=max_tokens):
        parts.append(chunk)
        sink(chunk)
    return "".join(parts)


def _emit_answer(answer: str) -> str:
    """不需要 LLM 的兜底答案也走一次 sink，前端就不用区分两种渲染路径。"""
    sink = current_token_sink()
    if sink is not None:
        sink(answer)
    return answer


def generate(state: dict) -> dict:
    query = state.get("query", "")
    history = state.get("history") or []

    if state.get("intent") == "chitchat":
        messages = [{"role": "system", "content": CHITCHAT_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})
        return {
            "answer": _chat_or_stream(messages, max_tokens=300),
            "trace": ["generate → 闲聊回答"],
            "products": [],
        }

    docs = collect_docs(state.get("tool_results") or [])
    docs = _drop_over_budget(docs, (state.get("profile") or {}).get("budget_max"))
    if not docs:
        # planner 没调工具就直接给了话术（比如问你想要什么风格），沿用它，别硬报「没找到」
        fallback = _last_assistant_text(state)
        if fallback:
            return {
                "answer": _emit_answer(fallback),
                "trace": ["generate → 沿用 planner 话术"],
                "products": [],
            }
        return {
            "answer": _emit_answer(
                "没有检索到匹配的商品。可以试试：\n"
                "1. 换个说法或去掉限制条件\n"
                "2. 放宽预算\n"
                "3. 告诉我品牌或具体功效"
            ),
            "trace": ["generate → 无结果"],
            "products": [],
        }

    messages = rag.build_messages(
        query,
        docs,
        history,
        system_prompt=_generate_prompt()
        + f"\n\n用户画像：{memory.profile_summary(state.get('profile') or {})}",
    )
    answer = _chat_or_stream(messages, max_tokens=800)
    return {
        "answer": answer,
        "trace": [f"generate → 基于 {len(docs)} 条商品资料"],
        "products": _product_refs(docs),
    }


def _product_refs(docs: list[dict]) -> list[dict]:
    """回答里引用的商品清单，只带前端跳转需要的字段。"""
    return [
        {
            "asin": doc["asin"],
            "title": doc.get("title") or "",
            "price": doc.get("price"),
        }
        for doc in docs
    ]


def _last_assistant_text(state: dict) -> str:
    for message in reversed(state.get("messages") or []):
        if message.get("role") == "assistant" and (message.get("content") or "").strip():
            return message["content"].strip()
    return ""


def _drop_over_budget(docs: list[dict], budget) -> list[dict]:
    """确定性过滤：超预算的别指望模型自己删干净（实测会算错）。

    全超预算时保留原样——至少让模型能说「都不在预算内，最接近的是 $32」。
    """
    try:
        budget = float(budget) if budget else None
    except (TypeError, ValueError):
        budget = None
    if not budget:
        return docs

    within = [d for d in docs if d.get("price") is None or float(d["price"]) <= budget]
    return within if within else docs


def collect_docs(tool_results: list[dict]) -> list[dict]:
    """把各种工具输出统一成 rag.answer 要的 {asin,title,price,content}，按 ASIN 去重。"""
    docs: dict[str, dict] = {}

    for item in tool_results:
        output = item.get("output") or {}
        result = output.get("result")
        if result is None:
            continue
        rows = result if isinstance(result, list) else [result]

        for row in rows:
            if not isinstance(row, dict) or not row.get("asin") or "error" in row:
                continue
            asin = row["asin"]
            content = row.get("content") or row.get("snippet") or row.get("description") or ""
            features = row.get("features") or ""
            if features:
                content = f"{content}\n卖点: {features}"

            if asin in docs:
                # 同一商品可能先来自检索片段、后来自详情，取更长的描述
                if len(content) > len(docs[asin]["content"]):
                    docs[asin]["content"] = content[:1500]
                continue

            docs[asin] = {
                "asin": asin,
                "title": row.get("title") or "",
                "price": row.get("price"),
                "content": content[:1500],
            }

    return list(docs.values())[:MAX_DOCS]


# ---------------- recall ----------------


def recall(state: dict) -> dict:
    """在规划之前把本轮提到的长期偏好落盘。

    放在前面而不是回合结束，是因为本轮说的预算/偏好必须立刻影响
    工具参数和最终答案，否则会出现「用户刚说了 $30，回答却说没记录」。
    """
    profile = memory.recall(state.get("query", ""))
    changed = profile != (state.get("profile") or {})
    return {
        "profile": profile,
        "trace": ["memory → 画像已更新" if changed else "memory → 画像无变化"],
    }
