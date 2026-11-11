"""Agent 状态定义。

一步对话在图上流动时共享这一个 dict。
带 Annotated[..., operator.add] 的字段按「追加」合并，其余是覆盖。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    query: str                       # 本轮用户输入
    history: list[dict]              # 短期记忆：最近几轮对话
    profile: dict                    # 长期记忆：用户画像
    intent: str                      # router 的判定结果

    messages: Annotated[list[dict], operator.add]   # assistant/tool 消息流
    tool_results: Annotated[list[dict], operator.add]
    trace: Annotated[list[str], operator.add]       # 人看的执行轨迹

    pending_tool_calls: list[dict]   # planner 产出、tool_exec 消费
    steps: int
    answer: str
    products: list[dict]             # 本轮回答引用的商品（asin/title/price），前端做跳转
