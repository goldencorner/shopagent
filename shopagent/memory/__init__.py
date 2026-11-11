"""两级记忆。

- 长期：用户画像（预算、偏好、品牌偏好、禁忌），落 data/user_profile.json，
       跨会话保留，进 prompt 时压成一行摘要。
- 短期：会话历史，落 data/sessions/<id>.json，只保留最近 N 轮。

没有上向量记忆：这个规模下（画像就几个字段）语义检索是过度设计，
面试时能说清"为什么不用"比硬塞一个更值分。
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from ..config import settings
from ..llm import chat_json

PROFILE_PATH = settings.data_dir / "user_profile.json"
SESSION_DIR = settings.data_dir / "sessions"
MAX_TURNS = 10  # 保留最近 20 条消息

PROFILE_FIELDS = ("budget_max", "skin_type", "preferred_brand", "avoid", "notes")

_EXTRACT_PROMPT = """从用户这句话里抽取**长期有效**的购物偏好。只抽取用户明确说过的，不要推测。

用户说：{user}

字段说明：
- budget_max: 预算上限，数字（美元）。没说就是 null
- skin_type: 肤质/发质/适用人群，如 "sensitive", "oily", "dry"
- preferred_brand: 明确表示喜欢的品牌
- avoid: 明确排斥的成分或品类（数组）
- notes: 其他**长期**偏好，一句话

不要记录一次性请求，例如「再便宜点」「多推荐几个」「换个牌子」——
这些只属于当前这轮对话，不是画像。
但用户在购物需求里明确给出的预算上限是长期偏好，要记录：
「预算 30 美元」→ budget_max: 30；「敏感肌能用的」→ skin_type: "sensitive"。
只有相对表达（「再便宜点」「别太贵」）不记录。

只返回有内容的字段，没有就给空对象 {{}}。
返回 JSON：{{"budget_max": null, "skin_type": null, "preferred_brand": null, "avoid": [], "notes": null}}"""


# ---------------- 长期：用户画像 ----------------


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_profile(profile: dict, new_facts: dict) -> dict:
    """新事实覆盖旧值，空值不覆盖。"""
    merged = dict(profile)
    for key, value in (new_facts or {}).items():
        if key not in PROFILE_FIELDS or value in (None, "", [], {}):
            continue
        merged[key] = value
    return merged


def profile_summary(profile: dict) -> str:
    if not profile:
        return "（还没有用户偏好信息）"
    parts: list[str] = []
    if profile.get("budget_max"):
        parts.append(f"预算上限 ${profile['budget_max']}")
    if profile.get("skin_type"):
        parts.append(f"偏好/适用: {profile['skin_type']}")
    if profile.get("preferred_brand"):
        parts.append(f"偏好品牌: {profile['preferred_brand']}")
    if profile.get("avoid"):
        parts.append(f"避开: {', '.join(map(str, profile['avoid']))}")
    if profile.get("notes"):
        parts.append(f"备注: {profile['notes']}")
    return "；".join(parts) if parts else "（还没有用户偏好信息）"


def recall(user_text: str) -> dict:
    """从本轮用户原话里抽偏好并落盘，返回合并后的完整画像。

    只看用户说了什么，不看助手回答——助手的话里全是「再便宜点」这类
    一次性请求，写进长期记忆就成了噪声。
    """
    prompt = _EXTRACT_PROMPT.format(user=(user_text or "")[:800])
    try:
        facts = chat_json([{"role": "user", "content": prompt}])
    except Exception:
        return load_profile()

    facts = {k: v for k, v in (facts or {}).items() if k in PROFILE_FIELDS}
    if not facts:
        return load_profile()

    profile = merge_profile(load_profile(), facts)
    save_profile(profile)
    return profile


# ---------------- 短期：会话历史 ----------------


class Session:
    def __init__(self, session_id: str = "default", messages: list[dict] | None = None):
        self.session_id = _safe_name(session_id)
        self.messages: list[dict] = messages or []

    @classmethod
    def load(cls, session_id: str = "default") -> "Session":
        path = SESSION_DIR / f"{_safe_name(session_id)}.json"
        if not path.exists():
            return cls(session_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls(session_id)
        return cls(session_id, data.get("messages", []))

    def append(self, role: str, content: str, meta: dict | None = None) -> None:
        message: dict = {"role": role, "content": content}
        if meta:
            message["meta"] = meta
        self.messages.append(message)
        self.messages = self.messages[-MAX_TURNS * 2 :]

    def history(self) -> list[dict]:
        # 只把 role/content 交给 LLM；meta 是前端展示用的，不能进 prompt。
        return [
            {"role": message.get("role", ""), "content": message.get("content", "")}
            for message in self.messages
        ]

    def messages_with_meta(self) -> list[dict]:
        """给 HTTP 前端读的完整消息，包含商品引用等展示元信息。"""
        return [dict(message) for message in self.messages]

    def save(self) -> None:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSION_DIR / f"{self.session_id}.json"
        path.write_text(
            json.dumps({"messages": self.messages}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def list_sessions(limit: int = 50) -> list[dict]:
    """历史会话列表：标题取第一条用户消息，时间用文件 mtime（旧文件没有时间字段）。"""
    if not SESSION_DIR.is_dir():
        return []
    items: list[tuple[float, dict]] = []
    for path in SESSION_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (json.JSONDecodeError, OSError):
            continue
        messages = data.get("messages") or []
        if not messages:
            continue
        title = next(
            (str(m.get("content") or "") for m in messages if m.get("role") == "user"),
            "",
        )
        items.append(
            (
                mtime,
                {
                    "session_id": path.stem,
                    "title": (title.strip() or "（无标题）")[:80],
                    "message_count": len(messages),
                    "updated_at": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                },
            )
        )
    items.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in items[:limit]]


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", (name or "default").strip())
    return cleaned or "default"


def clear_session(session_id: str = "default") -> bool:
    path = SESSION_DIR / f"{_safe_name(session_id)}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def clear_all(session_id: str = "default") -> None:
    """调试用：清画像和指定会话。"""
    save_profile({})
    clear_session(session_id)
