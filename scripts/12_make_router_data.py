"""生成 router LoRA 的训练数据：中文 query → {意图, 英文检索词}。

教师模型是 DeepSeek，按「类目 × 话术风格」批量生成，每批 10 条。输出字段：

    query      用户中文原话
    intent     shop / compare / chitchat
    en_query   给英文商品库的检索词（chitchat 为空）
    category   生成时指定的类目
    style      话术风格
    batch_id   批次哈希，用来按批切分 train/valid/test，避免同批泄漏

用法:
    python scripts/12_make_router_data.py --target 3000
    python scripts/12_make_router_data.py --target 100 --workers 2     # 只验证脚本

支持断点续跑：已完成的 batch_id 记在 data/router/raw.jsonl，重跑会跳过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shopagent import llm  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

INTENTS = {"shop", "compare", "chitchat"}

CATEGORIES = [
    ("yoga mat", "瑜伽垫"),
    ("camping tent", "露营帐篷"),
    ("dumbbell", "哑铃"),
    ("water bottle", "水杯"),
    ("fishing rod", "鱼竿"),
    ("bike helmet", "自行车头盔"),
    ("golf ball", "高尔夫球"),
    ("resistance band", "弹力带"),
    ("camping chair", "露营椅"),
    ("kettle", "电热水壶"),
    ("cat food", "猫粮"),
    ("cat litter", "猫砂"),
    ("baby formula", "婴儿奶粉"),
    ("dog leash", "狗绳"),
    ("backpack", "双肩包"),
    ("running shoes", "跑鞋"),
    ("headphones", "耳机"),
    ("coffee maker", "咖啡机"),
    ("desk lamp", "台灯"),
    ("air fryer", "空气炸锅"),
    ("skincare serum", "护肤精华"),
    ("sunscreen", "防晒霜"),
    ("keyboard", "键盘"),
    ("office chair", "办公椅"),
]

STYLES = {
    "explicit_shop": "明确说要买/要找某类商品，可以带具体属性",
    "constrained": "带预算、品牌、材质、尺寸或使用场景等约束，但不直接说“买”",
    "implicit_need": "只描述使用场景或遇到的问题，不提“买”“推荐”这类词",
    "explicit_compare": "明确要求在两三个商品/型号/品牌之间对比或选一个",
    "implicit_compare": "用“哪个更值”“拿不定主意”“有什么区别”表达选择，但不出现“对比”二字",
    "chitchat": "打招呼、问身份、闲聊、感谢、明确表示不买或不需要推荐",
    "followup": "接续上一轮对话的指代，如“那款”“它”“再便宜点”“刚才那个”，但单句自带一点上下文",
    "mixed": "一句话里既有寒暄/闲聊，又有真实的购物需求，购物需求是主体",
    # 长约束 query 单独生成，用来覆盖「用户认真描述需求」的场景（也是中文检索评测集的分布）
    "detailed": "30-70 字的长句，把使用场景、材质、尺寸、预算等多个约束一次说清楚",
}

# 基础批次沿用 8 种短话术，保证已生成数据的 batch_id 不变、可以断点续跑
BASE_STYLES = [name for name in STYLES if name != "detailed"]

PROMPT = """你在为电商导购 Agent 构造训练数据。请生成 {n} 条**中文**用户话术，类目是「{cat_cn}」，风格是：{style_desc}。

要求：
1. 每条都像真实用户打出来的话，{length_hint}，允许口语、省略、标点缺失；
2. intent 只能是 shop / compare / chitchat 之一：
   - shop：想买、想找商品、问推荐、问价格/功效
   - compare：要求在多个商品之间选一个、问区别
   - chitchat：打招呼、闲聊、问身份、明确表示不买或不需要推荐
3. en_query 是给英文商品库的检索词：保留品类、属性、场景、预算，保留用户给出的数字、不要换算币种，不补充用户没说的条件；chitchat 填空字符串；
4. 句式要多样，不要重复；不要出现“用户：”“示例”之类的标签；
5. 只返回 JSON，不要解释：
{{"items": [{{"query": "...", "intent": "shop", "en_query": "..."}}]}}"""


def norm(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", (text or "").lower())


def protected_queries() -> set[str]:
    """评测集里的 query 不能进训练集，否则指标虚高。"""
    paths = [
        ROOT / "eval" / "agent_tasks.jsonl",
        ROOT / "eval" / "router_hard.jsonl",
        ROOT / "eval" / "retrieval_eval_handwritten50.jsonl",
        ROOT / "eval" / "retrieval_eval_generated50.jsonl",
        ROOT / "eval" / "retrieval_eval_zh50.jsonl",
    ]
    out: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("query"):
                out.add(norm(str(item["query"])))
    return out


def valid_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    query = str(item.get("query") or "").strip()
    intent = str(item.get("intent") or "").strip().lower()
    en_query = str(item.get("en_query") or "").strip()
    if not query or len(query) > 120 or intent not in INTENTS:
        return False
    if intent == "chitchat":
        return True
    if not en_query:
        return False
    letters = sum(ch.isascii() and ch.isalpha() for ch in en_query)
    return letters >= max(3, len(en_query.replace(" ", "")) * 0.5)


def make_batch(
    cat_en: str, cat_cn: str, style: str, style_desc: str, n: int, length_hint: str
) -> list[dict]:
    messages = [
        {
            "role": "user",
            "content": PROMPT.format(
                n=n, cat_cn=cat_cn, style_desc=style_desc, length_hint=length_hint
            ),
        }
    ]
    with llm.use_model("deepseek"):
        obj = llm.chat_json(messages, temperature=0.9)
    items = obj.get("items") if isinstance(obj, dict) else None
    return [item for item in (items or []) if valid_item(item)]


def split_of(batch_id: str) -> str:
    bucket = int(hashlib.sha1(batch_id.encode()).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "valid"
    if bucket == 1:
        return "test"
    return "train"


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["batch_id"])
        except Exception:
            continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=3000, help="短话术目标条数（近似值）")
    parser.add_argument(
        "--extra-long",
        type=int,
        default=500,
        dest="extra_long",
        help="额外生成的长约束 query 条数（0 表示不加）",
    )
    parser.add_argument("--per-batch", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "router"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"

    n_batches = max(1, args.target // args.per_batch)
    rng = random.Random(args.seed)
    combos = [(cat_en, cat_cn, style) for cat_en, cat_cn in CATEGORIES for style in BASE_STYLES]
    rng.shuffle(combos)
    jobs = []
    for i in range(n_batches):
        cat_en, cat_cn, style = combos[i % len(combos)]
        batch_id = hashlib.sha1(f"{cat_en}|{style}|{i}".encode()).hexdigest()[:16]
        jobs.append((batch_id, cat_en, cat_cn, style, "3-30 字"))

    # 长约束 query：独立批次 ID，不影响已有短话术数据的断点续跑
    n_long = max(0, args.extra_long) // args.per_batch
    long_combos = list(CATEGORIES)
    rng.shuffle(long_combos)
    for i in range(n_long):
        cat_en, cat_cn = long_combos[i % len(long_combos)]
        batch_id = hashlib.sha1(f"long|{cat_en}|detailed|{i}".encode()).hexdigest()[:16]
        jobs.append((batch_id, cat_en, cat_cn, "detailed", "30-70 字"))

    done = load_done(raw_path)
    todo = [job for job in jobs if job[0] not in done]
    print(f"target={args.target} batches={len(jobs)} done={len(done)} todo={len(todo)}")
    if todo:
        lock = threading.Lock()
        written = 0
        failed = 0

        def work(job):
            batch_id, cat_en, cat_cn, style, length_hint = job
            items = make_batch(
                cat_en, cat_cn, style, STYLES[style], args.per_batch, length_hint
            )
            return job, items

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(work, job) for job in todo]
            for idx, future in enumerate(as_completed(futures), start=1):
                try:
                    (batch_id, cat_en, cat_cn, style, _length_hint), items = future.result()
                except Exception as exc:
                    failed += 1
                    print(f"  [{idx}/{len(todo)}] 失败：{type(exc).__name__}: {exc}")
                    continue
                if not items:
                    failed += 1
                    print(f"  [{idx}/{len(todo)}] 空结果：{cat_cn}/{style}")
                    continue
                with lock:
                    with open(raw_path, "a", encoding="utf-8") as f:
                        for item in items:
                            f.write(
                                json.dumps(
                                    {
                                        "batch_id": batch_id,
                                        "category": cat_en,
                                        "style": style,
                                        "query": str(item["query"]).strip(),
                                        "intent": str(item["intent"]).strip().lower(),
                                        "en_query": str(item.get("en_query") or "").strip(),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    written += len(items)
                print(f"  [{idx}/{len(todo)}] {cat_cn}/{style} +{len(items)}")

        print(f"本次写入 {written} 条，失败批次 {failed}")

    # 重新读全量 raw，去重、剔除评测 query、按批次切分
    protected = protected_queries()
    rows: list[dict] = []
    seen: set[str] = set()
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = norm(row.get("query", ""))
            if not key or key in seen or key in protected:
                continue
            seen.add(key)
            rows.append(row)

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for row in rows:
        splits[split_of(row["batch_id"])].append(row)

    for name, items in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"去重后 {len(rows)} 条，切分：" + str({k: len(v) for k, v in splits.items()}))
    for name, items in splits.items():
        counts: dict[str, int] = {}
        for row in items:
            counts[row["intent"]] = counts.get(row["intent"], 0) + 1
        print(f"  {name}: {counts}")


if __name__ == "__main__":
    main()
