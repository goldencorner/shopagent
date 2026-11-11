"""给检索评测集标注「query 与目标商品原文重合的词」，并生成人工审阅表。

为什么需要：LLM 从商品描述反写出来的 query 会复用原文用词，BM25 等于抄答案，
指标会虚高。把重合词标出来，人工 review 时一眼能看出哪条 query 是「作弊」的。

产物:
  - 就地更新评测集 jsonl，补 shared_title_words / shared_text_words /
    overlap_title_ratio / overlap_text_ratio / brand_in_query 字段
  - 生成 Markdown 审阅表，query 里与标题重合的词用 **加粗**，
    只与描述/卖点重合的词用 *斜体*

用法:
    python scripts/06_annotate_eval.py --eval eval/retrieval_eval_generated50.jsonl \
        --review eval/review_generated50.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "products.jsonl"

# 纯功能词：它们出现在标题里不代表词面泄漏，不计入重合统计
STOP = {
    "a", "an", "the", "and", "or", "for", "in", "on", "to", "of", "with", "that",
    "it", "my", "me", "is", "are", "be", "as", "at", "by", "from", "up", "out",
    "about", "around", "per", "not", "no", "more", "than", "into", "over", "when",
    "what", "which", "who", "this", "these", "their", "they", "you", "your", "i",
    "we", "our", "can", "need", "want", "has", "have", "its", "if", "so", "but",
    "under", "less", "just", "also", "one", "two", "three", "s", "t",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokens(text: str | None) -> list[str]:
    if not text:
        return []
    return [t.lower().strip("'") for t in TOKEN_RE.findall(text) if t]


def content_tokens(text: str | None) -> list[str]:
    return [t for t in tokens(text) if t and t not in STOP]


def product_text(p: dict) -> str:
    feats = p.get("features") or []
    if isinstance(feats, (list, tuple)):
        feats = " ".join(str(x) for x in feats)
    return " ".join(str(x) for x in (p.get("title"), p.get("description"), feats) if x)


def load_catalog(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                out[p["asin"]] = p
    return out


def norm(text: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def annotate(record: dict, product: dict) -> dict:
    q = tokens(record.get("query"))
    q_content = [t for t in q if t not in STOP]
    title = set(content_tokens(product.get("title")))
    full = set(content_tokens(product_text(product)))
    desc_only = full - title

    shared_title = sorted(set(q_content) & title)
    shared_text = sorted(set(q_content) & desc_only)

    record["shared_title_words"] = shared_title
    record["shared_text_words"] = shared_text
    record["overlap_title_ratio"] = round(len(shared_title) / max(len(q_content), 1), 3)
    record["overlap_text_ratio"] = round(
        len(set(q_content) & full) / max(len(q_content), 1), 3
    )
    # 只认完整品牌名，避免 "Wicked Edge" 的 edge、"White Boutique" 的 white
    # 这类通用词把普通 query 误判成品牌泄漏
    brand_phrase = norm(product.get("brand"))
    record["brand_in_query"] = bool(brand_phrase) and f" {brand_phrase} " in f" {norm(record['query'])} "
    return record


def mark_query(query: str, title_words: set[str], text_words: set[str]) -> str:
    """把 query 里与标题重合的词加粗、只与描述重合的词斜体。"""
    def repl(m: re.Match) -> str:
        word = m.group(0)
        low = word.lower().strip("'")
        if low in title_words:
            return f"**{word}**"
        if low in text_words:
            return f"*{word}*"
        return word
    return TOKEN_RE.sub(repl, query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)

    records: list[dict] = []
    missing = 0
    with open(args.eval, encoding="utf-8") as f:
        for idx, line in enumerate((ln for ln in f if ln.strip()), start=1):
            r = json.loads(line)
            r.setdefault("id", f"g_{idx:02d}")  # 生成集没有 id，按行号补一个
            asin = r["relevant_asins"][0]
            p = catalog.get(asin)
            if p is None:
                missing += 1
                print(f"  警告：{asin} 不在商品库里，跳过标注")
                continue
            records.append(annotate(r, p))

    with open(args.eval, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    avg_title = sum(r["overlap_title_ratio"] for r in records) / max(n, 1)
    avg_text = sum(r["overlap_text_ratio"] for r in records) / max(n, 1)
    brand_hits = sum(1 for r in records if r.get("brand_in_query"))

    lines: list[str] = []
    lines.append(f"# 评测集审阅表 · {args.eval.name}\n")
    lines.append(f"- 条目数：{n}（未标注 {missing}）")
    lines.append(f"- 平均与**标题**重合率：{avg_title:.0%}")
    lines.append(f"- 平均与**完整商品文本**重合率：{avg_text:.0%}")
    lines.append(f"- query 里出现品牌的条目：{brand_hits} 条")
    lines.append("")
    lines.append("标记说明：query 里 **加粗** = 与目标标题重合的词；*斜体* = 只与描述/卖点重合的词。"
                 "重合率只统计实词，忽略 the / for / under 这类功能词。\n")
    lines.append("## 总览\n")
    lines.append("| id | 标题重合 | 全文重合 | 品牌入 query | 价格 | 目标商品 |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        p = catalog[r["relevant_asins"][0]]
        price = p.get("price")
        price_s = f"${price}" if price is not None else "—"
        rid = r.get("id", "—")
        lines.append(
            f"| {rid} | {r['overlap_title_ratio']:.0%} ({len(r['shared_title_words'])}) "
            f"| {r['overlap_text_ratio']:.0%} | {'⚠️' if r.get('brand_in_query') else ''} "
            f"| {price_s} | {p.get('title')} |"
        )
    lines.append("")

    for r in records:
        p = catalog[r["relevant_asins"][0]]
        rid = r.get("id", r["query"][:20])
        price = p.get("price")
        price_s = f"${price}" if price is not None else "—"
        title_words = set(r["shared_title_words"])
        text_words = set(r["shared_text_words"])
        feats = p.get("features") or []
        if isinstance(feats, (list, tuple)):
            feats = " | ".join(str(x) for x in feats)

        lines.append(f"## {rid}\n")
        lines.append(f"**query（重合词已标注）**：{mark_query(r['query'], title_words, text_words)}\n")
        lines.append(f"- 目标：{p.get('title')}")
        lines.append(f"- 品牌 / 价格 / 类目：{p.get('brand')} / {price_s} / {p.get('category')}")
        lines.append(f"- 与标题重合（{len(title_words)}）：{', '.join(r['shared_title_words']) or '无'}")
        lines.append(f"- 与描述重合（{len(text_words)}）：{', '.join(r['shared_text_words']) or '无'}")
        if r.get("brand_in_query"):
            lines.append("- ⚠️ query 里出现了品牌词，建议删掉")
        if r.get("review_notes"):
            lines.append(f"- 设计说明：{r['review_notes']}")
        lines.append(f"\n描述摘录：{(p.get('description') or '').replace(chr(10), ' ')[:420]}\n")
        if feats:
            lines.append(f"卖点摘录：{str(feats).replace(chr(10), ' ')[:260]}\n")
        lines.append("人工修改：query 措辞 ▢　补充可接受 ASIN ▢　删除本条 ▢\n")

    args.review.parent.mkdir(parents=True, exist_ok=True)
    args.review.write_text("\n".join(lines), encoding="utf-8")

    print(f"标注完成：{n} 条（标题重合均值 {avg_title:.0%}，全文重合均值 {avg_text:.0%}）")
    print(f"审阅表 -> {args.review}")
    if missing:
        print(f"注意：{missing} 条目标商品不在商品库里，未写入审阅表")


if __name__ == "__main__":
    main()
