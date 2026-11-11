"""把上一版手写 20 条和新增手写 30 条合并成 50 条手写评测集。

上一版 20 条（eval/retrieval_eval_handwritten_draft.jsonl）已经人工设计过，
这里原样保留；新增 30 条的 query 刻意避开标题里的名词短语，只用使用场景 +
属性约束来表达需求，价格预算只在商品标价存在时给出。

用法:
    python scripts/10_make_handwritten_50.py
    python scripts/10_make_handwritten_50.py --out eval/retrieval_eval_handwritten50.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "products.jsonl"
BASE = ROOT / "eval" / "retrieval_eval_handwritten_draft.jsonl"

# (id, asin, query, 人工备注)
NEW_ITEMS = [
    ("hw_21", "B07S1RDC21",
     "compact hunting launcher for small game that cocks by hand and ships with three broad heads plus three practice arrows, under $60",
     "80 磅、225 fps、自带上弦结构、6 支箭，靠这些区分同价位十字弓"),
    ("hw_22", "B017KHNC00",
     "tight fabric sleeve for my throwing arm that keeps it warm between innings and dries fast",
     "压缩面料 + 排汗，臂套类目里靠棒球场景区分"),
    ("hw_23", "B07Q5Z92DT",
     "strap-on weights for the lower leg that let me add resistance gradually during leg raises, under $20",
     "可调负重沙袋，绑带式，两条一组"),
    ("hw_24", "B00KXXH2SM",
     "smooth replacement tire for commuting on pavement and light gravel, 35mm wide",
     "700x35c、通勤胎、排水纹路；城市/砾石路场景"),
    ("hw_25", "B0083R8UWC",
     "cubes that get a campfire going in wind and rain and don't need lighter fluid, six pack",
     "6 块装、防风防雨、烧尽成灰、无石油馏分"),
    ("hw_26", "B07CV7791P",
     "daypack for a woman with a padded slot for a laptop and a breathable back panel for commuting",
     "女款、笔记本+平板夹层、透气背板、可拆腰带"),
    ("hw_27", "B01AIRLLX4",
     "reversible nylon belt with a plastic buckle that won't trip metal detectors, under $20",
     "1.5 英寸、双层可翻转、非金属扣，机场/执法场景"),
    ("hw_28", "B07663KW19",
     "jointed hard lure that swims side to side and stays stable when trolled fast for bass",
     "分段硬饵、高速拖钓稳定、4-3/8 英寸"),
    ("hw_29", "B008277XE4",
     "narrow cargo tray for roof bars with plenty of anchor points to strap down gear and ski holders",
     "半长 59x25 英寸、钢制网篮、绑点最多的卖点"),
    ("hw_30", "B07FDV6T6H",
     "padded wall hook that holds a longboard or a guitar, screws included, under $15",
     "墙挂架、软垫挂钩、可挂滑板/吉他、含螺丝"),
    ("hw_31", "B07JCLPTTR",
     "stretchy bottoms for stretching class that are snug at the knee and looser at the ankle, under $35",
     "92% 棉 8% 氨纶、膝下收、脚踝放宽、31.5 英寸内长"),
    ("hw_32", "B008OLSIT2",
     "bit for an advanced rider that works off nose pressure with soft rubber and copper in the mouth",
     "组合式无口衔水勒、橡胶鼻带、铜丝口衔、6 英寸柄"),
    ("hw_33", "B000RZ4XH4",
     "ear protection that amplifies quiet sounds but cuts gunshots to a safe level and folds up small",
     "4 麦克风、85dB 截断、可折叠；射击场景"),
    ("hw_34", "B077NNNYC5",
     "leak-proof insulated cup for a toddler with a straw they can open one-handed, 16 ounces",
     "双层保温、防漏翻盖、单手弹吸管、带提环"),
    ("hw_35", "B0046ZZL18",
     "replacement cloth for an eight-foot billiard table in charcoal, heavy enough for commercial use, under $150",
     "8 英尺、20/21 盎司、75% 羊毛 25% 尼龙、商用级"),
    ("hw_36", "B07Z24JXJ3",
     "starter kit for walking on deep snow that includes poles and leg covers for one price",
     "雪鞋+登山杖+雪套三件套，入门款、上翘尾部"),
    ("hw_37", "B01KYHUEHW",
     "seven-and-a-half foot rod for casting light lures in freshwater with line up to 12-pound test",
     "7'6\"、中调、6-12 磅线、1/4-3/4 盎司饵"),
    ("hw_38", "B07H3NMGGJ",
     "loud squeeze horn for a child's trike or scooter that mounts with one screw, under $15",
     "金属+塑料、20-23mm 车把、单螺丝安装"),
    ("hw_39", "B008HYLW8K",
     "hand-tuned wooden call for duck hunting with a metal band around it, under $20",
     "竹制、手工调音、金属加固环；路易斯安那产"),
    ("hw_40", "B01MD2K50V",
     "short athletic socks that squeeze the foot and cushion the heel and toe for running",
     "压缩袜、脚跟/脚趾加垫、隐形袜、4 双装"),
    ("hw_41", "B07FBP4PDS",
     "clips that hold a flashlight or trekking pole onto one-inch webbing and release with one hand",
     "2 个装、1 英寸织带、单手锁扣、可换弹力绳"),
    ("hw_42", "B00EBB6VDE",
     "thin braided line with high strength for casting distance that stays invisible in green water",
     "8 股编织、低可见度绿色、高强度细径"),
    ("hw_43", "B00EUUFPII",
     "soft foam ball and bat set that lets a kid practice curveballs without getting hurt",
     "EVA/PE 泡棉、棒球缝线+高尔夫凹点、儿童训练"),
    ("hw_44", "B000ND9956",
     "firm-ground soccer shoes with fixed studs that keep feet dry, men's size 12",
     "US 12、12 颗固定钉、防水、MD PU 鞋底"),
    ("hw_45", "B001AMTI7I",
     "machine that launches clay targets in pairs for practice and folds out a seat for the operator",
     "双靶臂、可抛对靶、下折座椅、可插拖车接口"),
    ("hw_46", "B075VN7ZRX",
     "freestanding frame for pull-ups and dips at home that adjusts in height and holds heavy use",
     "高度可调、引体+双杠、钢架、三角加固"),
    ("hw_47", "B00SRCS3WA",
     "short-sleeve swim shirt with sun protection for snorkeling and aqua aerobics",
     "UPF 50+、短袖、女款；浮潜/水上健身场景"),
    ("hw_48", "B00LU2H8DY",
     "surfboard wax for cold water sold as two bars, one base coat and one top coat",
     "双支装、冷暖配方、底蜡+面蜡搭配"),
    ("hw_49", "B00DINO4R6",
     "forged alloy replacement crank arms for a mountain bike that fit a 100mm bottom bracket",
     "175mm、6066-T6 锻造、100mm 中轴、i-lock 钢嵌件"),
    ("hw_50", "B085HPR69P",
     "fishing bag with five compartments and rubber mesh pouches that sits stable on a boat deck",
     "600D 涤纶、5 个隔层、4 个橡胶网袋、防滑底座"),
]


def load_catalog(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            out[p["asin"]] = p
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "eval" / "retrieval_eval_handwritten50.jsonl")
    args = parser.parse_args()

    catalog = load_catalog(CATALOG)

    base: list[dict] = []
    with open(BASE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                base.append(json.loads(line))

    records = list(base)
    for rid, asin, query, note in NEW_ITEMS:
        p = catalog.get(asin)
        if p is None:
            raise SystemExit(f"{rid}: ASIN {asin} 不在 {CATALOG} 里")
        records.append({
            "id": rid,
            "query": query,
            "relevant_asins": [asin],
            "product_title": p.get("title"),
            "brand": p.get("brand"),
            "price": p.get("price"),
            "category": p.get("category"),
            "review_notes": note,
        })

    records.sort(key=lambda r: int(r["id"].split("_")[1]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写入 {len(records)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
