# 检索评测集总览（2026-11-10）

两套 50 条评测集，用途不同：生成集看「系统上限」，手写集看「没见过的问法下能到哪」。
命中口径统一为 **top-3**：目标 ASIN 排进前 3 名就算命中。

## 文件

| 用途 | 文件 |
|---|---|
| 商品改写集（50 条） | `eval/retrieval_eval_generated50.jsonl` |
| 手写集（50 条） | `eval/retrieval_eval_handwritten50.jsonl` |
| 生成集逐条审阅表 | `eval/review_generated50.md` |
| 手写集逐条审阅表 | `eval/review_handwritten50.md` |
| 逐条通道排名 | `eval/ranks_generated50.json`、`eval/ranks_handwritten50.json` |
| 聚合结果 | `eval/results_20261110_125144.json`（生成集）、`eval/results_20261110_125457.json`（手写集） |

旧的 `retrieval_eval.jsonl`（40 条商品反写）和 `retrieval_eval_handwritten_draft.jsonl`（上一版手写 20 条）
保留作历史对照；`10_make_handwritten_50.py` 会把后者的 20 条并进新手写集。

## 两套集子的差别

| | 商品改写集 | 手写集 |
|---|---|---|
| 来源 | DeepSeek 读商品 description 后反写（`scripts/05_make_eval_set.py`，seed 7.13，从 `data/products.jsonl` 随机抽 50 个描述 >200 字符的商品） | 人工写。hw_01–hw_20 沿用上一版，hw_21–hw_50 新增（`scripts/10_make_handwritten_50.py`） |
| query 语体 | 商品文案式名词短语 | 口语化使用场景 + 预算约束 |
| 平均与目标**标题**的重合词占比 | 44% | 15% |
| 平均与目标**完整文本**（标题+描述+卖点）的重合占比 | 75% | 52% |
| query 里出现完整品牌名 | 1 条（g_29 的 Zumba） | 0 条 |
| 每条 gold ASIN | 1 个 | 1 个 |

重合率只统计实词，忽略 `the / for / under / around` 这类功能词。生成集里四分之三的实词都能在目标商品原文里原样找到——这正是它指标偏高的原因。

## 重合词标注

`scripts/06_annotate_eval.py` 给每条记录补了这些字段，并生成审阅表：

| 字段 | 含义 |
|---|---|
| `shared_title_words` | query 里与目标**标题**重合的实词 |
| `shared_text_words` | query 里只与**描述/卖点**重合（标题里没有）的实词 |
| `overlap_title_ratio` | 标题重合词 / query 实词数 |
| `overlap_text_ratio` | 全文重合词 / query 实词数 |
| `brand_in_query` | query 是否出现了品牌词 |

审阅表里 query 上的标记：**加粗** = 与标题重合，*斜体* = 只与描述重合。

## 命中口径

`scripts/04_eval_retrieval.py` 已改为按 top-3 判定命中：

- `Hit@3`：目标进入前 3 名的 query 占比
- `MRR@3` / `nDCG@3`：同样只在前 3 名里计算，口径与命中一致
- `--k`（默认 5）只决定向检索器要回多少条，用来观察目标掉在 4/5 名的位置
- `--hit-k`（默认 3）可改命中窗口

```bash
# 生成集
python scripts/04_eval_retrieval.py --k 5 --hit-k 3 \
  --eval eval/retrieval_eval_generated50.jsonl --store pg
# 手写集
python scripts/04_eval_retrieval.py --k 5 --hit-k 3 \
  --eval eval/retrieval_eval_handwritten50.jsonl --store pg
```

本地没有 Docker 时把 `--store pg` 换成 `--store memory`。

## 当前结果（pg 后端，Hit@3）

| 模式 | 生成集 Hit@3 | 生成集 MRR@3 | 生成集 nDCG@3 | 手写集 Hit@3 | 手写集 MRR@3 | 手写集 nDCG@3 |
|---|---|---|---|---|---|---|
| bm25 | 0.7200 | 0.6100 | 0.6383 | 0.4200 | 0.3767 | 0.3879 |
| vector | 0.5200 | 0.4700 | 0.4826 | 0.1800 | 0.1000 | 0.1205 |
| hybrid (RRF) | 0.6200 | 0.5500 | 0.5679 | 0.3400 | 0.2867 | 0.3005 |
| hybrid_rerank | **0.7800** | **0.6867** | **0.7105** | **0.5600** | **0.4833** | **0.5031** |

## 目标排名分布（50 条 / 每格条数）

| 模式 | 生成集 1 / 2 / 3 / 4–5 / 未进前5 | 手写集 1 / 2 / 3 / 4–5 / 未进前5 |
|---|---|---|
| bm25 | 26 / 7 / 3 / 3 / 11 | 17 / 3 / 1 / 1 / 28 |
| vector | 22 / 1 / 3 / 4 / 20 | 2 / 4 / 3 / 5 / 36 |
| hybrid | 25 / 3 / 3 / 7 / 12 | 12 / 4 / 1 / 4 / 29 |
| hybrid_rerank | 31 / 4 / 4 / 5 / 6 | 21 / 5 / 2 / 1 / 21 |

## 手写集失败归因

用融合通道取回 top-50 逐条核对（`RERANK_CANDIDATES=20`）：

| 情况 | 生成集 | 手写集 |
|---|---|---|
| 进了精排候选池（融合前 20） | 46 | 33 |
| 融合 21–50，被候选截断 | 1 | 8 |
| 融合 top-50 里也没有目标 | 3 | 9 |

手写集 33 条进了精排池的条目里，28 条最终排进前 3——精排本身没问题，瓶颈在召回：
9 条（hw_03/07/17/23/25/31/36/37/42）两条通道都召不回，8 条被 top-20 候选数截掉。

## 精排候选数扫描（20 / 30 / 40）

同一套系统、同一批 query，只改 `RERANK_CANDIDATES`：

| 候选数 | 生成集 Hit@3 | 生成集 MRR@3 | 手写集 Hit@3 | 手写集 MRR@3 |
|---|---|---|---|---|
| **20** | **0.78** | **0.6867** | **0.56** | **0.4833** |
| 30 | 0.74 | 0.6600 | 0.54 | 0.4767 |
| 40 | 0.72 | 0.6533 | 0.56 | 0.4933 |

结论：加大候选数救不回被截断的那 8 条，反而有代价。

- 手写集 20→40：救回 hw_13、hw_46（融合排名 40/39，正好在 40 以内），但 hw_14、hw_45 被新进来的干扰项挤到前 3 之外，净变化为 0；
- 生成集 20→40：g_01、g_03、g_41 三条被挤掉，Hit@3 从 0.78 掉到 0.72；
- 候选数越多，精排耗时线性上升（部署机约 265 ms/对）。

所以保持默认 20。真正该优化的是召回：9 条 query 在融合 top-50 里根本没有目标商品。
20/30/40 的逐条排名在 `eval/ranks_{generated,handwritten}50_rc{20,30,40}.json`。

## 注意

- 两套集的 gold 都只有一个 ASIN。哑铃、泡沫轴、软冰袋这类通用品，前 3 名里出现同质商品其实合理，被判为「未命中」有评测设计成分。人工 review 时建议给这些条目补 1–2 个可接受 ASIN。
- 手写集部分 query 带预算从句（`under $15` 等），但检索层没有价格过滤，这些词目前只是噪声，不要当成系统缺陷来归因。
- 改完评测集后重跑 `scripts/06_annotate_eval.py` 刷新标注和审阅表，再跑 `04_eval_retrieval.py`。
