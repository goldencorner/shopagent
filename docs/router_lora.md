# 用 LoRA 小模型接管 router：意图分类 + 中文→英文检索词

## 1. 目标

原来的 router 用 DeepSeek 判意图（一次 API 调用，P50 约 700–900ms），planner 再把中文需求翻成英文检索词。这个方案有两个问题：

- 每次对话多一次云端调用，router 的延迟占比不低；
- 改写和意图判断分散在两个 prompt 里，prompt 越长越容易漂。

这里用一个 1.5B 的 LoRA 模型一次前向同时输出：

```json
{"intent": "shop|compare|chitchat", "en_query": "english search keywords"}
```

接入原则和本地 27B 一样：**能用就用，不能用就回退老链路**。本地模型未加载、超时、JSON 非法或置信度低，都会自动回退 DeepSeek，不影响服务可用性。

## 2. 数据

`scripts/12_make_router_data.py` 用 DeepSeek 做教师，按「24 个类目 × 8 种话术风格」批量生成，另加 500 条 30–70 字的长约束 query。当前数据：

| split | 条数 | shop | compare | chitchat |
|---|---:|---:|---:|---:|
| train | 2502 | 1191 | 916 | 395 |
| valid | 321 | 159 | 114 | 48 |
| test | 333 | 160 | 118 | 55 |

- 按生成批次切分，避免同一模板同时进 train/test；
- 自动剔除 21 条 agent 任务、39 条意图边界集、100 条英文检索评测、50 条中文检索评测里的 query（去重后实际从训练集剔掉 2 条与边界集重复的样本）；
- 人工边界集在 `eval/router_hard.jsonl`（39 条），覆盖闲聊夹商品词、隐含对比、否定、指代。

重新生成（需要 `.env` 里的 DeepSeek Key）：

```bash
python scripts/12_make_router_data.py --target 3000 --extra-long 500
```

## 3. 训练（AutoDL 4090）

本地是 Windows + AMD，不适合训练；用 AutoDL 的 4090 镜像。实测全流程约 25 分钟（装依赖 + 下模型 + 训练 + 转 GGUF），费用 2 元以内。

训练包已打好：`dist/router_lora_bundle.zip`（约 150KB），只含数据、训练脚本和 prompt 模块，不含 `.env`、商品数据和索引。上传解压后目录结构：

```
router_lora_bundle/
├── README_AUTODL.md
├── data/router/{train,valid,test}.jsonl
├── scripts/13_train_router_lora.py
└── shopagent/{__init__.py,router_prompt.py}
```

如果改了脚本或数据，把 `shopagent/{__init__,router_prompt}.py`、`scripts/13_train_router_lora.py`、`data/router/{train,valid,test}.jsonl` 按原路径重新拷进 `dist/router_lora_bundle/` 再压缩即可。

```bash
# AutoDL 上（压缩包已在 /root/autodl-tmp）
cd /root/autodl-tmp && unzip router_lora_bundle.zip && cd router_lora_bundle

# AutoDL 镜像常见 torch 2.1.x：最新 transformers 5.x 要求 torch >= 2.6，
# 直接 -U 会禁用 PyTorch。按 torch 版本二选一：
pip install --no-cache-dir "transformers==4.46.3" "peft==0.13.2" \
  "accelerate==0.34.2" "datasets==2.21.0"        # torch 2.1.x
# pip install -U transformers peft datasets accelerate  # torch >= 2.6

export HF_ENDPOINT=https://hf-mirror.com

# 基座：hf-mirror 实测很慢（16KB/s），建议改走 ModelScope，然后 --model 指本地目录
pip install --no-cache-dir modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct')"

# 训练 + 合并 adapter
python scripts/13_train_router_lora.py \
  --model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct --epochs 3 --merge
```

关键超参：Qwen2.5-1.5B-Instruct，LoRA r=16 / alpha=32 / dropout=0.05，target 为 q/k/v/o/gate/up/down，lr 2e-4，3 epoch，bf16，max_len 512，batch 8 × 梯度累积 2。max_len 不能设太小：prompt 模板本身约 290 字符，加上最长 120 字的 query，256 会把答案截掉。

产物：

- `data/router/lora_out/`：adapter + tokenizer + `train_meta.json`
- `data/router/merged/`：合并后的完整模型（转 GGUF 用）

训练前可先跑 `--dry-run` 检查数据构造（只需要 tokenizer，不需要 GPU）；`--smoke` 是 200 条 × 1 epoch 的跑通检查。

### 3.1 实测（2026-11-15，RTX 4090 24GB）

- **基座下载**：`hf-mirror.com` 实测 16KB/s，3.1GB 要 50 分钟以上，不可用。改用 ModelScope：
  `modelscope download --model Qwen/Qwen2.5-1.5B-Instruct --local_dir <dir>`，实测 10.7MB/s，约 5 分钟。
  下完之后用 `--model <本地目录>` 启动训练，不再碰 HF。
- **训练**：2502 条 / 有效 batch 16 ≈ 156 step/epoch，3 epoch 共 468 step，242 秒；train_loss 0.242。
- **valid eval_loss**：0.361 → 0.329 → 0.356。第 3 个 epoch 已轻微回升，说明这个数据量 2 epoch 就够；
  结果按原定 3 epoch 保留，没有回滚。
- **合并 + 转 GGUF**：约 1 分钟，`router-q8.gguf` 1.53GiB（1,646,572,416 字节）。
- **llama.cpp 源码**：GitHub 直连 `raw` 只有 10KB/s，codeload 约 27KB/s。用 `https://gh-proxy.com/` 前缀拉
  `b5200` tag 的 tarball（21MB，5MB/s 以上）。选 `b5200` 是因为它的 `convert_hf_to_gguf.py` 是单文件自包含，
  master 已经拆成 `conversion/` 包，只下一个文件跑不起来。
- **财务**：实例开机约 25 分钟（含装依赖、下模型、训练、转换），按 4090 单价不到 2 元。

## 4. 转 GGUF 并加载到 LM Studio

LM Studio 只认 GGUF，需要先把 adapter 合并进基座（上面的 `--merge` 已做），再转换：

```bash
# GitHub 直连太慢时走代理镜像；b5200 的 convert 脚本是单文件自包含
curl -L -o llama.cpp.tar.gz \
  https://gh-proxy.com/https://github.com/ggerganov/llama.cpp/archive/refs/tags/b5200.tar.gz
tar xzf llama.cpp.tar.gz && cd llama.cpp-b5200

# 不用 pip install -r requirements.txt：脚本优先用仓库自带的 gguf-py，
# 装依赖反而可能把 AutoDL 镜像里的 torch/transformers 顶掉
python convert_hf_to_gguf.py ../data/router/merged \
    --outfile router-q8.gguf --outtype q8_0
```

把 `router-q8.gguf` 下载到本地，在 LM Studio 里导入/加载（Q8_0 实测 1.53GiB，和 27B 可以共存；建议 router 模型常驻）。

```bash
lms import -c --user-repo shopagent/router-lora-q8 router-q8.gguf
lms server start
lms load router-lora --gpu max -y      # 模型 id 由文件名派生，就是 router-lora
```

**一个坑**：如果 `~/.lmstudio/models` 是软链（比如整体指到 `E:\models`），运行中的 LM Studio 不会对软链目标做实时
扫描，`lms import` 拷进去的文件在 `lms ls` 里看不到，重启 HTTP server 也没用——必须重启一次 LM Studio 应用，
启动时才会重建索引。索引建好之后再用 `lms import` 加新模型就正常了。

## 5. 配置

```bash
# 开启本地 router；默认仍是 llm，不影响现有行为
SHOPAGENT_ROUTER=local

# LM Studio 里的实际模型 id（lms ls 或 /v1/models 里复制；本次实测是 router-lora）
SHOPAGENT_LOCAL_ROUTER_MODEL=router-lora

# 可选
SHOPAGENT_LOCAL_ROUTER_BASE_URL=http://127.0.0.1:1234/v1
SHOPAGENT_LOCAL_ROUTER_MIN_CONF=0.55
SHOPAGENT_LOCAL_ROUTER_TIMEOUT=8
# 走反向 SSH 隧道时再配；本地开发不要配，否则每次探测白等超时
# SHOPAGENT_LOCAL_ROUTER_HEALTH_URL=http://127.0.0.1:1235/health
```

页面输入框上方、模型选择器旁的「LoRA 路由」开关，或 `GET/POST /config/router` 可以热切
`llm` / `local`；`.env` 只是默认值，重启回到默认（和精排候选数滑块同一套设计）。

回退链路（`shopagent/agent/nodes.py::router`）：

1. 探测本地服务与模型加载状态（缓存：可用 30s / 不可用 5s）；
2. 可用 → 调本地模型；
3. 未加载 / 超时 / JSON 非法 → 回退 DeepSeek；
4. 置信度 < 阈值 → 回退 DeepSeek；
5. trace 里记录原因，例如 `router → shop（local 0.91）`、`router → shop（local，本地不可用（模型未加载），回退 llm）`。

planner 会在系统 prompt 里拿到 `rewritten_query`。**注意**：当前措辞是「优先使用它」，
实测 46/50 条原样照抄，planner 基本不会修正小模型的改写（见 6.2.1）。

## 6. 评测

### 6.1 意图与改写质量

```bash
python scripts/14_eval_router.py --sets hard,test \
    --modes keyword,deepseek,local --local-model <模型 id>
```

当前基线（未微调，2026-11-13）：

| 集合 | 方案 | 准确率 | macro-F1 | P50 | 改写 Jaccard |
|---|---|---:|---:|---:|---:|
| hard (39) | keyword | 0.641 | 0.573 | 0ms | — |
| hard (39) | DeepSeek | 0.923 | 0.922 | 876ms | 0.657 |
| hard (39) | base 1.5B | 0.769 | 0.740 | 134ms | 0.440 |

微调后（2026-11-15，同一台机器上重跑，DeepSeek 有随机性，数值和 11-13 不完全一致）：

| 集合 | 方案 | 准确率 | macro-F1 | P50 | P95 | 改写 Jaccard |
|---|---|---:|---:|---:|---:|---:|
| hard (39) | keyword | 0.641 | 0.573 | 0ms | 0ms | — |
| hard (39) | DeepSeek | 0.897 | 0.896 | 654ms | 904ms | 0.644 |
| hard (39) | base 1.5B | 0.769 | 0.740 | 134ms | 183ms | 0.440 |
| hard (39) | **LoRA 1.5B** | **0.846** | **0.847** | **128ms** | **163ms** | 0.431 |
| test (333) | keyword | 0.676 | 0.617 | 0ms | 0ms | — |
| test (333) | DeepSeek | 0.985 | 0.981 | 774ms | 1132ms | 0.629 |
| test (333) | **LoRA 1.5B** | **0.979** | **0.976** | **157ms** | **232ms** | **0.663** |

结论：

- 边界集从基座 0.769 提到 0.846，超过关键词规则（0.641），但没到 DeepSeek 的 0.897，**P50 快 5 倍**；
- test 集学生和教师只差 0.6 个点，改写 Jaccard 还反超教师（0.663 vs 0.629），说明学到的模式和教师一致；
- hard 集剩下的 6 个错例有共同点：**闲聊里夹着商品词或流程词**（“谢谢，我先看看瑜伽垫”）、
  **否定和全选**（“别给我推荐贵的”“这两款都要”）、**问能力**（“你能帮我下单吗”）、
  **否定式比较**（“有没有比 $50 的哑铃更好的”）。这些是 1.5B 的语用理解短板，加这类样本比扩量更有效。

### 6.2 改写对检索的影响

`scripts/15_make_zh_retrieval.py` 把 50 条英文检索评测翻成中文（保留 ASIN 标签），`scripts/16_eval_router_retrieval.py` 对比几种改写方式（含微调前后的本地模型）：

```bash
python scripts/15_make_zh_retrieval.py
python scripts/16_eval_router_retrieval.py --store pg --local-model <模型 id>
```

hybrid_rerank，Hit@3。微调前后各跑了一次，planner 那行受 LLM 随机性影响在 0.44–0.46 之间：

| 方式 | Hit@3 | MRR | nDCG@3 | P50 改写 |
|---|---:|---:|---:|---:|
| 中文直接检索 | 0.000 | 0.000 | 0.000 | 0ms |
| DeepSeek 现场翻译 | 0.440 | 0.377 | 0.393 | 911ms |
| base 1.5B 改写（微调前） | 0.120 | 0.100 | — | 252ms |
| **LoRA 1.5B 改写（微调后）** | **0.320** | **0.270** | **0.283** | **216ms** |
| 原始英文 query（上界） | 0.560 | 0.483 | 0.503 | 0ms |

结论：

- 改写带来的收益是实打实的：不做改写 0.000，微调后 0.320，**超过 base 1.5B 的 0.120 两倍多**；
- 但仍明显落后 DeepSeek 的 0.440，差距集中在 zh50 这类 30–70 字的长约束 query：1.5B 能翻对品类，
  但容易丢“材质/尺寸/兼容性”这些次要约束；
- 所以接入策略上应该是**双通道**：短 query 走本地（216ms），长约束或低置信度回退 DeepSeek。
  注意当前代码只实现了置信度阈值回退，**按长度分流还没做**（见 §7 最后一条）；
  小模型该吃哪部分流量的结论已经有数据支撑，实现留到下一步。

50 条样本的置信区间大约 ±0.13（95%），0.32 vs 0.44 属于方向性证据，不是精确排名。

#### 6.2.1 planner 兜住了多少（2026-11-15 补测）

上面 0.320 是**绕过 planner**、直接拿 `en_query` 检索的数字。线上 `en_query` 只是 planner
prompt 的第 7 条提示，planner 可以改它。`scripts/17_eval_planner_absorb.py` 用
`nodes.py` 里真实的 planner prompt 跑了一遍（50 条同一评测集，hit 口径 top-3）：

| 方式 | Hit@3 | MRR | nDCG@3 | planner P50 | 采纳提示（Jaccard≥0.6） |
|---|---:|---:|---:|---:|---:|
| 原始英文 query（上界） | 0.560 | 0.483 | 0.503 | — | — |
| `lora_direct` | 0.320 | 0.270 | 0.283 | — | — |
| `planner_llm`（老链路） | 0.400 | 0.370 | 0.378 | 1030ms | — |
| `planner_lora`（当前线上提示） | 0.400 | 0.337 | 0.353 | 913ms | 46/50 |
| `planner_lora_soft`（"初稿，请核对修正"） | 0.460 | 0.390 | 0.408 | 934ms | 10/50 |

- **命中率与老链路持平**（0.400 vs 0.400，逐条 +7 / -7，p=1.0），MRR 低 0.033；
- 但**不是 planner 纠正了改写**：46/50 条原样照抄（Jaccard 中位数 0.857），只补了 `max_price`。
  `lora_direct → planner_lora` 的 +4 里有 3 条检索词几乎没变，是预算过滤带来的；
  只有 1 条真改对了品类（`2 tennis balls ...` → `tennis racket backpack ...`）；
- 提示措辞本身是瓶颈：「优先使用它」把 planner 锁死在不完美的改写上，换成
  「初稿、可能漏约束、请核对修正」后命中 0.460，收益集中在带预算的 27 条（0.333 → 0.444）。
  样本 50 条、CI ±0.13，属于方向性证据。
- 结论：当前提示下"命中率不差"是成立的，但**不能说是 planner 兜住了改写质量**；
  真正该做的是改措辞 + 在输入侧按否定词/长度分流。

### 6.3 端到端消融

`scripts/11_ablation.py agent` 已加入 `router=local` 配置，和 baseline / keyword / shop / guard=off 一起跑
21 个 agent 任务，输出通过率、LLM 调用次数、P50。

实测（2026-11-15）：

| 配置 | 任务通过 | 意图 | 平均 LLM 调用 | P50 | P95 |
|---|---|---|---|---|---|
| baseline（DeepSeek 路由） | 21/21 | 2/2 | 4.91 | 9.4s | 13.8s |
| **router=local（LoRA）** | **21/21** | **2/2** | **3.95** | **8.5s** | 15.3s |
| keyword 规则 | 21/21 | 2/2 | 3.91 | 8.7s | 10.9s |
| shop（去掉 router） | 19/21 | 0/2 | 3.90 | 8.7s | 12.5s |
| guard=off | 16/21 | 2/2 | 4.90 | 9.7s | 12.2s |

- 省下的约 1 次 LLM 调用就是 router 那次；通过率和 baseline 持平，说明端到端没有被小模型带坏；
- 端到端 P50 只快了 0.8s，因为检索（精排）占了 5–6s，router 在整条链路里权重不大；
- P95 更高是单次运行的噪声（21 条任务、含一次 15s 的慢检索），不作结论。

## 7. 预期与风险（实测之后）

- 边界集 0.846：超过了“1.5B 微调能到 90% 左右”的乐观预期下限，也说明**不能用 test 集的 0.979 当宣传数字**——
  那是教师标签、同分布，只能说明蒸馏成功，不能说明线上能力。
- 改写仍是短板：检索 Hit@3 0.320 vs DeepSeek 0.440。**长约束 query 的回退策略不能省**。
- 置信度来自 logprobs，LM Studio 能返回，但**在这份数据上几乎没有区分度**：实测 6 个 hard 错例的置信度是
  0.81–1.0（“别给我推荐贵的” 0.9998、“你能帮我下单吗” 0.9986），和正确样本一样高。
  所以阈值回退只能拦住格式问题，拦不住语用错误；**下一步应该按 query 长度/是否含否定词做输入侧分流**，
  而不是把阈值调高（调高只会误伤正确样本）。
- 3 epoch 时 valid loss 已回升（0.329 → 0.356），下次可以试 2 epoch + 更大 batch，或者加 hard 样本再训。
- 本地 router 常驻约 1.5GiB 显存，27B 同时加载时注意 KV cache。

## 8. 简历可写的点

- 用 DeepSeek 教师蒸馏 3.1k 条中文导购 query，构造「意图（shop/compare/chitchat）+ 英文检索词」双输出数据集，
  覆盖 24 个类目 × 8 种话术风格，按生成批次切分 train/valid/test 防模板泄漏，评测集 query 全部从训练集剔除；
- LoRA 微调 Qwen2.5-1.5B（r=16，7 个投影层，18.5M 可训练参数 / 1.18%），merge 后转 GGUF Q8_0 部署到 LM Studio；
  router P50 从 774ms 降到 157ms（-80%），人工边界集准确率从规则 64.1% / 基座 76.9% 提升到 84.6%，
  同分布 test 集 97.9%，与教师模型只差 0.6 个点；
- 设计可用性探测 + 置信度回退：本地未加载 / 超时 / JSON 非法 / 置信度低时自动回退 DeepSeek，服务可用性不受影响；
  实测错例的置信度也在 0.81–1.0，说明阈值只能拦住格式问题、拦不住语用错误；
- 在 50 条中文长约束检索评测上量化改写收益：不做改写 Hit@3 = 0 → base 1.5B 12.0% → LoRA 改写 32.0% →
  DeepSeek 44.0%（英文原句上界 56%），改写延迟 216ms vs 911ms，给出「短 query 本地、长约束回退大模型」的接入结论。
