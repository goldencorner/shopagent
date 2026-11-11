from __future__ import annotations

import os
from functools import lru_cache

import torch

# config 必须先导入，用于设置 HF_ENDPOINT
from ..config import settings
from sentence_transformers import CrossEncoder

# bf16 能把权重减半，但 CPU 没有 bf16 指令集时代价很大：
# 本机 30 对 fp32 3.2s、bf16 19.6s（约 6 倍），部署机是 AMD EPYC 7K62（Zen 2），
# 同样没有 AVX512-BF16/AMX。所以默认 fp32，换到支持 bf16 的机器再开：
# SHOPAGENT_RERANK_DTYPE=bfloat16
_ENV_DTYPE = os.getenv("SHOPAGENT_RERANK_DTYPE", "float32").lower()
_RERANK_DTYPE = torch.bfloat16 if _ENV_DTYPE in ("bf16", "bfloat16") else torch.float32

# chunk 本身最长 1000 字符（p50 767），1024 等于不截断。精排耗时随输入长度
# 近似线性：本机 30 对 1024→512 字符从 3.4s 降到 1.9s。但实测截到 512 后
# Recall@5 从 0.90 掉到 0.775，所以默认不截断，只作为实验开关。
_RERANK_MAX_CHARS = int(os.getenv("SHOPAGENT_RERANK_MAX_CHARS", "1024"))

# 动态 int8 量化：只量化 Linear，CPU 上通常快 1.5–2 倍。质量要用 04_eval 复测。
_QUANT = os.getenv("SHOPAGENT_RERANK_QUANT", "").lower() in ("1", "int8", "true")

# 一次喂给模型的 query-doc 对数。默认 32 时 padding 到最长样本，4 核机器上
# 反而慢：部署机实测 20 对 5.7s→batch=8 时 3.1s，接近 1.8 倍。
_BATCH_SIZE = int(os.getenv("SHOPAGENT_RERANK_BATCH", "8"))


class _Int8Reranker:
    """动态 int8 量化的 cross-encoder。

    sentence-transformers 的 CrossEncoder 在量化后调用约定对不上（会把
    BatchEncoding 当成 input_ids），这里直接走 transformers，接口保持 predict()。
    """

    def __init__(self, path: str) -> None:
        import torch.nn as nn
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        self.model.eval()

    def predict(self, pairs: list[tuple[str, str]], batch_size: int | None = None):
        pairs = list(pairs)
        size = batch_size or len(pairs) or 1
        out = []
        for i in range(0, len(pairs), size):
            part = pairs[i : i + size]
            inputs = self.tokenizer(
                [q for q, _ in part],
                [d for _, d in part],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits.view(-1).float()
            out.append(logits)
        return torch.cat(out).numpy() if out else torch.tensor([]).numpy()


@lru_cache(maxsize=1)
def get_reranker():
    if _QUANT:
        return _Int8Reranker(settings.rerank_model)
    kwargs = {}
    if _RERANK_DTYPE != torch.float32:
        kwargs["torch_dtype"] = _RERANK_DTYPE
    return CrossEncoder(settings.rerank_model, model_kwargs=kwargs)


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Cross-encoder 精排。输入 query-doc 对，比双塔召回更准但更慢。"""
    if not candidates:
        return []
    model = get_reranker()
    pairs = [(query, c["content"][:_RERANK_MAX_CHARS]) for c in candidates]
    scores = model.predict(pairs, batch_size=_BATCH_SIZE)

    ranked = []
    for cand, score in zip(candidates, scores):
        doc = dict(cand)
        doc["rerank_score"] = float(score)
        ranked.append(doc)
    ranked.sort(key=lambda d: -d["rerank_score"])
    return ranked[:top_k]
