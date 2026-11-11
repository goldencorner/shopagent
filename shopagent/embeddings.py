from __future__ import annotations

from functools import lru_cache

import numpy as np

# config 必须在 sentence_transformers 之前导入，以设置 HF_ENDPOINT
from .config import settings
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """返回 (n, dim) 的归一化向量，配合 cosine 距离使用。"""
    model = get_embedder()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 64,
        convert_to_numpy=True,
    )
