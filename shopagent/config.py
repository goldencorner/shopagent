"""集中读取 .env。

注意：HF_ENDPOINT 必须在本模块导入时写入环境变量，
且任何 `import sentence_transformers` 都要排在本模块之后。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    """环境变量里写了非法值时退回默认值，不让一个笔误把服务打挂。"""
    try:
        value = int(_get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, value)


def _get_weights(name: str, default: str = "1,1") -> tuple[float, float]:
    """RRF 两路权重（向量, BM25），支持 `1,1` 或 `1:1` 写法。"""
    raw = _get(name, default).replace(":", ",")
    try:
        parts = [float(x) for x in raw.split(",") if x.strip()]
        if len(parts) != 2:
            raise ValueError
        return parts[0], parts[1]
    except ValueError:
        return 1.0, 1.0


# 让 huggingface_hub 走镜像；必须在下载模型前生效
_hf_endpoint = _get("HF_ENDPOINT")
if _hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", _hf_endpoint)

# 加载模型时的进度条/权重日志走 stderr，会插进 CLI 对话里；
# MCP server 更不能有多余输出。统一压掉，出错信息仍然保留。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    # 本地 Qwen（OpenAI 兼容）：默认指向 LM Studio 的 1234 端口
    qwen_base_url: str
    qwen_api_key: str
    qwen_model: str
    qwen_timeout: float
    # qwen-proxy 的健康检查端口：只判断反向 SSH 隧道是否连通
    qwen_health_url: str
    pg_dsn: str
    embed_model: str
    embed_dim: int
    rerank_model: str
    # 送进精排的 RRF 候选数（50 融合池里截前 N 条），见 README §4.3
    rerank_candidates: int
    # RRF 的平滑常数和两路权重（向量, BM25），见 README §4.7 消融
    rrf_k: int
    rrf_weights: tuple[float, float]
    root_dir: Path
    data_dir: Path
    eval_dir: Path


settings = Settings(
    deepseek_api_key=_get("DEEPSEEK_API_KEY"),
    deepseek_base_url=_get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    deepseek_model=_get("DEEPSEEK_MODEL", "deepseek-chat"),
    qwen_base_url=_get("QWEN_BASE_URL", "http://127.0.0.1:1234/v1"),
    qwen_api_key=_get("QWEN_API_KEY", "lm-studio"),
    # 留空时从 /v1/models 里自动挑第一个 qwen 模型
    qwen_model=_get("QWEN_MODEL"),
    qwen_timeout=float(_get("QWEN_TIMEOUT", "120")),
    qwen_health_url=_get("QWEN_HEALTH_URL", "http://127.0.0.1:1235/health"),
    pg_dsn=_get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/shopagent"),
    embed_model=_get("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
    embed_dim=int(_get("EMBED_DIM", "384")),
    rerank_model=_get("RERANK_MODEL", "BAAI/bge-reranker-base"),
    rerank_candidates=_get_int("SHOPAGENT_RERANK_CANDIDATES", 20),
    rrf_k=_get_int("SHOPAGENT_RRF_K", 60),
    rrf_weights=_get_weights("SHOPAGENT_RRF_WEIGHTS", "1,1"),
    root_dir=ROOT_DIR,
    data_dir=ROOT_DIR / "data",
    eval_dir=ROOT_DIR / "eval",
)


def _hub_cache_dir() -> Path:
    hub = _get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    home = _get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _model_available(model: str) -> bool:
    """本地目录直接用；Hub ID 则看缓存里有没有完整 snapshot。"""
    if Path(model).is_dir():
        return True
    snapshots = _hub_cache_dir() / ("models--" + model.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(
        (p / "config.json").exists() for p in snapshots.iterdir() if p.is_dir()
    )


# 两个模型都在本地缓存时强制离线。huggingface_hub 每次 from_pretrained 都会
# 反问 Hub 校验 commit，网络不通时按 1/2/4/8s 重试 5 次，第一次问答被拖几分钟。
# 只在有模型缺失时才联网——那本来就是首次下载，等待是合理的。
if (
    not _get("HF_HUB_OFFLINE")
    and _model_available(settings.embed_model)
    and _model_available(settings.rerank_model)
):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
