FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# libgomp1 是 torch 的运行时依赖；curl 给 HEALTHCHECK 用
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# 先装 CPU 版 torch：直接 pip install torch 会拖进 2-3GB 的 CUDA 依赖
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY shopagent ./shopagent
# 数据准备和建索引也放进镜像，新机器上可以直接 docker compose run 完成初始化
COPY scripts/01_load_data.py scripts/02_build_index.py scripts/10_serve.py ./scripts/
COPY sql ./sql

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "scripts/10_serve.py", "--host", "0.0.0.0", "--store", "pg"]
