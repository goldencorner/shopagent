"""从 ModelScope 下载模型到本地目录（服务器上没有 HuggingFace 时用）。

用法:
    python3 download_models.py BAAI/bge-small-en-v1.5 /opt/shopagent/models/bge-small-en-v1.5
    python3 download_models.py --list BAAI/bge-reranker-base

只依赖标准库，服务器上有 python3 就能跑。
ModelScope 的 resolve 地址: https://modelscope.cn/models/<id>/resolve/master/<path>
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://modelscope.cn"
UA = {"User-Agent": "shopagent-deploy/1.0"}
SKIP_SUFFIXES = (".md", ".gitattributes")
SKIP_DIRS = ("onnx",)


def repo_files(model_id: str, root: str = "") -> list[tuple[str, str, int]]:
    """返回 [(type, path, size)]，type 是 blob 或 tree，目录递归展开。"""
    query = urllib.parse.urlencode({"Revision": "master", "Root": root})
    url = f"{BASE}/api/v1/models/{model_id}/repo/files?{query}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    out: list[tuple[str, str, int]] = []
    for item in (data.get("Data") or {}).get("Files") or []:
        kind = item.get("Type")
        path = item.get("Path", "")
        size = int(item.get("Size") or 0)
        if not path:
            continue
        if kind == "tree":
            if path.split("/")[0] in SKIP_DIRS:
                continue
            out.append(("tree", path, 0))
            out.extend(repo_files(model_id, path))
        else:
            out.append(("blob", path, size))
    return out


def download(model_id: str, target: Path) -> None:
    entries = repo_files(model_id)
    blobs = [(path, size) for kind, path, size in entries if kind == "blob"]
    if not blobs:
        raise SystemExit(f"{model_id} 文件列表为空，检查模型 ID 或网络")
    # safetensors 和 bin 二选一，优先 safetensors（体积一样，加载更快）
    has_safetensors = any(path.endswith("model.safetensors") for path, _ in blobs)
    wanted = [
        (path, size)
        for path, size in blobs
        if not path.endswith(SKIP_SUFFIXES)
        and not (has_safetensors and path.endswith("pytorch_model.bin"))
    ]
    if not wanted:
        raise SystemExit(f"{model_id} 没有可下载的文件")

    target.mkdir(parents=True, exist_ok=True)
    for path, expected in wanted:
        dest = target / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/models/{model_id}/resolve/master/{urllib.parse.quote(path)}"
        print(f"↓ {path}", flush=True)
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                f.write(block)
        size = dest.stat().st_size
        if size != expected:
            raise SystemExit(f"{path} 大小不符：期望 {expected}，实际 {size}")
        print(f"  {size / 1e6:.1f} MB", flush=True)
    print(f"完成 -> {target}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--list":
        for kind, path, size in repo_files(args[1]):
            if kind == "blob":
                print(f"{path}\t{size}")
        return
    if len(args) != 2:
        raise SystemExit(__doc__)
    download(args[0], Path(args[1]))


if __name__ == "__main__":
    main()
