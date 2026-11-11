#!/usr/bin/env python3
"""把 Docker 网桥上的 1234 端口转发到服务器 127.0.0.1:1234。

本机 LM Studio 通过 `ssh -R 127.0.0.1:1234:127.0.0.1:1234` 落到服务器回环，
但容器访问不到宿主机的 127.0.0.1。这里在所有 docker 网桥地址（docker0 和
compose 的 br-*）上开一个口，容器无论走哪条网桥都能连到本地 Qwen。

只监听网桥地址，不监听 0.0.0.0，公网访问不到。每 30 秒重新探测一次网桥，
docker 重启后网桥 IP 变了也会自动补上新地址。
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LISTEN_PORT = 1234
HEALTH_PORT = 1235
TARGET = ("127.0.0.1", LISTEN_PORT)


def bridge_ips() -> list[str]:
    """docker0 + 所有 compose 的 br-* 网桥地址。"""
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"],
            capture_output=True, text=True, check=True,
        ).stdout
    except Exception:
        return ["172.17.0.1"]

    ips: list[str] = []
    for line in out.splitlines():
        match = re.match(r"\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)/", line)
        if match and (match.group(1) == "docker0" or match.group(1).startswith("br-")):
            ips.append(match.group(2))
    return ips or ["172.17.0.1"]


class HealthHandler(BaseHTTPRequestHandler):
    """GET /health：探测服务器回环 1234（即反向 SSH 隧道）是否可连接。

    隧道断掉时 sshd 不再监听 127.0.0.1:1234，TCP 连接会立刻失败；隧道在、
    但本机 LM Studio 没开时，1234 仍然可连。因此这个接口只回答「隧道通不通」，
    模型是否加载由 API 再走一次 /api/v0/models 判断。
    """

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.split("?", 1)[0].rstrip("/") not in ("/health", "/healthz"):
            self._send(404, {"error": "not found"})
            return

        started = time.perf_counter()
        error = ""
        try:
            with socket.create_connection(TARGET, timeout=1.0):
                ok = True
        except OSError as exc:
            ok = False
            error = type(exc).__name__

        body = {
            "tunnel": ok,
            "target": f"{TARGET[0]}:{TARGET[1]}",
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
        if error:
            body["error"] = error
        self._send(200 if ok else 503, body)

    def _send(self, code: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        return


def start_health_server(host: str) -> ThreadingHTTPServer | None:
    try:
        httpd = ThreadingHTTPServer((host, HEALTH_PORT), HealthHandler)
    except OSError as exc:
        print(f"qwen health bind failed on {host}:{HEALTH_PORT}: {exc}", flush=True)
        return None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"qwen health listening on {host}:{HEALTH_PORT}", flush=True)
    return httpd


def pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def handle(client: socket.socket) -> None:
    try:
        upstream = socket.create_connection(TARGET, timeout=5)
    except OSError:
        client.close()
        return
    client.settimeout(None)
    upstream.settimeout(None)
    a = threading.Thread(target=pipe, args=(client, upstream), daemon=True)
    b = threading.Thread(target=pipe, args=(upstream, client), daemon=True)
    a.start()
    b.start()
    a.join()
    b.join()
    client.close()
    upstream.close()


def serve(server: socket.socket, host: str) -> None:
    print(f"qwen proxy listening on {host}:{LISTEN_PORT} -> {TARGET[0]}:{TARGET[1]}", flush=True)
    while True:
        try:
            client, _ = server.accept()
        except OSError:
            return
        threading.Thread(target=handle, args=(client,), daemon=True).start()


def main() -> None:
    listeners: dict[str, socket.socket] = {}
    health_servers: dict[str, ThreadingHTTPServer] = {}
    while True:
        for host in bridge_ips():
            if host not in health_servers:
                httpd = start_health_server(host)
                if httpd is not None:
                    health_servers[host] = httpd
            if host in listeners:
                continue
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((host, LISTEN_PORT))
                server.listen(64)
            except OSError:
                server.close()
                continue
            listeners[host] = server
            threading.Thread(target=serve, args=(server, host), daemon=True).start()
        time.sleep(30)


if __name__ == "__main__":
    main()
