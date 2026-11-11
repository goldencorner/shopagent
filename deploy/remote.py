"""通过 SSH 在云服务器上执行命令 / 上传文件（Windows 本机没有 sshpass 时用）。

密码从环境变量 SHOPAGENT_SSH_PASSWORD 读，不放在命令行里。

用法:
    python remote.py exec --host 1.2.3.4 --user ubuntu --cmd "uname -a"
    python remote.py put  --host 1.2.3.4 --user ubuntu --src deploy/images.tar.gz --dst /opt/shopagent/images.tar.gz
    python remote.py sudo --host 1.2.3.4 --user ubuntu --cmd "apt-get update"
"""

from __future__ import annotations

import argparse
import os
import sys

import paramiko


def connect(host: str, user: str, port: int, key: str | None = None) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict = {"hostname": host, "port": port, "username": user, "timeout": 25}
    if key:
        kwargs["key_filename"] = key
    else:
        password = os.environ.get("SHOPAGENT_SSH_PASSWORD")
        if not password:
            raise SystemExit("缺少 SHOPAGENT_SSH_PASSWORD 环境变量（或改用 --key）")
        kwargs["password"] = password
    client.connect(**kwargs)
    return client


def run(client: paramiko.SSHClient, cmd: str, timeout: float | None = None) -> int:
    print(f"$ {cmd}", flush=True)
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=True)
    for line in iter(stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
    code = stdout.channel.recv_exit_status()
    err = stderr.read().decode("utf-8", "replace").strip()
    if err:
        print(err, file=sys.stderr)
    return code


def put(client: paramiko.SSHClient, src: str, dst: str) -> None:
    sftp = client.open_sftp()
    size = os.path.getsize(src)
    print(f"上传 {src} -> {dst} ({size / 1e6:.0f} MB)", flush=True)
    last = [-1]

    def progress(done: int, total: int) -> None:
        pct = int(done * 100 / total)
        if pct >= last[0] + 5 or done == total:
            last[0] = pct
            print(f"  {pct}%  {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)

    sftp.put(src, dst, callback=progress)
    sftp.close()
    print("上传完成", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["exec", "put", "sudo"])
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--key", help="私钥路径；不传则用 SHOPAGENT_SSH_PASSWORD")
    parser.add_argument("--cmd")
    parser.add_argument("--src")
    parser.add_argument("--dst")
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()

    client = connect(args.host, args.user, args.port, args.key)
    try:
        if args.action == "put":
            if not args.src or not args.dst:
                raise SystemExit("put 需要 --src 和 --dst")
            put(client, args.src, args.dst)
            return
        if not args.cmd:
            raise SystemExit("exec/sudo 需要 --cmd")
        cmd = f"sudo -n bash -lc {shquote(args.cmd)}" if args.action == "sudo" else args.cmd
        code = run(client, cmd, args.timeout)
        sys.exit(code)
    finally:
        client.close()


def shquote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
