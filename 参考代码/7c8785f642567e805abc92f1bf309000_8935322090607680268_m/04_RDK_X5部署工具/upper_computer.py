from __future__ import annotations

import argparse
import getpass
import shlex
import time
import urllib.request
import webbrowser

import paramiko


def connect(host: str, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=8)
    return client


def remote(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=15)
    code = stdout.channel.recv_exit_status()
    return (
        code,
        stdout.read().decode("utf-8", "replace").strip(),
        stderr.read().decode("utf-8", "replace").strip(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 RDK A4 拼图视觉上位机")
    parser.add_argument("--host", default="192.168.1.9")
    parser.add_argument("--user", default="sunrise")
    parser.add_argument(
        "--password",
        default=None,
        help="SSH password; omit to enter it securely",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--mode",
        choices=("fixed", "unknown-white", "unknown-pattern"),
        default="fixed",
    )
    parser.add_argument(
        "--source-region",
        choices=("upper", "lower", "auto"),
        default="upper",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="启动服务但不自动打开浏览器",
    )
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()

    password = args.password or getpass.getpass("RDK SSH password: ")
    client = connect(args.host, args.user, password)
    try:
        remote(client, "pkill -f '[v]ision_server.py' || true")
        if args.stop:
            print("地瓜派视觉服务已停止。")
            return 0
        command = (
            "cd /home/sunrise/puzzle_vision && "
            "setsid -f python3 vision_server.py "
            f"--port {int(args.port)} --mode {shlex.quote(args.mode)} "
            f"--source-region {shlex.quote(args.source_region)} "
            "--use-color-hints >> vision_server.log 2>&1"
        )
        code, _, error = remote(client, command)
        if code != 0:
            raise RuntimeError("地瓜派视觉服务启动失败：" + error)
        _, process, _ = remote(
            client,
            "ps -eo pid,args | grep '[p]ython3 vision_server.py' | head -n 1",
        )
        print("地瓜派视觉服务已提交：" + (process or "正在启动"))
    finally:
        client.close()

    url = f"http://{args.host}:{args.port}/"
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}api/status", timeout=1.0) as response:
                if response.status == 200:
                    print(f"上位机已启动：{url}")
                    if not args.no_browser:
                        webbrowser.open(url)
                    return 0
        except OSError:
            time.sleep(0.5)
    client = connect(args.host, args.user, password)
    try:
        _, log, _ = remote(
            client,
            "cd /home/sunrise/puzzle_vision && "
            "tail -n 80 vision_server.log 2>/dev/null || true",
        )
    finally:
        client.close()
    raise RuntimeError(
        "地瓜派视觉服务未在 15 秒内启动。板端日志：\n"
        + (log or "（日志为空）")
    )


if __name__ == "__main__":
    raise SystemExit(main())
