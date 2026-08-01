from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def wait_until_ready(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"本地视觉服务提前退出，返回码 {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"{url}api/status", timeout=1.0
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("本地视觉服务未在 20 秒内启动")


def main() -> int:
    parser = argparse.ArgumentParser(description="在电脑本地运行 A4 拼图视觉上位机")
    parser.add_argument(
        "--source",
        default="usb:0",
        help="摄像头 usb:0/usb:1，或用于离线测试的 JPG/PNG 路径",
    )
    parser.add_argument("--host", default="127.0.0.1")
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
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=0.0,
        help="调试时自动运行指定秒数后退出；0 表示持续运行",
    )
    args = parser.parse_args()

    command = [
        sys.executable,
        str(PROJECT_DIR / "vision_server.py"),
        "--source",
        args.source,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--mode",
        args.mode,
        "--source-region",
        args.source_region,
        "--use-color-hints",
    ]
    process = subprocess.Popen(command, cwd=PROJECT_DIR)
    url = f"http://{args.host}:{args.port}/"
    try:
        wait_until_ready(url, process)
        print(f"电脑版上位机已启动：{url}")
        print("按 Ctrl+C 停止本地视觉服务。")
        if not args.no_browser:
            webbrowser.open(url)
        if args.run_seconds > 0:
            time.sleep(args.run_seconds)
            return 0
        return process.wait()
    except KeyboardInterrupt:
        print("\n正在停止电脑版上位机……")
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
