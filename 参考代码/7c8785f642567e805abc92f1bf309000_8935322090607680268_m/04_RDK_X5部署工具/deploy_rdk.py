from __future__ import annotations

import argparse
import getpass
import os
import posixpath
import shlex
from pathlib import Path

import paramiko


PROJECT_FILES = (
    "config.json",
    "main.py",
    "vision_server.py",
    "simulate_playing_cards.py",
    "taught_layout.json",
    "README.md",
    "README_无上位机先看.md",
    "启动RDK_X5服务.sh",
    "停止RDK_X5服务.sh",
    "启动有上位机模式.sh",
    "启动无上位机模式.sh",
    "停止无上位机服务.sh",
    "安装无上位机开机自启.sh",
)
PROJECT_DIRECTORIES = ("puzzle_vision", "web")


def _mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = ""
    for part in path.strip("/").split("/"):
        current = f"{current}/{part}"
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def _iter_files(root: Path) -> list[Path]:
    paths = [
        root / name for name in PROJECT_FILES if (root / name).is_file()
    ]
    for directory in PROJECT_DIRECTORIES:
        base = root / directory
        if base.is_dir():
            paths.extend(
                path
                for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    return paths


def _run(
    client: paramiko.SSHClient, command: str
) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command)
    return (
        stdout.channel.recv_exit_status(),
        stdout.read().decode("utf-8", errors="replace"),
        stderr.read().decode("utf-8", errors="replace"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the current puzzle-vision sources to an RDK X5"
    )
    parser.add_argument("--host", default="192.168.1.9")
    parser.add_argument("--user", default="sunrise")
    parser.add_argument(
        "--remote-dir", default="/home/sunrise/puzzle_vision"
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help=(
            "directory containing the RDK runtime files; defaults to the "
            "directory containing deploy_rdk.py"
        ),
    )
    parser.add_argument(
        "--password-env",
        default="RDK_PASSWORD",
        help="environment variable containing the SSH password",
    )
    parser.add_argument(
        "--mode",
        choices=("fixed", "unknown-white", "unknown-pattern"),
        default="unknown-pattern",
    )
    parser.add_argument(
        "--bind-host",
        choices=("0.0.0.0", "127.0.0.1"),
        default="0.0.0.0",
        help="0.0.0.0 enables the web console; 127.0.0.1 is headless-only",
    )
    parser.add_argument(
        "--camera-source",
        default="usb:/dev/video0",
        help="camera source passed to vision_server.py",
    )
    parser.add_argument("--no-restart", action="store_true")
    args = parser.parse_args()

    password = os.environ.get(args.password_env)
    if password is None:
        password = getpass.getpass("RDK SSH password: ")
    root = (
        Path(args.project_dir).expanduser().resolve()
        if args.project_dir
        else Path(__file__).resolve().parent
    )
    if not (root / "vision_server.py").is_file():
        raise FileNotFoundError(
            f"RDK project directory is invalid: {root}"
        )
    files = _iter_files(root)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.host,
        username=args.user,
        password=password,
        timeout=10,
    )
    try:
        sftp = client.open_sftp()
        try:
            _mkdirs(sftp, args.remote_dir)
            for local in files:
                relative = local.relative_to(root).as_posix()
                remote = posixpath.join(args.remote_dir, relative)
                _mkdirs(sftp, posixpath.dirname(remote))
                sftp.put(str(local), remote)
                print(f"uploaded {relative}")
        finally:
            sftp.close()

        code, output, error = _run(
            client,
            (
                f"cd {args.remote_dir} && "
                "python3 -m compileall -q puzzle_vision "
                "main.py vision_server.py simulate_playing_cards.py"
            ),
        )
        if code != 0:
            raise RuntimeError(error or output)
        if not args.no_restart:
            command = (
                "pids=$(pgrep -f '^python3 vision_server.py' || true); "
                'if [ -n "$pids" ]; then '
                "kill $pids; "
                "for n in $(seq 1 20); do "
                "pgrep -f '^python3 vision_server.py' >/dev/null "
                "|| break; sleep 0.5; "
                "done; "
                "pids=$(pgrep -f '^python3 vision_server.py' || true); "
                'if [ -n "$pids" ]; then kill -9 $pids; fi; '
                "fi; "
                "fuser -k 8000/tcp >/dev/null 2>&1 || true; "
                "sleep 2; "
                f"cd {args.remote_dir} && "
                "setsid -f python3 vision_server.py "
                f"--host {shlex.quote(args.bind_host)} --port 8000 "
                f"--source {shlex.quote(args.camera_source)} "
                f"--mode {args.mode} --source-region upper "
                "--use-color-hints "
                "</dev/null >vision_server.log 2>&1"
            )
            code, output, error = _run(client, command)
            if code != 0:
                raise RuntimeError(error or output)
        code, output, error = _run(
            client,
            (
                "sleep 2; "
                "v4l2-ctl -d /dev/video0 "
                "--set-ctrl=gain=32,auto_exposure=3,"
                "exposure_dynamic_framerate=1 >/dev/null 2>&1 || true; "
                "pgrep -af '^python3 vision_server.py'; "
                f"cd {args.remote_dir} && tail -n 12 vision_server.log"
            ),
        )
        if code != 0:
            raise RuntimeError(error or output)
        print(output.rstrip())
    finally:
        client.close()


if __name__ == "__main__":
    main()
