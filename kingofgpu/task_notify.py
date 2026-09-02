"""Standalone long-running task notifier; intentionally independent from GPU code."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import FeishuConfig
from .feishu import send_text


@dataclass(frozen=True)
class TaskNotifyConfig:
    feishu: FeishuConfig

    @classmethod
    def load(cls, path: Path) -> "TaskNotifyConfig":
        if not path.is_file():
            raise FileNotFoundError(
                f"任务通知配置不存在: {path}; 请先 cp task-notify.example.json task-notify.json"
            )
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("purpose") != "task-notify":
            raise ValueError("任务通知配置必须包含 purpose: task-notify")
        feishu_raw = raw.get("feishu", {})
        if not isinstance(feishu_raw, dict):
            raise ValueError("任务通知配置中的 feishu 必须是对象")
        webhook_url = str(feishu_raw.get("webhook_url", "")).strip()
        if not webhook_url or "REPLACE_ME" in webhook_url:
            raise ValueError("请在 task-notify.json 中填写有效的 feishu.webhook_url")
        return cls(
            feishu=FeishuConfig(
                webhook_url=webhook_url,
                secret=str(feishu_raw.get("secret", "")),
                at_all=bool(feishu_raw.get("at_all", False)),
            )
        )


def _task_name(command: list[str], explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    if len(command) > 1 and Path(command[0]).name in {"bash", "sh"}:
        return Path(command[1]).name
    return Path(command[0]).name


def _duration_text(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时{minutes}分{seconds}秒"
    if minutes:
        return f"{minutes}分{seconds}秒"
    return f"{seconds}秒"


def _load_config(args: argparse.Namespace) -> TaskNotifyConfig | None:
    try:
        return TaskNotifyConfig.load(Path(args.config).resolve())
    except (OSError, ValueError, TypeError) as exc:
        print(f"无法加载任务飞书通知配置：{exc}", file=sys.stderr)
        return None


def _send_task_text(config: TaskNotifyConfig, result_text: str) -> None:
    send_text(config.feishu, "📨 长任务通知\n\n📌 任务结果\n" + result_text)


def run(args: argparse.Namespace) -> int:
    command = list(args.run_command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("run 需要在 -- 后提供要执行的命令。", file=sys.stderr)
        return 2

    config = _load_config(args)
    if config is None:
        return 2

    task_name = _task_name(command, args.name)
    started_at = time.monotonic()
    interrupted_by: int | None = None
    process: subprocess.Popen[object] | None = None

    def forward_signal(signum: int, _frame: object) -> None:
        nonlocal interrupted_by
        if interrupted_by is None:
            interrupted_by = signum
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for signum in previous_handlers:
            signal.signal(signum, forward_signal)
        try:
            process = subprocess.Popen(command, start_new_session=True)
        except OSError as exc:
            return_code = 127
            error = f"无法启动任务：{exc}"
        else:
            return_code = process.wait()
            error = None
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)

    duration = _duration_text(time.monotonic() - started_at)
    if interrupted_by is not None:
        outcome = f"⚠️ 任务已中断（{signal.Signals(interrupted_by).name}）"
    elif return_code == 0:
        outcome = "✅ 任务已完成"
    else:
        outcome = "❌ 任务失败"
    message = (
        f"{outcome}\n"
        f"🏷️ 任务：{task_name}\n"
        f"⏱️ 耗时：{duration}\n"
        f"🚪 退出码：{return_code}"
    )
    if error:
        message += f"\n⚠️ {error}"
    try:
        _send_task_text(config, message)
    except Exception as exc:
        print(f"飞书任务通知发送失败：{exc}", file=sys.stderr)
    return return_code


def test_notify(args: argparse.Namespace) -> int:
    config = _load_config(args)
    if config is None:
        return 2
    _send_task_text(
        config,
        "✅ 任务机器人通知测试成功\n"
        "此机器人仅用于跨项目长任务的完成、失败和中断提醒。",
    )
    print("sent")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Standalone Feishu notifier for long-running tasks")
    parser.add_argument("--config", required=True, help="task-notify.json 的绝对路径")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="执行任务，并在结束后通知任务机器人")
    run_parser.add_argument("--name", help="飞书通知中的任务名称")
    run_parser.add_argument("run_command", nargs=argparse.REMAINDER, metavar="-- COMMAND")
    run_parser.set_defaults(func=run)
    test_parser = subparsers.add_parser("test-notify", help="测试任务机器人通知")
    test_parser.set_defaults(func=test_notify)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
