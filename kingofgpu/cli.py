from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import Config
from .feishu import send_text
from .gpu import NvidiaSmiError, list_compute_processes, list_gpus, process_command, process_user
from .monitor import run_monitor


def _project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _config(args: argparse.Namespace) -> Config:
    config = Config.load(Path(args.config).resolve())
    max_gpus = getattr(args, "max_gpus", None)
    if max_gpus is not None:
        config = replace(config, max_gpus=max(0, max_gpus))
    return config


def _state_path(args: argparse.Namespace) -> Path:
    try:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        state_file = str(raw.get("state_file", "run/state.json"))
    except (OSError, ValueError, TypeError):
        state_file = "run/state.json"
    return _project_dir() / state_file


def _mark_slow_gpus(
    args: argparse.Namespace,
    gpu_ids: set[int],
) -> None:
    path = _state_path(args)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        raw = {}
    slow = {int(index): float(deadline) for index, deadline in raw.get("slow_gpus", {}).items()}
    try:
        config_raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        interval = max(60, int(config_raw.get("released_poll_seconds", 60)))
    except (OSError, ValueError, TypeError):
        interval = 60
    deadline = time.time() + interval
    for gpu_id in gpu_ids:
        slow[gpu_id] = deadline
    path.parent.mkdir(parents=True, exist_ok=True)
    raw["slow_gpus"] = {str(index): value for index, value in slow.items()}
    path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _register_pending_user_binding(args: argparse.Namespace, gpu_id: int) -> None:
    """Reserve one manually released GPU for the configured user's next task."""

    path = _state_path(args)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        raw = {}
    config = _config(args)
    bindings = raw.get("user_task_bindings", {})
    if not isinstance(bindings, dict):
        bindings = {}
    bindings[str(gpu_id)] = {
        "user": config.release_bind_user,
        "pids": [],
        "pending_until": time.time() + config.bind_wait_seconds,
    }
    raw["user_task_bindings"] = bindings
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_state(args: argparse.Namespace) -> dict[str, object]:
    path = _state_path(args)
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        raw = {}
    return raw if isinstance(raw, dict) else {}


def _is_own_occupier(command: str | None) -> bool:
    return bool(command and "kingofgpu.occupier" in command)


def _status_payload(args: argparse.Namespace) -> dict[str, object]:
    config = _config(args)
    state = _load_state(args)
    bindings = state.get("user_task_bindings", {})
    if not isinstance(bindings, dict):
        bindings = {}
    required_mib = getattr(args, "required_mib", None)
    gpu_payload: list[dict[str, object]] = []
    for gpu in list_gpus():
        processes: list[dict[str, object]] = []
        own_occupier_mib = 0
        non_kingofgpu_process_mib = 0
        for process in list_compute_processes(gpu.index):
            command = process_command(process.pid)
            own_occupier = _is_own_occupier(command)
            if own_occupier:
                own_occupier_mib += process.used_mib
            else:
                non_kingofgpu_process_mib += process.used_mib
            processes.append(
                {
                    "pid": process.pid,
                    "owner": process_user(process.pid),
                    "name": process.name,
                    "command": command,
                    "used_mib": process.used_mib,
                    "is_kingofgpu_occupier": own_occupier,
                }
            )
        # NVML's whole-GPU total and its per-compute-process accounting can
        # legitimately disagree under MPS or between two consecutive queries.
        # Keep both facts instead of pretending their difference identifies a
        # particular user process.
        unattributed_used_mib = max(0, gpu.used_mib - own_occupier_mib - non_kingofgpu_process_mib)
        free_after_release_mib = min(gpu.total_mib, gpu.free_mib + own_occupier_mib)
        entry: dict[str, object] = {
            "index": gpu.index,
            "name": gpu.name,
            "total_mib": gpu.total_mib,
            "used_mib": gpu.used_mib,
            "free_mib": gpu.free_mib,
            "utilization": gpu.utilization,
            "kingofgpu_occupier_mib": own_occupier_mib,
            "non_kingofgpu_process_mib": non_kingofgpu_process_mib,
            "unattributed_used_mib": unattributed_used_mib,
            "free_after_releasing_kingofgpu_mib": free_after_release_mib,
            "user_task_binding": bindings.get(str(gpu.index)),
            "processes": processes,
        }
        if required_mib is not None:
            entry["fits_required_mib_now"] = gpu.free_mib >= required_mib
            entry["fits_required_mib_after_releasing_kingofgpu"] = free_after_release_mib >= required_mib
        gpu_payload.append(entry)
    return {
        "schema": "kingofgpu.status.v1",
        "required_mib": required_mib,
        "config": {
            "max_gpus": config.max_gpus,
            "release_bind_user": config.release_bind_user,
            "bind_wait_seconds": config.bind_wait_seconds,
        },
        "state": {
            "slow_gpus": state.get("slow_gpus", {}),
            "user_task_bindings": bindings,
        },
        "gpus": gpu_payload,
    }


def status(args: argparse.Namespace) -> int:
    try:
        payload = _status_payload(args)
    except NvidiaSmiError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    required_mib = payload["required_mib"]
    for gpu in payload["gpus"]:
        assert isinstance(gpu, dict)
        processes = gpu["processes"]
        assert isinstance(processes, list)
        ptext = ", ".join(
            f"{p['pid']}:{p['owner'] or '?'}:{p['name']}({p['used_mib']}MiB){'[KOG]' if p['is_kingofgpu_occupier'] else ''}"
            for p in processes
        ) or "none"
        fit = ""
        if required_mib is not None:
            fit = (
                f"; fit_now={gpu['fits_required_mib_now']}"
                f"; fit_after_kog_release={gpu['fits_required_mib_after_releasing_kingofgpu']}"
            )
        print(
            f"GPU {gpu['index']}: {gpu['name']}; free={gpu['free_mib']}MiB; used={gpu['used_mib']}MiB; "
            f"non_kog_processes={gpu['non_kingofgpu_process_mib']}MiB; kog={gpu['kingofgpu_occupier_mib']}MiB; "
            f"unattributed={gpu['unattributed_used_mib']}MiB; "
            f"free_after_kog_release={gpu['free_after_releasing_kingofgpu_mib']}MiB; util={gpu['utilization']}%; "
            f"processes={ptext}{fit}"
        )
    return 0


def release(args: argparse.Namespace) -> int:
    # Only signal occupiers whose command line contains this project's module.
    result = subprocess.run(
        ["pgrep", "-af", "kingofgpu.occupier"],
        check=False,
        text=True,
        capture_output=True,
    )
    matches: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2 and "kingofgpu.occupier" in fields[1]:
            try:
                matches.append((int(fields[0]), fields[1]))
            except ValueError:
                pass
    selected = matches if args.all else [item for item in matches if f"CUDA_VISIBLE_DEVICES={args.gpu}" in item[1]]
    if not selected and not args.all:
        # CUDA_VISIBLE_DEVICES is not present in all command-line renderings;
        # use nvidia-smi to map the project's occupier PID to the requested GPU.
        try:
            pids = {p.pid for p in list_compute_processes(args.gpu)}
            selected = [item for item in matches if item[0] in pids]
        except NvidiaSmiError:
            pass
    owned_pids = {pid for pid, _cmd in matches}
    slow_gpu_ids: set[int] = set()
    if selected and not args.all:
        slow_gpu_ids.add(args.gpu)
    elif selected and args.all:
        try:
            for gpu in list_gpus():
                if any(process.pid in owned_pids for process in list_compute_processes(gpu.index)):
                    slow_gpu_ids.add(gpu.index)
        except NvidiaSmiError:
            pass
    if args.all:
        if slow_gpu_ids:
            _mark_slow_gpus(args, slow_gpu_ids)
    else:
        # An explicit release is a hand-off to the configured local user, not
        # merely a fixed-duration cooldown.  The monitor binds the first
        # matching GPU task and leaves the card alone until that PID exits.
        _mark_slow_gpus(args, {args.gpu})
        _register_pending_user_binding(args, args.gpu)
    for pid, _cmd in selected:
        os.kill(pid, signal.SIGTERM)
        print(f"sent SIGTERM to own occupier PID {pid}")
    if not selected:
        print("没有找到匹配的本项目占用器。")
    return 0


def test_notify(args: argparse.Namespace) -> int:
    config = _config(args)
    send_text(
        config.feishu,
        "✅ 飞书通知测试成功\n\n"
        "🛠️ 释放指定 GPU：\n"
        "cd /home/xujunyi/KingOfGpu && python3 -m kingofgpu release --gpu <GPU编号>\n\n"
        "📦 释放本项目占用的全部 GPU：\n"
        "cd /home/xujunyi/KingOfGpu && python3 -m kingofgpu release --all\n\n"
        "🔍 释放后请使用 nvidia-smi 确认显存，再启动你的真实代码。\n"
        "🔒 以上命令只停止 KingOfGpu 自己启动的占用器，不会停止其他程序。",
    )
    print("sent")
    return 0


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


def notify_run(args: argparse.Namespace) -> int:
    """Run one command and report its outcome without exposing its arguments."""

    command = list(args.run_command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("notify-run 需要在 -- 后提供要执行的命令。", file=sys.stderr)
        return 2

    try:
        config = _config(args)
    except (OSError, ValueError) as exc:
        print(f"无法加载飞书通知配置：{exc}", file=sys.stderr)
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
        send_text(config.feishu, message, include_gpu_status=False)
    except Exception as exc:
        print(f"飞书任务通知发送失败：{exc}", file=sys.stderr)
    return return_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KingOfGpu safe GPU occupier")
    parser.add_argument("--config")
    subparsers = parser.add_subparsers(dest="command", required=True)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument(
        "--max_gpus",
        "--max-gpus",
        dest="max_gpus",
        type=int,
        default=None,
        metavar="N",
        help="仅本次运行最多占用的 GPU 数量；0 表示不限制（不修改 config.json）",
    )
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--json", action="store_true", help="输出机器可读的显存与进程归因 JSON")
    status_parser.add_argument(
        "--required-mib",
        type=int,
        default=None,
        metavar="MIB",
        help="额外报告当前及释放 KingOfGpu 后能否容纳该显存预算",
    )
    notify_parser = subparsers.add_parser("test-notify")
    notify_parser.set_defaults(func=test_notify)
    notify_run_parser = subparsers.add_parser(
        "notify-run", help="执行任务，并在结束后向飞书发送简洁通知"
    )
    notify_run_parser.add_argument("--name", help="飞书通知中的任务名称")
    notify_run_parser.add_argument("run_command", nargs=argparse.REMAINDER, metavar="-- COMMAND")
    notify_run_parser.set_defaults(func=notify_run)
    release_parser = subparsers.add_parser("release")
    release_group = release_parser.add_mutually_exclusive_group(required=True)
    release_group.add_argument("--gpu", type=int)
    release_group.add_argument("--all", action="store_true")
    release_parser.set_defaults(func=release)
    args = parser.parse_args(argv)
    if args.command == "notify-run" and not args.config:
        parser.error("notify-run 必须显式提供 --config /path/to/config.json")
    if args.config is None:
        args.config = str(_project_dir() / "config.json")
    if args.command == "monitor":
        run_monitor(_project_dir(), _config(args))
        return 0
    if args.command == "status":
        return status(args)
    return args.func(args)
