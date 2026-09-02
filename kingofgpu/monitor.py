from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .gpu_notify import send_gpu_text
from .gpu import ComputeProcess, Gpu, NvidiaSmiError, list_compute_processes, list_gpus, process_user


@dataclass
class Lease:
    gpu: Gpu
    process: subprocess.Popen[str]
    known_external_pids: set[int] = field(default_factory=set)
    started_at: float = field(default_factory=time.time)
    released: bool = False


@dataclass
class UserTaskBinding:
    """A manually released GPU waiting for, or held by, one user's task."""

    user: str
    pids: set[int] = field(default_factory=set)
    pending_until: float = 0.0


def _is_background_process(process: ComputeProcess) -> bool:
    """Ignore infrastructure processes that persist independently of a user's job."""
    return "nvidia-cuda-mps-server" in process.name


def _external_processes(gpu_index: int) -> list[ComputeProcess]:
    return [
        process
        for process in list_compute_processes(gpu_index)
        if not _is_background_process(process)
    ]


def _release_instructions(gpu_index: int | None = None) -> str:
    target = str(gpu_index) if gpu_index is not None else "<GPU编号>"
    return (
        "🛠️ 释放操作\n"
        "请在服务器另开一个终端执行：\n"
        f"cd /home/xujunyi/KingOfGpu && python3 -m kingofgpu release --gpu {target}\n"
        "\n📦 释放本项目占用的全部 GPU：\n"
        "cd /home/xujunyi/KingOfGpu && python3 -m kingofgpu release --all\n"
        "\n🔒 以上命令只停止 KingOfGpu 自己启动的占用器，不会停止其他程序。"
    )


class Monitor:
    def __init__(self, project_dir: Path, config: Config) -> None:
        self.project_dir = project_dir
        self.config = config
        self.leases: dict[int, Lease] = {}
        self.logger = logging.getLogger("kingofgpu")
        self.state_path = self.project_dir / self.config.state_file
        self.slow_gpu_until, self.user_task_bindings = self._load_state()
        self.failed_gpus: set[int] = set()
        self.stop_requested = False

    def _load_state(self) -> tuple[dict[int, float], dict[int, UserTaskBinding]]:
        if not self.state_path.is_file():
            return {}, {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            slow = data.get("slow_gpus", {})
            bindings: dict[int, UserTaskBinding] = {}
            for index, raw_binding in data.get("user_task_bindings", {}).items():
                if not isinstance(raw_binding, dict):
                    continue
                user = str(raw_binding.get("user", "")).strip()
                if not user:
                    continue
                try:
                    pids = {int(pid) for pid in raw_binding.get("pids", [])}
                    bindings[int(index)] = UserTaskBinding(
                        user=user,
                        pids=pids,
                        pending_until=float(raw_binding.get("pending_until", 0.0)),
                    )
                except (TypeError, ValueError):
                    continue
            if slow:
                return {int(index): float(deadline) for index, deadline in slow.items()}, bindings
            # Migrate the short-lived state format from the previous revision.
            legacy = {int(item) for item in data.get("held_gpus", [])}
            if legacy:
                return {index: time.time() + self.config.released_poll_seconds for index in legacy}, bindings
            return {}, bindings
        except (OSError, ValueError, TypeError):
            self.logger.exception("cannot read slow-poll state %s", self.state_path)
            return {}, {}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "slow_gpus": {str(index): deadline for index, deadline in self.slow_gpu_until.items()},
                    "user_task_bindings": {
                        str(index): {"user": binding.user, "pids": sorted(binding.pids), "pending_until": binding.pending_until}
                        for index, binding in self.user_task_bindings.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _mark_slow_gpu(self, gpu_index: int) -> None:
        self.slow_gpu_until[gpu_index] = time.time() + self.config.released_poll_seconds
        self._save_state()

    def _clear_slow_gpu(self, gpu_index: int) -> None:
        if gpu_index in self.slow_gpu_until:
            self.slow_gpu_until.pop(gpu_index, None)
            self._save_state()

    def _sync_state(self) -> None:
        self.slow_gpu_until, self.user_task_bindings = self._load_state()

    def _is_bound_user_process(self, process: ComputeProcess, user: str) -> bool:
        return process_user(process.pid) == user

    def _refresh_user_task_bindings(self, now: float) -> None:
        changed = False
        for gpu_index, binding in list(self.user_task_bindings.items()):
            if binding.pids:
                active_pids = {pid for pid in binding.pids if process_user(pid) == binding.user}
                if active_pids:
                    if active_pids != binding.pids:
                        binding.pids = active_pids
                        changed = True
                    continue
                self.user_task_bindings.pop(gpu_index, None)
                # The released card belonged to the configured user's task.
                # Once every bound PID has exited, make it eligible again in
                # this same monitor iteration instead of retaining the manual
                # release cooldown.
                self.slow_gpu_until.pop(gpu_index, None)
                changed = True
                continue
            if now > binding.pending_until:
                self.user_task_bindings.pop(gpu_index, None)
                changed = True
                continue
            try:
                candidates = _external_processes(gpu_index)
            except NvidiaSmiError:
                continue
            matched = {process.pid for process in candidates if self._is_bound_user_process(process, binding.user)}
            if matched:
                binding.pids = matched
                changed = True
        if changed:
            self._save_state()

    def _bind_user_task(self, gpu_index: int, processes: list[ComputeProcess]) -> None:
        pids = {process.pid for process in processes if self._is_bound_user_process(process, self.config.release_bind_user)}
        if not pids:
            return
        self.user_task_bindings[gpu_index] = UserTaskBinding(self.config.release_bind_user, pids=pids)
        self._save_state()

    def notify(self, text: str) -> None:
        try:
            send_gpu_text(self.config.feishu, text)
            self.logger.info("Feishu notification sent")
        except Exception:
            self.logger.exception("Feishu notification failed")

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop_requested = True

    def _capacity_available(self) -> bool:
        return self.config.max_gpus == 0 or len(self.leases) < self.config.max_gpus

    def _launch(self, gpu: Gpu, external: list[ComputeProcess]) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu.index)
        env["KOGGPU_TARGET_GPU"] = str(gpu.index)
        env["KOGGPU_RESERVE_MIB"] = str(self.config.reserve_mib)
        command = [self.config.occupier_python, "-m", "kingofgpu.occupier"]
        self.logger.info("launching own occupier on GPU %d", gpu.index)
        process = subprocess.Popen(
            command,
            cwd=self.project_dir,
            env=env,
            start_new_session=True,
            text=True,
        )
        self.leases[gpu.index] = Lease(gpu, process, {item.pid for item in external})
        external_text = ", ".join(f"{item.pid}:{item.name}" for item in external) or "无"
        self.notify(
            f"🎯 发现候选 GPU {gpu.index}（{gpu.name}）\n\n"
            f"💾 候选时剩余显存：{gpu.free_mib / 1024:.2f} GiB / "
            f"{gpu.total_mib / 1024:.2f} GiB\n"
            f"🧩 现有计算进程：{external_text}\n"
            f"🔒 已启动本项目占用器，PID={process.pid}\n\n"
            f"{_release_instructions(gpu.index)}"
        )

    def _release(self, gpu_index: int, reason: str, slow_poll_after: bool = False) -> None:
        lease = self.leases.pop(gpu_index, None)
        if lease is None or lease.released:
            return
        lease.released = True
        self.logger.info("releasing own occupier on GPU %d: %s", gpu_index, reason)
        if lease.process.poll() is None:
            lease.process.send_signal(signal.SIGTERM)
            try:
                lease.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # This is the process started by this project, never an external PID.
                lease.process.kill()
                lease.process.wait(timeout=5)
        if slow_poll_after:
            self._mark_slow_gpu(gpu_index)
        self.notify(
            f"✅ 已释放 GPU {gpu_index}\n\n"
            f"📝 原因：{reason}\n"
            "🚀 现在可以手动启动你的真实代码。"
        )

    def release_all(self, reason: str = "监控器停止") -> None:
        for gpu_index in list(self.leases):
            self._release(gpu_index, reason)

    def _check_lease(self, gpu_index: int, now: float) -> None:
        lease = self.leases.get(gpu_index)
        if lease is None:
            return
        if lease.process.poll() is not None:
            self.leases.pop(gpu_index, None)
            return_code = lease.process.returncode
            self.logger.warning("own occupier on GPU %d exited with code %s", gpu_index, return_code)
            if return_code and gpu_index not in self.failed_gpus:
                self.failed_gpus.add(gpu_index)
                self.notify(
                    f"❌ GPU {gpu_index} 的占用器启动失败\n\n"
                    f"📌 退出码：{return_code}\n"
                    "🔧 请检查 config.json 的 occupier_python，并重启监控器。\n"
                    "⏸️ 该 GPU 本次运行不会继续重试。"
                )
            elif not return_code and gpu_index not in self.slow_gpu_until:
                self.notify(
                    f"ℹ️ GPU {gpu_index} 的本项目占用器已退出\n\n"
                    "🚀 现在可以手动启动你的真实代码。"
                )
            return
        if now - lease.started_at < self.config.startup_grace_seconds:
            return
        external = _external_processes(gpu_index)
        external_pids = {item.pid for item in external if item.pid != lease.process.pid}
        new_pids = external_pids - lease.known_external_pids
        authorized_new = [
            item for item in external
            if item.pid in new_pids and self._is_bound_user_process(item, self.config.release_bind_user)
        ]
        if self.config.release_on_new_process and authorized_new:
            self._bind_user_task(gpu_index, authorized_new)
            self._release(
                gpu_index,
                f"检测到用户 {self.config.release_bind_user} 的新 GPU 进程: {sorted(item.pid for item in authorized_new)}",
                slow_poll_after=True,
            )

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        self.notify(
            "🚀 监控器已启动\n\n"
            "🔎 候选条件：剩余显存 >= %.2f GiB\n\n%s"
            % (self.config.min_free_mib / 1024, _release_instructions())
        )
        while not self.stop_requested:
            try:
                # release 命令可能在监控器启动后写入状态文件；每轮同步，避免立即重新占用。
                self._sync_state()
                gpus = list_gpus()
                now = time.time()
                self._refresh_user_task_bindings(now)
                for gpu_index in list(self.leases):
                    self._check_lease(gpu_index, now)
                if self._capacity_available():
                    for gpu in sorted(gpus, key=lambda item: item.free_mib, reverse=True):
                        if gpu.index in self.leases or gpu.index in self.failed_gpus or gpu.index in self.user_task_bindings:
                            continue
                        slow_deadline = self.slow_gpu_until.get(gpu.index)
                        if slow_deadline is not None:
                            if time.time() < slow_deadline:
                                continue
                            if gpu.free_mib < self.config.min_free_mib:
                                self._mark_slow_gpu(gpu.index)
                                continue
                            external = _external_processes(gpu.index)
                            self._clear_slow_gpu(gpu.index)
                            self._launch(gpu, external)
                            if not self._capacity_available():
                                break
                            continue
                        if gpu.free_mib < self.config.min_free_mib:
                            continue
                        external = _external_processes(gpu.index)
                        self._launch(gpu, external)
                        if not self._capacity_available():
                            break
            except NvidiaSmiError:
                self.logger.exception("GPU query failed")
            except Exception:
                self.logger.exception("monitor loop failed")
            time.sleep(self.config.poll_seconds)
        self.release_all()
        self.notify("⏹️ 监控器已停止\n\n🔓 已释放本项目启动的占用器。")


def configure_logging(project_dir: Path, config: Config) -> None:
    log_path = project_dir / config.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )


def run_monitor(project_dir: Path, config: Config) -> None:
    configure_logging(project_dir, config)
    Monitor(project_dir, config).run()
