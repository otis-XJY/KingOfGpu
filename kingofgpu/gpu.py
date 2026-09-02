from __future__ import annotations

import csv
import io
import subprocess
from dataclasses import dataclass


class NvidiaSmiError(RuntimeError):
    pass


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization: int


@dataclass(frozen=True)
class ComputeProcess:
    gpu_index: int
    pid: int
    name: str
    used_mib: int


def _run_query(query: str) -> str:
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise NvidiaSmiError(f"无法查询 nvidia-smi: {detail.strip()}") from exc
    return result.stdout


def list_gpus() -> list[Gpu]:
    rows = csv.reader(io.StringIO(_run_query("index,name,memory.total,memory.used,memory.free,utilization.gpu")))
    return [
        Gpu(
            index=int(row[0].strip()),
            name=row[1].strip(),
            total_mib=int(row[2].strip()),
            used_mib=int(row[3].strip()),
            free_mib=int(row[4].strip()),
            utilization=int(row[5].strip()),
        )
        for row in rows
        if len(row) >= 6
    ]


def list_compute_processes(gpu_index: int) -> list[ComputeProcess]:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise NvidiaSmiError(f"无法查询 GPU {gpu_index} 的进程: {detail.strip()}") from exc
    processes: list[ComputeProcess] = []
    for row in csv.reader(io.StringIO(result.stdout)):
        if not row or not row[0].strip():
            continue
        try:
            processes.append(
                ComputeProcess(
                    gpu_index=gpu_index,
                    pid=int(row[0].strip()),
                    name=row[1].strip() if len(row) > 1 else "unknown",
                    used_mib=int(row[2].strip()) if len(row) > 2 else 0,
                )
            )
        except ValueError:
            continue
    return processes


def process_user(pid: int) -> str | None:
    """Return the OS owner for one PID, or ``None`` when it no longer exists."""

    try:
        result = subprocess.run(
            ["ps", "-o", "user=", "-p", str(pid)],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    user = result.stdout.strip()
    return user or None


def process_command(pid: int) -> str | None:
    """Return the full command line for one PID when it is still available."""

    try:
        result = subprocess.run(
            ["ps", "-o", "args=", "-p", str(pid)],
            check=True,
            text=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    command = result.stdout.strip()
    return command or None
