from __future__ import annotations

from .config import FeishuConfig
from .feishu import send_text
from .gpu import NvidiaSmiError, list_gpus


def _gpu_status_text() -> str:
    try:
        gpus = list_gpus()
    except NvidiaSmiError as exc:
        return f"📊 GPU 使用情况\n⚠️ 暂时无法获取（{exc}）"

    if not gpus:
        return "📊 GPU 使用情况\n⚠️ 未发现 GPU"

    lines = [
        "📊 GPU 使用情况",
        "┌─────┬──────────────────────┬─────────────────┬────────────┬────────┐",
        "│ GPU │ 型号                 │ 已用 / 总量     │ 剩余       │ 利用率 │",
        "├─────┼──────────────────────┼─────────────────┼────────────┼────────┤",
    ]
    for gpu in gpus:
        name = gpu.name.replace("│", "/").replace("\n", " ")[:20]
        if gpu.utilization >= 80:
            utilization = f"🔴 {gpu.utilization}%"
        elif gpu.utilization >= 50:
            utilization = f"🟡 {gpu.utilization}%"
        else:
            utilization = f"🟢 {gpu.utilization}%"
        lines.append(
            f"│ {gpu.index:^3} │ {name:<20} │ "
            f"{gpu.used_mib / 1024:>6.2f} / {gpu.total_mib / 1024:<6.2f} │ "
            f"{gpu.free_mib / 1024:>8.2f} │ {utilization:^6} │"
        )
    lines.append("└─────┴──────────────────────┴─────────────────┴────────────┴────────┘")
    lines.append("图例：🟢 < 50%   🟡 50%–79%   🔴 ≥ 80%")
    return "\n".join(lines)


def send_gpu_text(config: FeishuConfig, event_text: str) -> None:
    """Send a GPU-monitor event with its current GPU-status appendix."""

    send_text(
        config,
        "📨 KingOfGpu GPU 通知\n\n"
        "📌 事件信息\n"
        f"{event_text}\n\n"
        "━━━━━━━━━━━━━━\n"
        f"{_gpu_status_text()}",
    )
