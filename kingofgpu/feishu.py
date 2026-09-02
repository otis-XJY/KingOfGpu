from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.request

from .config import FeishuConfig
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


def send_text(config: FeishuConfig, text: str, *, include_gpu_status: bool = True) -> None:
    """Send a Feishu text message, optionally appending the current GPU state."""

    text = "📨 KingOfGpu 通知\n\n📌 事件信息\n" + text
    if include_gpu_status:
        text += f"\n\n━━━━━━━━━━━━━━\n{_gpu_status_text()}"
    payload: dict[str, object] = {
        "msg_type": "text",
        "content": {"text": text},
    }
    if config.at_all:
        payload["content"] = {"text": f"{text}\n@_user_1"}
        payload["at"] = {"is_at_all": True}
    if config.secret:
        timestamp = str(int(time.time()))
        string_to_sign = f"{timestamp}\n{config.secret}"
        # 飞书自定义机器人签名：以 timestamp\nsecret 作为 HMAC-SHA256 的 key，
        # 消息体为空。此前这里把 key 和 message 参数顺序写反，会导致 sign match fail。
        digest = hmac.new(
            string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
        ).digest()
        payload["timestamp"] = timestamp
        payload["sign"] = base64.b64encode(digest).decode("ascii")
    request = urllib.request.Request(
        config.webhook_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read().decode("utf-8", errors="replace")
    result = json.loads(body)
    if result.get("code", 0) not in (0, None):
        raise RuntimeError(f"飞书 Webhook 返回错误: {body}")
