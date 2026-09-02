from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.request

from .config import FeishuConfig


def send_text(config: FeishuConfig, text: str) -> None:
    """Send a text payload through a Feishu custom bot without enrichment."""
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
