from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeishuConfig:
    webhook_url: str
    secret: str = ""
    at_all: bool = False


@dataclass(frozen=True)
class Config:
    feishu: FeishuConfig
    occupier_python: str = "/home/xujunyi/anaconda3/envs/dubins/bin/python"
    min_free_mib: int = 30720
    reserve_mib: int = 512
    poll_seconds: int = 15
    released_poll_seconds: int = 60
    max_gpus: int = 1
    log_file: str = "logs/kingofgpu.log"
    state_file: str = "run/state.json"
    release_on_new_process: bool = True
    release_bind_user: str = "xujunyi"
    bind_wait_seconds: int = 600
    startup_grace_seconds: int = 20
    occupier_start_timeout_seconds: int = 90

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.is_file():
            raise FileNotFoundError(
                f"配置文件不存在: {path}; 请先 cp config.example.json config.json"
            )
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        feishu_raw = raw.get("feishu", {})
        webhook_url = str(feishu_raw.get("webhook_url", "")).strip()
        if not webhook_url or "REPLACE_ME" in webhook_url:
            raise ValueError("请在 config.json 中填写有效的 feishu.webhook_url")
        config = cls(
            feishu=FeishuConfig(
                webhook_url=webhook_url,
                secret=str(feishu_raw.get("secret", "")),
                at_all=bool(feishu_raw.get("at_all", False)),
            ),
            occupier_python=str(
                raw.get("occupier_python", "/home/xujunyi/anaconda3/envs/dubins/bin/python")
            ),
            min_free_mib=int(raw.get("min_free_mib", 30720)),
            reserve_mib=int(raw.get("reserve_mib", 512)),
            poll_seconds=max(2, int(raw.get("poll_seconds", 15))),
            released_poll_seconds=max(60, int(raw.get("released_poll_seconds", 60))),
            max_gpus=max(0, int(raw.get("max_gpus", 1))),
            log_file=str(raw.get("log_file", "logs/kingofgpu.log")),
            state_file=str(raw.get("state_file", "run/state.json")),
            release_on_new_process=bool(raw.get("release_on_new_process", True)),
            release_bind_user=str(raw.get("release_bind_user", "xujunyi")).strip(),
            bind_wait_seconds=max(60, int(raw.get("bind_wait_seconds", raw.get("released_poll_seconds", 600)))),
            startup_grace_seconds=max(0, int(raw.get("startup_grace_seconds", 20))),
            occupier_start_timeout_seconds=max(
                10, int(raw.get("occupier_start_timeout_seconds", 90))
            ),
        )
        if config.min_free_mib <= config.reserve_mib:
            raise ValueError("min_free_mib 必须大于 reserve_mib")
        if not config.release_bind_user:
            raise ValueError("release_bind_user 不能为空")
        return config
