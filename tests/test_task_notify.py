from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from kingofgpu.config import FeishuConfig
from kingofgpu import gpu_notify, task_notify


def _config() -> task_notify.TaskNotifyConfig:
    return task_notify.TaskNotifyConfig(
        feishu=FeishuConfig(webhook_url="https://example.invalid/task-hook")
    )


def _args(*, command: list[str], name: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(run_command=command, name=name, config="unused.json")


def test_task_run_reports_success_without_gpu_dependencies() -> None:
    process = Mock(pid=321)
    process.wait.return_value = 0
    with patch("kingofgpu.task_notify._load_config", return_value=_config()), patch(
        "kingofgpu.task_notify.subprocess.Popen", return_value=process
    ) as popen, patch("kingofgpu.task_notify.time.monotonic", side_effect=[10.0, 75.0]), patch(
        "kingofgpu.task_notify._send_task_text"
    ) as send:
        assert task_notify.run(_args(command=["bash", "pipeline.sh"])) == 0
    popen.assert_called_once_with(["bash", "pipeline.sh"], start_new_session=True)
    message = send.call_args.args[1]
    assert "任务已完成" in message
    assert "任务：pipeline.sh" in message
    assert "耗时：1分5秒" in message
    assert "退出码：0" in message


def test_task_run_preserves_failed_exit_code_and_explicit_name() -> None:
    process = Mock(pid=321)
    process.wait.return_value = 7
    with patch("kingofgpu.task_notify._load_config", return_value=_config()), patch(
        "kingofgpu.task_notify.subprocess.Popen", return_value=process
    ), patch("kingofgpu.task_notify.time.monotonic", side_effect=[10.0, 11.0]), patch(
        "kingofgpu.task_notify._send_task_text"
    ) as send:
        assert task_notify.run(_args(command=["python", "train.py"], name="experiment-42")) == 7
    message = send.call_args.args[1]
    assert "任务失败" in message
    assert "任务：experiment-42" in message
    assert "退出码：7" in message


def test_task_run_forwards_interrupt_and_notifies_once() -> None:
    process = Mock(pid=321)
    process.poll.return_value = None
    handlers: dict[int, object] = {}

    def record_signal(signum: int, handler: object) -> None:
        handlers[signum] = handler

    def interrupt_wait() -> int:
        handler = handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)
        return -signal.SIGTERM

    process.wait.side_effect = interrupt_wait
    with patch("kingofgpu.task_notify._load_config", return_value=_config()), patch(
        "kingofgpu.task_notify.subprocess.Popen", return_value=process
    ), patch("kingofgpu.task_notify.signal.getsignal", return_value=object()), patch(
        "kingofgpu.task_notify.signal.signal", side_effect=record_signal
    ), patch("kingofgpu.task_notify.os.killpg") as killpg, patch(
        "kingofgpu.task_notify.time.monotonic", side_effect=[10.0, 11.0]
    ), patch("kingofgpu.task_notify._send_task_text") as send:
        assert task_notify.run(_args(command=["python", "train.py"])) == -signal.SIGTERM
    killpg.assert_called_once_with(321, signal.SIGTERM)
    assert send.call_count == 1
    assert "任务已中断" in send.call_args.args[1]
    assert "SIGTERM" in send.call_args.args[1]


def test_task_run_keeps_exit_code_when_notification_fails(capsys) -> None:
    process = Mock(pid=321)
    process.wait.return_value = 4
    with patch("kingofgpu.task_notify._load_config", return_value=_config()), patch(
        "kingofgpu.task_notify.subprocess.Popen", return_value=process
    ), patch("kingofgpu.task_notify.time.monotonic", side_effect=[10.0, 11.0]), patch(
        "kingofgpu.task_notify._send_task_text", side_effect=RuntimeError("network down")
    ):
        assert task_notify.run(_args(command=["python", "train.py"])) == 4
    assert "飞书任务通知发送失败：network down" in capsys.readouterr().err


def test_task_run_rejects_bad_config_before_starting_task(capsys) -> None:
    with patch("kingofgpu.task_notify._load_config", return_value=None), patch(
        "kingofgpu.task_notify.subprocess.Popen"
    ) as popen:
        assert task_notify.run(_args(command=["python", "train.py"])) == 2
    popen.assert_not_called()
    assert capsys.readouterr().err == ""


def test_task_config_rejects_gpu_config_and_requires_task_purpose() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(
            json.dumps({"purpose": "gpu-monitor", "feishu": {"webhook_url": "https://example.invalid/hook"}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="purpose: task-notify"):
            task_notify.TaskNotifyConfig.load(path)


def test_task_cli_passes_command_after_separator() -> None:
    process = Mock(pid=321)
    process.wait.return_value = 0
    with TemporaryDirectory() as directory:
        config_path = Path(directory) / "task-notify.json"
        config_path.write_text(
            json.dumps(
                {
                    "purpose": "task-notify",
                    "feishu": {"webhook_url": "https://example.invalid/task-hook"},
                }
            ),
            encoding="utf-8",
        )
        with patch("kingofgpu.task_notify.subprocess.Popen", return_value=process) as popen, patch(
            "kingofgpu.task_notify._send_task_text"
        ), patch("kingofgpu.task_notify.time.monotonic", side_effect=[10.0, 11.0]):
            assert task_notify.main(
                [
                    "--config",
                    str(config_path),
                    "run",
                    "--name",
                    "demo",
                    "--",
                    "bash",
                    "pipeline.sh",
                ]
            ) == 0
    popen.assert_called_once_with(["bash", "pipeline.sh"], start_new_session=True)


def test_gpu_notifier_adds_status_without_affecting_task_notifier() -> None:
    gpu = SimpleNamespace(index=0, name="GPU", used_mib=100, total_mib=1000, free_mib=900, utilization=0)
    with patch("kingofgpu.gpu_notify.list_gpus", return_value=[gpu]), patch(
        "kingofgpu.gpu_notify.send_text"
    ) as send:
        gpu_notify.send_gpu_text(FeishuConfig(webhook_url="https://example.invalid/gpu-hook"), "GPU 可用")
    assert "GPU 使用情况" in send.call_args.args[1]
    assert "GPU 可用" in send.call_args.args[1]


def test_task_notifier_does_not_import_gpu_module() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import kingofgpu.task_notify; print('kingofgpu.gpu' in sys.modules)",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.strip() == "False"
