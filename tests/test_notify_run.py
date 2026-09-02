from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest

from kingofgpu import cli
from kingofgpu.config import Config, FeishuConfig
from kingofgpu.feishu import send_text


def _config() -> Config:
    return Config(feishu=FeishuConfig(webhook_url="https://example.invalid/hook"))


def _args(*, command: list[str], name: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(run_command=command, name=name, config="unused.json")


def test_notify_run_reports_success_without_gpu_status() -> None:
    process = Mock(pid=321)
    process.wait.return_value = 0
    with patch("kingofgpu.cli._config", return_value=_config()), patch(
        "kingofgpu.cli.subprocess.Popen", return_value=process
    ) as popen, patch("kingofgpu.cli.time.monotonic", side_effect=[10.0, 75.0]), patch(
        "kingofgpu.cli.send_text"
    ) as send:
        assert cli.notify_run(_args(command=["bash", "pipeline.sh"])) == 0
    popen.assert_called_once_with(["bash", "pipeline.sh"], start_new_session=True)
    message = send.call_args.args[1]
    assert "任务已完成" in message
    assert "任务：pipeline.sh" in message
    assert "耗时：1分5秒" in message
    assert "退出码：0" in message
    assert send.call_args.kwargs["include_gpu_status"] is False


def test_notify_run_preserves_failed_exit_code_and_explicit_name() -> None:
    process = Mock(pid=321)
    process.wait.return_value = 7
    with patch("kingofgpu.cli._config", return_value=_config()), patch(
        "kingofgpu.cli.subprocess.Popen", return_value=process
    ), patch("kingofgpu.cli.time.monotonic", side_effect=[10.0, 11.0]), patch(
        "kingofgpu.cli.send_text"
    ) as send:
        assert cli.notify_run(_args(command=["python", "train.py"], name="experiment-42")) == 7
    message = send.call_args.args[1]
    assert "任务失败" in message
    assert "任务：experiment-42" in message
    assert "退出码：7" in message


def test_notify_run_forwards_interrupt_and_notifies_once() -> None:
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
    with patch("kingofgpu.cli._config", return_value=_config()), patch(
        "kingofgpu.cli.subprocess.Popen", return_value=process
    ), patch("kingofgpu.cli.signal.getsignal", return_value=object()), patch(
        "kingofgpu.cli.signal.signal", side_effect=record_signal
    ), patch("kingofgpu.cli.os.killpg") as killpg, patch(
        "kingofgpu.cli.time.monotonic", side_effect=[10.0, 11.0]
    ), patch("kingofgpu.cli.send_text") as send:
        assert cli.notify_run(_args(command=["python", "train.py"])) == -signal.SIGTERM
    killpg.assert_called_once_with(321, signal.SIGTERM)
    assert send.call_count == 1
    message = send.call_args.args[1]
    assert "任务已中断" in message
    assert "SIGTERM" in message


def test_notify_run_keeps_task_exit_code_when_notification_fails(capsys) -> None:
    process = Mock(pid=321)
    process.wait.return_value = 4
    with patch("kingofgpu.cli._config", return_value=_config()), patch(
        "kingofgpu.cli.subprocess.Popen", return_value=process
    ), patch("kingofgpu.cli.time.monotonic", side_effect=[10.0, 11.0]), patch(
        "kingofgpu.cli.send_text", side_effect=RuntimeError("network down")
    ):
        assert cli.notify_run(_args(command=["python", "train.py"])) == 4
    assert "飞书任务通知发送失败：network down" in capsys.readouterr().err


def test_notify_run_rejects_invalid_config_before_starting_task(capsys) -> None:
    with patch("kingofgpu.cli._config", side_effect=FileNotFoundError("missing config")), patch(
        "kingofgpu.cli.subprocess.Popen"
    ) as popen:
        assert cli.notify_run(_args(command=["python", "train.py"])) == 2
    popen.assert_not_called()
    assert "无法加载飞书通知配置：missing config" in capsys.readouterr().err


def test_notify_run_requires_explicit_config(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["notify-run", "--", "bash", "pipeline.sh"])
    assert exc_info.value.code == 2
    assert "notify-run 必须显式提供 --config" in capsys.readouterr().err


def test_notify_run_cli_passes_the_command_after_separator() -> None:
    process = Mock(pid=321)
    process.wait.return_value = 0
    with TemporaryDirectory() as directory:
        config_path = Path(directory) / "config.json"
        config_path.write_text(
            json.dumps({"feishu": {"webhook_url": "https://example.invalid/hook"}}),
            encoding="utf-8",
        )
        with patch("kingofgpu.cli.subprocess.Popen", return_value=process) as popen, patch(
            "kingofgpu.cli.send_text"
        ), patch("kingofgpu.cli.time.monotonic", side_effect=[10.0, 11.0]):
            assert cli.main(
                ["--config", str(config_path), "notify-run", "--name", "demo", "--", "bash", "pipeline.sh"]
            ) == 0
    popen.assert_called_once_with(["bash", "pipeline.sh"], start_new_session=True)


def test_compact_feishu_message_does_not_query_gpu_status() -> None:
    response = MagicMock()
    response.read.return_value = b'{"code": 0}'
    response.__enter__.return_value = response
    config = FeishuConfig(webhook_url="https://example.invalid/hook")
    with patch("kingofgpu.feishu.list_gpus", side_effect=AssertionError("unexpected GPU query")), patch(
        "kingofgpu.feishu.urllib.request.urlopen", return_value=response
    ) as urlopen:
        send_text(config, "✅ 任务已完成", include_gpu_status=False)
    request = urlopen.call_args.args[0]
    text = json.loads(request.data.decode("utf-8"))["content"]["text"]
    assert "任务已完成" in text
    assert "GPU 使用情况" not in text
