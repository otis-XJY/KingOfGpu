from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import argparse
import json

from kingofgpu import cli
from kingofgpu.config import Config, FeishuConfig
from kingofgpu.gpu import ComputeProcess, Gpu
from kingofgpu.monitor import Lease, Monitor, UserTaskBinding


def _config() -> Config:
    return Config(
        feishu=FeishuConfig(webhook_url="https://example.invalid/hook"),
        release_bind_user="xujunyi",
        bind_wait_seconds=600,
        startup_grace_seconds=0,
    )


def test_pending_binding_tracks_only_configured_user_until_pid_exits() -> None:
    with TemporaryDirectory() as directory:
        monitor = Monitor(Path(directory), _config())
        monitor.user_task_bindings[4] = UserTaskBinding("xujunyi", pending_until=9999.0)
        process = ComputeProcess(4, 123, "python", 100)
        with patch("kingofgpu.monitor._external_processes", return_value=[process]), patch(
            "kingofgpu.monitor.process_user", return_value="xujunyi"
        ):
            monitor._refresh_user_task_bindings(1.0)
        assert monitor.user_task_bindings[4].pids == {123}
        with patch("kingofgpu.monitor.process_user", return_value=None):
            monitor._refresh_user_task_bindings(2.0)
        assert 4 not in monitor.user_task_bindings


def test_only_xujunyi_new_process_releases_existing_occupier() -> None:
    with TemporaryDirectory() as directory:
        monitor = Monitor(Path(directory), _config())
        lease_process = Mock()
        lease_process.poll.return_value = None
        monitor.leases[4] = Lease(
            Gpu(4, "gpu", 1000, 0, 1000, 0), lease_process, known_external_pids=set(), started_at=0.0
        )
        candidate = ComputeProcess(4, 456, "python", 100)
        monitor._release = Mock()  # type: ignore[method-assign]
        monitor._bind_user_task = Mock()  # type: ignore[method-assign]
        with patch("kingofgpu.monitor._external_processes", return_value=[candidate]), patch(
            "kingofgpu.monitor.process_user", return_value="other"
        ):
            monitor._check_lease(4, 1.0)
        monitor._release.assert_not_called()
        with patch("kingofgpu.monitor._external_processes", return_value=[candidate]), patch(
            "kingofgpu.monitor.process_user", return_value="xujunyi"
        ):
            monitor._check_lease(4, 1.0)
        monitor._bind_user_task.assert_called_once_with(4, [candidate])
        monitor._release.assert_called_once()


def test_explicit_release_registers_xujunyi_pending_binding() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "config.json"
        config_path.write_text(json.dumps({
            "feishu": {"webhook_url": "https://example.invalid/hook"},
            "state_file": "run/state.json",
            "release_bind_user": "xujunyi",
            "bind_wait_seconds": 600,
        }), encoding="utf-8")
        args = argparse.Namespace(config=str(config_path))
        with patch("kingofgpu.cli._project_dir", return_value=root):
            cli._register_pending_user_binding(args, 4)
        state = json.loads((root / "run/state.json").read_text(encoding="utf-8"))
        assert state["user_task_bindings"]["4"]["user"] == "xujunyi"
        assert state["user_task_bindings"]["4"]["pids"] == []


def test_status_separates_kingofgpu_reservation_from_external_memory(capsys) -> None:
    args = argparse.Namespace(config="unused.json", required_mib=700, json=True)
    gpus = [Gpu(4, "gpu", 1000, 800, 200, 0)]
    processes = [
        ComputeProcess(4, 1, "python", 600),
        ComputeProcess(4, 2, "python", 100),
    ]
    with patch("kingofgpu.cli._config", return_value=_config()), patch(
        "kingofgpu.cli._load_state", return_value={}
    ), patch("kingofgpu.cli.list_gpus", return_value=gpus), patch(
        "kingofgpu.cli.list_compute_processes", return_value=processes
    ), patch("kingofgpu.cli.process_user", side_effect=["xujunyi", "other"]), patch(
        "kingofgpu.cli.process_command", side_effect=["python -m kingofgpu.occupier", "python train.py"]
    ):
        assert cli.status(args) == 0
    payload = json.loads(capsys.readouterr().out)
    gpu = payload["gpus"][0]
    assert gpu["kingofgpu_occupier_mib"] == 600
    assert gpu["non_kingofgpu_process_mib"] == 100
    assert gpu["unattributed_used_mib"] == 100
    assert gpu["free_after_releasing_kingofgpu_mib"] == 800
    assert gpu["fits_required_mib_now"] is False
    assert gpu["fits_required_mib_after_releasing_kingofgpu"] is True
