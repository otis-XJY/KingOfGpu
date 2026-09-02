from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from kingofgpu.config import Config, FeishuConfig
from kingofgpu.monitor import Monitor, UserTaskBinding


class UserTaskBindingTest(unittest.TestCase):
    def test_finished_bound_task_clears_release_cooldown(self) -> None:
        with TemporaryDirectory() as directory:
            monitor = Monitor(
                Path(directory),
                Config(
                    feishu=FeishuConfig(webhook_url="https://example.invalid/webhook"),
                    state_file="state.json",
                ),
            )
            monitor.slow_gpu_until[2] = time.time() + 600
            monitor.user_task_bindings[2] = UserTaskBinding(user="xujunyi", pids={12345})

            with patch("kingofgpu.monitor.process_user", return_value=None):
                monitor._refresh_user_task_bindings(time.time())

            self.assertNotIn(2, monitor.user_task_bindings)
            self.assertNotIn(2, monitor.slow_gpu_until)


if __name__ == "__main__":
    unittest.main()
