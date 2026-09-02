from __future__ import annotations

import logging
import os
import signal
import time


def _install_signal_handlers() -> None:
    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _install_signal_handlers()
    try:
        import torch
    except ImportError as exc:
        logging.error("占用器需要 CUDA 版 PyTorch: %s", exc)
        return 2
    if not torch.cuda.is_available():
        logging.error("当前 Python 环境中 CUDA 不可用")
        return 2

    reserve_mib = int(os.environ.get("KOGGPU_RESERVE_MIB", "512"))
    target_gpu = os.environ.get("KOGGPU_TARGET_GPU", "unknown")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    allocations: list[object] = []
    chunk_bytes = 256 * 1024 * 1024
    reserve_bytes = reserve_mib * 1024 * 1024
    logging.info("occupier started for physical GPU %s", target_gpu)

    try:
        while True:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            available = free_bytes - reserve_bytes
            if available <= 4 * 1024 * 1024:
                break
            request = min(chunk_bytes, available)
            allocated = False
            while request >= 4 * 1024 * 1024:
                try:
                    allocations.append(torch.empty(request // 4, dtype=torch.float32, device=device))
                    allocated = True
                    break
                except RuntimeError as exc:
                    logging.warning("allocation of %d MiB failed: %s", request // 1024 // 1024, exc)
                    request //= 2
                    torch.cuda.empty_cache()
            if not allocated:
                break
        logging.info("reserved approximately %.2f GiB; waiting for monitor", sum(x.numel() * x.element_size() for x in allocations) / 1024**3)
        while True:
            time.sleep(60)
    except SystemExit:
        logging.info("stopping own allocation on request")
        return 0
    finally:
        allocations.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
