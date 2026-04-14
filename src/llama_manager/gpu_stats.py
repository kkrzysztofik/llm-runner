# GPU statistics collection via nvtop/psutil

import time
from collections.abc import Callable
from typing import Any

import psutil


def _psutil_only_collector(device_index: int) -> dict[str, Any]:
    """Pure psutil-based GPU collector with no subprocess usage.

    This is the default collector when no custom collector is injected.
    It provides basic CPU/memory stats without GPU-specific metrics.
    """
    return {
        "device": f"GPU {device_index}",
        "gpu_util": "N/A",
        "mem_util": "N/A",
        "temp": "N/A",
        "power": "N/A",
        "cpu": f"{psutil.cpu_percent():.0f}%",
        "mem": f"{psutil.virtual_memory().percent:.0f}%",
    }


class GPUStats:
    """Collect GPU stats from nvtop and psutil.

    The GPU collector is injected via the collector parameter to allow
    testing without actual subprocess calls. The default collector is
    provided by llama_cli.gpu_collectors.collect_nvtop_stats.

    Args:
        device_index: Index of the GPU device to query.
        collector: Optional callable that returns GPU stats dict.
                   If None, uses a psutil-only fallback (no subprocess).
    """

    def __init__(
        self,
        device_index: int = 0,
        collector: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.device_index = device_index
        self.stats: dict[str, Any] = {}
        self.last_update: float = 0.0
        self.update_interval: float = 0.5
        # Injectable collector callable (defaults to psutil-only)
        self._collector: Callable[[], dict[str, Any]] = (
            collector if collector is not None else lambda: _psutil_only_collector(device_index)
        )

    def update(self) -> None:
        """Update GPU stats from nvtop or psutil."""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        self.stats = self._collector()
        self.last_update = current_time

    def get_stats_snapshot(self) -> dict[str, Any]:
        """Get current GPU stats as pure data."""
        self.update()
        return dict(self.stats)

    def format_stats_text(self) -> str:
        """Get plain-text representation of current GPU stats."""
        stats = self.get_stats_snapshot()
        lines = [f"Device: {stats.get('device', 'N/A')}"]

        # Value-based check: if gpu_util is "N/A", use CPU fallback branch
        if stats.get("gpu_util", "N/A") != "N/A":
            lines.append(
                f"GPU: {stats.get('gpu_util', 'N/A')} | Mem: {stats.get('mem_util', 'N/A')}"
            )
        else:
            lines.append(f"CPU: {stats.get('cpu', 'N/A')} | Mem: {stats.get('mem', 'N/A')}")

        if stats.get("temp", "N/A") != "N/A":
            lines.append(f"Temp: {stats.get('temp', 'N/A')}")

        if "power" in stats and stats["power"] != "N/A":
            lines.append(f"Power: {stats['power']}")

        return "\n".join(lines)

    @property
    def gpu_util(self) -> str:
        """Get GPU utilization string"""
        self.update()
        return self.stats.get("gpu_util", "N/A")

    @property
    def memory_util(self) -> str:
        """Get memory utilization string"""
        self.update()
        return self.stats.get("mem_util", "N/A")
