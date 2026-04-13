# GPU statistics collection via nvtop/psutil


import json
import subprocess
import time
from typing import Any

import psutil


class GPUStats:
    """Collect GPU stats from nvtop and psutil"""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.stats: dict[str, Any] = {}
        self.last_update = 0
        self.update_interval = 0.5

    def update(self) -> None:
        """Update GPU stats from nvtop or psutil"""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        self.stats = self._get_nvtop_stats()
        self.last_update = current_time

    def _get_nvtop_stats(self) -> dict[str, Any]:
        """Get stats from nvtop JSON output"""
        try:
            result = subprocess.run(
                ["nvtop", "-s"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            all_gpus = json.loads(result.stdout)
            if self.device_index < len(all_gpus):
                gpu = all_gpus[self.device_index]
                return {
                    "device": gpu.get("device_name", "Unknown"),
                    "gpu_util": gpu.get("gpu_util", "N/A"),
                    "mem_util": gpu.get("mem_util", "N/A"),
                    "temp": gpu.get("temp", "N/A"),
                    "power": gpu.get("power_draw", "N/A"),
                }
        except Exception:
            pass

        # Fallback to psutil
        return {
            "device": f"GPU {self.device_index}",
            "cpu": f"{psutil.cpu_percent():.0f}%",
            "mem": f"{psutil.virtual_memory().percent:.0f}%",
        }

    def get_stats_snapshot(self) -> dict[str, Any]:
        """Get current GPU stats as pure data."""
        self.update()
        return dict(self.stats)

    def format_stats_text(self) -> str:
        """Get plain-text representation of current GPU stats."""
        stats = self.get_stats_snapshot()
        lines = [f"Device: {stats.get('device', 'N/A')}"]

        if "gpu_util" in stats:
            lines.append(
                f"GPU: {stats.get('gpu_util', 'N/A')} | Mem: {stats.get('mem_util', 'N/A')}"
            )
        else:
            lines.append(f"CPU: {stats.get('cpu', 'N/A')} | Mem: {stats.get('mem', 'N/A')}")

        if "temp" in stats:
            lines.append(f"Temp: {stats.get('temp', 'N/A')}")

        if "power" in stats and stats["power"] != "N/A":
            lines.append(f"Power: {stats['power']}")

        return "\n".join(lines)

    @property
    def gpu_util(self) -> str:
        """Get GPU utilization string"""
        return self.stats.get("gpu_util", "N/A")

    @property
    def memory_util(self) -> str:
        """Get memory utilization string"""
        return self.stats.get("mem_util", "N/A")
