# GPU statistics collection via nvtop/psutil


import json
import subprocess
import time

import psutil
from rich.panel import Panel
from rich.text import Text


class GPUStats:
    """Collect GPU stats from nvtop and psutil"""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.stats: dict = {}
        self.last_update = 0
        self.update_interval = 0.5

    def update(self) -> None:
        """Update GPU stats from nvtop or psutil"""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        self.stats = self._get_nvtop_stats()
        self.last_update = current_time

    def _get_nvtop_stats(self) -> dict:
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

    def get_rich_renderable(self) -> Panel:
        """Get a panel with GPU stats"""
        self.update()

        stats_text = Text()
        stats_text.append("Device: ", style="bold")
        stats_text.append(self.stats.get("device", "N/A"), style="cyan")

        if "gpu_util" in self.stats:
            stats_text.append("GPU: ", style="bold")
            stats_text.append(str(self.stats.get("gpu_util", "N/A")), style="green")
            stats_text.append(" | Mem: ", style="bold")
            stats_text.append(str(self.stats.get("mem_util", "N/A")), style="yellow")

        if "temp" in self.stats:
            stats_text.append("\nTemp: ", style="bold")
            stats_text.append(str(self.stats.get("temp", "N/A")), style="red")

        if "power" in self.stats and self.stats["power"] != "N/A":
            stats_text.append("\nPower: ", style="bold")
            stats_text.append(str(self.stats["power"]), style="magenta")

        return Panel(
            stats_text,
            title="[bold yellow]GPU Stats[/]",
            border_style="yellow",
        )

    @property
    def gpu_util(self) -> str:
        """Get GPU utilization string"""
        return self.stats.get("gpu_util", "N/A")

    @property
    def memory_util(self) -> str:
        """Get memory utilization string"""
        return self.stats.get("mem_util", "N/A")
