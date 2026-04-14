"""GPU statistics collectors for llm-runner TUI.

This module provides GPU statistics collection via nvtop subprocess.
It is owned by the CLI layer to keep llama_manager free of subprocess usage.
"""

import json
import subprocess
import sys
from typing import Any

import psutil


def collect_nvtop_stats(device_index: int = 0) -> dict[str, Any]:
    """Collect GPU stats using nvtop subprocess.

    Args:
        device_index: Index of the GPU device to query.

    Returns:
        Dictionary with GPU statistics.
    """
    try:
        result = subprocess.run(
            ["nvtop", "-s"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvtop exited with code {result.returncode}: {result.stderr}")
        all_gpus = json.loads(result.stdout)
        if 0 <= device_index < len(all_gpus):
            gpu = all_gpus[device_index]
            return {
                "device": gpu.get("device_name", "Unknown"),
                "gpu_util": gpu.get("gpu_util", "N/A"),
                "mem_util": gpu.get("mem_util", "N/A"),
                "temp": gpu.get("temp", "N/A"),
                "power": gpu.get("power_draw", "N/A"),
                "cpu": f"{psutil.cpu_percent():.0f}%",
                "mem": f"{psutil.virtual_memory().percent:.0f}%",
            }
        print(
            f"warning: device_index {device_index} out of range for {len(all_gpus)} GPU(s)",
            file=sys.stderr,
        )
    except subprocess.TimeoutExpired:
        pass
    except json.JSONDecodeError as e:
        print(f"warning: nvtop JSON parse error: {e}", file=sys.stderr)
    except RuntimeError as e:
        print(f"warning: nvtop error: {e}", file=sys.stderr)
    except (ValueError, OSError) as e:
        print(f"warning: nvtop error: {e}", file=sys.stderr)

    # Fallback to psutil
    return {
        "device": f"GPU {device_index}",
        "gpu_util": "N/A",
        "mem_util": "N/A",
        "temp": "N/A",
        "power": "N/A",
        "cpu": f"{psutil.cpu_percent():.0f}%",
        "mem": f"{psutil.virtual_memory().percent:.0f}%",
    }
