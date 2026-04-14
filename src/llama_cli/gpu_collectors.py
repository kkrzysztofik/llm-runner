"""GPU statistics collectors for llm-runner TUI.

This module provides GPU statistics collection via nvtop subprocess.
It is owned by the CLI layer to keep llama_manager free of subprocess usage.
"""

import json
import subprocess
import sys
from typing import Any

import psutil


def _get_cpu_percent() -> float:
    """Get CPU percentage with safe fallback on exception."""
    try:
        return float(psutil.cpu_percent(interval=0.1))
    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
        return 0.0


def _get_memory_percent() -> float:
    """Get memory percentage with safe fallback on exception."""
    try:
        return float(psutil.virtual_memory().percent)
    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
        return 0.0


def _format_metric(value: Any) -> str:
    """Normalize collector metrics to string values."""
    return "N/A" if value is None else str(value)


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
                "device": _format_metric(gpu.get("device_name", "Unknown")),
                "gpu_util": _format_metric(gpu.get("gpu_util", "N/A")),
                "mem_util": _format_metric(gpu.get("mem_util", "N/A")),
                "temp": _format_metric(gpu.get("temp", "N/A")),
                "power": _format_metric(gpu.get("power_draw", "N/A")),
                "cpu": _format_metric(f"{_get_cpu_percent():.0f}%"),
                "mem": _format_metric(f"{_get_memory_percent():.0f}%"),
            }
        print(
            f"warning: device_index {device_index} out of range for {len(all_gpus)} GPU(s)",
            file=sys.stderr,
        )
    except subprocess.TimeoutExpired as e:
        print(f"warning: nvtop timeout: {e}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"warning: nvtop JSON parse error: {e}", file=sys.stderr)
    except RuntimeError as e:
        print(f"warning: nvtop error: {e}", file=sys.stderr)
    except (ValueError, OSError) as e:
        print(f"warning: nvtop error: {e}", file=sys.stderr)

    # Fallback to psutil
    return {
        "device": _format_metric(f"GPU {device_index}"),
        "gpu_util": _format_metric("N/A"),
        "mem_util": _format_metric("N/A"),
        "temp": _format_metric("N/A"),
        "power": _format_metric("N/A"),
        "cpu": _format_metric(f"{_get_cpu_percent():.0f}%"),
        "mem": _format_metric(f"{_get_memory_percent():.0f}%"),
    }
