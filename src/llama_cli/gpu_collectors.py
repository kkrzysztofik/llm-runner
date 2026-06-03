"""GPU statistics collectors for llm-runner TUI.

DEPRECATED: This module re-exports from llama_manager.gpu_telemetry.
The nvtop collector logic lives in the pure library.
"""

import psutil


def _get_cpu_percent() -> float:
    """Get CPU percentage with safe fallback on exception."""
    try:
        return float(psutil.cpu_percent(interval=0.1))
    except Exception:  # noqa: S110
        return 0.0


def _get_memory_percent() -> float:
    """Get memory percentage with safe fallback on exception."""
    try:
        return float(psutil.virtual_memory().percent)
    except Exception:  # noqa: S110
        return 0.0


from llama_manager.gpu_telemetry import collect_nvtop_stats

__all__ = ["collect_nvtop_stats", "_get_cpu_percent", "_get_memory_percent"]
