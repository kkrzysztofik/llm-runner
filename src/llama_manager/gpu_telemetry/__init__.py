"""GPU telemetry component package."""

from .common import GpuTelemetrySelector, parse_gpu_telemetry_selector
from .level_zero import collect_level_zero_stats
from .stats import (
    GPUStats,
    collect_gpu_stats,
    collector_for_config,
    get_gpu_identifier,
    make_gpu_collector,
    selector_for_config,
)
from .vendor import (
    collect_nvidia_smi_stats,
    collect_nvtop_stats,
    collect_nvtop_stats_for_selector,
    collect_xpu_smi_stats,
)

__all__ = [
    "GPUStats",
    "GpuTelemetrySelector",
    "collect_gpu_stats",
    "collect_level_zero_stats",
    "collect_nvtop_stats",
    "collect_nvtop_stats_for_selector",
    "collect_nvidia_smi_stats",
    "collect_xpu_smi_stats",
    "collector_for_config",
    "get_gpu_identifier",
    "make_gpu_collector",
    "parse_gpu_telemetry_selector",
    "selector_for_config",
]
