"""Subprocess-based GPU telemetry collectors."""

import json
import logging
import subprocess
from typing import Any

from .common import (
    GpuTelemetrySelector,
    format_metric,
    format_percent,
    format_power,
    format_temp,
    parse_float,
    psutil_only_collector,
    with_host_stats,
)

logger = logging.getLogger(__name__)

_nvtop_fallback_logged: set[int] = set()


def collect_nvidia_smi_stats(selector: GpuTelemetrySelector) -> dict[str, Any] | None:
    """Collect NVIDIA stats using nvidia-smi by CUDA ordinal."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        if index != selector.ordinal:
            continue
        used_mib = parse_float(parts[3])
        total_mib = parse_float(parts[4])
        mem_pct = (
            100.0 * used_mib / total_mib
            if used_mib is not None and total_mib is not None and total_mib > 0
            else None
        )
        stats = {
            "device": parts[1] or f"NVIDIA GPU {index}",
            "gpu_util": format_percent(parse_float(parts[2])),
            "mem_util": format_percent(mem_pct),
            "vram": (
                f"{used_mib / 1024:.1f}G/{total_mib / 1024:.1f}G"
                if used_mib is not None and total_mib is not None
                else "N/A"
            ),
            "temp": format_temp(parse_float(parts[5])),
            "power": format_power(parse_float(parts[6])),
            "source": "nvidia-smi",
        }
        return with_host_stats(stats)
    return None


def collect_xpu_smi_stats(selector: GpuTelemetrySelector) -> dict[str, Any] | None:
    """Collect Intel fallback stats using xpu-smi JSON output."""
    if selector.backend != "sycl":
        return None
    try:
        result = subprocess.run(
            ["xpu-smi", "stats", "-d", str(selector.ordinal), "-j"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    metrics = {
        item.get("metrics_type"): item.get("value")
        for item in payload.get("device_level", [])
        if isinstance(item, dict)
    }
    mem_pct = parse_float(metrics.get("XPUM_STATS_MEMORY_UTILIZATION"))
    mem_used_mib = parse_float(metrics.get("XPUM_STATS_MEMORY_USED"))
    gpu_util_raw = parse_float(metrics.get("XPUM_STATS_GPU_UTILIZATION"))
    temp_raw = parse_float(metrics.get("XPUM_STATS_TEMPERATURE"))
    stats = {
        "device": f"Intel GPU {selector.ordinal}",
        "gpu_util": format_percent(gpu_util_raw),
        "mem_util": format_percent(mem_pct),
        "vram": f"{mem_used_mib / 1024:.1f}G" if mem_used_mib is not None else "N/A",
        "temp": format_temp(temp_raw),
        "power": format_power(parse_float(metrics.get("XPUM_STATS_POWER"))),
        "source": "xpu-smi",
    }
    return with_host_stats(stats)


def _nvtop_device_matches(gpu: dict[str, Any], selector: GpuTelemetrySelector) -> bool:
    name = str(gpu.get("device_name", "")).lower()
    if selector.backend == "sycl":
        return any(token in name for token in ("intel", "arc", "battlemage"))
    if any(token in name for token in ("nvidia", "geforce", "rtx", "gtx")):
        return True
    return selector.backend == "cuda"


def collect_nvtop_stats_for_selector(selector: GpuTelemetrySelector) -> dict[str, Any] | None:
    """Collect nvtop stats by matching device identity, not dashboard position."""
    try:
        result = subprocess.run(
            ["nvtop", "-s"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode != 0:
            return None
        all_gpus = json.loads(result.stdout)
    except OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError:
        return None
    if not isinstance(all_gpus, list):
        return None
    matches = [
        gpu for gpu in all_gpus if isinstance(gpu, dict) and _nvtop_device_matches(gpu, selector)
    ]
    if not matches:
        return None
    gpu = matches[min(selector.ordinal, len(matches) - 1)]
    return with_host_stats(
        {
            "device": format_metric(gpu.get("device_name", "Unknown")),
            "gpu_util": format_metric(gpu.get("gpu_util", "N/A")),
            "mem_util": format_metric(gpu.get("mem_util", "N/A")),
            "temp": format_metric(gpu.get("temp", "N/A")),
            "power": format_metric(gpu.get("power_draw", "N/A")),
            "source": "nvtop",
        }
    )


def collect_nvtop_stats(device_index: int = 0) -> dict[str, Any]:
    """Collect legacy nvtop stats by raw device index."""
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
        if not isinstance(all_gpus, list):
            raise ValueError(f"nvtop JSON output is not a list, got {type(all_gpus).__name__}")
        if 0 <= device_index < len(all_gpus):
            gpu = all_gpus[device_index]
            if not isinstance(gpu, dict):
                raise ValueError(
                    f"gpu entry at index {device_index} is not a dict, got {type(gpu).__name__}"
                )
            return {
                "device": format_metric(gpu.get("device_name", "Unknown")),
                "gpu_util": format_metric(gpu.get("gpu_util", "N/A")),
                "mem_util": format_metric(gpu.get("mem_util", "N/A")),
                "temp": format_metric(gpu.get("temp", "N/A")),
                "power": format_metric(gpu.get("power_draw", "N/A")),
                **with_host_stats({}),
            }
    except subprocess.TimeoutExpired:
        pass
    except json.JSONDecodeError:
        pass
    except RuntimeError:
        pass
    except ValueError, OSError:
        pass

    if device_index not in _nvtop_fallback_logged:
        _nvtop_fallback_logged.add(device_index)
        logger.debug("nvtop unavailable - falling back to psutil for GPU %s", device_index)
    return psutil_only_collector(device_index)
