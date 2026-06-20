"""Shared GPU telemetry types and formatting helpers."""

from dataclasses import dataclass
from typing import Any

import psutil

INTEL_VENDOR_ID = 0x8086


@dataclass(frozen=True)
class GpuTelemetrySelector:
    """Stable selector for a dashboard GPU telemetry source."""

    backend: str
    ordinal: int = 0
    device: str = ""
    pci_bdf: str | None = None
    vendor_id: int | None = None
    device_id: int | None = None


def parse_gpu_telemetry_selector(
    device: str,
    main_gpu: int = 0,
    *,
    vendor_id: int | None = None,
    device_id: int | None = None,
    pci_bdf: str | None = None,
) -> GpuTelemetrySelector:
    """Parse a llama.cpp device string into a stable telemetry selector."""
    raw = device.strip()
    upper = raw.upper()
    if upper.startswith("SYCL"):
        ordinal_text = raw[4:].lstrip(":")
        try:
            ordinal = int(ordinal_text.split(":", maxsplit=1)[0] or "0")
        except ValueError:
            ordinal = 0
        return GpuTelemetrySelector(
            backend="sycl",
            ordinal=ordinal,
            device=raw,
            vendor_id=vendor_id or INTEL_VENDOR_ID,
            device_id=device_id,
            pci_bdf=pci_bdf,
        )
    if upper.startswith("CUDA"):
        _, _, ordinal_text = raw.partition(":")
        ordinals = [int(part.strip()) for part in ordinal_text.split(",") if part.strip().isdigit()]
        ordinal = _resolve_cuda_ordinal(main_gpu, ordinals)
        return GpuTelemetrySelector(backend="cuda", ordinal=ordinal, device=raw)
    return GpuTelemetrySelector(backend="cuda", ordinal=main_gpu, device=raw)


def parse_gpu_telemetry_selectors(
    device: str,
    main_gpu: int = 0,
    *,
    vendor_id: int | None = None,
    device_id: int | None = None,
    pci_bdf: str | None = None,
) -> list[GpuTelemetrySelector]:
    """Parse a llama.cpp device string into all telemetry selectors it references."""
    primary = parse_gpu_telemetry_selector(
        device,
        main_gpu,
        vendor_id=vendor_id,
        device_id=device_id,
        pci_bdf=pci_bdf,
    )
    raw = device.strip()
    if not raw.upper().startswith("CUDA"):
        return [primary]

    _, _, ordinal_text = raw.partition(":")
    ordinals = [int(part.strip()) for part in ordinal_text.split(",") if part.strip().isdigit()]
    if not ordinals:
        return [primary]

    ordered_ordinals = [primary.ordinal]
    ordered_ordinals.extend(ordinal for ordinal in ordinals if ordinal != primary.ordinal)
    return [
        GpuTelemetrySelector(backend="cuda", ordinal=ordinal, device=raw)
        for ordinal in ordered_ordinals
    ]


def _resolve_cuda_ordinal(main_gpu: int, ordinals: list[int]) -> int:
    """Pick the preferred ordinal: main_gpu when listed, else the first listed, else main_gpu."""
    if main_gpu in ordinals:
        return main_gpu
    return ordinals[0] if ordinals else main_gpu


def format_metric(value: Any) -> str:
    """Normalize collector metrics to string values."""
    return "N/A" if value is None else str(value)


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0f}%"


def format_power(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0f}W"


def format_temp(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.0f}C"


def format_memory_used(used_bytes: int | None, total_bytes: int | None) -> str:
    if used_bytes is None or total_bytes is None or total_bytes <= 0:
        return "N/A"
    used_gb = used_bytes / (1024**3)
    total_gb = total_bytes / (1024**3)
    return f"{used_gb:.1f}G/{total_gb:.1f}G"


def with_host_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Add host CPU and memory stats to a GPU stats dictionary."""
    stats["cpu"] = format_metric(f"{psutil.cpu_percent():.0f}%")
    stats["mem"] = format_metric(f"{psutil.virtual_memory().percent:.0f}%")
    return stats


def psutil_only_collector(device_index: int) -> dict[str, Any]:
    """Pure psutil-based GPU collector with no subprocess usage."""
    return {
        "device": f"GPU {device_index}",
        "gpu_util": "N/A",
        "mem_util": "N/A",
        "temp": "N/A",
        "power": "N/A",
        "cpu": f"{psutil.cpu_percent():.0f}%",
        "mem": f"{psutil.virtual_memory().percent:.0f}%",
    }


def parse_float(value: Any) -> float | None:
    try:
        return float(str(value).replace("%", "").replace("W", "").strip())
    except ValueError:
        return None
