"""GPU telemetry orchestration and state wrapper."""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from .common import (
    GpuTelemetrySelector,
    parse_gpu_telemetry_selector,
    parse_gpu_telemetry_selectors,
    psutil_only_collector,
)
from .level_zero import collect_level_zero_stats
from .vendor import (
    collect_nvidia_smi_stats,
    collect_nvtop_stats_for_selector,
    collect_xpu_smi_stats,
)

logger = logging.getLogger(__name__)


class GPUStats:
    """Collect GPU stats from a bound telemetry selector."""

    def __init__(
        self,
        device_index: int = 0,
        collector: Callable[[], dict[str, Any]] | None = None,
        selector: GpuTelemetrySelector | None = None,
    ) -> None:
        self.device_index = device_index
        self.selector = selector or GpuTelemetrySelector(backend="cuda", ordinal=device_index)
        self.stats: dict[str, Any] = {}
        self._stats_lock = threading.Lock()
        self.last_update: float = 0.0
        self.update_interval: float = 0.5
        self._prev_gpu_util: float | None = None
        self._collector: Callable[[], dict[str, Any]] = (
            collector if collector is not None else lambda: psutil_only_collector(device_index)
        )

    def update(self) -> None:
        """Update GPU stats from the bound collector."""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        new_stats = self._collector()
        with self._stats_lock:
            self.stats = new_stats
            snapshot = dict(self.stats)
        self.last_update = current_time

        new_util = snapshot.get("gpu_util")
        if isinstance(new_util, str) and "%" in new_util:
            try:
                current_pct = float(new_util.replace("%", ""))
            except ValueError:
                current_pct = 0.0
        elif isinstance(new_util, (int, float)):
            current_pct = float(new_util)
        else:
            current_pct = 0.0

        if self._prev_gpu_util is not None and current_pct > 0:
            delta = abs(current_pct - self._prev_gpu_util)
            if delta > 5.0:
                logger.debug(
                    "GPU %d util delta=%.1f%% (%.1f%% -> %.1f%%)",
                    self.device_index,
                    delta,
                    self._prev_gpu_util,
                    current_pct,
                )
        self._prev_gpu_util = current_pct if current_pct > 0 else None

    def get_stats_snapshot(self) -> dict[str, Any]:
        """Get current GPU stats as pure data."""
        self.update()
        return self.get_cached_stats_snapshot()

    def get_cached_stats_snapshot(self) -> dict[str, Any]:
        """Get current GPU stats without running the collector."""
        with self._stats_lock:
            return dict(self.stats)

    def format_stats_text(self) -> str:
        """Get plain-text representation of current GPU stats."""
        stats = self.get_stats_snapshot()
        lines = [f"Device: {stats.get('device', 'N/A')}"]

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
        """Current GPU utilization percentage or ``"N/A"``."""
        self.update()
        with self._stats_lock:
            return self.stats.get("gpu_util", "N/A")

    @property
    def memory_util(self) -> str:
        """Current GPU memory utilization percentage or ``"N/A"``."""
        self.update()
        with self._stats_lock:
            return self.stats.get("mem_util", "N/A")


def get_gpu_identifier(
    backend: str,
    gpu_collector: Callable[[], list[dict[str, Any]]] | None = None,
) -> str:
    """Compute a filesystem-safe GPU identifier for the given backend."""
    from llama_manager.config.profile_cache import compute_gpu_identifier

    if backend not in ("cuda", "sycl"):
        raise ValueError(
            f"unsupported backend: {backend!r}; expected one of: cuda, sycl",
        )

    def _default_collector() -> list[dict[str, Any]]:
        return [{"name": "Unknown", "index": 0}]

    collector = gpu_collector if gpu_collector is not None else _default_collector
    devices = collector()

    if not devices:
        raise IndexError(
            f"collector returned empty device list for backend {backend!r}",
        )

    first_device = devices[0]
    gpu_name: str = first_device["name"]
    device_index: int = first_device["index"]

    return compute_gpu_identifier(backend, gpu_name, device_index)


def _is_real_value(value: Any) -> bool:
    """Check if a metric value is real (not missing/unknown)."""
    return value not in (None, "N/A", "")


def collect_gpu_stats(selector: GpuTelemetrySelector) -> dict[str, Any]:
    """Collect GPU stats by merging results from available collectors.

    Collectors run in order — L0 → nvtop → xpu-smi for SYCL, nvidia-smi → nvtop
    for CUDA — and each fills whichever metrics it can provide. The first
    non-None value for each key wins. Collection stops once ``gpu_util`` has a
    real value, avoiding unnecessary subprocess calls.
    """
    collectors: tuple[Callable[[GpuTelemetrySelector], dict[str, Any] | None], ...]
    if selector.backend == "sycl":
        collectors = (
            collect_level_zero_stats,
            collect_nvtop_stats_for_selector,
            collect_xpu_smi_stats,
        )
    else:
        collectors = (collect_nvidia_smi_stats, collect_nvtop_stats_for_selector)
    merged: dict[str, Any] = {}
    for collector in collectors:
        stats = collector(selector)
        if stats is None:
            continue
        for key, value in stats.items():
            if key not in merged or not _is_real_value(merged.get(key)):
                merged[key] = value
        if _is_real_value(merged.get("gpu_util")):
            break
    if merged:
        return merged
    return psutil_only_collector(selector.ordinal)


def make_gpu_collector(selector: GpuTelemetrySelector) -> Callable[[], dict[str, Any]]:
    """Return a zero-argument collector bound to a stable GPU selector."""
    return lambda: collect_gpu_stats(selector)


def make_multi_gpu_collector(
    selectors: list[GpuTelemetrySelector],
) -> Callable[[], dict[str, Any]]:
    """Return a collector that includes all GPUs referenced by one server config."""

    def _collect() -> dict[str, Any]:
        if len(selectors) == 1:
            return collect_gpu_stats(selectors[0])
        snapshots = [_stats_with_selector_label(selector) for selector in selectors]
        if not snapshots:
            return {}
        primary = dict(snapshots[0])
        primary["devices"] = snapshots
        return primary

    return _collect


def _stats_with_selector_label(selector: GpuTelemetrySelector) -> dict[str, Any]:
    stats = collect_gpu_stats(selector)
    labelled = dict(stats)
    device_name = str(labelled.get("device", "N/A")).strip() or "N/A"
    labelled["device"] = f"{selector.backend.upper()}:{selector.ordinal} {device_name}"
    labelled["backend"] = selector.backend
    labelled["ordinal"] = selector.ordinal
    return labelled


def selector_for_config(cfg: Any) -> GpuTelemetrySelector:
    """Build a telemetry selector from a ServerConfig-like object."""
    return parse_gpu_telemetry_selector(
        str(getattr(cfg, "device", "")),
        int(getattr(cfg, "main_gpu", 0) or 0),
    )


def selectors_for_config(cfg: Any) -> list[GpuTelemetrySelector]:
    """Build all telemetry selectors referenced by a ServerConfig-like object."""
    return parse_gpu_telemetry_selectors(
        str(getattr(cfg, "device", "")),
        int(getattr(cfg, "main_gpu", 0) or 0),
    )


def collector_for_config(cfg: Any) -> Callable[[], dict[str, Any]]:
    """Return a collector bound to a ServerConfig-like object."""
    return make_multi_gpu_collector(selectors_for_config(cfg))
