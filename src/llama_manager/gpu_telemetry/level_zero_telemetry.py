"""Level Zero telemetry collection."""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Any

from .common import (
    GpuTelemetrySelector,
    format_memory_used,
    format_percent,
    format_power,
    format_temp,
    with_host_stats,
)
from .level_zero_device import (
    _discover_level_zero_devices,
    _enum_handles,
    _load_level_zero,
    _select_level_zero_device,
)
from .level_zero_engine import _collect_native_engine_util
from .level_zero_sysfs import _collect_sysfs_temp
from .level_zero_types import (
    _ZE_RESULT_SUCCESS,
    _ZES_STRUCTURE_TYPE_MEMORY_STATE,
    _LevelZeroDevice,
    _ZesMemState,
    _ZesPowerEnergyCounter,
)

logger = logging.getLogger(__name__)

_level_zero_fallback_logged = False


def _collect_level_zero_memory(
    lib: ctypes.CDLL,
    device: _LevelZeroDevice,
) -> tuple[int | None, int | None]:
    total = 0
    free = 0
    found = False
    for handle in _enum_handles(lib.zesDeviceEnumMemoryModules, device.handle):
        state = _ZesMemState()
        state.stype = _ZES_STRUCTURE_TYPE_MEMORY_STATE
        if lib.zesMemoryGetState(handle, ctypes.byref(state)) != _ZE_RESULT_SUCCESS:
            continue
        if state.size > 0:
            total += int(state.size)
            free += int(state.free)
            found = True
    if not found or total <= 0:
        return None, None
    return total - free, total


def _read_power_counters(
    lib: ctypes.CDLL,
    handles: list[ctypes.c_void_p],
) -> list[_ZesPowerEnergyCounter]:
    counters: list[_ZesPowerEnergyCounter] = []
    for handle in handles:
        counter = _ZesPowerEnergyCounter()
        if lib.zesPowerGetEnergyCounter(handle, ctypes.byref(counter)) == _ZE_RESULT_SUCCESS:
            counters.append(counter)
    return counters


def _collect_level_zero_power(lib: ctypes.CDLL, device: _LevelZeroDevice) -> float | None:
    handles = _enum_handles(lib.zesDeviceEnumPowerDomains, device.handle)
    if not handles:
        return None
    first = _read_power_counters(lib, handles)
    if not first:
        return None
    time.sleep(0.05)
    second = _read_power_counters(lib, handles)
    watts: list[float] = []
    for before, after in zip(first, second, strict=False):
        timestamp_delta = after.timestamp - before.timestamp
        if timestamp_delta <= 0:
            continue
        energy_delta = after.energy - before.energy
        watts.append(max(0.0, energy_delta / timestamp_delta))
    if not watts:
        return None
    return sum(watts)


def _collect_level_zero_temp(lib: ctypes.CDLL, device: _LevelZeroDevice) -> float | None:
    handles = _enum_handles(lib.zesDeviceEnumTemperatureSensors, device.handle)
    temps: list[float] = []
    for handle in handles:
        temp = ctypes.c_double(0.0)
        if lib.zesTemperatureGetState(handle, ctypes.byref(temp)) == _ZE_RESULT_SUCCESS:
            temps.append(float(temp.value))
    if not temps:
        return None
    return max(temps)


def collect_level_zero_stats(selector: GpuTelemetrySelector) -> dict[str, Any] | None:
    """Collect Intel GPU stats using Level Zero Sysman."""
    global _level_zero_fallback_logged  # noqa: PLW0603

    lib = _load_level_zero()
    if lib is None:
        if not _level_zero_fallback_logged:
            logger.debug("Level Zero loader unavailable; falling back for %s", selector.device)
            _level_zero_fallback_logged = True
        return None
    try:
        device = _select_level_zero_device(_discover_level_zero_devices(lib), selector)
        if device is None:
            return None
        used_bytes, total_bytes = _collect_level_zero_memory(lib, device)
        mem_pct = (
            (100.0 * used_bytes / total_bytes)
            if used_bytes is not None and total_bytes is not None and total_bytes > 0
            else None
        )
        gpu_util = _collect_native_engine_util(lib, device)
        temp = _collect_level_zero_temp(lib, device)
        if temp is None:
            temp = _collect_sysfs_temp(device)
        stats = {
            "device": device.name,
            "gpu_util": format_percent(gpu_util),
            "mem_util": format_percent(mem_pct),
            "vram": format_memory_used(used_bytes, total_bytes),
            "temp": format_temp(temp),
            "power": format_power(_collect_level_zero_power(lib, device)),
            "source": "level-zero",
            "pci_bdf": device.pci_bdf or "N/A",
            "uuid": device.uuid or "N/A",
        }
        return with_host_stats(stats)
    except AttributeError, OSError, ValueError, ctypes.ArgumentError:
        logger.debug("Level Zero telemetry failed", exc_info=True)
        return None
