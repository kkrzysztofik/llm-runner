"""Level Zero engine utilization collection."""

from __future__ import annotations

import ctypes
import time

from .level_zero_device import _enum_handles
from .level_zero_fdinfo import _collect_fdinfo_engine_util
from .level_zero_sysfs import _collect_sysfs_engine_util
from .level_zero_types import _ZE_RESULT_SUCCESS, _LevelZeroDevice, _ZesEngineStats


def _read_engine_stats(lib: ctypes.CDLL, handles: list[ctypes.c_void_p]) -> list[_ZesEngineStats]:
    snapshots: list[_ZesEngineStats] = []
    for handle in handles:
        stats = _ZesEngineStats()
        if lib.zesEngineGetActivity(handle, ctypes.byref(stats)) == _ZE_RESULT_SUCCESS:
            snapshots.append(stats)
    return snapshots


def _collect_level_zero_engine_util(lib: ctypes.CDLL, device: _LevelZeroDevice) -> float | None:
    handles = _enum_handles(lib.zesDeviceEnumEngineGroups, device.handle)
    if not handles:
        return None
    first = _read_engine_stats(lib, handles)
    if not first:
        return None
    time.sleep(0.05)
    second = _read_engine_stats(lib, handles)
    values: list[float] = []
    for before, after in zip(first, second, strict=False):
        timestamp_delta = after.timestamp - before.timestamp
        if timestamp_delta <= 0:
            continue
        active_delta = after.activeTime - before.activeTime
        values.append(max(0.0, min(100.0, 100.0 * active_delta / timestamp_delta)))
    if not values:
        return None
    return max(values)


def _collect_native_engine_util(lib: ctypes.CDLL, device: _LevelZeroDevice) -> float | None:
    util = _collect_level_zero_engine_util(lib, device)
    if util is not None:
        return util
    util = _collect_sysfs_engine_util(device)
    if util is not None:
        return util
    return _collect_fdinfo_engine_util(device)
