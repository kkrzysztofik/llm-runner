"""Level Zero Sysman telemetry collector for Intel GPUs."""

import time

from . import level_zero_fdinfo as _fdinfo
from . import level_zero_sysfs as _sysfs
from .level_zero_device import (
    _discover_level_zero_devices,
    _enum_handles,
    _load_level_zero,
    _select_level_zero_device,
)
from .level_zero_telemetry import collect_level_zero_stats
from .level_zero_types import (
    _decode_c_string,
    _FdinfoCounters,
    _LevelZeroDevice,
    _pci_bdf,
    _uuid_to_string,
    _ZeDeviceProperties,
    _ZeDeviceUuid,
    _ZesDeviceExtProperties,
    _ZesDeviceProperties,
    _ZesEngineStats,
    _ZesMemState,
    _ZesPciAddress,
    _ZesPciProperties,
    _ZesPciSpeed,
    _ZesPowerEnergyCounter,
)

_SYS_BUS_PCI_DEVICES = _sysfs._SYS_BUS_PCI_DEVICES
_SYS_CLASS_DRM = _sysfs._SYS_CLASS_DRM
_PROC_ROOT = _fdinfo._PROC_ROOT


def _collect_sysfs_temp(device: _LevelZeroDevice) -> float | None:
    _sysfs._SYS_BUS_PCI_DEVICES = _SYS_BUS_PCI_DEVICES
    _sysfs._SYS_CLASS_DRM = _SYS_CLASS_DRM
    return _sysfs._collect_sysfs_temp(device)


def _collect_fdinfo_engine_util(device: _LevelZeroDevice) -> float | None:
    _fdinfo._PROC_ROOT = _PROC_ROOT
    return _fdinfo._collect_fdinfo_engine_util(device)


__all__ = [
    "_FdinfoCounters",
    "_LevelZeroDevice",
    "_ZeDeviceProperties",
    "_ZeDeviceUuid",
    "_ZesDeviceExtProperties",
    "_ZesDeviceProperties",
    "_ZesEngineStats",
    "_ZesMemState",
    "_ZesPciAddress",
    "_ZesPciProperties",
    "_ZesPciSpeed",
    "_ZesPowerEnergyCounter",
    "_collect_fdinfo_engine_util",
    "_collect_sysfs_temp",
    "_decode_c_string",
    "_discover_level_zero_devices",
    "_enum_handles",
    "_load_level_zero",
    "_pci_bdf",
    "_PROC_ROOT",
    "_select_level_zero_device",
    "_SYS_BUS_PCI_DEVICES",
    "_SYS_CLASS_DRM",
    "_uuid_to_string",
    "collect_level_zero_stats",
    "time",
]
