"""Sysfs, DRM, and fdinfo fallbacks for Intel GPU telemetry."""

from __future__ import annotations

import os
import time

from .common import INTEL_VENDOR_ID
from .level_zero_types import _LevelZeroDevice

_SYS_BUS_PCI_DEVICES = "/sys/bus/pci/devices"
_SYS_CLASS_DRM = "/sys/class/drm"
_INTEL_HWMON_NAMES = {"xe", "i915", "intel_gpu", "intel-gpu"}
_PREFERRED_TEMP_LABELS = ("pkg", "card", "gpu", "junction")


def _safe_read_text(path: str) -> str | None:
    try:
        with open(path) as file_obj:
            return file_obj.read().strip()
    except OSError:
        return None


def _safe_read_int(path: str) -> int | None:
    text = _safe_read_text(path)
    if text is None:
        return None
    try:
        return int(text.split(maxsplit=1)[0])
    except ValueError:
        return None


def _normalize_temp(raw_value: int) -> float | None:
    value = raw_value / 1000.0 if abs(raw_value) >= 1000 else float(raw_value)
    if value <= 0 or value >= 150:
        return None
    return value


def _device_root_matches(path: str, device: _LevelZeroDevice) -> bool:
    vendor = _safe_read_text(os.path.join(path, "vendor"))
    if vendor is not None and vendor.lower() != f"0x{INTEL_VENDOR_ID:04x}":
        return False
    device_id = _safe_read_text(os.path.join(path, "device"))
    if device_id is not None:
        try:
            if int(device_id, 16) != device.device_id:
                return False
        except ValueError:
            return False
    if device.pci_bdf:
        return os.path.basename(os.path.realpath(path)).lower() == device.pci_bdf.lower()
    return vendor is not None or device_id is not None


def _iter_device_roots(device: _LevelZeroDevice) -> list[str]:
    roots: list[str] = []
    if device.pci_bdf:
        roots.append(os.path.join(_SYS_BUS_PCI_DEVICES, device.pci_bdf))
    try:
        drm_entries = sorted(os.listdir(_SYS_CLASS_DRM))
    except OSError:
        drm_entries = []
    for entry in drm_entries:
        if not entry.startswith(("card", "renderD")):
            continue
        if entry.startswith("card") and "-" in entry:
            continue
        device_path = os.path.join(_SYS_CLASS_DRM, entry, "device")
        if os.path.exists(device_path) and _device_root_matches(device_path, device):
            roots.append(device_path)
    return _unique_existing_dirs(roots)


def _unique_existing_dirs(paths: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen or not os.path.isdir(path):
            continue
        seen.add(real)
        unique.append(path)
    return unique


def _iter_hwmon_paths(device: _LevelZeroDevice) -> list[str]:
    hwmon_paths: list[str] = []
    for root in _iter_device_roots(device):
        try:
            entries = sorted(os.listdir(os.path.join(root, "hwmon")))
        except OSError:
            continue
        for entry in entries:
            path = os.path.join(root, "hwmon", entry)
            if os.path.isdir(path):
                hwmon_paths.append(path)
    return hwmon_paths


def _read_hwmon_temperature(hwmon_path: str) -> float | None:
    if not _is_intel_hwmon(hwmon_path):
        return None
    readings = _collect_hwmon_temp_readings(hwmon_path)
    if not readings:
        return None
    for preferred in _PREFERRED_TEMP_LABELS:
        for label, temp in readings:
            if label == preferred:
                return temp
    return max(temp for _, temp in readings)


def _is_intel_hwmon(hwmon_path: str) -> bool:
    name = (_safe_read_text(os.path.join(hwmon_path, "name")) or "").lower()
    return not name or name in _INTEL_HWMON_NAMES


def _collect_hwmon_temp_readings(hwmon_path: str) -> list[tuple[str, float]]:
    readings: list[tuple[str, float]] = []
    entries = _list_hwmon_entries(hwmon_path)
    for entry in entries:
        reading = _read_single_temp_entry(hwmon_path, entry)
        if reading is not None:
            readings.append(reading)
    return readings


def _list_hwmon_entries(hwmon_path: str) -> list[str]:
    try:
        return sorted(os.listdir(hwmon_path))
    except OSError:
        return []


def _read_single_temp_entry(hwmon_path: str, entry: str) -> tuple[str, float] | None:
    if not entry.startswith("temp") or not entry.endswith("_input"):
        return None
    index = entry.removeprefix("temp").removesuffix("_input")
    raw_value = _safe_read_int(os.path.join(hwmon_path, entry))
    if raw_value is None:
        return None
    temp = _normalize_temp(raw_value)
    if temp is None:
        return None
    label = (_safe_read_text(os.path.join(hwmon_path, f"temp{index}_label")) or "").lower()
    return (label, temp)


def _iter_drm_paths(device: _LevelZeroDevice) -> list[str]:
    paths: list[str] = []
    paths.extend(_drm_paths_from_device_roots(device))
    paths.extend(_drm_paths_from_class_drm(device))
    return _unique_paths(paths)


def _drm_paths_from_device_roots(device: _LevelZeroDevice) -> list[str]:
    paths: list[str] = []
    for root in _iter_device_roots(device):
        drm_root = os.path.join(root, "drm")
        try:
            entries = sorted(os.listdir(drm_root))
        except OSError:
            continue
        for entry in entries:
            path = _drm_dir_path(drm_root, entry)
            if path is not None:
                paths.append(path)
    return paths


def _drm_paths_from_class_drm(device: _LevelZeroDevice) -> list[str]:
    try:
        entries = sorted(os.listdir(_SYS_CLASS_DRM))
    except OSError:
        return []
    paths: list[str] = []
    for entry in entries:
        if not entry.startswith("card") or "-" in entry:
            continue
        path = os.path.join(_SYS_CLASS_DRM, entry)
        device_path = os.path.join(path, "device")
        if os.path.exists(device_path) and _device_root_matches(device_path, device):
            paths.append(path)
    return paths


def _drm_dir_path(drm_root: str, entry: str) -> str | None:
    if not entry.startswith("card") or "-" in entry:
        return None
    path = os.path.join(drm_root, entry)
    if os.path.isdir(path):
        return path
    return None


def _unique_paths(paths: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        unique.append(path)
    return unique


def _read_drm_engine_busy(drm_paths: list[str]) -> list[int]:
    counters: list[int] = []
    for drm_path in drm_paths:
        try:
            engines = sorted(os.listdir(os.path.join(drm_path, "engine")))
        except OSError:
            continue
        for engine in engines:
            busy = _safe_read_int(os.path.join(drm_path, "engine", engine, "busy"))
            if busy is not None:
                counters.append(busy)
    return counters


def _collect_sysfs_engine_util(device: _LevelZeroDevice) -> float | None:
    drm_paths = _iter_drm_paths(device)
    if not drm_paths:
        return None
    start_time = time.monotonic_ns()
    first = _read_drm_engine_busy(drm_paths)
    if not first:
        return None
    time.sleep(0.05)
    elapsed = time.monotonic_ns() - start_time
    if elapsed <= 0:
        return None
    values = [
        100.0 * (after - before) / elapsed
        for before, after in zip(first, _read_drm_engine_busy(drm_paths), strict=False)
        if after >= before
    ]
    if not values:
        return None
    return max(0.0, min(100.0, max(values)))


def _collect_sysfs_temp(device: _LevelZeroDevice) -> float | None:
    """Fallback: read GPU temperature from sysfs hwmon when L0 Sysman fails."""
    for hwmon_path in _iter_hwmon_paths(device):
        temp = _read_hwmon_temperature(hwmon_path)
        if temp is not None:
            return temp
    return None
