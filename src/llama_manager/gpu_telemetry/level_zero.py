"""Level Zero Sysman telemetry collector for Intel GPUs."""

import ctypes
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .common import (
    INTEL_VENDOR_ID,
    GpuTelemetrySelector,
    format_memory_used,
    format_percent,
    format_power,
    format_temp,
    with_host_stats,
)

logger = logging.getLogger(__name__)

_level_zero_fallback_logged = False

_ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x1
_ZES_STRUCTURE_TYPE_PCI_PROPERTIES = 0x2
_ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES = 0x2D
_ZES_STRUCTURE_TYPE_MEMORY_STATE = 0x1D
_ZES_STRING_PROPERTY_SIZE = 64
_ZE_MAX_DEVICE_UUID_SIZE = 16
_ZE_MAX_DEVICE_NAME = 256
_ZE_RESULT_SUCCESS = 0
_SYS_BUS_PCI_DEVICES = "/sys/bus/pci/devices"
_SYS_CLASS_DRM = "/sys/class/drm"
_PROC_ROOT = "/proc"
_INTEL_HWMON_NAMES = {"xe", "i915", "intel_gpu", "intel-gpu"}
_PREFERRED_TEMP_LABELS = ("pkg", "card", "gpu", "junction")
_FDINFO_ENGINE_RE = re.compile(r"^drm-engine-[^:]+:\s*(\d+)")
_FDINFO_CYCLES_RE = re.compile(r"^drm-cycles-([^:]+):\s*(\d+)")
_FDINFO_TOTAL_CYCLES_RE = re.compile(r"^drm-total-cycles-([^:]+):\s*(\d+)")


class _ZeDeviceUuid(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint8 * _ZE_MAX_DEVICE_UUID_SIZE)]


class _ZeDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", _ZeDeviceUuid),
        ("name", ctypes.c_char * _ZE_MAX_DEVICE_NAME),
    ]


class _ZesDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("core", _ZeDeviceProperties),
        ("numSubdevices", ctypes.c_uint32),
        ("serialNumber", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("boardNumber", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("brandName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("modelName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("vendorName", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
        ("driverVersion", ctypes.c_char * _ZES_STRING_PROPERTY_SIZE),
    ]


class _ZesDeviceExtProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("uuid", _ZeDeviceUuid),
        ("type", ctypes.c_int),
        ("flags", ctypes.c_uint32),
    ]


class _ZesPciAddress(ctypes.Structure):
    _fields_ = [
        ("domain", ctypes.c_uint32),
        ("bus", ctypes.c_uint32),
        ("device", ctypes.c_uint32),
        ("function", ctypes.c_uint32),
    ]


class _ZesPciSpeed(ctypes.Structure):
    _fields_ = [
        ("gen", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("maxBandwidth", ctypes.c_int64),
    ]


class _ZesPciProperties(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("address", _ZesPciAddress),
        ("maxSpeed", _ZesPciSpeed),
        ("haveBandwidthCounters", ctypes.c_uint32),
        ("havePacketCounters", ctypes.c_uint32),
        ("haveReplayCounters", ctypes.c_uint32),
    ]


class _ZesMemState(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("health", ctypes.c_int),
        ("free", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _ZesEngineStats(ctypes.Structure):
    _fields_ = [("activeTime", ctypes.c_uint64), ("timestamp", ctypes.c_uint64)]


class _ZesPowerEnergyCounter(ctypes.Structure):
    _fields_ = [("energy", ctypes.c_uint64), ("timestamp", ctypes.c_uint64)]


@dataclass(frozen=True)
class _FdinfoCounters:
    engine_ns: int
    cycles: dict[str, int]
    total_cycles: dict[str, int]


@dataclass(frozen=True)
class _LevelZeroDevice:
    handle: ctypes.c_void_p
    ordinal: int
    name: str
    vendor_id: int
    device_id: int
    pci_bdf: str | None
    uuid: str


def _decode_c_string(value: bytes | ctypes.Array[ctypes.c_char]) -> str:
    raw = bytes(value)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


def _uuid_to_string(uuid: _ZeDeviceUuid) -> str:
    raw = bytes(uuid.id)
    return "".join(f"{byte:02x}" for byte in raw)


def _pci_bdf(address: _ZesPciAddress) -> str:
    return f"{address.domain:04x}:{address.bus:02x}:{address.device:02x}.{address.function:x}"


def _load_level_zero() -> ctypes.CDLL | None:
    try:
        lib = ctypes.CDLL("libze_loader.so.1")
    except OSError:
        try:
            lib = ctypes.CDLL("libze_loader.so")
        except OSError:
            return None

    lib.zesInit.argtypes = [ctypes.c_uint32]
    lib.zesInit.restype = ctypes.c_int
    lib.zesDriverGet.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
    lib.zesDriverGet.restype = ctypes.c_int
    lib.zesDeviceGet.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.zesDeviceGet.restype = ctypes.c_int
    lib.zesDeviceGetProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZesDeviceProperties)]
    lib.zesDeviceGetProperties.restype = ctypes.c_int
    lib.zesDevicePciGetProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZesPciProperties)]
    lib.zesDevicePciGetProperties.restype = ctypes.c_int
    lib.zesDeviceEnumMemoryModules.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.zesDeviceEnumMemoryModules.restype = ctypes.c_int
    lib.zesMemoryGetState.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZesMemState)]
    lib.zesMemoryGetState.restype = ctypes.c_int
    lib.zesDeviceEnumEngineGroups.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.zesDeviceEnumEngineGroups.restype = ctypes.c_int
    lib.zesEngineGetActivity.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ZesEngineStats)]
    lib.zesEngineGetActivity.restype = ctypes.c_int
    lib.zesDeviceEnumPowerDomains.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.zesDeviceEnumPowerDomains.restype = ctypes.c_int
    lib.zesPowerGetEnergyCounter.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ZesPowerEnergyCounter),
    ]
    lib.zesPowerGetEnergyCounter.restype = ctypes.c_int
    lib.zesDeviceEnumTemperatureSensors.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.zesDeviceEnumTemperatureSensors.restype = ctypes.c_int
    lib.zesTemperatureGetState.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.zesTemperatureGetState.restype = ctypes.c_int
    return lib


def _enum_handles(
    enum_fn: Callable[[ctypes.c_void_p, Any, Any], int],
    owner: ctypes.c_void_p,
) -> list[ctypes.c_void_p]:
    count = ctypes.c_uint32(0)
    if enum_fn(owner, ctypes.byref(count), None) != _ZE_RESULT_SUCCESS or count.value <= 0:
        return []
    handles = (ctypes.c_void_p * count.value)()
    if enum_fn(owner, ctypes.byref(count), handles) != _ZE_RESULT_SUCCESS:
        return []
    return list(handles[: count.value])


def _discover_level_zero_devices(lib: ctypes.CDLL) -> list[_LevelZeroDevice]:
    if lib.zesInit(0) != _ZE_RESULT_SUCCESS:
        return []

    driver_count = ctypes.c_uint32(0)
    if lib.zesDriverGet(ctypes.byref(driver_count), None) != _ZE_RESULT_SUCCESS:
        return []
    if driver_count.value <= 0:
        return []

    drivers = (ctypes.c_void_p * driver_count.value)()
    if lib.zesDriverGet(ctypes.byref(driver_count), drivers) != _ZE_RESULT_SUCCESS:
        return []

    devices: list[_LevelZeroDevice] = []
    for driver in drivers[: driver_count.value]:
        device_count = ctypes.c_uint32(0)
        if lib.zesDeviceGet(driver, ctypes.byref(device_count), None) != _ZE_RESULT_SUCCESS:
            continue
        if device_count.value <= 0:
            continue
        device_handles = (ctypes.c_void_p * device_count.value)()
        if (
            lib.zesDeviceGet(driver, ctypes.byref(device_count), device_handles)
            != _ZE_RESULT_SUCCESS
        ):
            continue

        for handle in device_handles[: device_count.value]:
            ext_props = _ZesDeviceExtProperties()
            ext_props.stype = _ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES
            props = _ZesDeviceProperties()
            props.stype = _ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
            props.pNext = ctypes.cast(ctypes.pointer(ext_props), ctypes.c_void_p)
            if lib.zesDeviceGetProperties(handle, ctypes.byref(props)) != _ZE_RESULT_SUCCESS:
                continue

            pci_bdf: str | None = None
            pci_props = _ZesPciProperties()
            pci_props.stype = _ZES_STRUCTURE_TYPE_PCI_PROPERTIES
            if lib.zesDevicePciGetProperties(handle, ctypes.byref(pci_props)) == _ZE_RESULT_SUCCESS:
                pci_bdf = _pci_bdf(pci_props.address)

            name = _decode_c_string(props.modelName) or _decode_c_string(props.core.name)
            devices.append(
                _LevelZeroDevice(
                    handle=handle,
                    ordinal=len(devices),
                    name=name or "Intel GPU",
                    vendor_id=int(props.core.vendorId),
                    device_id=int(props.core.deviceId),
                    pci_bdf=pci_bdf,
                    uuid=_uuid_to_string(ext_props.uuid or props.core.uuid),
                )
            )
    return devices


def _select_level_zero_device(
    devices: list[_LevelZeroDevice],
    selector: GpuTelemetrySelector,
) -> _LevelZeroDevice | None:
    if selector.pci_bdf:
        wanted_bdf = selector.pci_bdf.lower()
        for device in devices:
            if device.pci_bdf and device.pci_bdf.lower() == wanted_bdf:
                return device
    if selector.device_id is not None:
        for device in devices:
            if device.device_id == selector.device_id:
                return device
    intel_devices = [
        device
        for device in devices
        if device.vendor_id == (selector.vendor_id or INTEL_VENDOR_ID)
        or "intel" in device.name.lower()
        or "arc" in device.name.lower()
        or "battlemage" in device.name.lower()
    ]
    candidates = intel_devices or devices
    if 0 <= selector.ordinal < len(candidates):
        return candidates[selector.ordinal]
    return candidates[0] if candidates else None


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


def _safe_read_text(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
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
        if not (entry.startswith("card") or entry.startswith("renderD")):
            continue
        if entry.startswith("card") and "-" in entry:
            continue
        device_path = os.path.join(_SYS_CLASS_DRM, entry, "device")
        if os.path.exists(device_path) and _device_root_matches(device_path, device):
            roots.append(device_path)

    unique_roots: list[str] = []
    seen: set[str] = set()
    for root in roots:
        real = os.path.realpath(root)
        if real in seen or not os.path.isdir(root):
            continue
        seen.add(real)
        unique_roots.append(root)
    return unique_roots


def _iter_hwmon_paths(device: _LevelZeroDevice) -> list[str]:
    hwmon_paths: list[str] = []
    for root in _iter_device_roots(device):
        direct = os.path.join(root, "hwmon")
        try:
            entries = sorted(os.listdir(direct))
        except OSError:
            continue
        for entry in entries:
            path = os.path.join(direct, entry)
            if os.path.isdir(path):
                hwmon_paths.append(path)
    return hwmon_paths


def _read_hwmon_temperature(hwmon_path: str) -> float | None:
    name = (_safe_read_text(os.path.join(hwmon_path, "name")) or "").lower()
    if name and name not in _INTEL_HWMON_NAMES:
        return None

    readings: list[tuple[str, float]] = []
    try:
        entries = sorted(os.listdir(hwmon_path))
    except OSError:
        return None
    for entry in entries:
        if not entry.startswith("temp") or not entry.endswith("_input"):
            continue
        index = entry.removeprefix("temp").removesuffix("_input")
        raw_value = _safe_read_int(os.path.join(hwmon_path, entry))
        if raw_value is None:
            continue
        temp = _normalize_temp(raw_value)
        if temp is None:
            continue
        label = (_safe_read_text(os.path.join(hwmon_path, f"temp{index}_label")) or "").lower()
        readings.append((label, temp))
    if not readings:
        return None
    for preferred in _PREFERRED_TEMP_LABELS:
        for label, temp in readings:
            if label == preferred:
                return temp
    return max(temp for _, temp in readings)


def _iter_drm_paths(device: _LevelZeroDevice) -> list[str]:
    paths: list[str] = []
    for root in _iter_device_roots(device):
        drm_root = os.path.join(root, "drm")
        try:
            entries = sorted(os.listdir(drm_root))
        except OSError:
            continue
        for entry in entries:
            if entry.startswith("card") and "-" not in entry:
                path = os.path.join(drm_root, entry)
                if os.path.isdir(path):
                    paths.append(path)

    try:
        entries = sorted(os.listdir(_SYS_CLASS_DRM))
    except OSError:
        entries = []
    for entry in entries:
        if not entry.startswith("card") or "-" in entry:
            continue
        path = os.path.join(_SYS_CLASS_DRM, entry)
        device_path = os.path.join(path, "device")
        if os.path.exists(device_path) and _device_root_matches(device_path, device):
            paths.append(path)

    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        unique_paths.append(path)
    return unique_paths


def _read_drm_engine_busy(drm_paths: list[str]) -> list[int]:
    counters: list[int] = []
    for drm_path in drm_paths:
        engine_root = os.path.join(drm_path, "engine")
        try:
            engines = sorted(os.listdir(engine_root))
        except OSError:
            continue
        for engine in engines:
            busy = _safe_read_int(os.path.join(engine_root, engine, "busy"))
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
    second = _read_drm_engine_busy(drm_paths)
    values: list[float] = []
    for before, after in zip(first, second, strict=False):
        if after >= before:
            values.append(100.0 * (after - before) / elapsed)
    if not values:
        return None
    return max(0.0, min(100.0, max(values)))


def _fdinfo_matches_device(path: str, lines: list[str], device: _LevelZeroDevice) -> bool:
    driver = ""
    pdev = ""
    for line in lines:
        if line.startswith("drm-driver:"):
            driver = line.partition(":")[2].strip().lower()
        elif line.startswith("drm-pdev:"):
            pdev = line.partition(":")[2].strip().lower()
    if driver and driver not in ("xe", "i915"):
        return False
    if device.pci_bdf:
        return pdev == device.pci_bdf.lower()
    return driver in ("xe", "i915") or "drm" in path


def _read_fdinfo_counters(device: _LevelZeroDevice) -> dict[str, _FdinfoCounters]:
    snapshots: dict[str, _FdinfoCounters] = {}
    proc_root = _PROC_ROOT
    try:
        proc_entries = sorted(os.listdir(proc_root))
    except OSError:
        return snapshots
    for pid in proc_entries:
        if not pid.isdigit():
            continue
        fdinfo_root = os.path.join(proc_root, pid, "fdinfo")
        try:
            fd_entries = sorted(os.listdir(fdinfo_root))
        except OSError:
            continue
        for fd in fd_entries:
            path = os.path.join(fdinfo_root, fd)
            text = _safe_read_text(path)
            if not text:
                continue
            lines = text.splitlines()
            if not _fdinfo_matches_device(path, lines, device):
                continue
            engine_ns = 0
            cycles: dict[str, int] = {}
            total_cycles: dict[str, int] = {}
            for line in lines:
                if match := _FDINFO_ENGINE_RE.match(line):
                    engine_ns += int(match.group(1))
                    continue
                if match := _FDINFO_CYCLES_RE.match(line):
                    cycles[match.group(1)] = cycles.get(match.group(1), 0) + int(match.group(2))
                    continue
                if match := _FDINFO_TOTAL_CYCLES_RE.match(line):
                    total_cycles[match.group(1)] = total_cycles.get(match.group(1), 0) + int(
                        match.group(2)
                    )
            if engine_ns > 0 or cycles or total_cycles:
                snapshots[path] = _FdinfoCounters(engine_ns, cycles, total_cycles)
    return snapshots


def _collect_fdinfo_engine_util(device: _LevelZeroDevice) -> float | None:
    start_time = time.monotonic_ns()
    first = _read_fdinfo_counters(device)
    if not first:
        return None
    time.sleep(0.05)
    elapsed = time.monotonic_ns() - start_time
    if elapsed <= 0:
        return None
    second = _read_fdinfo_counters(device)
    percentages: list[float] = []
    for path, after in second.items():
        before = first.get(path)
        if before is None:
            continue
        if after.engine_ns >= before.engine_ns and after.engine_ns > 0:
            percentages.append(100.0 * (after.engine_ns - before.engine_ns) / elapsed)
        for engine, after_cycles in after.cycles.items():
            before_cycles = before.cycles.get(engine)
            before_total = before.total_cycles.get(engine)
            after_total = after.total_cycles.get(engine)
            if before_cycles is None or before_total is None or after_total is None:
                continue
            cycle_delta = after_cycles - before_cycles
            total_delta = after_total - before_total
            if cycle_delta >= 0 and total_delta > 0:
                percentages.append(100.0 * cycle_delta / total_delta)
    if not percentages:
        return None
    return max(0.0, min(100.0, sum(percentages)))


def _collect_native_engine_util(lib: ctypes.CDLL, device: _LevelZeroDevice) -> float | None:
    util = _collect_level_zero_engine_util(lib, device)
    if util is not None:
        return util
    util = _collect_sysfs_engine_util(device)
    if util is not None:
        return util
    return _collect_fdinfo_engine_util(device)


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


def _collect_sysfs_temp(device: _LevelZeroDevice) -> float | None:
    """Fallback: read GPU temperature from sysfs hwmon when L0 Sysman fails."""
    for hwmon_path in _iter_hwmon_paths(device):
        temp = _read_hwmon_temperature(hwmon_path)
        if temp is not None:
            return temp
    return None


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
