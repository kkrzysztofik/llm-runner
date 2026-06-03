"""Level Zero Sysman telemetry collector for Intel GPUs."""

import ctypes
import logging
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
    return sum(values) / len(values)


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
        stats = {
            "device": device.name,
            "gpu_util": format_percent(_collect_level_zero_engine_util(lib, device)),
            "mem_util": format_percent(mem_pct),
            "vram": format_memory_used(used_bytes, total_bytes),
            "temp": format_temp(_collect_level_zero_temp(lib, device)),
            "power": format_power(_collect_level_zero_power(lib, device)),
            "source": "level-zero",
            "pci_bdf": device.pci_bdf or "N/A",
            "uuid": device.uuid or "N/A",
        }
        return with_host_stats(stats)
    except AttributeError, OSError, ValueError, ctypes.ArgumentError:
        logger.debug("Level Zero telemetry failed", exc_info=True)
        return None
