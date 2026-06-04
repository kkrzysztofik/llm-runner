"""Level Zero device discovery."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

from .common import INTEL_VENDOR_ID, GpuTelemetrySelector
from .level_zero_types import (
    _ZE_RESULT_SUCCESS,
    _ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES,
    _ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES,
    _ZES_STRUCTURE_TYPE_PCI_PROPERTIES,
    _decode_c_string,
    _LevelZeroDevice,
    _pci_bdf,
    _uuid_to_string,
    _ZeDeviceUuid,
    _ZesDeviceExtProperties,
    _ZesDeviceProperties,
    _ZesEngineStats,
    _ZesMemState,
    _ZesPciProperties,
    _ZesPowerEnergyCounter,
)


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
            uuid = ext_props.uuid if isinstance(ext_props.uuid, _ZeDeviceUuid) else props.core.uuid
            devices.append(
                _LevelZeroDevice(
                    handle=handle,
                    ordinal=len(devices),
                    name=name or "Intel GPU",
                    vendor_id=int(props.core.vendorId),
                    device_id=int(props.core.deviceId),
                    pci_bdf=pci_bdf,
                    uuid=_uuid_to_string(uuid),
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
