"""ctypes types and small value helpers for Level Zero telemetry."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

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
