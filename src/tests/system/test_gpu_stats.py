"""Tests for Intel GPU sysfs/fdinfo telemetry fallbacks."""

import ctypes
import pathlib

import pytest

from llama_manager.gpu_telemetry import level_zero_fdinfo, level_zero_sysfs
from llama_manager.gpu_telemetry.level_zero_types import _LevelZeroDevice


def _level_zero_device() -> _LevelZeroDevice:
    return _LevelZeroDevice(
        handle=ctypes.c_void_p(1),
        ordinal=0,
        name="Intel GPU",
        vendor_id=0x8086,
        device_id=0xE20B,
        pci_bdf="0000:30:00.0",
        uuid="abc",
    )


def test_level_zero_sysfs_temp_accepts_xe_pkg_sensor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sysfs fallback should read the XE package temperature used by nvtop."""
    pci_root = tmp_path / "pci"
    hwmon = pci_root / "0000:30:00.0" / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("xe\n")
    (hwmon / "temp1_input").write_text("42000\n")
    (hwmon / "temp1_label").write_text("card\n")
    (hwmon / "temp2_input").write_text("49000\n")
    (hwmon / "temp2_label").write_text("pkg\n")
    monkeypatch.setattr(level_zero_sysfs, "_SYS_BUS_PCI_DEVICES", str(pci_root))

    assert level_zero_sysfs._collect_sysfs_temp(_level_zero_device()) == 49.0


def test_level_zero_sysfs_temp_accepts_i915_sensor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sysfs fallback should not reject i915 hwmon names."""
    pci_root = tmp_path / "pci"
    hwmon = pci_root / "0000:30:00.0" / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("i915\n")
    (hwmon / "temp1_input").write_text("47000\n")
    monkeypatch.setattr(level_zero_sysfs, "_SYS_BUS_PCI_DEVICES", str(pci_root))

    assert level_zero_sysfs._collect_sysfs_temp(_level_zero_device()) == 47.0


def test_level_zero_fdinfo_util_samples_i915_engine_ns(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DRM fdinfo fallback should compute process-summed i915 engine utilization."""
    proc_root = tmp_path / "proc"
    fdinfo = proc_root / "123" / "fdinfo"
    fdinfo.mkdir(parents=True)
    fdinfo_file = fdinfo / "7"
    fdinfo_file.write_text(
        "drm-driver:\ti915\ndrm-pdev:\t0000:30:00.0\ndrm-engine-render:\t100000000 ns\n",
    )

    def advance_sample(_seconds: float) -> None:
        fdinfo_file.write_text(
            "drm-driver:\ti915\ndrm-pdev:\t0000:30:00.0\ndrm-engine-render:\t125000000 ns\n",
        )

    monkeypatch.setattr(level_zero_fdinfo, "_PROC_ROOT", str(proc_root))
    monkeypatch.setattr(level_zero_fdinfo.time, "sleep", advance_sample)
    monotonic_values = iter((1_000_000_000, 1_100_000_000))
    monkeypatch.setattr(level_zero_fdinfo.time, "monotonic_ns", lambda: next(monotonic_values))

    assert level_zero_fdinfo._collect_fdinfo_engine_util(_level_zero_device()) == 25.0
