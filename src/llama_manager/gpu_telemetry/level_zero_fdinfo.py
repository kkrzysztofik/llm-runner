"""fdinfo engine utilization fallback for Intel GPU telemetry."""

from __future__ import annotations

import os
import re
import time

from .level_zero_types import _FdinfoCounters, _LevelZeroDevice

_PROC_ROOT = "/proc"
_FDINFO_ENGINE_RE = re.compile(r"^drm-engine-[^:]+:\s*(\d+)")
_FDINFO_CYCLES_RE = re.compile(r"^drm-cycles-([^:]+):\s*(\d+)")
_FDINFO_TOTAL_CYCLES_RE = re.compile(r"^drm-total-cycles-([^:]+):\s*(\d+)")


def _safe_read_text(path: str) -> str | None:
    try:
        with open(path) as file_obj:
            return file_obj.read().strip()
    except OSError:
        return None


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
    try:
        proc_entries = sorted(os.listdir(_PROC_ROOT))
    except OSError:
        return snapshots
    for pid in proc_entries:
        if pid.isdigit():
            _read_pid_fdinfo_counters(os.path.join(_PROC_ROOT, pid, "fdinfo"), device, snapshots)
    return snapshots


def _read_pid_fdinfo_counters(
    fdinfo_root: str,
    device: _LevelZeroDevice,
    snapshots: dict[str, _FdinfoCounters],
) -> None:
    try:
        fd_entries = sorted(os.listdir(fdinfo_root))
    except OSError:
        return
    for fd in fd_entries:
        path = os.path.join(fdinfo_root, fd)
        text = _safe_read_text(path)
        if not text:
            continue
        lines = text.splitlines()
        if not _fdinfo_matches_device(path, lines, device):
            continue
        counter = _fdinfo_counter_from_lines(lines)
        if counter.engine_ns > 0 or counter.cycles or counter.total_cycles:
            snapshots[path] = counter


def _fdinfo_counter_from_lines(lines: list[str]) -> _FdinfoCounters:
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
            total_cycles[match.group(1)] = total_cycles.get(match.group(1), 0) + int(match.group(2))
    return _FdinfoCounters(engine_ns, cycles, total_cycles)


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
        percentages.extend(_cycle_percentages(before, after))
    if not percentages:
        return None
    return max(0.0, min(100.0, sum(percentages)))


def _cycle_percentages(before: _FdinfoCounters, after: _FdinfoCounters) -> list[float]:
    percentages: list[float] = []
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
    return percentages
