"""Slot-level lockfile management — acquire, release, staleness, shutdown."""

import os
import signal
import time

import psutil

from ..config import Config, ErrorDetail
from .launch import _lockfile_error
from .lockfile import (
    _get_lock_path,
    check_lockfile_integrity,
    create_lock,
    read_lock,
    release_lock,
    resolve_runtime_dir,
    verify_shutdown_ownership,
)
from .types import LOCKFILE_FIX_SUGGESTION


def acquire_slot_lock(slot_id: str, port: int, server_pid: int | None = None) -> str:
    """Acquire a lockfile for a slot. Returns the lock path."""

    runtime_dir = resolve_runtime_dir()
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if lock_path.exists():
        integrity = check_lockfile_integrity(runtime_dir, slot_id)
        if integrity is not None:
            raise _lockfile_error(
                integrity.why_blocked,
                LOCKFILE_FIX_SUGGESTION,
            )

    pid = server_pid if server_pid is not None else os.getpid()
    return str(create_lock(runtime_dir, slot_id, pid, port))


def release_slot_lock(slot_id: str) -> None:
    """Release lockfile for a slot."""

    runtime_dir = resolve_runtime_dir()
    release_lock(runtime_dir, slot_id)


def check_lock_stale(slot_id: str) -> bool:
    """Check if a lockfile is stale."""

    runtime_dir = resolve_runtime_dir()
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if not lock_path.exists():
        return False

    metadata_result = read_lock(runtime_dir, slot_id, require_valid=False)
    if metadata_result is None or isinstance(metadata_result, ErrorDetail):
        return False

    metadata = metadata_result
    age = time.time() - metadata.started_at
    stale_threshold = Config().lock_stale_threshold_s
    return age > stale_threshold


def shutdown_slot(slot_id: str, timeout: float = 10.0) -> bool:
    """Gracefully shut down a slot's server process."""

    runtime_dir = resolve_runtime_dir()
    metadata_result = read_lock(runtime_dir, slot_id, require_valid=False)
    if metadata_result is None or isinstance(metadata_result, ErrorDetail):
        return True

    metadata = metadata_result
    pid = metadata.pid

    if pid is None:
        return True

    if not verify_shutdown_ownership(pid, metadata.port):
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        release_lock(runtime_dir, slot_id)
        return True

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            release_lock(runtime_dir, slot_id)
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        release_lock(runtime_dir, slot_id)
        return True

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            release_lock(runtime_dir, slot_id)
            return True
        time.sleep(0.1)

    return False
