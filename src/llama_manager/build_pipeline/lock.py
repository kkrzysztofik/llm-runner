"""Build lock management — atomic lock acquisition and release."""

import contextlib
import json
import os
import time
from pathlib import Path

import psutil
from loguru import logger

from ..common.file_ops import atomic_exclusive_create_json, atomic_write_json
from .models import BuildLock


def acquire_lock(lock_path: Path, backend: str, *, dry_run: bool = False) -> bool:
    """Acquire build lock atomically.

    Uses O_EXCL flag to ensure atomic lock acquisition, preventing TOCTOU race conditions.

    Args:
        lock_path: Path to lock file.
        backend: Backend name string written into the lock file.
        dry_run: When True, skips actual lock acquisition and returns True.

    Returns:
        True if lock acquired (or dry-run), False otherwise.
    """
    if dry_run:
        logger.debug("[lock] dry-run: skipping lock acquisition")
        return True

    logger.info("[lock] acquiring lock at %s", lock_path)

    # Ensure parent directory exists
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_data = {
        "pid": os.getpid(),
        "started_at": time.time(),
        "backend": backend,
    }

    try:
        # Primary path: atomic exclusive create — fails immediately if file exists.
        atomic_exclusive_create_json(lock_path, lock_data)
        logger.info("[lock] acquired for backend=%s pid=%s", backend, os.getpid())
        return True
    except FileExistsError:
        pass

    # Lock file exists; check whether it is stale.
    if not is_lock_stale(lock_path):
        logger.error("[lock] already held by another process: %s", lock_path)
        return False

    # Stale lock: replace atomically so we never delete-then-create (TOCTOU fix).
    logger.warning("[lock] replacing stale lock at %s", lock_path)
    try:
        atomic_write_json(lock_path, lock_data)
        logger.info("[lock] acquired (replaced stale) for backend=%s pid=%s", backend, os.getpid())
        return True
    except OSError as e:
        logger.error("[lock] failed to replace stale lock: %s", e)
        return False


def release_lock(lock_file: Path | None) -> None:
    """Release build lock by unlinking the lock file.

    Args:
        lock_file: Path to the lock file to remove, or None (no-op).
    """
    if lock_file and lock_file.exists():
        logger.info("[lock] releasing %s", lock_file)
        with contextlib.suppress(OSError):
            lock_file.unlink()
    else:
        logger.debug("[lock] no active lock to release")


def is_lock_stale(lock_path: Path) -> bool:
    """Check if a lock file is stale (PID invalid or timeout exceeded).

    Args:
        lock_path: Path to lock file.

    Returns:
        True if lock is stale or unreadable.
    """
    try:
        with open(lock_path) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return True

        pid = data.get("pid")
        started_at_str = data.get("started_at")

        if pid is None or started_at_str is None:
            return True

        started_at = float(started_at_str)
        lock = BuildLock(pid=pid, started_at=started_at, backend=data.get("backend", ""))

        # Check if PID is still valid
        pid_valid = psutil.pid_exists(pid)

        # Lock is stale if PID is invalid or timeout exceeded
        return not pid_valid or lock.is_stale()

    except json.JSONDecodeError, FileNotFoundError, ValueError, TypeError:
        return True


def get_lock_error_message(lock_path: Path) -> str:
    """Return a human-readable error message for lock contention.

    Args:
        lock_path: Path to the contested lock file.

    Returns:
        Descriptive error string.
    """
    try:
        with open(lock_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return "Build lock file exists but could not be read"
        pid = data.get("pid", "unknown")
        backend = data.get("backend", "unknown")
        return f"Build lock already held by PID {pid} (backend: {backend})"
    except OSError, json.JSONDecodeError, TypeError:
        return "Build lock file exists but could not be read"
