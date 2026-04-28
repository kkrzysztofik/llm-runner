"""Build lock management — atomic lock acquisition and release."""

import contextlib
import json
import logging
import os
import time
from pathlib import Path

from .models import BuildLock

logger = logging.getLogger(__name__)


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

    try:
        # First, check if there's a stale lock and remove it safely
        if lock_path.exists() and is_lock_stale(lock_path):
            logger.warning("[lock] removing stale lock at %s", lock_path)
            with contextlib.suppress(Exception):
                lock_path.unlink()

        # Ensure parent directory exists
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic lock acquisition using O_EXCL flag
        # O_CREAT|O_EXCL ensures the file is created only if it doesn't exist
        # This is atomic at the filesystem level
        lock_fd = os.open(
            str(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,  # Owner read/write only
        )

        # Write lock data to the file descriptor
        lock_data = {
            "pid": os.getpid(),
            "started_at": time.time(),
            "backend": backend,
        }
        try:
            os.write(lock_fd, json.dumps(lock_data).encode("utf-8"))
        finally:
            os.close(lock_fd)

        logger.info("[lock] acquired for backend=%s pid=%s", backend, os.getpid())
        return True

    except FileExistsError:
        # Another process holds the lock
        logger.error("[lock] already held by another process: %s", lock_path)
        return False
    except OSError as e:
        logger.error("[lock] failed to acquire: %s", e)
        return False
    except Exception as e:
        logger.error("[lock] failed to acquire: %s", e)
        return False


def release_lock(lock_file: Path | None) -> None:
    """Release build lock by unlinking the lock file.

    Args:
        lock_file: Path to the lock file to remove, or None (no-op).
    """
    if lock_file and lock_file.exists():
        logger.info("[lock] releasing %s", lock_file)
        with contextlib.suppress(Exception):
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

        pid = data.get("pid")
        started_at_str = data.get("started_at")

        if pid is None or started_at_str is None:
            return True

        started_at = float(started_at_str)
        lock = BuildLock(pid=pid, started_at=started_at, backend=data.get("backend", ""))

        # Check if PID is still valid
        try:
            os.kill(pid, 0)
            pid_valid = True
        except OSError:
            pid_valid = False

        # Lock is stale if PID is invalid or timeout exceeded
        return not pid_valid or lock.is_stale()

    except (json.JSONDecodeError, FileNotFoundError, KeyError):
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
        pid = data.get("pid", "unknown")
        backend = data.get("backend", "unknown")
        return f"Build lock already held by PID {pid} (backend: {backend})"
    except Exception:
        return "Build lock file exists but could not be read"
