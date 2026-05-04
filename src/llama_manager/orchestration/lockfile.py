"""Lockfile operations — create, read, update, release, and integrity checks."""

import contextlib
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from ..common.constants import DIR_MODE_OWNER_ONLY, FILE_MODE_OWNER_ONLY
from ..common.file_ops import atomic_exclusive_create_json, atomic_write_json
from ..config import Config, ErrorCode, ErrorDetail

if TYPE_CHECKING:
    from ..config import MultiValidationError

# Module-local string constants (lockfile-specific).
LOCKFILE_CHECK_NAME: str = "lockfile_integrity"
MAX_COLLISION_RETRIES: int = 10
INDENTERRMINATE_OWNER_MESSAGE: str = (
    "indeterminate_owner: lock exists but ownership verification is not definitive"
)
INDENTERRMINATE_OWNER_FIX: str = (
    "verify owning process and clear lock only after confirmed stale ownership"
)


@dataclass
class LockMetadata:
    """Lockfile metadata persisted in `slot-{slot_id}.lock` files."""

    pid: int
    port: int
    started_at: float


class ValidationException(Exception):
    """Exception wrapper for MultiValidationError to enable raising as exception."""

    def __init__(self, multi_error: "MultiValidationError") -> None:
        self.multi_error = multi_error
        if multi_error.errors:
            details = "; ".join(e.why_blocked for e in multi_error.errors)
            super().__init__(
                f"Validation failed with {len(multi_error.errors)} error(s): {details}"
            )
        else:
            super().__init__(f"Validation failed with {len(multi_error.errors)} error(s)")


def _make_lockfile_validation_error(
    error_code: ErrorCode,
    failed_check: str,
    why_blocked: str,
    how_to_fix: str,
) -> ValidationException:
    """Build a single-error ValidationException from raw fields."""
    from ..config import MultiValidationError

    detail = ErrorDetail(
        error_code=error_code,
        failed_check=failed_check,
        why_blocked=why_blocked,
        how_to_fix=how_to_fix,
    )
    return ValidationException(MultiValidationError(errors=[detail]))


def _lockfile_error(why_blocked: str, how_to_fix: str) -> ValidationException:
    return _make_lockfile_validation_error(
        ErrorCode.LOCKFILE_INTEGRITY_FAILURE, LOCKFILE_CHECK_NAME, why_blocked, how_to_fix
    )


def _make_validation_error(
    error_code: ErrorCode,
    failed_check: str,
    why_blocked: str,
    how_to_fix: str,
) -> ValidationException:
    """Build a ValidationException from raw fields (shared helper)."""
    return _make_lockfile_validation_error(error_code, failed_check, why_blocked, how_to_fix)


def resolve_runtime_dir() -> Path:
    """FR-009: Resolve runtime directory for lockfiles and artifacts."""
    env_dir = os.environ.get("LLM_RUNNER_RUNTIME_DIR")
    if env_dir:
        candidate = Path(env_dir)
        try:
            candidate.mkdir(parents=True, exist_ok=True, mode=DIR_MODE_OWNER_ONLY)
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                return candidate
        except (OSError, RuntimeError):
            pass

    xdg_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_dir:
        candidate = Path(xdg_dir) / "llm-runner"
        try:
            candidate.mkdir(parents=True, exist_ok=True, mode=DIR_MODE_OWNER_ONLY)
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                return candidate
        except (OSError, RuntimeError):
            pass

    raise _make_validation_error(
        ErrorCode.RUNTIME_DIR_UNAVAILABLE,
        "runtime_dir_resolution",
        "neither LLM_RUNNER_RUNTIME_DIR env var nor XDG_RUNTIME_DIR/llm-runner directory exists and directory creation required",
        "set LLM_RUNNER_RUNTIME_DIR to writable path or create directory structure",
    )


def _get_lock_path(runtime_dir: Path, slot_id: str) -> Path:
    """Get lockfile path for a specific slot."""
    return runtime_dir / f"slot-{slot_id}.lock"


def create_lock(runtime_dir: Path, slot_id: str, pid: int, port: int) -> Path:
    """T011: Create lockfile with metadata and integrity checks."""
    lock_path = _get_lock_path(runtime_dir, slot_id)
    metadata = LockMetadata(pid=pid, port=port, started_at=time.time())

    lock_data = {
        "pid": metadata.pid,
        "port": metadata.port,
        "started_at": metadata.started_at,
        "version": "1.0",
    }

    try:
        atomic_exclusive_create_json(lock_path, lock_data)

        mode = stat.S_IMODE(os.stat(lock_path).st_mode)
        if mode != FILE_MODE_OWNER_ONLY:
            with contextlib.suppress(OSError):
                os.remove(lock_path)
            raise _lockfile_error(
                "lockfile persistence failed to enforce required owner-only permissions",
                "verify runtime path and filesystem permission support/chmod limitations before retry",
            )

        return lock_path
    except PermissionError as e:
        raise _lockfile_error(
            "lockfile creation failed due to permission denied",
            "ensure runtime directory is writable and supports chmod",
        ) from e
    except FileExistsError as e:
        raise FileExistsError(f"Lockfile already exists: {lock_path}") from e
    except OSError as e:
        raise _lockfile_error(
            f"lockfile persistence failed: {e}",
            "verify runtime path and filesystem permission support/chmod limitations before retry",
        ) from e


def read_lock(
    runtime_dir: Path, slot_id: str, require_valid: bool = False
) -> LockMetadata | ErrorDetail | None:
    """T011: Read lockfile metadata for a slot."""
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if not lock_path.exists():
        return None

    try:
        lock_data = json.loads(lock_path.read_text())
        return LockMetadata(
            pid=lock_data["pid"],
            port=lock_data["port"],
            started_at=lock_data["started_at"],
        )
    except (json.JSONDecodeError, KeyError, OSError) as e:
        if require_valid:
            return ErrorDetail(
                error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                failed_check=LOCKFILE_CHECK_NAME,
                why_blocked=f"malformed_content: {e}",
                how_to_fix="remove or repair the lockfile to proceed",
            )
        return None


def update_lock(runtime_dir: Path, slot_id: str, pid: int, port: int) -> None:
    """T011: Update existing lockfile with new metadata."""
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if not lock_path.exists():
        raise FileNotFoundError(f"Lockfile not found: {lock_path}")

    metadata = LockMetadata(pid=pid, port=port, started_at=time.time())

    lock_data = {
        "pid": metadata.pid,
        "port": metadata.port,
        "started_at": metadata.started_at,
        "version": "1.0",
    }

    try:
        atomic_write_json(lock_path, lock_data, FILE_MODE_OWNER_ONLY)
    except OSError as e:
        raise _lockfile_error(
            f"lockfile update failed: {e}",
            "verify runtime path and permission support/chmod limitations before retry",
        ) from e


def release_lock(runtime_dir: Path, slot_id: str) -> None:
    """T011: Release lockfile by deleting it."""
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if lock_path.exists():
        with contextlib.suppress(OSError):
            lock_path.unlink()


def check_lockfile_integrity(runtime_dir: Path, slot_id: str) -> ErrorDetail | None:
    """T011: Check lockfile integrity and ownership."""
    metadata_result = read_lock(runtime_dir, slot_id, require_valid=True)
    if isinstance(metadata_result, ErrorDetail):
        return metadata_result
    if metadata_result is None:
        return None
    metadata: LockMetadata = metadata_result

    if time.time() - metadata.started_at > Config().lock_stale_threshold_s:
        _clear_lockfile(runtime_dir, slot_id)
        return None

    if not psutil.pid_exists(metadata.pid):
        _clear_lockfile(runtime_dir, slot_id)
        return None

    return _verify_lock_owner(runtime_dir, slot_id, metadata)


def _clear_lockfile(runtime_dir: Path, slot_id: str) -> None:
    lock_path = _get_lock_path(runtime_dir, slot_id)
    with contextlib.suppress(OSError):
        if lock_path.exists():
            lock_path.unlink()


def _build_indeterminate_owner_error(
    why_blocked: str = INDENTERRMINATE_OWNER_MESSAGE,
) -> ErrorDetail:
    return ErrorDetail(
        error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
        failed_check=LOCKFILE_CHECK_NAME,
        why_blocked=why_blocked,
        how_to_fix=INDENTERRMINATE_OWNER_FIX,
    )


def _verify_lock_owner(
    runtime_dir: Path,
    slot_id: str,
    metadata: LockMetadata,
) -> ErrorDetail | None:
    try:
        if not psutil.pid_exists(metadata.pid):
            _clear_lockfile(runtime_dir, slot_id)
            return None

        try:
            psutil.Process(metadata.pid)
        except psutil.NoSuchProcess:
            _clear_lockfile(runtime_dir, slot_id)
            return None

        try:
            connections: list = psutil.net_connections(kind="inet")  # type: ignore[assignment]
            port_matches = any(
                conn.laddr.port == metadata.port and conn.pid == metadata.pid
                for conn in connections
                if conn.pid is not None
            )

            if not port_matches:
                return _build_indeterminate_owner_error()
        except (psutil.AccessDenied, OSError):
            return _build_indeterminate_owner_error()
    except (OSError, psutil.AccessDenied) as e:
        return _build_indeterminate_owner_error(why_blocked=f"indeterminate_owner: {e}")

    return None
