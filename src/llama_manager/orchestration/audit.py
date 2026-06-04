"""Audit logging for server lifecycle events."""

import contextlib
import os
import time
from pathlib import Path

from ..common.constants import FILE_MODE_OWNER_ONLY
from ..common.security import redact_text

# Audit log rotation threshold: 5 MiB
_AUDIT_LOG_MAX_BYTES = 5 * 1024 * 1024
# Maximum number of rotated log files to retain (including current)
_AUDIT_LOG_MAX_FILES = 5


def _rotate_audit_log(log_path: Path) -> None:
    """Rotate audit log files, keeping up to ``_AUDIT_LOG_MAX_FILES``."""
    oldest = log_path.with_suffix(f".{_AUDIT_LOG_MAX_FILES - 1}")
    with contextlib.suppress(OSError):
        oldest.unlink()

    for i in range(_AUDIT_LOG_MAX_FILES - 2, 0, -1):
        src = log_path.with_suffix(f".{i}")
        dst = log_path.with_suffix(f".{i + 1}")
        with contextlib.suppress(OSError):
            if src.exists():
                src.rename(dst)

    rotated = log_path.with_suffix(".1")
    with contextlib.suppress(OSError):
        log_path.rename(rotated)

    for i in range(1, _AUDIT_LOG_MAX_FILES):
        rotated_path = log_path.with_suffix(f".{i}")
        try:
            if rotated_path.exists():
                rotated_path.chmod(FILE_MODE_OWNER_ONLY)
        except OSError:
            pass


def _append_audit_log(
    log_path: Path,
    message: str,
    redact: bool = True,
) -> None:
    """Append a line to the audit log file, rotating if needed."""
    if log_path.exists():
        try:
            size = log_path.stat().st_size
            if size > _AUDIT_LOG_MAX_BYTES:
                _rotate_audit_log(log_path)
        except OSError:
            pass

    if redact:
        message = redact_text(message)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{timestamp} {message}\n"

    with open(log_path, "a", encoding="utf-8") as fh:
        os.fchmod(fh.fileno(), FILE_MODE_OWNER_ONLY)
        fh.write(line)


class AuditLogger:
    """Records server lifecycle events to an in-memory trail and optional file."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._log_path = log_path
        self._lifecycle_audit: list[dict] = []

    def record_event(self, event: str, pid: int | None = None, details: str | None = None) -> None:
        """Record a lifecycle event in the audit trail."""
        self._lifecycle_audit.append(
            {
                "event": event,
                "pid": pid,
                "details": details,
                "timestamp": time.time(),
            }
        )
        if self._log_path is not None:
            with contextlib.suppress(OSError):
                _append_audit_log(
                    self._log_path,
                    f"lifecycle:{event} pid={pid} {details or ''}",
                )

    @property
    def lifecycle_audit(self) -> list[dict]:
        """Return the in-memory audit trail."""
        return self._lifecycle_audit
