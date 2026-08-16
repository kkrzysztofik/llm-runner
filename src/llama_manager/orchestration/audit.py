"""Audit logging for server lifecycle events."""

import time


class AuditLogger:
    """Records server lifecycle events to an in-memory trail."""

    def __init__(self) -> None:
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

    @property
    def lifecycle_audit(self) -> list[dict]:
        """Return the in-memory audit trail."""
        return self._lifecycle_audit
