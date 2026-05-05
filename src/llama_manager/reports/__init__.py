"""reports package — build failure reporting and log rotation."""

from .failure import (
    FailureReport,
    MutatingActionLogEntry,
    log_mutating_action,
    write_failure_report,
)
from .redaction import redact_sensitive
from .rotation import rotate_reports

__all__ = [
    "redact_sensitive",
    "FailureReport",
    "MutatingActionLogEntry",
    "write_failure_report",
    "log_mutating_action",
    "rotate_reports",
]
