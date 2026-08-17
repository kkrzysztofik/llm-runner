"""reports package — build failure reporting and log rotation."""

from .failure import FailureReport, write_failure_report

__all__ = [
    "FailureReport",
    "write_failure_report",
]
