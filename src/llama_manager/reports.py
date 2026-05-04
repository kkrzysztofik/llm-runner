"""Backward compatibility shim. Import from llama_manager.reports submodules instead."""

from .reports import *  # noqa: F401, F403
from .reports.failure import (  # noqa: F401
    FailureReport,
    MutatingActionLogEntry,
    log_mutating_action,
    write_failure_report,
)
from .reports.redaction import redact_sensitive  # noqa: F401
from .reports.rotation import rotate_reports  # noqa: F401
