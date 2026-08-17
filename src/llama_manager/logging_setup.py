"""Logging infrastructure for llama_manager — Loguru backend with stdlib bridge.

Provides ``configure_logging()`` as the single entry point for setting up
structured logging across the entire application.  All user-facing
``print()`` calls remain untouched — this module only governs diagnostic
logging.

Usage
-----
    from llama_manager.logging_setup import configure_logging

    configure_logging(stderr_level="DEBUG", log_file="/var/log/llm-runner/app.log")
"""

import contextlib
import contextvars
import logging
import sys
from collections.abc import Iterator
from logging import LogRecord
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record as LoguruRecord

from llama_manager.common.security import redact_log_line

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# When True (TUI build worker), drop llama_manager.build_pipeline records from the stderr sink
# only so Loguru does not corrupt Textual's alternate screen. File sinks still receive them.
_SUPPRESS_BUILD_PIPELINE_ON_STDERR: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_SUPPRESS_BUILD_PIPELINE_ON_STDERR", default=False
)
_BUILD_PIPELINE_LOG_PREFIX = "llama_manager.build_pipeline"

# Loguru → stdlib level mapping (stdlib level name → loguru level name / int)
_LEVEL_MAP: dict[str, str | int] = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


def _validate_log_level(level: str | None) -> str | None:
    """Normalise and validate a log level name; ``None`` passes through."""
    if level is None:
        return None
    normalized = level.upper()
    if normalized not in _LEVEL_MAP:
        raise ValueError(f"unknown log level '{level}' — must be one of {list(_LEVEL_MAP)}")
    return normalized


# ---------------------------------------------------------------------------
# Redaction filter
# ---------------------------------------------------------------------------


def _redact_log_message(message: str) -> str:
    """Apply security redaction to log messages."""
    return redact_log_line(message)


def _redact_only_filter(record: LoguruRecord) -> bool:
    record["message"] = _redact_log_message(record["message"])
    return True


def _stderr_sink_filter(record: LoguruRecord) -> bool:
    rec_name = record["name"] or ""
    if _SUPPRESS_BUILD_PIPELINE_ON_STDERR.get() and rec_name.startswith(_BUILD_PIPELINE_LOG_PREFIX):
        return False
    record["message"] = _redact_log_message(record["message"])
    return True


# ---------------------------------------------------------------------------
# Stdlib → Loguru bridge
# ---------------------------------------------------------------------------


class _InterceptHandler(logging.Handler):
    """Forward stdlib ``logging`` records to Loguru.

    Enables third-party libraries (or internal stdlib ``logging`` calls) to
    flow through the same Loguru sinks without duplication.
    """

    def emit(self, record: LogRecord) -> None:
        # Determine the corresponding Loguru level
        level: str | int = _LEVEL_MAP.get(record.levelname, "MESSAGE")

        # Resolve the message — handle exceptions
        try:
            message = record.getMessage()
        except Exception:
            message = "<error formatting message>"

        # Forward to Loguru with enough depth to show the original caller
        logger.opt(depth=6, exception=record.exc_info).log(
            level,
            message,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    *,
    stderr_level: str | None = "INFO",
    file_level: str | None = None,
    log_file: str | None = None,
    json_logs: bool = False,
) -> None:
    """Configure the logging subsystem.

    *stderr_level* ``None`` disables the stderr sink. *file_level* ``None``
    follows *stderr_level* (or INFO when stderr is disabled).
    """
    logger.remove()

    stderr_norm: str | None = _validate_log_level(stderr_level)
    if log_file is not None:
        if file_level is None:
            file_norm = stderr_norm if stderr_norm is not None else "INFO"
        else:
            file_norm = _validate_log_level(file_level) or "INFO"
    else:
        file_norm = "INFO"

    text_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}"
    fmt = text_format if not json_logs else ""

    if stderr_norm is not None:
        logger.add(
            sys.stderr,
            level=stderr_norm,
            format=fmt,
            colorize=True,
            filter=_stderr_sink_filter,
            serialize=json_logs,
        )
    if log_file is not None:
        logger.add(
            log_file,
            level=file_norm,
            format=fmt,
            colorize=False,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            filter=_redact_only_filter,
            serialize=json_logs,
        )

    # --- Install stdlib → Loguru bridge (unchanged tail) ---
    root_logger = logging.getLogger()
    if not any(isinstance(h, _InterceptHandler) for h in root_logger.handlers):
        root_logger.addHandler(_InterceptHandler())
    for target_name in ("llama_manager", "llama_cli"):
        logging.getLogger(target_name).setLevel(logging.DEBUG)
        logging.getLogger(target_name).handlers = []


def update_stderr_level(level: str) -> None:
    """Update the stderr sink level at runtime.

    Parameters
    ----------
    level:
        New minimum log level for stderr (one of DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    level = level.upper()
    if level not in _LEVEL_MAP:
        raise ValueError(f"unknown log level '{level}' — must be one of {list(_LEVEL_MAP)}")

    sinks: dict[int, Any] = logger._core.handlers  # type: ignore[union-attr]
    for idx, handler in sinks.items():
        # Stderr sink is the first non-None handler (sys.stderr target)
        if handler._name is None and handler._sink._stream is sys.stderr:
            sinks[idx]._level = (level, _LEVEL_MAP[level])
            return


def update_file_level(level: str) -> None:
    """Update the file sink level at runtime.

    Parameters
    ----------
    level:
        New minimum log level for file sink.
    """
    level = level.upper()
    if level not in _LEVEL_MAP:
        raise ValueError(f"unknown log level '{level}' — must be one of {list(_LEVEL_MAP)}")

    sinks: dict[int, Any] = logger._core.handlers  # type: ignore[union-attr]
    for idx, handler in sinks.items():
        # File sink has a file path as _name
        if handler._name is not None and not handler._sink._stream.closed:
            sinks[idx]._level = (level, _LEVEL_MAP[level])
            return


@contextlib.contextmanager
def suppress_build_pipeline_stderr_for_tui() -> Iterator[None]:
    """Hide build_pipeline Loguru output on stderr during TUI-driven builds.

    File sinks (if configured) still record full diagnostics.
    """
    token = _SUPPRESS_BUILD_PIPELINE_ON_STDERR.set(True)
    try:
        yield
    finally:
        _SUPPRESS_BUILD_PIPELINE_ON_STDERR.reset(token)
