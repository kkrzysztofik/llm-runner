"""Logging infrastructure for llama_manager — Loguru backend with stdlib bridge.

Provides ``configure_logging()`` as the single entry point for setting up
structured logging across the entire application.  All user-facing
``print()`` calls remain untouched — this module only governs diagnostic
logging.

Usage
-----
    from llama_manager.logging_setup import configure_logging

    configure_logging(level="DEBUG", log_file="/var/log/llm-runner/app.log")
"""

import contextlib
import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
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

# ---------------------------------------------------------------------------
# JSON log record envelope
# ---------------------------------------------------------------------------


@dataclass
class _JsonLogEnvelope:
    """Structure of a JSON-encoded log record."""

    timestamp: str
    level: str
    name: str
    line: int
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the envelope as a plain dict (for json.dumps)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Redaction filter
# ---------------------------------------------------------------------------


def _redact_log_message(message: str) -> str:
    """Apply security redaction to log messages."""
    return redact_log_line(message)


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
    level: str = "INFO",
    log_file: str | None = None,
    json_logs: bool = False,
) -> None:
    """Configure the logging subsystem.

    Removes the default Loguru handler and installs:
    - A coloured stderr sink (always)
    - An optional rotating file sink (when *log_file* is provided)
    - A stdlib → Loguru bridge for ``logging.getLogger(...)`` consumers

    Parameters
    ----------
    level:
        Minimum log level for all sinks.  One of ``DEBUG``, ``INFO``,
        ``WARNING``, ``ERROR``, ``CRITICAL`` (case-insensitive).
    log_file:
        If provided, a rotating file sink is added at the given path.
        Files rotate at 10 MB and are retained for 30 days (gzipped).
    json_logs:
        If ``True``, both sinks emit JSON-encoded records instead of
        human-readable text.
    """
    # Remove default handler — we replace it entirely
    logger.remove()

    # Normalise level
    log_level = level.upper()
    if log_level not in _LEVEL_MAP:
        raise ValueError(f"unknown log level '{level}' — must be one of {list(_LEVEL_MAP)}")

    # Common format templates
    text_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}\n"

    fmt = text_format if not json_logs else ""

    def _redact_only_filter(record: LoguruRecord) -> bool:
        """Redact message in-place; keep all records."""
        original = record["message"]
        record["message"] = _redact_log_message(original)
        return True

    def _stderr_sink_filter(record: LoguruRecord) -> bool:
        """Redact; drop build_pipeline to stderr while TUI suppression is active."""
        rec_name = record["name"] or ""
        if _SUPPRESS_BUILD_PIPELINE_ON_STDERR.get() and rec_name.startswith(
            _BUILD_PIPELINE_LOG_PREFIX
        ):
            return False
        original = record["message"]
        record["message"] = _redact_log_message(original)
        return True

    # --- Stderr sink (always present) ---
    logger.add(
        sys.stderr,
        level=log_level,
        format=fmt,
        colorize=True,
        filter=_stderr_sink_filter,
        serialize=json_logs,
    )

    # --- Optional file sink ---
    if log_file is not None:
        logger.add(
            log_file,
            level=log_level,
            format=fmt,
            colorize=False,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            filter=_redact_only_filter,
            serialize=json_logs,
        )

    # --- Install stdlib → Loguru bridge ---
    intercept = _InterceptHandler()
    for target_name in ("llama_manager", "llama_cli"):
        logging.getLogger(target_name).setLevel(logging.DEBUG)
        logging.getLogger(target_name).handlers = []
    root_logger = logging.getLogger()
    root_logger.addHandler(intercept)


def configure_logging_split(
    *,
    stderr_level: str = "INFO",
    file_level: str = "DEBUG",
    log_file: str | None = None,
    json_logs: bool = False,
) -> None:
    """Configure logging with separate levels for stderr and file sinks.

    Removes the default Loguru handler and installs:
    - A coloured stderr sink at *stderr_level*
    - An optional rotating file sink at *file_level* (when *log_file* is provided)
    - A stdlib → Loguru bridge for ``logging.getLogger(...)`` consumers

    Parameters
    ----------
    stderr_level:
        Minimum log level for the stderr sink.
    file_level:
        Minimum log level for the file sink.
    log_file:
        If provided, a rotating file sink is added at the given path.
    json_logs:
        If ``True``, both sinks emit JSON-encoded records.
    """
    logger.remove()

    stderr_level = stderr_level.upper()
    if stderr_level not in _LEVEL_MAP:
        raise ValueError(
            f"unknown stderr level '{stderr_level}' — must be one of {list(_LEVEL_MAP)}"
        )

    file_level = file_level.upper()
    if file_level not in _LEVEL_MAP:
        raise ValueError(f"unknown file level '{file_level}' — must be one of {list(_LEVEL_MAP)}")

    text_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}\n"

    fmt = text_format if not json_logs else ""

    def _redact_only_filter(record: LoguruRecord) -> bool:
        original = record["message"]
        record["message"] = _redact_log_message(original)
        return True

    def _stderr_sink_filter(record: LoguruRecord) -> bool:
        rec_name = record["name"] or ""
        if _SUPPRESS_BUILD_PIPELINE_ON_STDERR.get() and rec_name.startswith(
            _BUILD_PIPELINE_LOG_PREFIX
        ):
            return False
        original = record["message"]
        record["message"] = _redact_log_message(original)
        return True

    # --- Stderr sink ---
    logger.add(
        sys.stderr,
        level=stderr_level,
        format=fmt,
        colorize=True,
        filter=_stderr_sink_filter,
        serialize=json_logs,
    )

    # --- Optional file sink ---
    if log_file is not None:
        logger.add(
            log_file,
            level=file_level,
            format=fmt,
            colorize=False,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            filter=_redact_only_filter,
            serialize=json_logs,
        )

    # --- Install stdlib → Loguru bridge ---
    intercept = _InterceptHandler()
    for target_name in ("llama_manager", "llama_cli"):
        logging.getLogger(target_name).setLevel(logging.DEBUG)
        logging.getLogger(target_name).handlers = []
    root_logger = logging.getLogger()
    root_logger.addHandler(intercept)


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
    for idx, handler in list(sinks.items()):
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
    for idx, handler in list(sinks.items()):
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


# ---------------------------------------------------------------------------
# JSON formatting helper
# ---------------------------------------------------------------------------


def _format_json(record: LoguruRecord) -> str:
    """Serialize a Loguru record dict to a JSON string."""
    ts = record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    extra = {
        k: v
        for k, v in record.get("extra", {}).items()
        if k not in ("id", "name", "level", "message", "time", "line", "function", "file", "module")
    }
    envelope = _JsonLogEnvelope(
        timestamp=ts,
        level=record["level"].name,
        name=record["name"] or "unknown",
        line=record["line"],
        message=record["message"],
        extra=extra,
    )
    return json.dumps(envelope.to_dict(), default=_json_default)


def _json_default(obj: Any) -> Any:
    """Fallback serializer for non-JSON-native types in extra fields."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return str(obj)
