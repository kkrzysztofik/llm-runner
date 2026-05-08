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

from __future__ import annotations

import json
import logging
import sys
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

    # Redaction filter factory
    def _redact_filter(record: LoguruRecord) -> bool:
        """Return True if the record should be kept (not suppressed)."""
        # Redact the message in-place
        original = record["message"]
        record["message"] = _redact_log_message(original)
        return True  # always keep — we only redact, never drop

    # --- Stderr sink (always present) ---
    logger.add(
        sys.stderr,
        level=log_level,
        format=fmt,
        colorize=True,
        filter=_redact_filter,
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
            filter=_redact_filter,
            serialize=json_logs,
        )

    # --- Install stdlib → Loguru bridge ---
    intercept = _InterceptHandler()
    for target_name in ("llama_manager", "llama_cli"):
        logging.getLogger(target_name).setLevel(logging.DEBUG)
        logging.getLogger(target_name).handlers = []
    root_logger = logging.getLogger()
    root_logger.addHandler(intercept)


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
