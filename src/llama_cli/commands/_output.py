"""Shared CLI output helpers.

Provides common formatting functions used across llama_cli commands
to ensure consistent output styling (error/success/header/json).

All consumers should import from this module rather than defining
local _print_* helpers.
"""

import json
from typing import Any

from llama_cli.ui_output import (
    emit_error,
    emit_heading,
    emit_plain,
    emit_success,
)


def print_error(message: str) -> None:
    """Print error message to stderr in red.

    Args:
        message: Error message to print.
    """
    emit_error(message)


def print_success(message: str) -> None:
    """Print success message to stdout.

    Args:
        message: Success message to print.
    """
    emit_success(message)


def print_header(message: str) -> None:
    """Print a bold blue header message.

    Args:
        message: Header message to print.
    """
    emit_heading(message)


def print_json(data: dict[str, Any]) -> None:
    """Print JSON data to stdout.

    Args:
        data: Dictionary to serialize to JSON.
    """
    emit_plain(json.dumps(data, indent=2, default=str))


def emit_json(data: dict[str, Any]) -> None:
    """Print a JSON dictionary to stdout.

    Args:
        data: Dictionary to serialize to JSON.
    """
    emit_plain(json.dumps(data, indent=2, default=str))


def emit_json_str(json_str: str) -> None:
    """Print a pre-serialized JSON string to stdout.

    Args:
        json_str: A JSON string to print directly.
    """
    emit_plain(json_str)


def emit_json_error(
    message: str,
    details: str | None = None,
    status: str | None = None,
) -> None:
    """Print a structured JSON error envelope.

    Args:
        message: Error message string.
        details: Optional additional detail string.
        status: Optional status field (e.g. "error").
    """
    payload: dict[str, str] = {"error": message}
    if details:
        payload["details"] = details
    if status:
        payload["status"] = status
    emit_plain(json.dumps(payload))


def emit_json_success(data: dict[str, Any]) -> None:
    """Print a JSON success envelope wrapping a data dict.

    Args:
        data: Dictionary to include under the "success" key.
    """
    emit_plain(json.dumps({"success": True, **data}, default=str))
