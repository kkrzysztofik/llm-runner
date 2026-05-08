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
