"""Shared CLI output helpers.

Provides common formatting functions used across llama_cli commands
to ensure consistent output styling (error/success/header/json).

All consumers should import from this module rather than defining
local _print_* helpers.
"""

import json
import sys
from typing import Any

from llama_cli.colors import Colors


def print_error(message: str) -> None:
    """Print error message to stderr in red.

    Args:
        message: Error message to print.
    """
    print(Colors.red(f"error: {message}"), file=sys.stderr)


def print_success(message: str) -> None:
    """Print success message to stdout.

    Args:
        message: Success message to print.
    """
    print(message)


def print_header(message: str) -> None:
    """Print a bold blue header message.

    Args:
        message: Header message to print.
    """
    print(Colors.bold(Colors.blue(message)))


def print_json(data: dict[str, Any]) -> None:
    """Print JSON data to stdout.

    Args:
        data: Dictionary to serialize to JSON.
    """
    print(json.dumps(data, indent=2, default=str))
