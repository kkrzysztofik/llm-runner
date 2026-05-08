"""User-facing output helpers for llama_cli — separate from diagnostic logging."""

from __future__ import annotations

import sys
from typing import TextIO

# ---------------------------------------------------------------------------
# Color constants (ANSI)
# ---------------------------------------------------------------------------

_COLORS: dict[str, str] = {
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def _tty() -> bool:
    """Return True if stdout is a TTY and colors should be used."""
    return sys.stdout.isatty()


def _out(err: bool = False) -> TextIO:
    """Return the output stream (stdout by default, stderr if err=True)."""
    return sys.stderr if err else sys.stdout


def _style(text: str, color: str | None) -> str:
    """Wrap text in ANSI color if TTY, otherwise return as-is."""
    if color and _tty():
        return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}"
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def emit_info(msg: str) -> None:
    """Print an informational message to stdout."""
    print(_style("info:", "cyan"), msg)


def emit_success(msg: str) -> None:
    """Print a success/status message to stdout."""
    print(_style("ok:", "green"), msg)


def emit_warn(msg: str) -> None:
    """Print a warning message to stderr."""
    print(_style("warn:", "yellow"), msg, file=sys.stderr)


def emit_error(msg: str) -> None:
    """Print an error message to stderr."""
    print(_style("error:", "red"), msg, file=sys.stderr)


def emit_plain(msg: str, *, err: bool = False) -> None:
    """Print raw text without prefix or coloring."""
    print(msg, file=_out(err))


def emit_heading(msg: str, *, level: int = 1) -> None:
    """Print a section heading (level 1 = #, 2 = ##, etc.) with dim styling."""
    prefix = "#" * level
    dimmed = _style(f"{prefix} ", "dim") if _tty() else f"{prefix} "
    print(f"{dimmed}{msg}")
