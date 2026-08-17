"""User-facing output helpers for llama_cli — separate from diagnostic logging."""

from rich.console import Console
from rich.text import Text

_STDOUT = Console(highlight=False)
_STDERR = Console(stderr=True, highlight=False)


def _emit(console: Console, prefix: str, color: str, msg: str | Text) -> None:
    console.print(Text(f"{prefix} ", color), msg, sep="", markup=False, soft_wrap=True)


def emit_info(msg: str | Text) -> None:
    """Print an informational message to stdout."""
    _emit(_STDOUT, "info:", "cyan", msg)


def emit_success(msg: str | Text) -> None:
    """Print a success/status message to stdout."""
    _emit(_STDOUT, "ok:", "green", msg)


def emit_warn(msg: str | Text) -> None:
    """Print a warning message to stderr."""
    _emit(_STDERR, "warn:", "yellow", msg)


def emit_error(msg: str | Text) -> None:
    """Print an error message to stderr."""
    _emit(_STDERR, "error:", "red", msg)


def emit_plain(msg: str | Text, *, err: bool = False) -> None:
    """Print raw text without prefix or coloring."""
    (_STDERR if err else _STDOUT).print(msg, markup=False, soft_wrap=True)


def emit_heading(msg: str, *, level: int = 1) -> None:
    """Print a section heading (level 1 = #, 2 = ##, etc.) dimmed."""
    _STDOUT.print(Text(f"{'#' * level} ", style="dim"), msg, sep="", markup=False, soft_wrap=True)
