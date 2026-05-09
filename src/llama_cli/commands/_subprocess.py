"""Shared subprocess helpers for CLI commands."""

import subprocess


def run_capture_command(
    argv: list[str],
    *,
    timeout_seconds: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command with captured text output and no shell."""
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        shell=False,
        timeout=timeout_seconds,
        check=check,
    )


def stream_to_text(value: str | bytes | None) -> str:
    """Normalize subprocess timeout streams to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
