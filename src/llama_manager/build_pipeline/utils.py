"""Build pipeline text-formatting and redaction utilities."""

import re
import shlex
from pathlib import Path

from ..common.security import REDACTED_VALUE
from ..reports import redact_sensitive

# Message constants used across pipeline stages
MSG_SOURCES_ALREADY_EXIST = "Sources already exist"

# Intel oneAPI environment setup script (default install location)
_INTEL_SETVARS_SH = Path("/opt/intel/oneapi/setvars.sh")
_MAX_OUTPUT_SUMMARY_LINES = 12


def _format_command(command: list[str]) -> str:
    """Return a shell-readable command string without executing it."""
    return _redact_build_text(shlex.join(command))


def _redact_build_text(text: str) -> str:
    """Redact secrets from command lines and captured build output."""
    redacted = redact_sensitive(text)
    return re.sub(r"(https?://)[^\s/@:]+:[^\s/@]+@", rf"\1{REDACTED_VALUE}@", redacted)


def _format_duration(seconds: float) -> str:
    """Format a duration for human-readable build logs."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _tail_lines(text: str, max_lines: int = _MAX_OUTPUT_SUMMARY_LINES) -> str:
    """Return a concise tail excerpt from command output."""
    lines = [line for line in _redact_build_text(text).strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _summarize_command_output(stdout: str, stderr: str) -> str:
    """Build a compact stdout/stderr excerpt for user-facing error messages."""
    excerpts: list[str] = []
    stdout_excerpt = _tail_lines(stdout)
    stderr_excerpt = _tail_lines(stderr)
    if stderr_excerpt:
        excerpts.append(f"stderr tail:\n{stderr_excerpt}")
    if stdout_excerpt:
        excerpts.append(f"stdout tail:\n{stdout_excerpt}")
    if not excerpts:
        return "No output captured."
    return "\n\n".join(excerpts)


def _format_command_failure(
    *,
    stage: str,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    """Format an actionable command failure summary."""
    output_summary = _summarize_command_output(stdout, stderr)
    return (
        f"{stage} command failed with exit code {returncode}: {_format_command(command)}\n"
        f"{output_summary}"
    )
