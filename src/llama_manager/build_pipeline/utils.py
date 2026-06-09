"""Build pipeline text-formatting and redaction utilities."""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from ..common.security import REDACTED_VALUE
from ..reports import redact_sensitive
from .models import BuildBackend

# Message constants used across pipeline stages
MSG_SOURCES_ALREADY_EXIST = "Sources already exist"
MSG_SOURCES_NOT_GIT_REPO = (
    "Source directory exists but is not a git repository; "
    "remove it or point source_dir at an empty/nonexistent path"
)

# Intel oneAPI environment setup script (default install location)
_INTEL_SETVARS_SH = Path("/opt/intel/oneapi/setvars.sh")
_MAX_OUTPUT_SUMMARY_LINES = 12
# After cancel, stop waiting for cmake/ninja/nvcc beyond this (compile stage only).
CANCEL_KILL_TIMEOUT_SECONDS = 15.0


def get_build_env_cmd(cmd: list[str], backend: BuildBackend) -> list[str]:
    """Wrap a command with the Intel oneAPI environment when building for SYCL.

    Sources ``/opt/intel/oneapi/setvars.sh`` via ``bash -c`` so that Intel
    compilers and libraries are on PATH. Uses ``--force`` because build
    subprocesses inherit ``SETVARS_COMPLETED=1`` from an already-initialized
    parent shell and setvars otherwise exits 3 without re-sourcing. Returns the
    command unchanged for non-SYCL backends or when the script is missing.
    """
    if backend != BuildBackend.SYCL:
        return cmd
    if not _INTEL_SETVARS_SH.exists():
        return cmd
    cmd_str = shlex.join(cmd)
    return [
        "bash",
        "-c",
        f'source "{_INTEL_SETVARS_SH}" --force && {cmd_str}',
    ]


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


def _get_process_group_id(proc: subprocess.Popen[str], use_process_group: bool) -> int | None:
    """Determine the process group ID for a subprocess.

    Returns the PID if the process was started with start_new_session=True
    and is the leader of its process group. Returns None if process group
    cannot be determined. Raises ProcessLookupError if the process no longer
    exists (caller should return early without terminating).
    """
    if not (use_process_group and isinstance(proc.pid, int) and proc.pid > 1):
        return None
    try:
        if os.getpgid(proc.pid) == proc.pid:
            return proc.pid
    except ProcessLookupError:
        raise  # Process gone — caller should return early without terminating
    except OSError:
        pass  # Fall through to return None
    return None


def _send_termination_signal(proc: subprocess.Popen[str], process_group_id: int | None) -> None:
    """Send SIGTERM to process or process group."""
    try:
        if process_group_id is not None:
            os.killpg(process_group_id, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass
    except OSError:
        proc.terminate()


def _send_kill_signal(proc: subprocess.Popen[str], process_group_id: int | None) -> None:
    """Send SIGKILL to process or process group."""
    try:
        if process_group_id is not None:
            os.killpg(process_group_id, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except OSError:
        proc.kill()


def terminate_process_tree(proc: subprocess.Popen[str], *, use_process_group: bool = True) -> None:
    """Stop a subprocess and its children (cmake/ninja/gcc workers).

    When ``use_process_group`` is true, the caller must have started the process
    with ``start_new_session=True`` so ``proc.pid`` is the process group id.
    """
    if proc.poll() is not None:
        return

    try:
        process_group_id = _get_process_group_id(proc, use_process_group)
    except ProcessLookupError:
        return  # Process already gone

    _send_termination_signal(proc, process_group_id)
    deadline = time.monotonic() + 2.0
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is not None:
        return
    _send_kill_signal(proc, process_group_id)


def _cancel_requested(cancel_event: threading.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _start_cancel_watcher(
    proc: subprocess.Popen[str],
    cancel_event: threading.Event | None,
) -> threading.Thread:
    def watch_cancel() -> None:
        while proc.poll() is None:
            if _cancel_requested(cancel_event):
                terminate_process_tree(proc, use_process_group=True)
                return
            time.sleep(0.1)

    cancel_watcher = threading.Thread(target=watch_cancel, name="build-cancel-watch", daemon=True)
    cancel_watcher.start()
    return cancel_watcher


def _read_stream(
    stream: TextIO | None,
    lines_container: list[str],
    line_callback: Callable[[str], None] | None,
) -> None:
    """Read lines from a stream and append to *lines_container*."""
    if stream is None:
        return
    for line in stream:
        lines_container.append(line.rstrip("\n"))
        if line_callback is not None:
            line_callback(line)


def _start_output_drainers(
    proc: subprocess.Popen[str],
    line_callback: Callable[[str], None] | None,
) -> tuple[list[str], list[str], threading.Thread, threading.Thread]:
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    stdout_t = threading.Thread(
        target=_read_stream, args=(proc.stdout, stdout_lines, line_callback)
    )
    stderr_t = threading.Thread(
        target=_read_stream, args=(proc.stderr, stderr_lines, line_callback)
    )
    stdout_t.start()
    stderr_t.start()
    return stdout_lines, stderr_lines, stdout_t, stderr_t


def _wait_for_process_exit(
    proc: subprocess.Popen[str],
    *,
    deadline: float,
    cancel_event: threading.Event | None,
    stdout_t: threading.Thread,
    stderr_t: threading.Thread,
    cancel_kill_timeout_seconds: float | None = None,
) -> int | None:
    """Wait for process exit.

    Return -1 on build timeout, -2 when cancel kill timeout elapses, else None.
    """
    cancel_kill_deadline: float | None = None
    while proc.poll() is None:
        if _cancel_requested(cancel_event):
            terminate_process_tree(proc, use_process_group=True)
            if cancel_kill_timeout_seconds is not None:
                if cancel_kill_deadline is None:
                    cancel_kill_deadline = time.monotonic() + cancel_kill_timeout_seconds
                elif time.monotonic() >= cancel_kill_deadline:
                    terminate_process_tree(proc, use_process_group=True)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        proc.wait(timeout=5)
                    stdout_t.join(timeout=5)
                    stderr_t.join(timeout=5)
                    return -2
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            terminate_process_tree(proc, use_process_group=True)
            proc.wait(timeout=5)
            stdout_t.join(timeout=5)
            stderr_t.join(timeout=5)
            return -1
        try:
            proc.wait(timeout=min(0.25, remaining))
        except subprocess.TimeoutExpired:
            continue
    return None


def run_command_with_cancel(
    cmd: list[str],
    *,
    cancel_event: threading.Event | None,
    set_active_proc: Callable[[subprocess.Popen[str] | None], None] | None = None,
    timeout_seconds: float,
    line_callback: Callable[[str], None] | None = None,
    cancel_kill_timeout_seconds: float | None = None,
) -> tuple[int, str, str]:
    """Run a command; terminate the process tree when *cancel_event* is set."""
    # Check for pre-existing cancellation before spawning
    if _cancel_requested(cancel_event):
        if set_active_proc is not None:
            set_active_proc(None)
        return (1, "", "build cancelled")

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
    ) as proc:
        if set_active_proc is not None:
            set_active_proc(proc)

        _start_cancel_watcher(proc, cancel_event)
        stdout_lines, stderr_lines, stdout_t, stderr_t = _start_output_drainers(proc, line_callback)
        deadline = time.monotonic() + timeout_seconds
        try:
            timed_out = _wait_for_process_exit(
                proc,
                deadline=deadline,
                cancel_event=cancel_event,
                stdout_t=stdout_t,
                stderr_t=stderr_t,
                cancel_kill_timeout_seconds=cancel_kill_timeout_seconds,
            )
            if timed_out is not None:
                return timed_out, "\n".join(stdout_lines), "\n".join(stderr_lines)
        finally:
            stdout_t.join(timeout=5)
            stderr_t.join(timeout=5)
            if set_active_proc is not None:
                set_active_proc(None)

    return proc.returncode or 0, "\n".join(stdout_lines), "\n".join(stderr_lines)


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
