"""Process launcher abstraction — decouples ServerManager from subprocess.

Exports:
    ProcessHandle  – protocol for a launched process (pid, stdout, stderr, poll, wait)
    ProcessLauncher – callable protocol for creating a ProcessHandle
    ProcessTimeoutError – raised by wait(timeout=...) when the process exceeds the limit
    DefaultProcessLauncher – concrete launcher backed by subprocess.Popen
"""

import contextlib
import logging
import os
import re
import signal
import subprocess
import time
import traceback
from collections.abc import Callable
from io import TextIOWrapper
from pathlib import Path
from typing import Protocol, TextIO

import psutil

from ..common.security import redact_text

_LLAMA_TS_PATTERN = re.compile(r"^\s*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\s+")

logger = logging.getLogger(__name__)


_INTEL_SETVARS_SH = Path("/opt/intel/oneapi/setvars.sh")
_SYCL_LAUNCH_SCRIPT = (
    'if ! source "$1" --force >/dev/null 2>&1; then '
    'echo "failed to source Intel oneAPI setvars for SYCL launch: $1" >&2; '
    "exit 127; "
    "fi; "
    "shift; "
    'exec "$@"'
)


class ProcessTimeoutError(Exception):
    """Raised when a process does not exit within the given timeout."""


class ProcessHandle(Protocol):
    """Minimal protocol matching the subset of ``subprocess.Popen`` that
    :class:`ServerManager` actually uses.

    Attributes:
        pid: OS process identifier.
        stdout: Readable text stream for stdout.
        stderr: Readable text stream for stderr.
    """

    pid: int
    stdout: TextIOWrapper
    stderr: TextIOWrapper

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class ProcessLauncher(Protocol):
    """Protocol for objects that can launch a process and return a handle."""

    def launch(self, cmd: list[str]) -> ProcessHandle: ...


class DefaultProcessLauncher:
    """Default concrete launcher backed by :mod:`subprocess.Popen`.

    Raises:
        ProcessTimeoutError: when *wait()* exceeds the timeout.
    """

    def launch(self, cmd: list[str]) -> ProcessHandle:
        return _SubprocessHandle(  # type: ignore[return-value]
            subprocess.Popen(  # noqa: S603
                # safe: argv is a validated list[str] (no shell injection),
                #   command source is built by build_server_cmd() from ServerConfig fields only
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        )


class _SubprocessHandle:
    """Thin wrapper around ``subprocess.Popen`` implementing :class:`ProcessHandle`."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._proc = proc

    @property
    def pid(self) -> int:
        return self._proc.pid

    @property
    def stdout(self) -> TextIOWrapper:
        return self._proc.stdout  # type: ignore[return-value]

    @property
    def stderr(self) -> TextIOWrapper:
        return self._proc.stderr  # type: ignore[return-value]

    def poll(self) -> int | None:
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        try:
            return self._proc.wait(timeout=timeout)  # type: ignore[arg-type]
        except subprocess.TimeoutExpired:
            raise ProcessTimeoutError(
                f"process {self.pid} did not exit within {timeout}s",
            ) from None


def _is_sycl_device(device: str) -> bool:
    """Return True if device string starts with SYCL (case-insensitive)."""
    return device.upper().startswith("SYCL")


def wrap_sycl_launch_cmd(
    cmd: list[str],
    device: str,
    setvars_path: Path | None = None,
) -> list[str]:
    """Wrap SYCL server launches with Intel oneAPI runtime setup.

    Returns the original *cmd* unchanged for non-SYCL devices or when
    *setvars_path* does not exist.  For SYCL devices where *setvars_path*
    exists, wraps *cmd* in a `bash -c` invocation that sources
    *setvars_path* before exec'ing the original command.
    """
    if not _is_sycl_device(device):
        return cmd
    if setvars_path is None:
        setvars_path = _INTEL_SETVARS_SH
    if not setvars_path.exists():
        return cmd
    return [
        "bash",
        "-c",
        _SYCL_LAUNCH_SCRIPT,
        "llm-runner-sycl-launch",
        str(setvars_path),
        *cmd,
    ]


# ---------------------------------------------------------------------------
# Process I/O and cleanup utilities (extracted from ServerManager)
# ---------------------------------------------------------------------------


def format_output(server_name: str, line: str) -> str:
    """Format output line with timestamp, stripping llama.cpp internals."""
    timestamp = time.strftime("%H:%M:%S")
    stripped = _LLAMA_TS_PATTERN.sub("", line)
    return f"[{timestamp}][{server_name}] {stripped}"


def stream_pipe(
    pipe: TextIO | None,
    server_name: str,
    is_stderr: bool = False,
    log_handler: Callable[[str], None] | None = None,
) -> None:
    """Stream pipe output to the main log and optionally to a UI log handler."""
    if pipe is None:
        return
    try:
        for line in iter(pipe.readline, ""):
            redacted = redact_text(line.rstrip("\n"))
            formatted = format_output(server_name, redacted)
            if log_handler is not None:
                log_handler(formatted)
            if is_stderr:
                logger.warning("%s", formatted)
            else:
                logger.info("%s", formatted)
    finally:
        pipe.close()


def wait_for_processes(
    servers: list[ProcessHandle],
    lifecycle_audit: list[dict],
) -> None:
    """Wait for all managed server processes to exit."""
    for proc in servers:
        try:
            proc.wait(timeout=5)
        except ProcessTimeoutError:
            pass
        except Exception as e:
            pid = proc.pid
            tb_str = traceback.format_exc()
            lifecycle_audit.append(
                {
                    "event": "wait_failed",
                    "pid": pid,
                    "details": f"{type(e).__name__}: {e}",
                    "traceback": tb_str,
                }
            )


def send_signals_to_pids(
    pids: list[int],
    signal_type: signal.Signals,
    label: str,
    record_event: Callable[..., None],
) -> None:
    """Send a signal to a list of PIDs with lifecycle logging."""
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal_type)
            record_event("kill", pid=pid, details=label)


def filter_owned_running_pids(
    pids: list[int],
    verify_ownership: Callable[[int], bool],
    record_event: Callable[..., None],
) -> list[int]:
    """Filter PIDs to only those that exist and are owned by us."""
    owned: list[int] = []
    for pid in pids:
        if not psutil.pid_exists(pid):
            record_event("skip", pid=pid, details="exited")
            continue
        if verify_ownership(pid):
            owned.append(pid)
        else:
            record_event("skip", pid=pid, details="ownership_failed")
    return owned
