"""Process launcher abstraction — decouples ServerManager from subprocess.

Exports:
    ProcessHandle  – protocol for a launched process (pid, stdout, stderr, poll, wait)
    ProcessLauncher – callable protocol for creating a ProcessHandle
    ProcessTimeoutError – raised by wait(timeout=...) when the process exceeds the limit
    DefaultProcessLauncher – concrete launcher backed by subprocess.Popen
"""

from __future__ import annotations

import subprocess
from io import TextIOWrapper
from typing import Protocol


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
        # noqa: S603 — safe: argv is a validated list[str] (no shell injection),
        #   command source is built by build_server_cmd() from ServerConfig fields only
        return _SubprocessHandle(  # type: ignore[return-value]
            subprocess.Popen(
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
