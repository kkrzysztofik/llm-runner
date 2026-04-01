# Server process management


import contextlib
import os
import signal
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

from .colors import Color

if TYPE_CHECKING:
    from .config import ServerConfig


class ServerManager:
    """Manages server processes"""

    def __init__(self):
        self.pids: list[int] = []
        self.shutting_down: bool = False
        self.servers: list[subprocess.Popen] = []

    def cleanup_servers(self) -> None:
        """Clean up all server processes"""
        if self.shutting_down:
            return
        self.shutting_down = True

        running_pids = []
        for pid in self.pids:
            try:
                os.kill(pid, 0)
                running_pids.append(pid)
            except OSError:
                pass

        if not running_pids:
            return

        print(
            f"warning: Sending TERM to {len(running_pids)} server(s)...",
            file=sys.stderr,
        )

        for pid in running_pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)

        time.sleep(1)

        stubborn_pids = []
        for pid in running_pids:
            try:
                os.kill(pid, 0)
                stubborn_pids.append(pid)
            except OSError:
                pass

        if stubborn_pids:
            print(
                f"warning: Killing {len(stubborn_pids)} stubborn server(s)...",
                file=sys.stderr,
            )
            for pid in stubborn_pids:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)

        for proc in self.servers:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

    def on_interrupt(self, signum, frame) -> None:
        """Handle SIGINT"""
        self.cleanup_servers()
        sys.exit(130)

    def on_terminate(self, signum, frame) -> None:
        """Handle SIGTERM"""
        self.cleanup_servers()
        sys.exit(143)

    def _stream_pipe(self, pipe, server_name: str, is_stderr: bool = False) -> None:
        """Stream pipe output with timestamp and color"""
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                formatted = self._format_output(server_name, line.rstrip("\n"))
                if is_stderr:
                    print(formatted, file=sys.stderr, flush=True)
                else:
                    print(formatted, flush=True)
        finally:
            pipe.close()

    def _format_output(self, server_name: str, line: str) -> str:
        """Format output line with timestamp and color"""
        timestamp = time.strftime("%H:%M:%S")
        color_code = Color.get_code(server_name)

        if color_code:
            return f"\033[1m[{timestamp}][{server_name}]\033[0m {line}"
        return f"[{timestamp}][{server_name}] {line}"

    def start_server_background(self, server_name: str, cmd: list[str]) -> subprocess.Popen:
        """Start a server in background with output redirection"""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        self.pids.append(proc.pid)
        self.servers.append(proc)

        threading.Thread(
            target=self._stream_pipe,
            args=(proc.stdout, server_name, False),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stream_pipe,
            args=(proc.stderr, server_name, True),
            daemon=True,
        ).start()

        return proc

    def run_server_foreground(self, server_name: str, cmd: list[str]) -> int:
        """Start a server in foreground and wait for it"""
        proc = self.start_server_background(server_name, cmd)
        return proc.wait()

    def wait_for_any(self) -> int:
        """Wait for any server to exit"""
        while True:
            for proc in self.servers:
                code = proc.poll()
                if code is not None:
                    return code
            time.sleep(0.2)

    def start_servers(self, configs: list["ServerConfig"]) -> list[subprocess.Popen]:
        """Start multiple servers and return their processes"""
        from .server import build_server_cmd

        processes = []
        for cfg in configs:
            cmd = build_server_cmd(cfg)
            proc = self.start_server_background(cfg.alias, cmd)
            processes.append(proc)
        return processes
