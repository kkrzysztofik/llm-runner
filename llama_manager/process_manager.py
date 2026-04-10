# Server process management


import contextlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from .colors import Color
from .config import ErrorCode, ErrorDetail, MultiValidationError, ServerConfig


@dataclass
class LockMetadata:
    """Lockfile metadata for T011 integrity checks"""

    pid: int
    port: int
    started_at: float


@dataclass
class ArtifactMetadata:
    """Artifact metadata for T012 persistence tracking"""

    artifact_type: str
    created_at: float
    slot_id: str | None = None
    additional_fields: dict | None = None

    def __post_init__(self) -> None:
        if self.additional_fields is None:
            self.additional_fields = {}


class ValidationException(Exception):
    """Exception wrapper for MultiValidationError to enable raising as exception"""

    def __init__(self, multi_error: MultiValidationError) -> None:
        self.multi_error = multi_error
        super().__init__(f"Validation failed with {len(multi_error.errors)} error(s)")


def resolve_runtime_dir() -> Path:
    """FR-009: Resolve runtime directory for lockfiles and artifacts.

    Fallback order:
    1. LLM_RUNNER_RUNTIME_DIR environment variable
    2. XDG_RUNTIME_DIR/llm-runner

    Raises:
        ValidationException: If no usable runtime directory can be determined,
                             with actionable FR-005 error details.
    """
    env_dir = os.environ.get("LLM_RUNNER_RUNTIME_DIR")
    if env_dir:
        candidate = Path(env_dir)
        try:
            candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                return candidate
        except (OSError, RuntimeError):
            pass

    xdg_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_dir:
        candidate = Path(xdg_dir) / "llm-runner"
        try:
            candidate.mkdir(parents=True, exist_ok=True, mode=0o700)
            if candidate.is_dir() and os.access(candidate, os.W_OK):
                return candidate
        except (OSError, RuntimeError):
            pass

    # Neither candidate usable - raise FR-005 actionable error
    error_detail = ErrorDetail(
        error_code=ErrorCode.RUNTIME_DIR_UNAVAILABLE,
        failed_check="runtime_dir_resolution",
        why_blocked="neither LLM_RUNNER_RUNTIME_DIR env var nor XDG_RUNTIME_DIR/llm-runner directory exists and directory creation required",
        how_to_fix="set LLM_RUNNER_RUNTIME_DIR to writable path or create directory structure",
    )
    raise ValidationException(MultiValidationError(errors=[error_detail]))


def _get_lock_path(runtime_dir: Path, slot_id: str) -> Path:
    """Get lockfile path for a specific slot."""
    return runtime_dir / f"lock-{slot_id}.json"


def create_lock(runtime_dir: Path, slot_id: str, pid: int, port: int) -> Path:
    """T011: Create lockfile with metadata and integrity checks.

    Args:
        runtime_dir: Runtime directory path
        slot_id: Slot identifier
        pid: Process ID of the server
        port: Port number the server is bound to

    Returns:
        Path to the created lockfile

    Raises:
        FileExistsError: If lockfile already exists
        ValidationException: If lockfile creation fails due to permission or persistence issues
    """
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if lock_path.exists():
        raise FileExistsError(f"Lockfile already exists: {lock_path}")

    metadata = LockMetadata(pid=pid, port=port, started_at=time.monotonic())

    # Serialize metadata
    lock_data = {
        "pid": metadata.pid,
        "port": metadata.port,
        "started_at": metadata.started_at,
        "version": "1.0",
    }

    try:
        # Write with 0600 permissions
        lock_path.write_text(json.dumps(lock_data, indent=2))
        os.chmod(lock_path, 0o600)

        # Verify permissions were set correctly
        mode = stat.S_IMODE(os.stat(lock_path).st_mode)
        if mode != 0o600:
            error_detail = ErrorDetail(
                error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                failed_check="lockfile_integrity",
                why_blocked="lockfile persistence failed to enforce required owner-only permissions",
                how_to_fix="verify runtime path writability and filesystem permission support",
            )
            raise ValidationException(MultiValidationError(errors=[error_detail]))

        return lock_path
    except PermissionError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
            failed_check="lockfile_integrity",
            why_blocked="lockfile creation failed due to permission denied",
            how_to_fix="ensure runtime directory is writable",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e
    except OSError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
            failed_check="lockfile_integrity",
            why_blocked=f"lockfile persistence failed: {e}",
            how_to_fix="verify runtime path and permission support before retry",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e


def read_lock(runtime_dir: Path, slot_id: str) -> LockMetadata | None:
    """T011: Read lockfile metadata for a slot.

    Args:
        runtime_dir: Runtime directory path
        slot_id: Slot identifier

    Returns:
        LockMetadata if lockfile exists and is valid, None otherwise
    """
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if not lock_path.exists():
        return None

    try:
        lock_data = json.loads(lock_path.read_text())
        return LockMetadata(
            pid=lock_data["pid"],
            port=lock_data["port"],
            started_at=lock_data["started_at"],
        )
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def update_lock(runtime_dir: Path, slot_id: str, pid: int, port: int) -> None:
    """T011: Update existing lockfile with new metadata.

    Args:
        runtime_dir: Runtime directory path
        slot_id: Slot identifier
        pid: Process ID of the server
        port: Port number the server is bound to

    Raises:
        FileNotFoundError: If lockfile does not exist
        ValidationException: If lockfile update fails
    """
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if not lock_path.exists():
        raise FileNotFoundError(f"Lockfile not found: {lock_path}")

    metadata = LockMetadata(pid=pid, port=port, started_at=time.monotonic())

    lock_data = {
        "pid": metadata.pid,
        "port": metadata.port,
        "started_at": metadata.started_at,
        "version": "1.0",
    }

    try:
        lock_path.write_text(json.dumps(lock_data, indent=2))
    except OSError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
            failed_check="lockfile_integrity",
            why_blocked=f"lockfile update failed: {e}",
            how_to_fix="verify runtime path and permission support",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e


def release_lock(runtime_dir: Path, slot_id: str) -> None:
    """T011: Release lockfile by deleting it.

    Args:
        runtime_dir: Runtime directory path
        slot_id: Slot identifier
    """
    lock_path = _get_lock_path(runtime_dir, slot_id)

    if lock_path.exists():
        with contextlib.suppress(OSError):
            lock_path.unlink()


def check_lockfile_integrity(runtime_dir: Path, slot_id: str) -> ErrorDetail | None:
    """T011: Check lockfile integrity and ownership.

    Handles indeterminate owner states by mapping to LOCKFILE_INTEGRITY_FAILURE
    with failed_check=lockfile_integrity.

    Args:
        runtime_dir: Runtime directory path
        slot_id: Slot identifier

    Returns:
        ErrorDetail if integrity check fails, None if valid
    """
    metadata = read_lock(runtime_dir, slot_id)
    if metadata is None:
        return None

    # Check if process exists
    if not psutil.pid_exists(metadata.pid):
        # Stale lock - auto-clearable
        try:
            lock_path = _get_lock_path(runtime_dir, slot_id)
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            pass
        return None

    # Process exists - check if it's bound to the same port
    # This is the indeterminate owner state check
    try:
        proc = psutil.Process(metadata.pid)
        try:
            connections = proc.connections()
            port_matches = any(conn.laddr.port == metadata.port for conn in connections)

            if not port_matches:
                # Indeterminate state: PID exists but port doesn't match
                return ErrorDetail(
                    error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                    failed_check="lockfile_integrity",
                    why_blocked="indeterminate_owner: lock exists but ownership verification is not definitive",
                    how_to_fix="verify owning process and clear lock only after confirmed stale ownership",
                )
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            # Access denied or process died between check - treat as indeterminate
            return ErrorDetail(
                error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                failed_check="lockfile_integrity",
                why_blocked="indeterminate_owner: lock exists but ownership verification is not definitive",
                how_to_fix="verify owning process and clear lock only after confirmed stale ownership",
            )
    except (OSError, psutil.AccessDenied) as e:
        return ErrorDetail(
            error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
            failed_check="lockfile_integrity",
            why_blocked=f"indeterminate_owner: {e}",
            how_to_fix="verify owning process and clear lock only after confirmed stale ownership",
        )

    return None


def write_artifact(runtime_dir: Path, slot_id: str, data: dict) -> Path:
    """T012: Write artifact with JSON serialization and 0700/0600 permission enforcement.

    FR-005 actionable error mapping for ARTIFACT_PERSISTENCE_FAILURE with
    failed_check=artifact_persistence.

    Args:
        runtime_dir: Runtime directory path
        slot_id: Slot identifier for artifact naming
        data: Artifact data to serialize

    Returns:
        Path to the created artifact file

    Raises:
        ValidationException: If artifact persistence fails due to permission issues
                             or filesystem doesn't support required permissions
    """
    artifact_dir = runtime_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Verify directory permissions
    dir_mode = stat.S_IMODE(os.stat(artifact_dir).st_mode)
    if dir_mode != 0o700:
        error_detail = ErrorDetail(
            error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
            failed_check="artifact_persistence",
            why_blocked="artifact persistence failed to enforce required owner-only permissions",
            how_to_fix="verify runtime path writability and filesystem permission support",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail]))

    # Create artifact filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_filename = f"artifact-{timestamp}.json"
    artifact_path = artifact_dir / artifact_filename

    try:
        # Redact sensitive environment variables in the data
        redacted_data = _redact_sensitive_in_dict(data)

        # Write with 0600 permissions
        artifact_path.write_text(json.dumps(redacted_data, indent=2))
        os.chmod(artifact_path, 0o600)

        # Verify file permissions
        file_mode = stat.S_IMODE(os.stat(artifact_path).st_mode)
        if file_mode != 0o600:
            error_detail = ErrorDetail(
                error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
                failed_check="artifact_persistence",
                why_blocked="artifact persistence failed to enforce required owner-only permissions",
                how_to_fix="verify runtime path writability and filesystem permission support",
            )
            raise ValidationException(MultiValidationError(errors=[error_detail]))

        return artifact_path
    except PermissionError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
            failed_check="artifact_persistence",
            why_blocked="artifact persistence failed due to permission denied",
            how_to_fix="verify runtime path and permission support before retry",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e
    except OSError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
            failed_check="artifact_persistence",
            why_blocked=f"artifact persistence failed: {e}",
            how_to_fix="verify runtime path and permission support before retry",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e
    except TypeError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
            failed_check="artifact_persistence",
            why_blocked=f"artifact serialization failed: {e}",
            how_to_fix="ensure artifact data is JSON-serializable",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e


def _redact_sensitive_in_dict(data: dict, env_key_prefix: str = "") -> dict:
    """Recursively redact sensitive environment variable values in a nested dict.

    Matches keys containing KEY|TOKEN|SECRET|PASSWORD|AUTH (case-insensitive).

    Args:
        data: Dictionary to redact
        env_key_prefix: Current key prefix for nested traversal

    Returns:
        Dictionary with sensitive values redacted
    """
    result = {}
    for key, value in data.items():
        full_key = f"{env_key_prefix}_{key}" if env_key_prefix else key
        if isinstance(value, dict):
            result[key] = _redact_sensitive_in_dict(value, full_key)
        elif isinstance(value, str) and re.search(
            r"(KEY|TOKEN|SECRET|PASSWORD|AUTH)", key, re.IGNORECASE
        ):
            result[key] = "[REDACTED]"
        else:
            result[key] = value
    return result


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
