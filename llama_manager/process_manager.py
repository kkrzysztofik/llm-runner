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
import uuid
from collections.abc import Callable
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
class ProcessMetadata:
    """Process ownership metadata for T001 security hardening"""

    pid: int
    create_time: float


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
                how_to_fix="verify runtime path writability and filesystem permission support/chmod limitations",
            )
            raise ValidationException(MultiValidationError(errors=[error_detail]))

        return lock_path
    except PermissionError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
            failed_check="lockfile_integrity",
            why_blocked="lockfile creation failed due to permission denied",
            how_to_fix="ensure runtime directory is writable and supports chmod",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e
    except OSError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
            failed_check="lockfile_integrity",
            why_blocked=f"lockfile persistence failed: {e}",
            how_to_fix="verify runtime path and permission support/chmod limitations before retry",
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
            how_to_fix="verify runtime path writability and filesystem permission support/chmod limitations",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail]))

    # Create artifact filename with timestamp + UUID for collision resistance
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    uuid_suffix = uuid.uuid4().hex[:8]  # 8 hex characters
    artifact_filename = f"artifact-{timestamp}-{uuid_suffix}.json"
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
                how_to_fix="verify runtime path writability and filesystem permission support/chmod limitations",
            )
            raise ValidationException(MultiValidationError(errors=[error_detail]))

        return artifact_path
    except PermissionError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
            failed_check="artifact_persistence",
            why_blocked="artifact persistence failed due to permission denied",
            how_to_fix="verify runtime path and permission support/chmod limitations before retry",
        )
        raise ValidationException(MultiValidationError(errors=[error_detail])) from e
    except OSError as e:
        error_detail = ErrorDetail(
            error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
            failed_check="artifact_persistence",
            why_blocked=f"artifact persistence failed: {e}",
            how_to_fix="verify runtime path and permission support/chmod limitations before retry",
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
    """Manages server processes with lifecycle audit trail."""

    def __init__(self):
        self.pids: list[int] = []
        self.shutting_down: bool = False
        self.servers: list[subprocess.Popen] = []
        self.pid_metadata: dict[int, float] = {}
        # In-memory lifecycle audit trail for start/cleanup/kill/skip decisions
        self._lifecycle_audit: list[dict] = []

    def cleanup_servers(self) -> None:
        """Clean up all server processes with ownership verification and audit trail."""
        if self.shutting_down:
            self._record_lifecycle_event("skip", details="already_shutting_down")
            return
        self.shutting_down = True

        self._record_lifecycle_event("cleanup", details="initiated")

        # Filter PIDs by ownership verification
        running_pids = []
        for pid in self.pids:
            if self._verify_process_ownership(pid):
                running_pids.append(pid)
            else:
                self._record_lifecycle_event("skip", pid=pid, details="ownership_failed")

        if not running_pids:
            self._record_lifecycle_event("cleanup", details="no_running_pids")
            return

        print(
            f"warning: Sending TERM to {len(running_pids)} server(s)...",
            file=sys.stderr,
        )

        for pid in running_pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGTERM)
                self._record_lifecycle_event("kill", pid=pid, details="SIGTERM")

        time.sleep(1)

        # Re-verify ownership after TERM (stubborn processes that still exist)
        stubborn_pids = []
        for pid in running_pids:
            if self._verify_process_ownership(pid):
                stubborn_pids.append(pid)
            else:
                self._record_lifecycle_event("skip", pid=pid, details="graceful_exit")

        if stubborn_pids:
            print(
                f"warning: Killing {len(stubborn_pids)} stubborn server(s)...",
                file=sys.stderr,
            )
            for pid in stubborn_pids:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGKILL)
                    self._record_lifecycle_event("kill", pid=pid, details="SIGKILL")

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

    def _stream_pipe(
        self,
        pipe,
        server_name: str,
        is_stderr: bool = False,
        log_handler: Callable[[str], None] | None = None,
    ) -> None:
        """Stream pipe output with timestamp and color, optionally to a log handler.

        Args:
            pipe: Pipe reader (stdout or stderr)
            server_name: Server alias for log formatting
            is_stderr: True if this is stderr output
            log_handler: Optional callable(str) to append to instead of printing
        """
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                formatted = self._format_output(server_name, line.rstrip("\n"))
                if log_handler is not None:
                    log_handler(formatted)
                else:
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

    def _verify_process_ownership(self, pid: int) -> bool:
        """Verify process ownership using creation time metadata and UID.

        Defense-in-depth verification:
        1. If metadata is available, verify creation time matches (0.1s tolerance)
        2. When metadata path is used, also verify owner UID matches current process
        3. Falls back to existence check if metadata unavailable or UID check fails

        Args:
            pid: Process ID to verify

        Returns:
            True if ownership is verified, False otherwise
        """
        # If metadata is available, use creation time verification
        if pid in self.pid_metadata:
            try:
                proc = psutil.Process(pid)
                current_create_time = proc.create_time()
                recorded_create_time = self.pid_metadata[pid]

                # Allow 0.1 second tolerance for floating-point precision
                if abs(current_create_time - recorded_create_time) > 0.1:
                    return False

                # Defense-in-depth: verify owner UID matches current process
                try:
                    current_uid = os.getuid()
                    # Check if the process has a uid() method (may not be available in all environments)
                    if hasattr(proc, "uids"):
                        proc_uid = proc.uids().real
                        if proc_uid != current_uid:
                            return False
                except (psutil.AccessDenied, AttributeError, TypeError):
                    # UID check not available or denied - still accept if time matches
                    pass

                return True
            except psutil.NoSuchProcess:
                return False
            except psutil.AccessDenied:
                # If we can't read the process, fall back to existence check
                pass

        # Fallback: just check if process exists
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _record_lifecycle_event(
        self, event: str, pid: int | None = None, details: str | None = None
    ) -> None:
        """Record a lifecycle event in the audit trail.

        Args:
            event: Event type (start, cleanup, kill, skip, etc.)
            pid: Process ID involved in the event
            details: Optional additional details
        """
        self._lifecycle_audit.append(
            {
                "event": event,
                "pid": pid,
                "details": details,
                "timestamp": time.time(),
            }
        )

    def start_server_background(
        self,
        server_name: str,
        cmd: list[str],
        log_handler: Callable[[str], None] | None = None,
    ) -> subprocess.Popen:
        """Start a server in background with output redirection.

        Args:
            server_name: Server alias for log formatting
            cmd: Command to execute
            log_handler: Optional callable(str) to append to instead of printing

        Returns:
            subprocess.Popen process object
        """
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        self.pids.append(proc.pid)
        self.servers.append(proc)
        self._record_lifecycle_event("start", pid=proc.pid, details=f"server={server_name}")

        # Capture process creation time for ownership verification
        try:
            proc_obj = psutil.Process(proc.pid)
            create_time = proc_obj.create_time()
            self.pid_metadata[proc.pid] = create_time
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            # Metadata unavailable - will fall back to existence check later
            pass

        threading.Thread(
            target=self._stream_pipe,
            args=(proc.stdout, server_name, False, log_handler),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stream_pipe,
            args=(proc.stderr, server_name, True, log_handler),
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

    def start_servers(
        self,
        configs: list["ServerConfig"],
        log_handlers: dict[str, Callable[[str], None]] | None = None,
    ) -> list[subprocess.Popen]:
        """Start multiple servers and return their processes.

        Args:
            configs: List of ServerConfig objects
            log_handlers: Optional dict mapping server aliases to callables.
                        If provided, logs are written via these handlers instead of stdout/stderr.

        Returns:
            List of subprocess.Popen process objects
        """
        from .server import build_server_cmd

        log_handlers = log_handlers or {}
        processes = []
        for cfg in configs:
            cmd = build_server_cmd(cfg)
            handler = log_handlers.get(cfg.alias) if log_handlers else None
            proc = self.start_server_background(cfg.alias, cmd, handler)
            processes.append(proc)
        return processes
