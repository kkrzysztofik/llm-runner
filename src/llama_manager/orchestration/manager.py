"""Server lifecycle management — ServerManager, launch_orchestrate, SlotRuntime."""

import contextlib
import os
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any, Final, TextIO

import psutil

from ..common.constants import FILE_MODE_OWNER_ONLY
from ..common.security import REDACTED_VALUE, SENSITIVE_KEY_NAME_PATTERN, SENSITIVE_WORD_PATTERN
from ..config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    SlotState,
    apply_profile_overrides,
)
from ..gpu_stats import GPUStats
from ..log_buffer import LogBuffer
from .launcher import ProcessHandle, ProcessLauncher, ProcessTimeoutError

__all__ = ["ServerManager", "launch_orchestrate"]

if TYPE_CHECKING:
    from ..risk_ack import RiskAckResult

from ..orchestration.lockfile import (
    LOCKFILE_CHECK_NAME,
    ValidationException,
)

# Module-local string constants (process_manager-specific).
ARTIFACT_CHECK_NAME: Final[str] = "artifact_persistence"
OWNER_ONLY_PERMISSIONS_FAILURE: Final[str] = (
    "artifact persistence failed to enforce required owner-only permissions"
)
PERMISSION_SUPPORT_HINT: Final[str] = (
    "verify runtime path and permission support/chmod limitations before retry"
)
PERMISSION_WRITABILITY_HINT: Final[str] = (
    "verify runtime path writability and filesystem permission support/chmod limitations"
)
MAX_COLLISION_RETRIES: Final[int] = 10


@dataclass
class ProcessMetadata:
    """Process ownership metadata for T001 security hardening."""

    pid: int
    create_time: float


def _make_validation_error(
    error_code: ErrorCode,
    failed_check: str,
    why_blocked: str,
    how_to_fix: str,
) -> ValidationException:
    """Build a single-error ValidationException from raw fields."""
    detail = ErrorDetail(
        error_code=error_code,
        failed_check=failed_check,
        why_blocked=why_blocked,
        how_to_fix=how_to_fix,
    )
    return ValidationException(MultiValidationError(errors=[detail]))


def _lockfile_error(why_blocked: str, how_to_fix: str) -> ValidationException:
    """Build a lockfile-integrity ValidationException."""
    return _make_validation_error(
        ErrorCode.LOCKFILE_INTEGRITY_FAILURE, LOCKFILE_CHECK_NAME, why_blocked, how_to_fix
    )


def _artifact_error(
    why_blocked: str, how_to_fix: str, check: str = ARTIFACT_CHECK_NAME
) -> ValidationException:
    """Build an artifact-persistence ValidationException."""
    return _make_validation_error(
        ErrorCode.ARTIFACT_PERSISTENCE_FAILURE, check, why_blocked, how_to_fix
    )


@dataclass
class LaunchResult:
    """Result of slot-based launch operation (T020)."""

    status: str
    launched: list[str] | None = None
    warnings: list[str] | None = None
    errors: MultiValidationError | None = None

    @property
    def launch_count(self) -> int:
        """Return the number of successfully launched slots."""
        return len(self.launched) if self.launched else 0

    def is_blocked(self) -> bool:
        """Check if launch was completely blocked."""
        return self.status == "blocked"

    def is_degraded(self) -> bool:
        """Check if launch was partially successful (degraded)."""
        return self.status == "degraded"

    def is_success(self) -> bool:
        """Check if launch was fully successful."""
        return self.status == "success"


@dataclass
class LaunchOrchestrationResult:
    """Structured result from launch orchestration."""

    updated_configs: list[ServerConfig]
    launch_result: LaunchResult | None
    processes: dict[str, Any]
    slot_states: dict[str, str]
    status_messages: list[str]
    risk_result: RiskAckResult
    empty: bool = False


@dataclass
class SlotRuntime:
    """Runtime state for a single model slot."""

    slot_id: str
    state: SlotState
    pid: int | None
    start_time: float
    logs: LogBuffer
    gpu_stats: GPUStats | None = None

    def transition_to(self, new_state: SlotState) -> None:
        """Transition to a new state, updating start_time if needed."""
        self.state = new_state
        if new_state in (SlotState.LAUNCHING, SlotState.RUNNING):
            self.start_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime state to a dictionary."""
        return {
            "slot_id": self.slot_id,
            "state": self.state.value,
            "pid": self.pid,
            "start_time": self.start_time,
            "gpu_stats": self.gpu_stats is not None,
        }


# Audit log rotation threshold: 5 MiB
_AUDIT_LOG_MAX_BYTES: Final[int] = 5 * 1024 * 1024
# Maximum number of rotated log files to retain (including current)
_AUDIT_LOG_MAX_FILES: Final[int] = 5


def _rotate_audit_log(log_path: Path) -> None:
    """Rotate audit log files, keeping up to ``_AUDIT_LOG_MAX_FILES``."""
    oldest = log_path.with_suffix(f".{_AUDIT_LOG_MAX_FILES - 1}")
    with contextlib.suppress(OSError):
        oldest.unlink()

    for i in range(_AUDIT_LOG_MAX_FILES - 2, 0, -1):
        src = log_path.with_suffix(f".{i}")
        dst = log_path.with_suffix(f".{i + 1}")
        with contextlib.suppress(OSError):
            if src.exists():
                src.rename(dst)

    rotated = log_path.with_suffix(".1")
    with contextlib.suppress(OSError):
        log_path.rename(rotated)

    for i in range(1, _AUDIT_LOG_MAX_FILES):
        rotated_path = log_path.with_suffix(f".{i}")
        try:
            if rotated_path.exists():
                rotated_path.chmod(FILE_MODE_OWNER_ONLY)
        except OSError:
            pass


def _redact_sensitive(text: str) -> str:
    """Redact sensitive patterns from text using module-level patterns."""
    if not isinstance(text, str):
        return text
    text = SENSITIVE_WORD_PATTERN.sub(REDACTED_VALUE, text)
    text = re.sub(
        r'(?i)\b[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD|AUTH_HEADER|AUTH)\s*=\s*"[^"]*"',
        REDACTED_VALUE,
        text,
    )
    text = re.sub(
        r"(?i)\b[A-Z_]*(?:KEY|TOKEN|SECRET|PASSWORD|AUTH_HEADER|AUTH)\s*=\s*'[^']*'",
        REDACTED_VALUE,
        text,
    )
    text = re.sub(
        r"(?i)\b[A-Z_]*Authorization\s*:\s*Bearer\s+\S+",
        REDACTED_VALUE,
        text,
    )
    text = SENSITIVE_KEY_NAME_PATTERN.sub(REDACTED_VALUE, text)
    return text


def _verify_shutdown_ownership(pid: int, port: int) -> bool:
    """Verify that *pid* owns the slot by checking port binding and UID."""
    if not psutil.pid_exists(pid):
        return False

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess, psutil.AccessDenied:
        return False

    try:
        connections: list = psutil.net_connections(kind="inet")  # type: ignore[assignment]
        if not any(
            conn.laddr.port == port and conn.pid == pid
            for conn in connections
            if conn.pid is not None
        ):
            return False
    except psutil.AccessDenied, OSError:
        return False

    try:
        current_uid = os.getuid()
        proc_uid = proc.uids().real
        if proc_uid != current_uid:
            return False
    except psutil.AccessDenied, psutil.NoSuchProcess, AttributeError, TypeError, OSError:
        return False

    return True


def _append_audit_log(
    log_path: Path,
    message: str,
    redact: bool = True,
) -> None:
    """Append a line to the audit log file, rotating if needed."""
    if log_path.exists():
        try:
            size = log_path.stat().st_size
            if size > _AUDIT_LOG_MAX_BYTES:
                _rotate_audit_log(log_path)
        except OSError:
            pass

    if redact:
        message = _redact_sensitive(message)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = f"{timestamp} {message}\n"

    # Use high-level open() with explicit mode to ensure owner-only permissions
    # on new files; also enforce on existing files via fchmod.
    with open(log_path, "a", encoding="utf-8") as fh:
        os.fchmod(fh.fileno(), FILE_MODE_OWNER_ONLY)
        fh.write(line)


def launch_orchestrate(
    configs: list[ServerConfig],
    base_config: Config,
    server_manager: "ServerManager",  # noqa: UP037
    log_buffers: Mapping[str, LogBuffer],
    get_driver_version: Callable[[str], str],
    acknowledged: bool = False,
) -> LaunchOrchestrationResult:
    """Orchestrate the full launch sequence for model slots."""
    from ..risk_ack import RiskAckResult, evaluate_risks
    from ..slot_state import compute_slot_transition

    updated_configs, profile_messages = apply_profile_overrides(
        configs, base_config, get_driver_version
    )

    if not updated_configs:
        return LaunchOrchestrationResult(
            updated_configs=[],
            launch_result=LaunchResult(status="success", launched=[]),
            processes={},
            slot_states={},
            status_messages=["No slots configured. Press 'a' to add a slot."],
            risk_result=RiskAckResult(),
            empty=True,
        )

    slots = [
        ModelSlot(slot_id=cfg.alias, model_path=cfg.model, port=cfg.port) for cfg in updated_configs
    ]

    launch_attempt_id = server_manager.begin_launch_attempt()
    ack_token = server_manager.issue_ack_token(launch_attempt_id)

    risk_result = evaluate_risks(
        updated_configs,
        server_manager,
        launch_attempt_id,
        ack_token,
        acknowledged,
    )

    if risk_result.has_risks and not risk_result.risks_acknowledged and risk_result.risk_details:
        status_messages: list[str] = list(profile_messages)
        status_messages.append(
            "Launch blocked: unacknowledged risks detected. "
            f"Details: {len(risk_result.risk_details)} risk(s) require"
            " acknowledgement.",
        )
        return LaunchOrchestrationResult(
            updated_configs=updated_configs,
            launch_result=None,
            processes={},
            slot_states={},
            status_messages=status_messages,
            risk_result=risk_result,
            empty=False,
        )

    launch_result = server_manager.launch_all_slots(slots, configs=updated_configs)

    status_messages: list[str] = list(profile_messages)

    if launch_result.is_blocked():
        status_messages.append("Launch blocked: no slots could be launched")
        if launch_result.errors is not None:
            for error_detail in launch_result.errors.errors:
                status_messages.append(f"  {error_detail.error_code} - {error_detail.why_blocked}")
        return LaunchOrchestrationResult(
            updated_configs=updated_configs,
            launch_result=launch_result,
            processes={},
            slot_states={},
            status_messages=status_messages,
            risk_result=risk_result,
            empty=False,
        )

    if launch_result.is_degraded():
        status_messages.append("Launch degraded: some slots blocked")
        for warning in launch_result.warnings or []:
            status_messages.append(f"  warning: {warning}")

    launched_slots = launch_result.launched or []
    launched_set = set(launched_slots)

    launched_configs = [cfg for cfg in updated_configs if cfg.alias in launched_set]
    launched_log_buffers = {
        alias: buf for alias, buf in log_buffers.items() if alias in launched_set
    }

    log_handlers: dict[str, Callable[[str], None]] = {}
    for cfg in launched_configs:
        buf = launched_log_buffers.get(cfg.alias)
        if buf is not None:
            log_handlers[cfg.alias] = lambda line, b=buf: b.add_line(line)

    processes: dict[str, Any] = {}
    try:
        processes_list = server_manager.start_servers(launched_configs, log_handlers)
    except Exception:
        server_manager.cleanup_servers()
        raise

    for cfg, proc in zip(launched_configs, processes_list, strict=True):
        processes[cfg.alias] = proc

    slot_states: dict[str, str] = {}
    for cfg in launched_configs:
        old_state = None
        new_state = SlotState.RUNNING
        transition = compute_slot_transition(cfg.alias, old_state, new_state)
        slot_states[cfg.alias] = new_state.value
        if transition is not None:
            message, _color = transition
            status_messages.append(message)

    return LaunchOrchestrationResult(
        updated_configs=updated_configs,
        launch_result=launch_result,
        processes=processes,
        slot_states=slot_states,
        status_messages=status_messages,
        risk_result=risk_result,
        empty=False,
    )


class ServerManager:
    """Manages server processes with lifecycle audit trail."""

    def __init__(
        self,
        audit_log_path: Path | None = None,
        process_launcher: ProcessLauncher | None = None,
    ) -> None:
        self.pids: list[int] = []
        self.shutting_down: bool = False
        self.servers: list[ProcessHandle] = []
        self.pid_metadata: dict[int, float] = {}
        self._launcher = process_launcher
        self._lifecycle_audit: list[dict] = []
        self._risky_acknowledged_cache: dict[str, set[str]] = {}
        self._current_launch_attempt_id: str | None = None
        self._audit_log_path = audit_log_path

    def begin_launch_attempt(self, launch_attempt_id: str | None = None) -> str:
        """Create/select launch attempt and initialize per-attempt ack cache."""
        attempt_id = launch_attempt_id or uuid.uuid4().hex
        self._current_launch_attempt_id = attempt_id
        self._risky_acknowledged_cache.setdefault(attempt_id, set())
        return attempt_id

    def issue_ack_token(self, launch_attempt_id: str | None = None) -> str:
        """Issue deterministic ack token bound to a launch attempt."""
        attempt_id = launch_attempt_id or self.begin_launch_attempt()
        return f"ack:{attempt_id}"

    def validate_ack_token(self, launch_attempt_id: str, ack_token: str | None) -> bool:
        """Validate that ack_token is bound to launch_attempt_id."""
        if ack_token is None:
            return False
        return ack_token == f"ack:{launch_attempt_id}"

    def acknowledge_risk(
        self,
        slot_id: str,
        risk_type: str,
        launch_attempt_id: str | None = None,
        ack_token: str | None = None,
    ) -> None:
        """Mark a risky operation as acknowledged for a specific slot."""
        attempt_id = launch_attempt_id or self._current_launch_attempt_id
        if attempt_id is None:
            attempt_id = self.begin_launch_attempt()

        if ack_token is not None and not self.validate_ack_token(attempt_id, ack_token):
            raise ValueError("ack_token does not match launch_attempt_id")

        self._risky_acknowledged_cache.setdefault(attempt_id, set()).add(f"{slot_id}:{risk_type}")

    def is_risk_acknowledged(
        self,
        slot_id: str,
        risk_type: str,
        launch_attempt_id: str | None = None,
    ) -> bool:
        """Check if a risky operation has been acknowledged for a specific slot."""
        attempt_id = launch_attempt_id or self._current_launch_attempt_id
        if attempt_id is None:
            return False
        return f"{slot_id}:{risk_type}" in self._risky_acknowledged_cache.get(attempt_id, set())

    def clear_risk_acknowledgements(self) -> None:
        """Clear all in-memory risk acknowledgement state."""
        self._risky_acknowledged_cache.clear()
        self._current_launch_attempt_id = None

    def _filter_owned_running_pids(self, pids: list[int]) -> list[int]:
        """Filter PIDs to only those that exist and are owned by us."""
        owned: list[int] = []
        for pid in pids:
            if not psutil.pid_exists(pid):
                self._record_lifecycle_event("skip", pid=pid, details="exited")
                continue
            if self._verify_process_ownership(pid):
                owned.append(pid)
            else:
                self._record_lifecycle_event("skip", pid=pid, details="ownership_failed")
        return owned

    def _send_signals_to_pids(
        self,
        pids: list[int],
        signal_type: signal.Signals,
        label: str,
    ) -> None:
        """Send a signal to a list of PIDs with lifecycle logging."""
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal_type)
                self._record_lifecycle_event("kill", pid=pid, details=label)

    def _wait_for_processes(self) -> None:
        """Wait for all managed server processes to exit."""
        for proc in self.servers:
            try:
                proc.wait(timeout=5)
            except ProcessTimeoutError:
                pass
            except Exception as e:
                pid = proc.pid
                tb_str = traceback.format_exc()
                self._lifecycle_audit.append(
                    {
                        "event": "wait_failed",
                        "pid": pid,
                        "details": f"{type(e).__name__}: {e}",
                        "traceback": tb_str,
                    }
                )

    def cleanup_servers(self) -> None:
        """Clean up all server processes with ownership verification and audit trail."""
        if self.shutting_down:
            self._record_lifecycle_event("skip", details="already_shutting_down")
            return
        self.shutting_down = True
        self.clear_risk_acknowledgements()
        self._record_lifecycle_event("cleanup", details="initiated")

        running_pids = self._filter_owned_running_pids(self.pids)
        if not running_pids:
            self._record_lifecycle_event("cleanup", details="no_running_pids")
            return

        self._record_lifecycle_event("terminate", details=f"count={len(running_pids)}")
        self._send_signals_to_pids(running_pids, signal.SIGTERM, "SIGTERM")

        time.sleep(1)

        stubborn_pids = self._filter_owned_running_pids(running_pids)
        for pid in running_pids:
            if pid not in stubborn_pids:
                self._record_lifecycle_event("skip", pid=pid, details="graceful_exit")

        if stubborn_pids:
            self._record_lifecycle_event("kill", details=f"count={len(stubborn_pids)}")
            self._send_signals_to_pids(stubborn_pids, signal.SIGKILL, "SIGKILL")

        self._wait_for_processes()

    def on_interrupt(self, _signum: int, _frame: FrameType | None) -> int:
        """Handle SIGINT cleanup and return the conventional exit code."""
        self.cleanup_servers()
        return 130

    def on_terminate(self, _signum: int, _frame: FrameType | None) -> int:
        """Handle SIGTERM cleanup and return the conventional exit code."""
        self.cleanup_servers()
        return 143

    def _stream_pipe(
        self,
        pipe: TextIO | None,
        server_name: str,
        is_stderr: bool = False,
        log_handler: Callable[[str], None] | None = None,
    ) -> None:
        """Stream pipe output with timestamp and color, optionally to a log handler."""
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                redacted = _redact_sensitive(line.rstrip("\n"))
                formatted = self._format_output(server_name, redacted)
                if log_handler is not None:
                    log_handler(formatted)
                else:
                    print(formatted, file=sys.stderr if is_stderr else sys.stdout, flush=True)
        finally:
            pipe.close()

    def _format_output(self, server_name: str, line: str) -> str:
        """Format output line with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}][{server_name}] {line}"

    def _verify_process_ownership(self, pid: int) -> bool:
        """Verify process ownership using creation time metadata and UID."""
        if pid in self.pid_metadata:
            try:
                proc = psutil.Process(pid)
                current_create_time = proc.create_time()
                recorded_create_time = self.pid_metadata[pid]

                if abs(current_create_time - recorded_create_time) > 0.1:
                    return False

                try:
                    current_uid = os.getuid()
                    if hasattr(proc, "uids"):
                        proc_uid = proc.uids().real
                        if proc_uid != current_uid:
                            return False
                except psutil.AccessDenied, AttributeError, TypeError:
                    pass

                return True
            except psutil.NoSuchProcess:
                return False
            except psutil.AccessDenied:
                pass

        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _record_lifecycle_event(
        self, event: str, pid: int | None = None, details: str | None = None
    ) -> None:
        """Record a lifecycle event in the audit trail."""
        self._lifecycle_audit.append(
            {
                "event": event,
                "pid": pid,
                "details": details,
                "timestamp": time.time(),
            }
        )
        if self._audit_log_path is not None:
            with contextlib.suppress(OSError):
                _append_audit_log(
                    self._audit_log_path,
                    f"lifecycle:{event} pid={pid} {details or ''}",
                )

    def start_server_background(
        self,
        server_name: str,
        cmd: list[str],
        log_handler: Callable[[str], None] | None = None,
    ) -> ProcessHandle:
        """Start a server in background with output redirection."""
        launcher = self._launcher
        if launcher is None:
            from .launcher import DefaultProcessLauncher

            launcher = DefaultProcessLauncher()

        proc = launcher.launch(cmd)

        self.pids.append(proc.pid)
        self.servers.append(proc)
        self._record_lifecycle_event("start", pid=proc.pid, details=f"server={server_name}")

        try:
            proc_obj = psutil.Process(proc.pid)
            create_time = proc_obj.create_time()
            self.pid_metadata[proc.pid] = create_time
        except psutil.AccessDenied, psutil.NoSuchProcess:
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
        """Start a server in the foreground and wait for it to finish."""
        proc = self.start_server_background(server_name, cmd)
        return proc.wait()

    def wait_for_any(self) -> int:
        """Wait for any server to exit."""
        while True:
            for proc in self.servers:
                code = proc.poll()
                if code is not None:
                    self.clear_risk_acknowledgements()
                    return code
            time.sleep(0.2)

    def start_servers(
        self,
        configs: list[ServerConfig],
        log_handlers: dict[str, Callable[[str], None]] | None = None,
    ) -> list[ProcessHandle]:
        """Start multiple servers and return their processes."""
        from ..validation.commands import build_server_cmd

        log_handlers = log_handlers or {}
        processes = []
        for cfg in configs:
            cmd = build_server_cmd(cfg)
            handler = log_handlers.get(cfg.alias) if log_handlers else None
            proc = self.start_server_background(cfg.alias, cmd, handler)
            processes.append(proc)
        return processes

    def launch_all_slots(
        self,
        slots: list[ModelSlot],
        runtime_dir: Path | None = None,
        configs: list[ServerConfig] | None = None,
    ) -> LaunchResult:
        """Launch model slots with lock collision detection (T017-T019)."""
        from .lockfile import check_lockfile_integrity, resolve_runtime_dir

        runtime_dir = runtime_dir or resolve_runtime_dir()

        launched: list[str] = []
        warnings: list[str] = []
        errors: list[ErrorDetail] = []

        config_map: dict[str, ServerConfig] = {}
        if configs:
            config_map = {cfg.alias: cfg for cfg in configs}

        for slot in slots:
            block = check_lockfile_integrity(runtime_dir, slot.slot_id)

            if block is not None:
                error_msg = f"slot {slot.slot_id}: {block.error_code.value} - {block.why_blocked}"
                warnings.append(error_msg)
                errors.append(block)
            else:
                cfg = config_map.get(slot.slot_id)
                port = cfg.port if cfg is not None else 0
                try:
                    self.acquire_lock(slot.slot_id, port)
                    launched.append(slot.slot_id)
                except Exception as exc:
                    error_msg = f"slot {slot.slot_id}: lock_acquire_failed - {exc}"
                    warnings.append(error_msg)
                    errors.append(
                        ErrorDetail(
                            error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                            failed_check=LOCKFILE_CHECK_NAME,
                            why_blocked=error_msg,
                            how_to_fix="verify the owning process or clear the lockfile",
                        )
                    )

        if not launched:
            if errors:
                multi_error = MultiValidationError(errors=errors)
                return LaunchResult(status="blocked", launched=[], errors=multi_error)
            else:
                return LaunchResult(status="blocked", launched=[], errors=None)
        elif len(launched) < len(slots):
            return LaunchResult(status="degraded", launched=launched, warnings=warnings)
        else:
            return LaunchResult(status="success", launched=launched)

    def acquire_lock(self, slot_id: str, port: int, server_pid: int | None = None) -> Path:
        """Acquire a lockfile for a slot."""
        from .lockfile import (
            _get_lock_path,
            check_lockfile_integrity,
            create_lock,
            resolve_runtime_dir,
        )

        runtime_dir = resolve_runtime_dir()
        lock_path = _get_lock_path(runtime_dir, slot_id)

        if lock_path.exists():
            integrity = check_lockfile_integrity(runtime_dir, slot_id)
            if integrity is not None:
                raise _lockfile_error(
                    integrity.why_blocked,
                    "verify the owning process or clear the lockfile",
                )

        pid = server_pid if server_pid is not None else os.getpid()
        return create_lock(runtime_dir, slot_id, pid, port)

    def release_lock(self, slot_id: str) -> None:
        """Release lockfile for a slot."""
        from .lockfile import release_lock as _rl
        from .lockfile import resolve_runtime_dir

        runtime_dir = resolve_runtime_dir()
        _rl(runtime_dir, slot_id)

    def check_lock_stale(self, slot_id: str) -> bool:
        """Check if a lockfile is stale."""
        from .lockfile import _get_lock_path, read_lock, resolve_runtime_dir

        runtime_dir = resolve_runtime_dir()
        lock_path = _get_lock_path(runtime_dir, slot_id)

        if not lock_path.exists():
            return False

        metadata_result = read_lock(runtime_dir, slot_id, require_valid=False)
        if metadata_result is None or isinstance(metadata_result, ErrorDetail):
            return False

        metadata = metadata_result  # type: ignore[assignment]
        age = time.time() - metadata.started_at
        stale_threshold = Config().lock_stale_threshold_s
        return age > stale_threshold

    def shutdown_slot(self, slot_id: str, timeout: float = 10.0) -> bool:
        """Gracefully shut down a slot's server process."""
        from .lockfile import (
            read_lock,
            release_lock,
            resolve_runtime_dir,
        )

        runtime_dir = resolve_runtime_dir()
        metadata_result = read_lock(runtime_dir, slot_id, require_valid=False)
        if metadata_result is None or isinstance(metadata_result, ErrorDetail):
            return True

        metadata = metadata_result  # type: ignore[assignment]
        pid = metadata.pid

        if pid is None:
            return True

        if not _verify_shutdown_ownership(pid, metadata.port):
            return False

        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            release_lock(runtime_dir, slot_id)
            return True

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                release_lock(runtime_dir, slot_id)
                return True
            time.sleep(0.1)

        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            release_lock(runtime_dir, slot_id)
            return True

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not psutil.pid_exists(pid):
                release_lock(runtime_dir, slot_id)
                return True
            time.sleep(0.1)

        return False
