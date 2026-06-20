"""Server lifecycle management — ServerManager."""

import contextlib
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import TextIO

import psutil

from ..config import (
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    ValidationException,
)
from .audit import AuditLogger
from .launcher import (
    ProcessHandle,
    ProcessLauncher,
    ProcessTimeoutError,
    filter_owned_running_pids,
    send_signals_to_pids,
    stream_pipe,
    wait_for_processes,
)
from .lockfile import (
    LOCKFILE_CHECK_NAME,
    check_lockfile_integrity,
    verify_process_ownership,
)
from .risk import RiskAckManager
from .types import LOCKFILE_FIX_SUGGESTION, LaunchResult

__all__ = ["ServerManager"]


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
        self.slot_processes: dict[str, ProcessHandle] = {}
        self.pid_metadata: dict[int, float] = {}
        self._launcher = process_launcher
        self._audit = AuditLogger(audit_log_path)
        self._risk = RiskAckManager()

    # -- Delegates to AuditLogger --

    @property
    def _lifecycle_audit(self) -> list[dict]:
        return self._audit.lifecycle_audit

    def _record_lifecycle_event(
        self, event: str, pid: int | None = None, details: str | None = None
    ) -> None:
        self._audit.record_event(event, pid=pid, details=details)

    # -- Delegates to RiskAckManager --

    def begin_launch_attempt(self, launch_attempt_id: str | None = None) -> str:
        return self._risk.begin_launch_attempt(launch_attempt_id)

    def issue_ack_token(self, launch_attempt_id: str | None = None) -> str:
        return self._risk.issue_ack_token(launch_attempt_id)

    def validate_ack_token(self, launch_attempt_id: str, ack_token: str | None) -> bool:
        return self._risk.validate_ack_token(launch_attempt_id, ack_token)

    def acknowledge_risk(
        self,
        slot_id: str,
        risk_type: str,
        launch_attempt_id: str | None = None,
        ack_token: str | None = None,
    ) -> None:
        self._risk.acknowledge_risk(slot_id, risk_type, launch_attempt_id, ack_token)

    def is_risk_acknowledged(
        self,
        slot_id: str,
        risk_type: str,
        launch_attempt_id: str | None = None,
    ) -> bool:
        return self._risk.is_risk_acknowledged(slot_id, risk_type, launch_attempt_id)

    def clear_risk_acknowledgements(self) -> None:
        self._risk.clear_all()

    def cleanup_servers(self) -> None:
        """Clean up all server processes with ownership verification and audit trail."""
        if self.shutting_down:
            self._record_lifecycle_event("skip", details="already_shutting_down")
            return
        self.shutting_down = True
        self.clear_risk_acknowledgements()
        self._record_lifecycle_event("cleanup", details="initiated")

        running_pids = filter_owned_running_pids(
            self.pids, self._verify_process_ownership, self._record_lifecycle_event
        )
        if not running_pids:
            self._record_lifecycle_event("cleanup", details="no_running_pids")
            return

        self._record_lifecycle_event("terminate", details=f"count={len(running_pids)}")
        send_signals_to_pids(running_pids, signal.SIGTERM, "SIGTERM", self._record_lifecycle_event)

        time.sleep(1)

        stubborn_pids = filter_owned_running_pids(
            running_pids, self._verify_process_ownership, self._record_lifecycle_event
        )
        for pid in running_pids:
            if pid not in stubborn_pids:
                self._record_lifecycle_event("skip", pid=pid, details="graceful_exit")

        if stubborn_pids:
            self._record_lifecycle_event("kill", details=f"count={len(stubborn_pids)}")
            send_signals_to_pids(
                stubborn_pids, signal.SIGKILL, "SIGKILL", self._record_lifecycle_event
            )

        wait_for_processes(self.servers, self._lifecycle_audit)

    def on_interrupt(self, _signum: int, _frame: FrameType | None) -> int:
        """Handle SIGINT cleanup and return the conventional exit code."""
        self.cleanup_servers()
        return 130

    def on_terminate(self, _signum: int, _frame: FrameType | None) -> int:
        """Handle SIGTERM cleanup and return the conventional exit code."""
        self.cleanup_servers()
        return 143

    def _verify_process_ownership(self, pid: int) -> bool:
        """Verify process ownership using creation time metadata and UID."""
        return verify_process_ownership(pid, self.pid_metadata)

    def _stream_pipe(
        self,
        pipe: TextIO | None,
        server_name: str,
        is_stderr: bool = False,
        log_handler: Callable[[str], None] | None = None,
    ) -> None:
        """Stream pipe output — delegates to launcher.stream_pipe."""
        stream_pipe(pipe, server_name, is_stderr, log_handler)

    def _wait_for_processes(self) -> None:
        """Wait for all managed server processes to exit — delegates to launcher.wait_for_processes."""
        wait_for_processes(self.servers, self._audit._lifecycle_audit)

    def _format_output(self, server_name: str, line: str) -> str:
        """Format output line with timestamp — delegates to launcher.format_output."""
        from .launcher import format_output

        return format_output(server_name, line)

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
            target=stream_pipe,
            args=(proc.stdout, server_name, False, log_handler),
            daemon=True,
        ).start()
        threading.Thread(
            target=stream_pipe,
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
        from .launcher import wrap_sycl_launch_cmd

        log_handlers = log_handlers or {}
        processes = []
        for cfg in configs:
            try:
                self._reserve_slot_lock(cfg)
                cmd = build_server_cmd(cfg)
                cmd = wrap_sycl_launch_cmd(cmd, cfg.device)
                handler = log_handlers.get(cfg.alias) if log_handlers else None
                proc = self.start_server_background(cfg.alias, cmd, handler)
            except Exception:
                self.release_lock(cfg.alias)
                raise
            try:
                self._record_started_slot_lock(cfg, proc.pid)
            except Exception:
                self._shutdown_process_handle(cfg.alias, proc, timeout=1.0)
                raise
            self.slot_processes[cfg.alias] = proc
            processes.append(proc)
        return processes

    def _reserve_slot_lock(self, cfg: ServerConfig) -> None:
        """Reserve a slot lock before launching a server process."""
        from .lockfile import LockMetadata, read_lock, resolve_runtime_dir
        from .slot_lockfile import acquire_slot_lock

        runtime_dir = resolve_runtime_dir()
        metadata_result = read_lock(runtime_dir, cfg.alias, require_valid=False)
        if isinstance(metadata_result, ErrorDetail):
            raise ValidationException(MultiValidationError(errors=[metadata_result]))
        if (
            isinstance(metadata_result, LockMetadata)
            and metadata_result.pid == os.getpid()
            and metadata_result.port == cfg.port
        ):
            return

        acquire_slot_lock(cfg.alias, cfg.port)

    def _record_started_slot_lock(self, cfg: ServerConfig, server_pid: int) -> None:
        """Replace the launch reservation PID with the actual server PID."""
        from .lockfile import LockMetadata, read_lock, resolve_runtime_dir, update_lock
        from .slot_lockfile import acquire_slot_lock

        runtime_dir = resolve_runtime_dir()
        metadata_result = read_lock(runtime_dir, cfg.alias, require_valid=False)
        if isinstance(metadata_result, ErrorDetail):
            raise ValidationException(MultiValidationError(errors=[metadata_result]))
        if metadata_result is None:
            acquire_slot_lock(cfg.alias, cfg.port, server_pid=server_pid)
            return
        if isinstance(metadata_result, LockMetadata) and metadata_result.pid not in {
            os.getpid(),
            server_pid,
        }:
            raise RuntimeError(f"slot lock for '{cfg.alias}' changed during launch")

        update_lock(runtime_dir, cfg.alias, server_pid, cfg.port)

    def _process_slot(
        self,
        slot: ModelSlot,
        config_map: dict[str, ServerConfig],
        runtime_dir: Path,
    ) -> tuple[str | None, ErrorDetail | None]:
        """Check lockfile integrity and acquire lock for a single slot.

        Returns ``(slot_id, None)`` on success or ``(None, error)`` if blocked.
        """
        block = check_lockfile_integrity(runtime_dir, slot.slot_id)

        if block is not None:
            block_error_code = block.error_code.value  # type: ignore[union-attr]
            block_why_blocked = block.why_blocked  # type: ignore[union-attr]
            error_msg = f"slot {slot.slot_id}: {block_error_code} - {block_why_blocked}"
            return None, ErrorDetail(
                error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                failed_check=LOCKFILE_CHECK_NAME,
                why_blocked=error_msg,
                how_to_fix=LOCKFILE_FIX_SUGGESTION,
            )

        cfg = config_map.get(slot.slot_id)
        port = cfg.port if cfg is not None else 0

        try:
            from .slot_lockfile import acquire_slot_lock

            acquire_slot_lock(slot.slot_id, port)
        except Exception as exc:
            error_msg = f"slot {slot.slot_id}: lock_acquire_failed - {exc}"
            return None, ErrorDetail(
                error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                failed_check=LOCKFILE_CHECK_NAME,
                why_blocked=error_msg,
                how_to_fix=LOCKFILE_FIX_SUGGESTION,
            )

        return slot.slot_id, None

    def launch_all_slots(
        self,
        slots: list[ModelSlot],
        runtime_dir: Path | None = None,
        configs: list[ServerConfig] | None = None,
    ) -> LaunchResult:
        """Launch model slots with lock collision detection (T017-T019)."""
        from .lockfile import resolve_runtime_dir

        runtime_dir = runtime_dir or resolve_runtime_dir()

        launched: list[str] = []
        warnings: list[str] = []
        errors: list[ErrorDetail] = []

        config_map: dict[str, ServerConfig] = {}
        if configs:
            config_map = {cfg.alias: cfg for cfg in configs}

        for slot in slots:
            slot_id, error = self._process_slot(slot, config_map, runtime_dir)
            if slot_id is None:
                error_msg = f"slot {slot.slot_id}: {error.why_blocked}"  # type: ignore[union-attr]
                warnings.append(error_msg)
                errors.append(error)  # type: ignore[arg-type]
            else:
                launched.append(slot_id)

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
        """Acquire a lockfile for a slot. Delegates to slot_lockfile.acquire_slot_lock."""
        from .slot_lockfile import acquire_slot_lock

        return Path(acquire_slot_lock(slot_id, port, server_pid))

    def release_lock(self, slot_id: str) -> None:
        """Release lockfile for a slot. Delegates to slot_lockfile.release_slot_lock."""
        from .slot_lockfile import release_slot_lock

        release_slot_lock(slot_id)

    def check_lock_stale(self, slot_id: str) -> bool:
        """Check if a lockfile is stale. Delegates to slot_lockfile.check_lock_stale."""
        from .slot_lockfile import check_lock_stale as _check_stale

        return _check_stale(slot_id)

    def shutdown_slot(self, slot_id: str, timeout: float = 10.0) -> bool:
        """Gracefully shut down a slot's server process. Delegates to slot_lockfile.shutdown_slot."""
        from .slot_lockfile import shutdown_slot as _shutdown

        process = self.slot_processes.get(slot_id)
        if process is not None:
            return self._shutdown_process_handle(slot_id, process, timeout)

        return _shutdown(slot_id, timeout)

    def _shutdown_process_handle(
        self,
        slot_id: str,
        process: ProcessHandle,
        timeout: float,
    ) -> bool:
        """Shut down a tracked server process and release its slot lock."""
        pid = process.pid
        if process.poll() is not None:
            self._forget_slot_process(slot_id, process)
            self.release_lock(slot_id)
            return True

        if pid in self.pid_metadata and not self._verify_process_ownership(pid):
            self._record_lifecycle_event("skip", pid=pid, details="ownership_failed")
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            self._record_lifecycle_event("kill", pid=pid, details="SIGTERM")
        except OSError:
            self._forget_slot_process(slot_id, process)
            self.release_lock(slot_id)
            return True

        try:
            process.wait(timeout=timeout)
            self._forget_slot_process(slot_id, process)
            self.release_lock(slot_id)
            return True
        except ProcessTimeoutError:
            pass

        try:
            os.kill(pid, signal.SIGKILL)
            self._record_lifecycle_event("kill", pid=pid, details="SIGKILL")
        except OSError:
            self._forget_slot_process(slot_id, process)
            self.release_lock(slot_id)
            return True

        try:
            process.wait(timeout=5.0)
        except ProcessTimeoutError:
            return False

        self._forget_slot_process(slot_id, process)
        self.release_lock(slot_id)
        return True

    def _forget_slot_process(self, slot_id: str, process: ProcessHandle) -> None:
        """Remove a stopped process from manager tracking structures."""
        pid = process.pid
        if self.slot_processes.get(slot_id) is process:
            self.slot_processes.pop(slot_id, None)
        with contextlib.suppress(ValueError):
            self.servers.remove(process)
        with contextlib.suppress(ValueError):
            self.pids.remove(pid)
        self.pid_metadata.pop(pid, None)
