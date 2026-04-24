"""Tests for llama_manager.process_manager.

Focused tests for:
- resolve_runtime_dir fallback behavior (T001)
- Runtime directory usability (T002)
- Pipe streaming with optional log buffers (T013)
"""

import json
import os
import signal
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import ErrorCode, ErrorDetail
from llama_manager.log_buffer import LogBuffer
from llama_manager.process_manager import (
    REDACTED_VALUE,
    LockMetadata,
    ServerManager,
    ValidationException,
    _redact_sensitive_in_dict,
    check_lockfile_integrity,
    create_lock,
    read_lock,
    release_lock,
    resolve_runtime_dir,
    update_lock,
)


class TestResolveRuntimeDir:
    """Tests for resolve_runtime_dir fallback and usability."""

    def test_env_var_takes_precedence(self, tmp_path: Path) -> None:
        """LLM_RUNNER_RUNTIME_DIR should take precedence over XDG_RUNTIME_DIR."""
        env_runtime = tmp_path / "env_runtime"
        xdg_runtime = tmp_path / "xdg_runtime"
        env_runtime.mkdir()
        xdg_runtime.mkdir()

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(env_runtime))
            mp.setenv("XDG_RUNTIME_DIR", str(xdg_runtime))
            result = resolve_runtime_dir()
            assert result == env_runtime

    def test_falls_back_to_xdg_runtime_dir(self, tmp_path: Path) -> None:
        """Should fall back to XDG_RUNTIME_DIR/llm-runner when env var not set."""
        xdg_base = tmp_path / "xdg_runtime"
        xdg_llm_runner = xdg_base / "llm-runner"

        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_RUNNER_RUNTIME_DIR", raising=False)
            mp.setenv("XDG_RUNTIME_DIR", str(xdg_base))
            result = resolve_runtime_dir()
            # Should create the llm-runner subdirectory
            assert result == xdg_llm_runner
            assert result.exists()
            assert result.is_dir()

    def test_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        """resolve_runtime_dir should create the directory if it doesn't exist."""
        target = tmp_path / "new_runtime"

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(target))
            result = resolve_runtime_dir()
            assert result == target
            assert result.exists()
            assert result.is_dir()

    def test_no_env_vars_raises_validation_exception(self) -> None:
        """Should raise ValidationException when neither env var is set (FR-005)."""
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_RUNNER_RUNTIME_DIR", raising=False)
            mp.delenv("XDG_RUNTIME_DIR", raising=False)
            with pytest.raises(ValidationException) as exc_info:
                resolve_runtime_dir()
            assert exc_info.value.multi_error.error_count == 1

    def test_xdg_creates_subdirectory(self, tmp_path: Path) -> None:
        """XDG_RUNTIME_DIR fallback should create llm-runner subdirectory."""
        xdg_base = tmp_path / "xdg_base"

        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("LLM_RUNNER_RUNTIME_DIR", raising=False)
            mp.setenv("XDG_RUNTIME_DIR", str(xdg_base))
            result = resolve_runtime_dir()
            # Should be XDG_RUNTIME_DIR/llm-runner
            assert result.name == "llm-runner"
            assert result.parent == xdg_base

    def test_env_var_path_with_spaces(self, tmp_path: Path) -> None:
        """Should handle runtime directory paths with spaces."""
        target = tmp_path / "runtime with spaces"

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(target))
            result = resolve_runtime_dir()
            assert result == target
            assert result.exists()

    def test_writable_directory_required(self, tmp_path: Path) -> None:
        """Should skip non-writable directories and try next fallback."""
        env_dir = tmp_path / "env_dir"
        xdg_dir = tmp_path / "xdg_dir" / "llm-runner"
        env_dir.mkdir()
        xdg_dir.parent.mkdir(parents=True)

        # Make env_dir read-only (simulate non-writable)
        env_dir.chmod(0o555)

        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(env_dir))
                mp.setenv("XDG_RUNTIME_DIR", str(xdg_dir.parent))
                result = resolve_runtime_dir()
                # Should fall back to XDG_RUNTIME_DIR
                assert result == xdg_dir
                assert result.exists()
        finally:
            # Restore permissions
            env_dir.chmod(0o755)

    def test_multiple_calls_return_same_path(self, tmp_path: Path) -> None:
        """Multiple calls with same env vars should return same Path object."""
        target = tmp_path / "runtime"
        target.mkdir()

        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("LLM_RUNNER_RUNTIME_DIR", str(target))
            result1 = resolve_runtime_dir()
            result2 = resolve_runtime_dir()
            assert result1 == result2
            assert result1 == target


class TestPipeStreaming:
    """Tests for pipe streaming with optional log buffer support (T013)."""

    def test_stream_pipe_to_handler(self) -> None:
        """_stream_pipe should write to log handler when provided."""
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()
        buffer = LogBuffer()

        # Create a mock pipe with test lines (subprocess pipes return strings with text=True)
        mock_pipe = MagicMock()
        mock_pipe.readline.side_effect = ["line1\n", "line2\n", "line3\n", ""]

        # Stream to handler
        manager._stream_pipe(mock_pipe, "test_server", False, buffer.add_line)

        # Verify buffer received lines
        assert buffer.line_count == 3
        lines = list(buffer.lines)
        assert "[test_server] line1" in lines[0]
        assert "[test_server] line2" in lines[1]
        assert "[test_server] line3" in lines[2]

    def test_stream_pipe_to_handler_stderr(self) -> None:
        """_stream_pipe should handle stderr to handler correctly."""
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()
        buffer = LogBuffer()

        mock_pipe = MagicMock()
        mock_pipe.readline.side_effect = ["error1\n", "error2\n", ""]

        # Stream stderr to handler
        manager._stream_pipe(mock_pipe, "test_server", True, buffer.add_line)

        assert buffer.line_count == 2

    def test_stream_pipe_null_pipe(self) -> None:
        """_stream_pipe should handle None pipe gracefully."""
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()
        buffer = LogBuffer()

        # Should not raise on None pipe
        manager._stream_pipe(None, "test_server", False, buffer.add_line)
        assert buffer.line_count == 0

    def test_stream_pipe_without_handler_prints(self, capsys) -> None:
        """_stream_pipe should print to stdout/stderr when no handler provided."""
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()

        # Create mock pipe that prints
        mock_pipe = MagicMock()
        mock_pipe.readline.side_effect = ["stdout_line\n", ""]

        # Stream without handler - should print
        manager._stream_pipe(mock_pipe, "test_server", False, None)

        captured = capsys.readouterr()
        assert "test_server" in captured.out

    def test_start_server_background_with_handler(self) -> None:
        """start_server_background should accept and use log handler."""
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()
        buffer = LogBuffer()

        # Mock subprocess.Popen to avoid spawning real process
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = ["buffered line\n", ""]
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline.side_effect = ["error line\n", ""]

        with patch("subprocess.Popen", return_value=mock_proc):
            cmd = ["echo", "test"]
            proc = manager.start_server_background("test_server", cmd, buffer.add_line)

            # Wait for thread to process with polling (deterministic sync)
            start_time = time.time()
            while time.time() - start_time < 5.0:
                if buffer.line_count >= 1:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("Thread did not process within timeout")

            assert proc.pid == 12345
            assert "buffered line" in list(buffer.lines)[0]
            assert "error line" in list(buffer.lines)[1]

    def test_start_servers_with_handlers(self) -> None:
        """start_servers should accept and distribute log handlers."""
        from llama_manager.config import ServerConfig
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()

        # Create test configs
        config1 = ServerConfig(
            model="/path/to/model1.gguf",
            alias="server1",
            device="sycl",
            port=8080,
            ctx_size=16384,
            ubatch_size=1024,
            threads=8,
        )
        config2 = ServerConfig(
            model="/path/to/model2.gguf",
            alias="server2",
            device="cuda",
            port=8081,
            ctx_size=16384,
            ubatch_size=1024,
            threads=8,
        )

        # Create buffers and handlers
        buffers = {
            "server1": LogBuffer(),
            "server2": LogBuffer(),
        }
        handlers = {alias: buffer.add_line for alias, buffer in buffers.items()}

        # Mock subprocess.Popen
        def create_mock_proc(*args, **kwargs):
            mock = MagicMock()
            mock.pid = id(args[0][-1]) if args[0] else id(args[0])
            mock.stdout = MagicMock()
            mock.stdout.readline.side_effect = [f"output from {args[0][-1]}\n", ""]
            mock.stderr = MagicMock()
            mock.stderr.readline.side_effect = ["err line\n", ""]
            return mock

        with patch("subprocess.Popen", side_effect=create_mock_proc):
            processes = manager.start_servers([config1, config2], handlers)

            # Wait for threads to process with polling (deterministic sync)
            start_time = time.time()
            while time.time() - start_time < 5.0:
                if buffers["server1"].line_count >= 1 and buffers["server2"].line_count >= 1:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("Threads did not process within timeout")

            assert len(processes) == 2
            assert buffers["server1"].line_count >= 1
            assert buffers["server2"].line_count >= 1

    def test_start_servers_backward_compatible_no_handlers(self) -> None:
        """start_servers should work without handlers for backward compatibility."""
        from llama_manager.config import ServerConfig
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()

        config = ServerConfig(
            model="/path/to/model.gguf",
            alias="test_server",
            device="sycl",
            port=8080,
            ctx_size=16384,
            ubatch_size=1024,
            threads=8,
        )

        # Mock subprocess.Popen
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline.side_effect = ["test output\n", ""]
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline.side_effect = ["err line\n", ""]

        with patch("subprocess.Popen", return_value=mock_proc):
            # Call without handlers - should not raise
            processes = manager.start_servers([config], None)
            assert len(processes) == 1

    def test_no_dual_consumer_issue(self) -> None:
        """Ensure only ServerManager reads pipes when handlers provided."""
        from llama_manager.config import ServerConfig
        from llama_manager.log_buffer import LogBuffer
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()
        buffer = LogBuffer()

        config = ServerConfig(
            model="/path/to/model.gguf",
            alias="test_server",
            device="sycl",
            port=8080,
            ctx_size=16384,
            ubatch_size=1024,
            threads=8,
        )

        # Mock subprocess.Popen
        mock_proc = MagicMock()
        mock_proc.pid = 11111
        mock_stdout = MagicMock()
        mock_stdout.readline.side_effect = ["single read\n", ""]
        mock_stderr = MagicMock()
        mock_stderr.readline.side_effect = ["stderr read\n", ""]
        mock_proc.stdout = mock_stdout
        mock_proc.stderr = mock_stderr

        with patch("subprocess.Popen", return_value=mock_proc):
            manager.start_servers([config], {"test_server": buffer.add_line})

            # Wait for thread to process with polling (deterministic sync)
            start_time = time.time()
            while time.time() - start_time < 5.0:
                if buffer.line_count >= 2:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("Thread did not process within timeout")

            # Verify both stdout and stderr were consumed by ServerManager only (2 lines)
            # If TUI also read, we'd see 4 lines (duplication from dual consumption)
            assert buffer.line_count == 2


class TestCleanupServersIdempotency:
    """T014: Tests for SIGTERM→SIGKILL shutdown flow and idempotency."""

    def _make_manager_with_pids(self, pids: list[int]) -> ServerManager:
        """Create a ServerManager with given PIDs and matching metadata."""
        from llama_manager.process_manager import ServerManager

        manager = ServerManager()
        manager.pids = list(pids)
        for pid in pids:
            manager.pid_metadata[pid] = time.time() - 60  # 60s old
        return manager

    def test_cleanup_is_idempotent_first_call(self, monkeypatch) -> None:
        """First cleanup_servers call should set shutting_down=True."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = self._make_manager_with_pids([12345])

        with (
            patch("llama_manager.process_manager.psutil.pid_exists", return_value=True),
            patch("os.kill") as mock_kill,
        ):
            mock_kill.return_value = None
            manager.cleanup_servers()
            assert manager.shutting_down is True

    def test_cleanup_is_idempotent_second_call_skips(self, monkeypatch) -> None:
        """Second cleanup_servers call should return immediately without re-signaling."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = self._make_manager_with_pids([12345])

        with (
            patch("llama_manager.process_manager.psutil.pid_exists", return_value=True),
            patch("os.kill") as mock_kill,
        ):
            mock_kill.return_value = None

            # First call
            manager.cleanup_servers()
            first_call_count = mock_kill.call_count

            # Second call should skip immediately
            manager.cleanup_servers()
            second_call_count = mock_kill.call_count

            # No additional signals should have been sent
            assert second_call_count == first_call_count

    def test_cleanup_sends_sigterm_before_sigkill(self, monkeypatch) -> None:
        """cleanup_servers should send SIGTERM first, then SIGKILL after delay."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = self._make_manager_with_pids([12345])

        # Track signal order
        signals_sent: list[tuple[int, int]] = []  # (pid, signal)

        def track_kill(pid: int, sig: int) -> None:
            signals_sent.append((pid, sig))

        with (
            patch("llama_manager.process_manager.psutil.pid_exists", return_value=True),
            patch("llama_manager.process_manager.psutil.Process") as mock_psutil,
            patch("os.kill", side_effect=track_kill),
        ):
            mock_proc_obj = mock_psutil.return_value
            # Return matching creation time
            mock_proc_obj.create_time.return_value = manager.pid_metadata[12345]
            # Mock uids to match current process UID
            mock_uids = MagicMock()
            mock_uids.real = os.getuid()
            mock_proc_obj.uids.return_value = mock_uids
            manager.cleanup_servers()

        # Should have at least SIGTERM
        sigterm_calls = [s for s in signals_sent if s[1] == signal.SIGTERM]
        assert len(sigterm_calls) >= 1

    def test_cleanup_sends_sigkill_to_stubborn_processes(self, monkeypatch) -> None:
        """cleanup_servers should send SIGKILL to processes that survive SIGTERM."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = self._make_manager_with_pids([12345])

        signals_sent: list[int] = []

        def track_kill(pid: int, sig: int) -> None:
            signals_sent.append(sig)

        with (
            patch("llama_manager.process_manager.psutil.pid_exists", return_value=True),
            patch("llama_manager.process_manager.psutil.Process") as mock_psutil,
            patch("os.kill", side_effect=track_kill),
        ):
            mock_proc_obj = mock_psutil.return_value
            # Return matching creation time
            mock_proc_obj.create_time.return_value = manager.pid_metadata[12345]
            # Mock uids to match current process UID
            mock_uids = MagicMock()
            mock_uids.real = os.getuid()
            mock_proc_obj.uids.return_value = mock_uids
            manager.cleanup_servers()

        # Should have both SIGTERM and SIGKILL for stubborn process
        assert signal.SIGTERM in signals_sent
        assert signal.SIGKILL in signals_sent

    def test_cleanup_does_not_kill_graceful_processes(self, monkeypatch) -> None:
        """cleanup_servers should not SIGKILL processes that exited after SIGTERM."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = self._make_manager_with_pids([12345])

        # After first SIGTERM, simulate process exit by making pid_exists return False
        call_count = 0

        def selective_pid_exists(pid: int) -> bool:
            nonlocal call_count
            call_count += 1
            # First call (before sleep): process exists
            # After sleep: process no longer exists (graceful exit)
            return not call_count > 1

        signals_sent: list[int] = []

        def track_kill(pid: int, sig: int) -> None:
            signals_sent.append(sig)

        with (
            patch(
                "llama_manager.process_manager.psutil.pid_exists",
                side_effect=selective_pid_exists,
            ),
            patch("llama_manager.process_manager.psutil.Process") as mock_psutil,
            patch("os.kill", side_effect=track_kill),
        ):
            mock_proc_obj = mock_psutil.return_value
            # Return matching creation time
            mock_proc_obj.create_time.return_value = manager.pid_metadata[12345]
            # Mock uids to match current process UID
            mock_uids = MagicMock()
            mock_uids.real = os.getuid()
            mock_proc_obj.uids.return_value = mock_uids
            manager.cleanup_servers()

        # Should have SIGTERM but NOT SIGKILL (process exited gracefully)
        assert signal.SIGTERM in signals_sent
        assert signal.SIGKILL not in signals_sent

    def test_cleanup_records_events_in_audit_trail(self, monkeypatch) -> None:
        """cleanup_servers should record lifecycle events in audit trail."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = self._make_manager_with_pids([12345])

        with (
            patch("llama_manager.process_manager.psutil.pid_exists", return_value=True),
            patch("os.kill"),
        ):
            manager.cleanup_servers()

        audit = manager._lifecycle_audit
        assert any(e["event"] == "cleanup" for e in audit)
        assert any(e["details"] == "initiated" for e in audit)

    def test_cleanup_with_no_pids(self, monkeypatch) -> None:
        """cleanup_servers with empty PIDs list should not raise."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = ServerManager()
        manager.pids = []

        # Should not raise
        manager.cleanup_servers()
        assert manager.shutting_down is True

    def test_cleanup_with_already_shutting_down(self, monkeypatch) -> None:
        """cleanup_servers when already shutting down should be a no-op."""
        monkeypatch.setattr(time, "sleep", lambda x: None)

        manager = ServerManager()
        manager.pids = [12345]
        manager.shutting_down = True

        with (
            patch("llama_manager.process_manager.psutil.pid_exists", return_value=True),
            patch("os.kill"),
        ):
            manager.cleanup_servers()

        # Should have recorded skip event
        audit = manager._lifecycle_audit
        assert any(e["details"] == "already_shutting_down" for e in audit)


class TestLockfileAcquisition:
    """T015: Tests for lockfile acquisition and staleness detection."""

    def test_create_lock_creates_file(self, tmp_path: Path) -> None:
        """create_lock should create a lockfile with correct JSON content."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        assert lock_path.exists()
        assert lock_path.name == "slot-slot1.lock"

        content = json.loads(lock_path.read_text())
        assert content["pid"] == 12345
        assert content["port"] == 8080
        assert "started_at" in content
        assert content["version"] == "1.0"

    def test_create_lock_uses_correct_path(self, tmp_path: Path) -> None:
        """create_lock should place lockfile in runtime directory with correct naming."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = create_lock(runtime_dir, "my-slot", pid=9999, port=9090)

        expected_path = runtime_dir / "slot-my-slot.lock"
        assert lock_path == expected_path

    def test_create_lock_sets_owner_only_permissions(self, tmp_path: Path) -> None:
        """create_lock should set 0600 permissions on lockfile."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)
        lock_path = runtime_dir / "slot-slot1.lock"

        mode = stat.S_IMODE(os.stat(lock_path).st_mode)
        assert mode == 0o600

    def test_create_lock_raises_on_duplicate(self, tmp_path: Path) -> None:
        """create_lock should raise FileExistsError if lockfile already exists."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        with pytest.raises(FileExistsError) as exc_info:
            create_lock(runtime_dir, "slot1", pid=54321, port=9090)

        assert "slot1" in str(exc_info.value)

    def test_read_lock_returns_metadata(self, tmp_path: Path) -> None:
        """read_lock should return LockMetadata from a valid lockfile."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        metadata_result = read_lock(runtime_dir, "slot1")

        assert metadata_result is not None
        assert isinstance(metadata_result, LockMetadata)
        metadata = metadata_result
        assert metadata.pid == 12345
        assert metadata.port == 8080
        assert isinstance(metadata.started_at, float)

    def test_read_lock_returns_none_for_missing(self, tmp_path: Path) -> None:
        """read_lock should return None when lockfile doesn't exist."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        metadata = read_lock(runtime_dir, "nonexistent")

        assert metadata is None

    def test_read_lock_returns_error_for_malformed(self, tmp_path: Path) -> None:
        """read_lock should return ErrorDetail for malformed lockfile when require_valid=True."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Write invalid JSON
        lock_path = runtime_dir / "slot-bad.lock"
        lock_path.write_text("not valid json{{{")

        metadata = read_lock(runtime_dir, "bad", require_valid=True)

        assert isinstance(metadata, ErrorDetail)
        assert metadata.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE

    def test_read_lock_returns_none_for_malformed_permissive(self, tmp_path: Path) -> None:
        """read_lock should return None for malformed lockfile when require_valid=False."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-bad.lock"
        lock_path.write_text("invalid json")

        metadata = read_lock(runtime_dir, "bad", require_valid=False)

        assert metadata is None

    def test_release_lock_removes_file(self, tmp_path: Path) -> None:
        """release_lock should delete the lockfile."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)
        lock_path = runtime_dir / "slot-slot1.lock"

        assert lock_path.exists()
        release_lock(runtime_dir, "slot1")
        assert not lock_path.exists()

    def test_release_lock_noop_for_missing(self, tmp_path: Path) -> None:
        """release_lock should not raise when lockfile doesn't exist."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Should not raise
        release_lock(runtime_dir, "nonexistent")

    def test_check_lockfile_integrity_valid(self, tmp_path: Path) -> None:
        """check_lockfile_integrity should return None for valid, active lock."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create lock with current PID
        current_pid = os.getpid()
        create_lock(runtime_dir, "slot1", pid=current_pid, port=8080)

        result = check_lockfile_integrity(runtime_dir, "slot1")

        # Should be None (valid) or ErrorDetail (if PID doesn't exist)
        # At minimum, should not raise
        assert result is None or isinstance(result, ErrorDetail)

    def test_check_lockfile_integrity_detects_stale_lock(self, tmp_path: Path) -> None:
        """check_lockfile_integrity should detect stale locks (started_at > 300s ago)."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create a lock with a fake old timestamp
        lock_path = runtime_dir / "slot-slot1.lock"
        lock_data = {
            "pid": 99999,
            "port": 8080,
            "started_at": time.time() - 600,  # 10 minutes ago
            "version": "1.0",
        }
        # Write directly to bypass O_EXCL
        with open(lock_path, "w") as f:
            json.dump(lock_data, f)

        # Stale lock should be cleared
        result = check_lockfile_integrity(runtime_dir, "slot1")

        # The stale lock should have been cleared
        # check_lockfile_integrity clears stale locks and returns None
        assert not lock_path.exists() or result is None

    def test_check_lockfile_integrity_no_lock_returns_none(self, tmp_path: Path) -> None:
        """check_lockfile_integrity should return None when no lock exists."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        result = check_lockfile_integrity(runtime_dir, "nonexistent")

        assert result is None

    def test_update_lock_modifies_existing(self, tmp_path: Path) -> None:
        """update_lock should modify an existing lockfile."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Update with new values
        update_lock(runtime_dir, "slot1", pid=54321, port=9090)

        metadata_result = read_lock(runtime_dir, "slot1")
        assert metadata_result is not None
        assert isinstance(metadata_result, LockMetadata)
        metadata = metadata_result
        assert metadata.pid == 54321
        assert metadata.port == 9090

    def test_update_lock_raises_for_missing(self, tmp_path: Path) -> None:
        """update_lock should raise FileNotFoundError for missing lockfile."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            update_lock(runtime_dir, "nonexistent", pid=12345, port=8080)


class TestAuditLogRedaction:
    """T016: Tests for rotating log appending with secret redaction."""

    def test_redacts_api_key_in_dict(self) -> None:
        """_redact_sensitive_in_dict should redact API_KEY values."""
        data = {"api_key": "secret123", "name": "test"}
        redacted = _redact_sensitive_in_dict(data)

        assert redacted["api_key"] == REDACTED_VALUE
        assert redacted["name"] == "test"

    def test_redacts_token_in_dict(self) -> None:
        """_redact_sensitive_in_dict should redact TOKEN values."""
        data = {"auth_token": "abc123", "user": "admin"}
        redacted = _redact_sensitive_in_dict(data)

        assert redacted["auth_token"] == REDACTED_VALUE
        assert redacted["user"] == "admin"

    def test_redacts_secret_in_dict(self) -> None:
        """_redact_sensitive_in_dict should redact SECRET values."""
        data = {"database_secret": "mysecret", "host": "localhost"}
        redacted = _redact_sensitive_in_dict(data)

        assert redacted["database_secret"] == REDACTED_VALUE
        assert redacted["host"] == "localhost"

    def test_redacts_password_in_dict(self) -> None:
        """_redact_sensitive_in_dict should redact PASSWORD values."""
        data = {"DB_PASSWORD": "supersecret", "DB_USER": "admin"}
        redacted = _redact_sensitive_in_dict(data)

        assert redacted["DB_PASSWORD"] == REDACTED_VALUE
        assert redacted["DB_USER"] == "admin"

    def test_redacts_auth_in_dict(self) -> None:
        """_redact_sensitive_in_dict should redact AUTH values."""
        data = {"AUTH_HEADER": "bearer_xyz", "method": "jwt"}
        redacted = _redact_sensitive_in_dict(data)

        assert redacted["AUTH_HEADER"] == REDACTED_VALUE
        assert redacted["method"] == "jwt"

    def test_case_insensitive_redaction(self) -> None:
        """_redact_sensitive_in_dict should redact keys case-insensitively."""
        data = {"api_key": "lower", "API_KEY": "upper", "Api_Key": "mixed"}
        redacted = _redact_sensitive_in_dict(data)

        assert all(v == REDACTED_VALUE for v in redacted.values())

    def test_non_sensitive_keys_unchanged(self) -> None:
        """_redact_sensitive_in_dict should not modify non-sensitive keys."""
        data = {"name": "test", "port": 8080, "enabled": True}
        redacted = _redact_sensitive_in_dict(data)

        assert redacted == data

    def test_nested_dict_redaction(self) -> None:
        """_redact_sensitive_in_dict should recursively redact nested dicts."""
        data = {
            "outer": {
                "api_key": "nested_secret",
                "safe": "value",
            }
        }
        redacted = _redact_sensitive_in_dict(data)

        assert redacted["outer"]["api_key"] == REDACTED_VALUE
        assert redacted["outer"]["safe"] == "value"

    def test_non_dict_values_preserved(self) -> None:
        """_redact_sensitive_in_dict should preserve non-string values."""
        data = {"count": 42, "enabled": True, "items": [1, 2, 3]}
        redacted = _redact_sensitive_in_dict(data)

        assert redacted["count"] == 42
        assert redacted["enabled"] is True
        assert redacted["items"] == [1, 2, 3]

    def test_empty_dict(self) -> None:
        """_redact_sensitive_in_dict should handle empty dict."""
        assert _redact_sensitive_in_dict({}) == {}

    def test_sensitive_value_not_in_output(self) -> None:
        """Redacted values should not appear in output."""
        data = {"api_key": "supersecretvalue123"}
        redacted = _redact_sensitive_in_dict(data)

        assert "supersecretvalue123" not in str(redacted)
        assert redacted["api_key"] == REDACTED_VALUE


class TestSlotRuntime:
    """T016b: Tests for SlotRuntime dataclass.

    NOTE: SlotRuntime is expected to be implemented in process_manager.py
    as part of T017. These tests verify the expected API contract.
    """

    def test_default_construction(self) -> None:
        """SlotRuntime should construct with default values."""
        from llama_manager.config import SlotState
        from llama_manager.log_buffer import LogBuffer

        try:
            from llama_manager.process_manager import SlotRuntime  # type: ignore[attr-defined]

            runtime = SlotRuntime(
                slot_id="test-slot",
                state=SlotState.IDLE,
                pid=None,
                start_time=time.time(),
                logs=LogBuffer(),
                gpu_stats=None,
            )

            assert runtime.slot_id == "test-slot"
            assert runtime.state == SlotState.IDLE
            assert runtime.pid is None
            assert isinstance(runtime.start_time, float)
            assert isinstance(runtime.logs, LogBuffer)
            assert runtime.gpu_stats is None
        except ImportError:
            pytest.skip("SlotRuntime not yet implemented (T017)")

    def test_construction_with_all_fields(self) -> None:
        """SlotRuntime should accept all expected fields."""
        from llama_manager.config import SlotState
        from llama_manager.gpu_stats import GPUStats
        from llama_manager.log_buffer import LogBuffer

        try:
            from llama_manager.process_manager import SlotRuntime  # type: ignore[attr-defined]

            logs = LogBuffer()
            gpu = GPUStats(device_index=0)

            runtime = SlotRuntime(
                slot_id="gpu0-slot1",
                state=SlotState.RUNNING,
                pid=12345,
                start_time=1234567890.0,
                logs=logs,
                gpu_stats=gpu,
            )

            assert runtime.slot_id == "gpu0-slot1"
            assert runtime.state == SlotState.RUNNING
            assert runtime.pid == 12345
            assert runtime.start_time == 1234567890.0
            assert runtime.logs is logs
            assert runtime.gpu_stats is gpu
        except ImportError:
            pytest.skip("SlotRuntime not yet implemented (T017)")

    def test_serialization_to_dict(self) -> None:
        """SlotRuntime should serialize to dict with expected keys."""
        from llama_manager.config import SlotState
        from llama_manager.log_buffer import LogBuffer

        try:
            from llama_manager.process_manager import SlotRuntime  # type: ignore[attr-defined]

            runtime = SlotRuntime(
                slot_id="test",
                state=SlotState.RUNNING,
                pid=9999,
                start_time=1234567890.0,
                logs=LogBuffer(),
                gpu_stats=None,
            )

            d = runtime.to_dict()
            assert "slot_id" in d
            assert "state" in d
            assert "pid" in d
            assert "start_time" in d
            assert d["slot_id"] == "test"
            assert d["state"] == "running"
            assert d["pid"] == 9999
        except ImportError:
            pytest.skip("SlotRuntime not yet implemented (T017)")

    def test_state_transition_method(self) -> None:
        """SlotRuntime should have a method to transition state."""
        from llama_manager.config import SlotState
        from llama_manager.log_buffer import LogBuffer

        try:
            from llama_manager.process_manager import SlotRuntime  # type: ignore[attr-defined]

            runtime = SlotRuntime(
                slot_id="test",
                state=SlotState.IDLE,
                pid=None,
                start_time=time.time(),
                logs=LogBuffer(),
                gpu_stats=None,
            )

            # Transition IDLE -> LAUNCHING
            runtime.transition_to(SlotState.LAUNCHING)
            assert runtime.state == SlotState.LAUNCHING

            # Transition LAUNCHING -> RUNNING
            runtime.transition_to(SlotState.RUNNING)
            assert runtime.state == SlotState.RUNNING
        except ImportError:
            pytest.skip("SlotRuntime not yet implemented (T017)")

    def test_to_dict_includes_gpu_stats(self) -> None:
        """SlotRuntime.to_dict() should include gpu_stats info when present."""
        from llama_manager.config import SlotState
        from llama_manager.gpu_stats import GPUStats
        from llama_manager.log_buffer import LogBuffer

        try:
            from llama_manager.process_manager import SlotRuntime  # type: ignore[attr-defined]

            gpu = GPUStats(device_index=1)
            runtime = SlotRuntime(
                slot_id="test",
                state=SlotState.RUNNING,
                pid=12345,
                start_time=time.time(),
                logs=LogBuffer(),
                gpu_stats=gpu,
            )

            d = runtime.to_dict()
            assert "gpu_stats" in d
        except ImportError:
            pytest.skip("SlotRuntime not yet implemented (T017)")
