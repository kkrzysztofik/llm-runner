"""Tests for the process launcher abstraction."""

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.orchestration.launcher import (
    DefaultProcessLauncher,
    ProcessHandle,
    ProcessTimeoutError,
    _SubprocessHandle,
    wrap_sycl_launch_cmd,
)


class TestSyclLaunchWrapper:
    """Tests for wrap_sycl_launch_cmd helper."""

    def test_wrap_sycl_launch_cmd_sources_setvars_when_device_is_sycl(self, tmp_path: Path) -> None:
        setvars = tmp_path / "setvars.sh"
        setvars.write_text("# test\n")
        cmd = ["/bin/echo", "hello world"]

        wrapped = wrap_sycl_launch_cmd(cmd, "SYCL0", setvars_path=setvars)

        assert wrapped[:3] == [
            "bash",
            "-c",
            (
                'if ! source "$1" --force >/dev/null 2>&1; then '
                'echo "failed to source Intel oneAPI setvars for SYCL launch: $1" >&2; '
                "exit 127; "
                "fi; "
                "shift; "
                'exec "$@"'
            ),
        ]
        assert wrapped[3] == "llm-runner-sycl-launch"
        assert wrapped[4] == str(setvars)
        assert wrapped[5:] == cmd
        assert cmd == ["/bin/echo", "hello world"]

    def test_wrap_sycl_launch_cmd_leaves_non_sycl_device_unchanged(self, tmp_path: Path) -> None:
        setvars = tmp_path / "setvars.sh"
        setvars.write_text("# test\n")
        cmd = ["/bin/echo", "cuda"]

        wrapped = wrap_sycl_launch_cmd(cmd, "CUDA0", setvars_path=setvars)

        assert wrapped is cmd

    def test_wrap_sycl_launch_cmd_leaves_command_unchanged_when_setvars_missing(
        self, tmp_path: Path
    ) -> None:
        cmd = ["/bin/echo", "sycl"]

        wrapped = wrap_sycl_launch_cmd(cmd, "SYCL0", setvars_path=tmp_path / "missing.sh")

        assert wrapped is cmd


class MockProcessHandle:
    """Test double for ProcessHandle.

    Uses Any for stdout/stderr types so MagicMock instances are
    structurally compatible with the ProcessHandle protocol.
    """

    pid: int = 12345
    stdout: Any = None  # pyright: ignore[reportAny]
    stderr: Any = None  # pyright: ignore[reportAny]

    def __init__(self, pid: int = 12345) -> None:
        self.pid = pid
        self.stdout = MagicMock()
        self.stdout.readline.side_effect = [""]
        self.stderr = MagicMock()
        self.stderr.readline.side_effect = [""]
        self._poll_return: int | None = None
        self._wait_return: int = 0

    def poll(self) -> int | None:
        return self._poll_return

    def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
        return self._wait_return


class MockProcessLauncher:
    """Test double for ProcessLauncher."""

    def __init__(self, handle: ProcessHandle | None = None) -> None:
        self._handle: ProcessHandle = handle or MockProcessHandle()  # type: ignore[assignment]
        self.launch_calls: list[list[str]] = []

    def launch(self, cmd: list[str]) -> ProcessHandle:
        self.launch_calls.append(cmd)
        return self._handle


class TestProcessTimeoutError:
    """Tests for ProcessTimeoutError exception."""

    def test_is_exception(self) -> None:
        assert issubclass(ProcessTimeoutError, Exception)

    def test_message_contains_pid_and_timeout(self) -> None:
        err = ProcessTimeoutError("process 42 did not exit within 5s")
        assert "42" in str(err)
        assert "5s" in str(err)


class TestDefaultProcessLauncher:
    """Tests for DefaultProcessLauncher."""

    def test_launch_returns_process_handle(self) -> None:
        launcher = DefaultProcessLauncher()
        handle = launcher.launch(["echo", "hello"])
        assert isinstance(handle, _SubprocessHandle)
        assert handle.pid is not None

    def test_handle_has_required_attributes(self) -> None:
        launcher = DefaultProcessLauncher()
        handle = launcher.launch(["echo", "hello"])
        assert hasattr(handle, "pid")
        assert hasattr(handle, "stdout")
        assert hasattr(handle, "stderr")
        assert hasattr(handle, "poll")
        assert hasattr(handle, "wait")

    def test_handle_poll_returns_none_for_running(self) -> None:
        launcher = DefaultProcessLauncher()
        handle = launcher.launch(["sleep", "1"])
        code = handle.poll()
        assert code is None
        handle._proc.terminate()  # pyright: ignore[reportAttributeAccessIssue]
        handle._proc.wait()  # pyright: ignore[reportAttributeAccessIssue]


class TestSubprocessHandleWaitTimeout:
    """Tests for _SubprocessHandle.wait() timeout behavior."""

    def test_wait_raises_timeout_error_on_timeout(self) -> None:
        launcher = DefaultProcessLauncher()
        handle = launcher.launch(["sleep", "300"])
        try:
            with pytest.raises(ProcessTimeoutError):
                handle.wait(timeout=0.1)
        finally:
            handle._proc.terminate()  # pyright: ignore[reportAttributeAccessIssue]
            handle._proc.wait()  # pyright: ignore[reportAttributeAccessIssue]

    def test_wait_returns_exit_code_on_success(self) -> None:
        launcher = DefaultProcessLauncher()
        handle = launcher.launch(["echo", "done"])
        code = handle.wait(timeout=5)
        assert code == 0


class TestProcessLauncherProtocol:
    """Tests demonstrating the ProcessLauncher protocol with injected mocks."""

    def test_injected_launcher_in_manager(self) -> None:
        """ServerManager should accept an injected ProcessLauncher."""
        from llama_manager.orchestration import ServerManager

        mock_launcher = MockProcessLauncher()
        manager = ServerManager(process_launcher=mock_launcher)

        # Verify launcher is stored
        assert manager._launcher is mock_launcher

        # Verify default launcher is NOT used when injected
        assert not isinstance(mock_launcher, DefaultProcessLauncher)

    def test_injected_launcher_uses_mock_handle(self) -> None:
        """Injected launcher should be used to start processes."""
        from llama_manager.orchestration import ServerManager

        mock_handle = MockProcessHandle(pid=99999)
        mock_launcher = MockProcessLauncher(handle=mock_handle)  # pyright: ignore[arg-type,reportArgumentType]
        manager = ServerManager(process_launcher=mock_launcher)

        # Start a server with the injected launcher
        proc = manager.start_server_background("test", ["echo", "hello"])

        # Verify the injected launcher was called
        assert len(mock_launcher.launch_calls) == 1
        assert mock_launcher.launch_calls[0] == ["echo", "hello"]

        # Verify the returned handle is from the mock
        assert proc.pid == 99999
        assert proc is mock_handle

    def test_default_launcher_used_when_none_injected(self) -> None:
        """ServerManager should use DefaultProcessLauncher when no launcher injected."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()

        # No launcher injected — should use default
        assert manager._launcher is None

        # start_server_background should create DefaultProcessLauncher internally
        # We mock psutil to avoid real process tracking issues
        with patch("psutil.Process") as mock_psutil:
            mock_proc_obj = mock_psutil.return_value
            mock_proc_obj.create_time.return_value = time.time()

            # This will actually spawn a subprocess (echo hello)
            # which exits immediately
            proc = manager.start_server_background("test", ["echo", "hello"])

            # Verify it's a _SubprocessHandle (from DefaultProcessLauncher)
            assert isinstance(proc, _SubprocessHandle)

    def test_multiple_servers_with_injected_launcher(self) -> None:
        """Multiple servers should each call the injected launcher."""
        from llama_manager.orchestration import ServerManager

        mock_handle = MockProcessHandle(pid=99999)
        mock_launcher = MockProcessLauncher(handle=mock_handle)  # pyright: ignore[arg-type]
        manager = ServerManager(process_launcher=mock_launcher)

        # Start multiple servers
        for i in range(3):
            manager.start_server_background(f"server{i}", ["echo", str(i)])

        assert len(mock_launcher.launch_calls) == 3
        pids = [proc.pid for proc in manager.servers]
        assert pids == [99999, 99999, 99999]

    def test_wait_for_any_uses_handle_poll(self) -> None:
        """wait_for_any should use ProcessHandle.poll()."""
        from llama_manager.orchestration import ServerManager

        mock_launcher = MockProcessLauncher()
        manager = ServerManager(process_launcher=mock_launcher)

        # Add a mock process that returns exit code 42 on poll
        mock_handle = MockProcessHandle(pid=11111)
        mock_handle._poll_return = 42
        manager.servers.append(mock_handle)  # pyright: ignore[arg-type,reportArgumentType]

        # wait_for_any should return the poll code
        code = manager.wait_for_any()
        assert code == 42

    def test_wait_for_any_blocks_until_process_exits(self) -> None:
        """wait_for_any should poll repeatedly until a process exits."""
        from llama_manager.orchestration import ServerManager

        mock_launcher = MockProcessLauncher()
        manager = ServerManager(process_launcher=mock_launcher)

        # Add a mock process that goes from running to exited
        mock_handle = MockProcessHandle(pid=22222)
        call_count = 0

        def side_effect() -> int | None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # still running
            return 0  # exited

        mock_handle.poll = side_effect  # pyright: ignore[reportGeneralTypeIssues]
        manager.servers.append(mock_handle)  # pyright: ignore[arg-type,reportArgumentType]

        code = manager.wait_for_any()
        assert code == 0
        assert call_count >= 2  # at least 2 polls (one returned None, one returned 0)

    def test_wait_for_processes_catches_timeout_error(self) -> None:
        """_wait_for_processes should catch ProcessTimeoutError without raising."""
        from llama_manager.orchestration import ServerManager

        mock_launcher = MockProcessLauncher()
        manager = ServerManager(process_launcher=mock_launcher)

        # Add a mock process that raises ProcessTimeoutError on wait
        mock_handle = MockProcessHandle(pid=33333)

        def raise_timeout(timeout: float) -> int:  # noqa: ARG001
            raise ProcessTimeoutError(f"process {mock_handle.pid} timed out")

        mock_handle.wait = raise_timeout  # pyright: ignore[reportGeneralTypeIssues,reportAttributeAccessIssue]
        manager.servers.append(mock_handle)  # pyright: ignore[arg-type,reportArgumentType]

        # Should not raise
        manager._wait_for_processes()
        assert len(manager.servers) == 1  # server not removed

    def test_start_servers_wraps_sycl_config_with_oneapi_setvars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llama_manager.orchestration import ServerManager
        from llama_manager.orchestration import launcher as launcher_module
        from tests.support.helpers import make_server_config

        setvars = tmp_path / "setvars.sh"
        setvars.write_text("# test\n")
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        monkeypatch.setenv("LLM_RUNNER_RUNTIME_DIR", str(runtime_dir))
        monkeypatch.setattr(launcher_module, "_INTEL_SETVARS_SH", setvars)

        mock_launcher = MockProcessLauncher()
        manager = ServerManager(process_launcher=mock_launcher)  # pyright: ignore[arg-type]
        cfg = make_server_config(alias="summary-balanced", device="SYCL0", server_bin="/bin/echo")

        manager.start_servers([cfg], {})

        launched = mock_launcher.launch_calls[0]
        assert launched[0:2] == ["bash", "-c"]
        assert launched[3] == "llm-runner-sycl-launch"
        assert launched[4] == str(setvars)
        assert launched[5] == "/bin/echo"

    def test_start_servers_does_not_wrap_cuda_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llama_manager.orchestration import ServerManager
        from llama_manager.orchestration import launcher as launcher_module
        from tests.support.helpers import make_server_config

        setvars = tmp_path / "setvars.sh"
        setvars.write_text("# test\n")
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        monkeypatch.setenv("LLM_RUNNER_RUNTIME_DIR", str(runtime_dir))
        monkeypatch.setattr(launcher_module, "_INTEL_SETVARS_SH", setvars)

        mock_launcher = MockProcessLauncher()
        manager = ServerManager(process_launcher=mock_launcher)  # pyright: ignore[arg-type]
        cfg = make_server_config(alias="qwen35-coding", device="CUDA0", server_bin="/bin/echo")

        manager.start_servers([cfg], {})

        assert mock_launcher.launch_calls[0][0] == "/bin/echo"

    def test_start_servers_records_server_pid_in_slot_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llama_manager.orchestration import LockMetadata, ServerManager, read_lock
        from tests.support.helpers import make_server_config

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        monkeypatch.setenv("LLM_RUNNER_RUNTIME_DIR", str(runtime_dir))
        mock_handle = MockProcessHandle(pid=24680)
        mock_launcher = MockProcessLauncher(handle=mock_handle)  # pyright: ignore[arg-type]
        manager = ServerManager(process_launcher=mock_launcher)
        cfg = make_server_config(alias="summary-balanced", port=8080, server_bin="/bin/echo")

        manager.start_servers([cfg], {})

        metadata = read_lock(runtime_dir, "summary-balanced")
        assert metadata is not None
        assert isinstance(metadata, LockMetadata)
        assert metadata.pid == 24680
        assert metadata.port == 8080

    def test_shutdown_slot_uses_tracked_process_and_releases_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from llama_manager.orchestration import ServerManager, read_lock
        from tests.support.helpers import make_server_config

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        monkeypatch.setenv("LLM_RUNNER_RUNTIME_DIR", str(runtime_dir))
        mock_handle = MockProcessHandle(pid=24681)
        mock_launcher = MockProcessLauncher(handle=mock_handle)  # pyright: ignore[arg-type]
        manager = ServerManager(process_launcher=mock_launcher)
        cfg = make_server_config(alias="summary-balanced", port=8080, server_bin="/bin/echo")
        manager.start_servers([cfg], {})

        with patch("llama_manager.orchestration.manager.os.kill") as kill:
            result = manager.shutdown_slot("summary-balanced")

        assert result is True
        kill.assert_called_once()
        assert read_lock(runtime_dir, "summary-balanced") is None
        assert "summary-balanced" not in manager.slot_processes
        assert mock_handle not in manager.servers
        assert mock_handle.pid not in manager.pids

    def test_launch_all_slots_empty_is_blocked_without_errors(self, tmp_path: Path) -> None:
        """launch_all_slots should return blocked without errors when no slots are requested."""
        from llama_manager.orchestration import ServerManager

        result = ServerManager().launch_all_slots([], runtime_dir=tmp_path)

        assert result.status == "blocked"
        assert result.launched == []
        assert result.errors is None

    def test_launch_all_slots_degraded_when_one_slot_lock_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """launch_all_slots should report degraded when only some slot locks are acquired."""
        from llama_manager.config import ModelSlot
        from llama_manager.orchestration import ServerManager

        def acquire_slot_lock(slot_id: str, port: int, server_pid: int | None = None) -> Path:
            if slot_id == "blocked":
                raise RuntimeError("lock busy")
            return tmp_path / f"slot-{slot_id}.lock"

        monkeypatch.setattr(
            "llama_manager.orchestration.slot_lockfile.acquire_slot_lock",
            acquire_slot_lock,
        )

        result = ServerManager().launch_all_slots(
            [
                ModelSlot(slot_id="ok", model_path="/models/a.gguf", port=8080),
                ModelSlot(slot_id="blocked", model_path="/models/b.gguf", port=8081),
            ],
            runtime_dir=tmp_path,
        )

        assert result.status == "degraded"
        assert result.launched == ["ok"]
        assert result.warnings is not None
        assert "lock_acquire_failed" in result.warnings[0]

    def test_start_server_background_ignores_psutil_access_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """start_server_background should still track handles when psutil metadata is denied."""
        import psutil

        from llama_manager.orchestration import ServerManager

        monkeypatch.setattr(
            "llama_manager.orchestration.manager.threading.Thread",
            lambda *args, **kwargs: MagicMock(start=MagicMock()),
        )
        monkeypatch.setattr(
            "llama_manager.orchestration.manager.psutil.Process",
            MagicMock(side_effect=psutil.AccessDenied(pid=123)),
        )
        mock_handle = MockProcessHandle(pid=123)
        manager = ServerManager(
            process_launcher=MockProcessLauncher(handle=mock_handle)  # pyright: ignore[arg-type]
        )

        result = manager.start_server_background("test", ["echo", "hello"])

        assert result is mock_handle
        assert manager.pids == [123]
        assert manager.servers == [mock_handle]
        assert manager.pid_metadata == {}

    def test_shutdown_process_handle_returns_true_for_already_exited(self) -> None:
        """_shutdown_process_handle should forget exited processes and release the lock."""
        from llama_manager.orchestration import ServerManager

        process = MockProcessHandle(pid=555)
        process._poll_return = 0
        manager = ServerManager()
        manager.slot_processes["slot"] = process  # pyright: ignore[assignment]
        manager.servers.append(process)  # pyright: ignore[arg-type]
        manager.pids.append(process.pid)
        manager.pid_metadata[process.pid] = 1.0
        manager.release_lock = MagicMock()  # type: ignore[method-assign]

        assert manager._shutdown_process_handle("slot", process, timeout=0.1) is True
        manager.release_lock.assert_called_once_with("slot")
        assert manager.slot_processes == {}
        assert manager.servers == []
        assert manager.pids == []
        assert manager.pid_metadata == {}

    def test_shutdown_process_handle_returns_false_on_ownership_failure(self) -> None:
        """_shutdown_process_handle should not signal a process it no longer owns."""
        from llama_manager.orchestration import ServerManager

        process = MockProcessHandle(pid=556)
        manager = ServerManager()
        manager.pid_metadata[process.pid] = 1.0
        manager._verify_process_ownership = MagicMock(return_value=False)  # type: ignore[method-assign]
        manager._record_lifecycle_event = MagicMock()  # type: ignore[method-assign]

        assert manager._shutdown_process_handle("slot", process, timeout=0.1) is False
        manager._record_lifecycle_event.assert_called_once_with(
            "skip", pid=556, details="ownership_failed"
        )

    def test_shutdown_process_handle_releases_lock_when_sigterm_process_missing(self) -> None:
        """_shutdown_process_handle should treat missing process on SIGTERM as stopped."""
        from llama_manager.orchestration import ServerManager

        process = MockProcessHandle(pid=557)
        manager = ServerManager()
        manager.slot_processes["slot"] = process  # pyright: ignore[assignment]
        manager.servers.append(process)  # pyright: ignore[arg-type]
        manager.pids.append(process.pid)
        manager.release_lock = MagicMock()  # type: ignore[method-assign]

        with patch("llama_manager.orchestration.manager.os.kill", side_effect=OSError):
            assert manager._shutdown_process_handle("slot", process, timeout=0.1) is True

        manager.release_lock.assert_called_once_with("slot")
        assert manager.slot_processes == {}

    def test_shutdown_process_handle_uses_sigkill_after_timeout(self) -> None:
        """_shutdown_process_handle should escalate to SIGKILL after SIGTERM timeout."""
        from llama_manager.orchestration import ServerManager

        process = MockProcessHandle(pid=558)
        wait_calls = 0

        def wait(timeout: float | None = None) -> int:
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls == 1:
                raise ProcessTimeoutError("term timeout")
            return 0

        process.wait = wait  # pyright: ignore[method-assign]
        manager = ServerManager()
        manager.slot_processes["slot"] = process  # pyright: ignore[assignment]
        manager.servers.append(process)  # pyright: ignore[arg-type]
        manager.pids.append(process.pid)
        manager.release_lock = MagicMock()  # type: ignore[method-assign]

        with patch("llama_manager.orchestration.manager.os.kill") as kill:
            assert manager._shutdown_process_handle("slot", process, timeout=0.1) is True

        assert [call.args[1] for call in kill.call_args_list] == [15, 9]
        manager.release_lock.assert_called_once_with("slot")

    def test_shutdown_process_handle_returns_false_when_sigkill_wait_times_out(self) -> None:
        """_shutdown_process_handle should return False if the process survives SIGKILL."""
        from llama_manager.orchestration import ServerManager

        process = MockProcessHandle(pid=559)

        def wait(timeout: float | None = None) -> int:
            raise ProcessTimeoutError("still running")

        process.wait = wait  # pyright: ignore[method-assign]
        manager = ServerManager()
        manager.release_lock = MagicMock()  # type: ignore[method-assign]

        with patch("llama_manager.orchestration.manager.os.kill"):
            assert manager._shutdown_process_handle("slot", process, timeout=0.1) is False

        manager.release_lock.assert_not_called()


class TestServerManagerAckFlow:
    """Tests for risk acknowledgement flow in ServerManager."""

    def test_validate_ack_token_matches(self) -> None:
        """validate_ack_token should return True when token matches attempt_id."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        attempt_id = manager.begin_launch_attempt()
        token = f"ack:{attempt_id}"

        assert manager.validate_ack_token(attempt_id, token) is True

    def test_validate_ack_token_none(self) -> None:
        """validate_ack_token should return False when token is None."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        attempt_id = manager.begin_launch_attempt()

        assert manager.validate_ack_token(attempt_id, None) is False

    def test_validate_ack_token_mismatch(self) -> None:
        """validate_ack_token should return False when token doesn't match."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        attempt_id = manager.begin_launch_attempt()

        assert manager.validate_ack_token(attempt_id, "wrong_token") is False

    def test_acknowledge_risk_stores_ack(self) -> None:
        """acknowledge_risk should store the slot:risk_type combination."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.begin_launch_attempt()
        manager.acknowledge_risk("slot1", "gpu_unavailable")

        assert manager.is_risk_acknowledged("slot1", "gpu_unavailable") is True

    def test_acknowledge_risk_different_slot(self) -> None:
        """acknowledge_risk should not affect other slots."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.begin_launch_attempt()
        manager.acknowledge_risk("slot1", "gpu_unavailable")

        assert manager.is_risk_acknowledged("slot2", "gpu_unavailable") is False

    def test_acknowledge_risk_different_risk(self) -> None:
        """acknowledge_risk should not affect other risk types."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.begin_launch_attempt()
        manager.acknowledge_risk("slot1", "gpu_unavailable")

        assert manager.is_risk_acknowledged("slot1", "port_conflict") is False

    def test_acknowledge_risk_with_explicit_attempt_id(self) -> None:
        """acknowledge_risk should work with explicit launch_attempt_id."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.acknowledge_risk("slot1", "gpu_unavailable", launch_attempt_id="abc123")

        assert (
            manager.is_risk_acknowledged("slot1", "gpu_unavailable", launch_attempt_id="abc123")
            is True
        )

    def test_acknowledge_risk_invalid_token_raises(self) -> None:
        """acknowledge_risk should raise ValueError for invalid ack_token."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.begin_launch_attempt()

        with pytest.raises(ValueError, match="ack_token does not match"):
            manager.acknowledge_risk("slot1", "gpu_unavailable", ack_token="wrong_token")  # noqa: S106

    def test_clear_risk_acknowledgements_clears_cache(self) -> None:
        """clear_risk_acknowledgements should clear all ack state."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.begin_launch_attempt()
        manager.acknowledge_risk("slot1", "gpu_unavailable")

        manager.clear_risk_acknowledgements()

        assert manager.is_risk_acknowledged("slot1", "gpu_unavailable") is False

    def test_clear_risk_acknowledgements_clears_attempt_id(self) -> None:
        """clear_risk_acknowledgements should clear current attempt_id."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.begin_launch_attempt()
        manager.clear_risk_acknowledgements()

        assert manager._risk._current_launch_attempt_id is None


class TestServerManagerFormatOutput:
    """Tests for _format_output helper method."""

    def test_format_output_includes_timestamp(self) -> None:
        """_format_output should include HH:MM:SS timestamp."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        result = manager._format_output("test_server", "hello world")

        # Should start with [HH:MM:SS] and include the server name prefix.
        assert result.startswith("[")
        assert "[test_server] hello world" in result
        assert "hello world" in result

    def test_format_output_preserves_content(self) -> None:
        """_format_output should preserve the content after stripping newlines (done by caller)."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        result = manager._format_output("test_server", "hello")

        assert result.endswith("hello")

    def test_format_output_strips_llama_cpp_timestamp(self) -> None:
        """_format_output should strip llama.cpp millisecond timestamps like 177.32.478.581."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        result = manager._format_output("test_server", "177.32.478.581 I srv  update_slots: idle")

        assert "177.32.478.581" not in result
        assert "I srv" in result

    def test_format_output_includes_server_name(self) -> None:
        """_format_output should include the server_name in the formatted line."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        result = manager._format_output("qwen35-coding", "hello world")

        assert "[qwen35-coding] hello world" in result


class TestServerManagerForeground:
    """Tests for run_server_foreground method."""

    def test_run_server_foreground_calls_wait_no_timeout(self) -> None:
        """run_server_foreground should call wait() without timeout (blocks till exit)."""
        from llama_manager.orchestration import ServerManager

        mock_handle = MockProcessHandle(pid=42)
        mock_handle._wait_return = 0
        wait_called = False

        def capture_wait(timeout: float | None = None) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
            nonlocal wait_called
            wait_called = True
            assert timeout is None
            return mock_handle._wait_return

        mock_handle.wait = capture_wait  # pyright: ignore[reportGeneralTypeIssues]

        mock_launcher = MockProcessLauncher(handle=mock_handle)  # pyright: ignore[arg-type]
        manager = ServerManager(process_launcher=mock_launcher)

        result = manager.run_server_foreground("test", ["echo", "hello"])

        assert result == 0
        assert wait_called is True


class TestServerManagerSignalHandlers:
    """Tests for signal handler methods (on_interrupt, on_terminate)."""

    def test_on_interrupt_calls_cleanup_and_returns_code(self) -> None:
        """on_interrupt should call cleanup_servers then return 130."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.pids = [12345]

        exit_code = manager.on_interrupt(2, None)  # SIGINT

        assert exit_code == 130
        assert manager.shutting_down is True

    def test_on_terminate_calls_cleanup_and_returns_code(self) -> None:
        """on_terminate should call cleanup_servers then return 143."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.pids = [12345]

        exit_code = manager.on_terminate(15, None)  # SIGTERM

        assert exit_code == 143
        assert manager.shutting_down is True
