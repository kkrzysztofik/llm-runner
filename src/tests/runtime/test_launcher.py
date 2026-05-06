from __future__ import annotations

"""Tests for the process launcher abstraction."""


import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.orchestration.launcher import (
    DefaultProcessLauncher,
    ProcessHandle,
    ProcessTimeoutError,
    _SubprocessHandle,
)


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

        assert manager._current_launch_attempt_id is None


class TestServerManagerFormatOutput:
    """Tests for _format_output helper method."""

    def test_format_output_includes_timestamp(self) -> None:
        """_format_output should include HH:MM:SS timestamp."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        result = manager._format_output("test_server", "hello world")

        # Should start with [HH:MM:SS][test_server]
        assert result.startswith("[")
        assert "][test_server] hello world" in result

    def test_format_output_preserves_content(self) -> None:
        """_format_output should preserve the content after stripping newlines (done by caller)."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        result = manager._format_output("test_server", "hello")

        assert result.endswith("hello")


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

    def test_on_interrupt_calls_cleanup_and_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """on_interrupt should call cleanup_servers then exit with 130."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.pids = [12345]

        with pytest.raises(SystemExit) as exc_info:
            manager.on_interrupt(2, None)  # SIGINT

        assert exc_info.value.code == 130
        assert manager.shutting_down is True

    def test_on_terminate_calls_cleanup_and_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        """on_terminate should call cleanup_servers then exit with 143."""
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()
        manager.pids = [12345]

        with pytest.raises(SystemExit) as exc_info:
            manager.on_terminate(15, None)  # SIGTERM

        assert exc_info.value.code == 143
        assert manager.shutting_down is True
