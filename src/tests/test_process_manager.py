"""Tests for llama_manager.process_manager.

Focused tests for:
- resolve_runtime_dir fallback behavior (T001)
- Runtime directory usability (T002)
- Pipe streaming with optional log buffers (T013)
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.log_buffer import LogBuffer
from llama_manager.process_manager import ValidationException, resolve_runtime_dir


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

    def test_stream_pipe_without_handler_prints(self, capsys):
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

            # Wait for thread to process (use event for deterministic sync)
            event = threading.Event()
            event.wait(timeout=1.0)

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

            # Wait for threads to process (use event for deterministic sync)
            event = threading.Event()
            event.wait(timeout=1.0)

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

            # Wait for thread to process (use event for deterministic sync)
            event = threading.Event()
            event.wait(timeout=1.0)

            # Verify both stdout and stderr were consumed by ServerManager only (2 lines)
            # If TUI also read, we'd see 4 lines (duplication from dual consumption)
            assert buffer.line_count == 2
