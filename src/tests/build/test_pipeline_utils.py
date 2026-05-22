"""Tests for build_pipeline/utils.py helper functions."""

from __future__ import annotations

import signal
import threading
from unittest.mock import MagicMock, call, patch

from llama_manager.build_pipeline.models import BuildBackend
from llama_manager.build_pipeline.utils import (
    _cancel_requested,
    _format_command,
    _format_command_failure,
    _format_duration,
    _redact_build_text,
    _summarize_command_output,
    _tail_lines,
    get_build_env_cmd,
    terminate_process_tree,
)

# ── get_build_env_cmd ──────────────────────────────────────────────────────


class TestGetBuildEnvCmd:
    def test_cuda_unchanged(self) -> None:
        cmd = ["cmake", "-S", "src"]
        result = get_build_env_cmd(cmd, BuildBackend.CUDA)
        assert result == cmd

    def test_sycl_no_setvars(self) -> None:
        cmd = ["cmake", "-S", "src"]
        with patch("llama_manager.build_pipeline.utils._INTEL_SETVARS_SH") as mock:
            mock.exists.return_value = False
            result = get_build_env_cmd(cmd, BuildBackend.SYCL)
        assert result == cmd

    def test_sycl_with_setvars(self) -> None:

        cmd = ["cmake", "-S", "src"]
        # Patch the module-level variable to a mock Path that reports exists=True
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_path.__str__ = MagicMock(return_value="/opt/intel/oneapi/setvars.sh")
        with patch("llama_manager.build_pipeline.utils._INTEL_SETVARS_SH", mock_path):
            result = get_build_env_cmd(cmd, BuildBackend.SYCL)
        assert result[0] == "bash"
        assert result[1] == "-c"
        assert "setvars.sh" in result[2]
        assert "cmake" in result[2]


# ── _format_command ────────────────────────────────────────────────────────


class TestFormatCommand:
    def test_simple(self) -> None:
        result = _format_command(["ls", "-la"])
        assert "ls" in result
        assert "-la" in result

    def test_with_spaces(self) -> None:
        result = _format_command(["echo", "hello world"])
        assert "hello world" in result

    def test_empty(self) -> None:
        result = _format_command([])
        assert result == ""


# ── _redact_build_text ─────────────────────────────────────────────────────


class TestRedactBuildText:
    def test_no_change(self) -> None:
        result = _redact_build_text("normal text")
        assert result == "normal text"

    def test_empty(self) -> None:
        result = _redact_build_text("")
        assert result == ""


# ── _format_duration ───────────────────────────────────────────────────────


class TestFormatDuration:
    def test_sub_second(self) -> None:
        result = _format_duration(0.5)
        assert "ms" in result

    def test_seconds(self) -> None:
        result = _format_duration(45.2)
        assert "45.2s" in result

    def test_minutes(self) -> None:
        result = _format_duration(120.0)
        assert "2m 0s" in result

    def test_hours(self) -> None:
        result = _format_duration(3661.0)
        assert "m" in result and "s" in result

    def test_zero(self) -> None:
        result = _format_duration(0.0)
        assert "0ms" in result


# ── _tail_lines ────────────────────────────────────────────────────────────


class TestTailLines:
    def test_fewer_than_limit(self) -> None:
        text = "a\nb\nc\n"
        result = _tail_lines(text, max_lines=10)
        assert result == "a\nb\nc"

    def test_more_than_limit(self) -> None:
        lines = [f"line{i}" for i in range(60)]
        text = "\n".join(lines)
        result = _tail_lines(text, max_lines=5)
        expected = "\n".join(lines[-5:])
        assert result == expected

    def test_empty(self) -> None:
        result = _tail_lines("")
        assert result == ""


# ── _summarize_command_output ──────────────────────────────────────────────


class TestSummarizeCommandOutput:
    def test_stdout_only(self) -> None:
        result = _summarize_command_output("out\n", "")
        assert "stdout tail:" in result

    def test_stderr_only(self) -> None:
        result = _summarize_command_output("", "err\n")
        assert "stderr tail:" in result

    def test_both(self) -> None:
        result = _summarize_command_output("out\n", "err\n")
        assert "stderr tail:" in result
        assert "stdout tail:" in result

    def test_empty(self) -> None:
        result = _summarize_command_output("", "")
        assert "No output captured." in result


# ── terminate_process_tree ─────────────────────────────────────────────────


class TestTerminateProcessTree:
    def test_already_dead(self) -> None:
        proc = MagicMock()
        proc.poll.return_value = 0
        terminate_process_tree(proc)
        proc.poll.assert_called_once()

    def test_killpg_success(self) -> None:
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.side_effect = [None, 0, 0]

        with (
            patch("llama_manager.build_pipeline.utils.os.getpgid", return_value=12345),
            patch("llama_manager.build_pipeline.utils.os.killpg") as killpg,
        ):
            terminate_process_tree(proc)

        killpg.assert_called_once_with(12345, signal.SIGTERM)
        proc.terminate.assert_not_called()

    def test_magicmock_pid_falls_back_without_real_killpg(self) -> None:
        proc = MagicMock()
        proc.poll.side_effect = [None, 0, 0]

        with patch("llama_manager.build_pipeline.utils.os.killpg") as killpg:
            terminate_process_tree(proc)

        killpg.assert_not_called()
        proc.terminate.assert_called_once()

    def test_process_lookup_error(self) -> None:
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None

        with patch("llama_manager.build_pipeline.utils.os.getpgid", side_effect=ProcessLookupError):
            terminate_process_tree(proc)

        assert proc.poll.called
        proc.terminate.assert_not_called()

    def test_os_error_fallback(self) -> None:
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.return_value = None

        with patch("llama_manager.build_pipeline.utils.os.getpgid", side_effect=OSError("nope")):
            terminate_process_tree(proc)

        assert proc.terminate.called

    def test_sigkill_needed(self) -> None:
        proc = MagicMock()
        proc.pid = 12345
        proc.poll.side_effect = [None, None, None, None]

        with (
            patch("llama_manager.build_pipeline.utils.os.getpgid", return_value=12345),
            patch("llama_manager.build_pipeline.utils.os.killpg") as killpg,
            patch("llama_manager.build_pipeline.utils.time.sleep"),
            patch("llama_manager.build_pipeline.utils.time.monotonic", side_effect=[0.0, 3.0]),
        ):
            terminate_process_tree(proc, use_process_group=True)

        assert killpg.call_args_list == [
            call(12345, signal.SIGTERM),
            call(12345, signal.SIGKILL),
        ]


# ── _cancel_requested ──────────────────────────────────────────────────────


class TestCancelRequested:
    def test_none(self) -> None:
        assert _cancel_requested(None) is False

    def test_unset(self) -> None:
        evt = threading.Event()
        assert _cancel_requested(evt) is False

    def test_set(self) -> None:
        evt = threading.Event()
        evt.set()
        assert _cancel_requested(evt) is True


# ── _format_command_failure ────────────────────────────────────────────────


class TestFormatCommandFailure:
    def test_basic(self) -> None:
        result = _format_command_failure(
            stage="configure",
            command=["cmake", "-S", "src"],
            returncode=1,
            stdout="",
            stderr="",
        )
        assert "configure" in result
        assert "exit code 1" in result

    def test_with_output(self) -> None:
        result = _format_command_failure(
            stage="build",
            command=["cmake", "--build", "."],
            returncode=2,
            stdout="some output\n",
            stderr="",
        )
        assert "build" in result
        assert "exit code 2" in result
        assert "some output" in result
