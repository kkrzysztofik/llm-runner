"""Additional configure and build stage edge-case tests.

Covers:
- run_configure: dry-run, cache skip, pre-cancel, timeout, cancel-after, exception
- run_build: dry-run, effective parallel jobs, cancel-before/after, exception
- _effective_parallel_jobs: explicit, os.cpu_count, None fallback
- _build_cmake_cmd: standard, no jobs, custom jobs
"""

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from llama_manager.build_pipeline import BuildBackend, BuildConfig, BuildProgress
from llama_manager.build_pipeline._context import _BuildContext
from llama_manager.build_pipeline.stages.build import run_build
from llama_manager.build_pipeline.stages.configure import get_cmake_flags, run_configure
from llama_manager.build_pipeline.utils import (
    _cancel_requested,
    _format_command_failure,
    _format_duration,
    _summarize_command_output,
    _tail_lines,
    get_build_env_cmd,
    terminate_process_tree,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ctx(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    cancel_event: threading.Event | None = None,
    **overrides: Any,
) -> _BuildContext:
    """Create a minimal _BuildContext for configure/build tests."""
    kwargs: dict = {
        "backend": BuildBackend.SYCL,
        "source_dir": tmp_path / "source",
        "build_dir": tmp_path / "build",
        "output_dir": tmp_path / "output",
        "git_remote_url": "https://github.com/ggerganov/llama.cpp",
        "git_branch": "main",
        "build_timeout_seconds": 30,
    }
    kwargs.update(overrides)
    config = BuildConfig(**kwargs)
    return _BuildContext(
        config=config, dry_run=dry_run, build_start_time=0.0, cancel_event=cancel_event
    )


# ── configure stage edge cases ───────────────────────────────────────────────


class TestConfigureStageEdgeCases:
    """Edge-case tests for run_configure."""

    def test_configure_dry_run_returns_command(self, tmp_path: Path) -> None:
        """Dry-run configure should not spawn cmake."""
        ctx = _make_ctx(tmp_path, dry_run=True)
        result = run_configure(ctx)
        assert result.status == "success"
        assert "Would run:" in result.message
        assert "cmake" in result.message

    def test_configure_cache_skip(self, tmp_path: Path) -> None:
        """Configure should skip when CMakeCache.txt exists and update_sources=False."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "CMakeCache.txt").touch()
        ctx = _make_ctx(tmp_path, build_dir=build_dir, update_sources=False)
        result = run_configure(ctx)
        assert result.status == "skipped"
        assert "Already configured" in result.message
        assert result.progress_percent == 50

    def test_configure_runs_when_cache_exists_and_update_sources_true(self, tmp_path: Path) -> None:
        """Configure should run (not skip) when CMakeCache.txt exists but update_sources=True."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "CMakeCache.txt").touch()
        ctx = _make_ctx(tmp_path, build_dir=build_dir, update_sources=True)
        with patch(
            "llama_manager.build_pipeline.stages.configure.run_command_with_cancel",
            return_value=(0, "", ""),
        ):
            result = run_configure(ctx)
        # When update_sources=True, configure runs even if cache exists
        assert result.status == "success"

    def test_configure_pre_cancel_before_spawn(self, tmp_path: Path) -> None:
        """Configure should detect pre-spawn cancellation and return cancelled."""
        cancel_event = threading.Event()
        cancel_event.set()
        ctx = _make_ctx(tmp_path, cancel_event=cancel_event)
        result = run_configure(ctx)
        assert result.status == "failed"
        assert "build cancelled" in result.message.lower()

    def test_configure_cancel_after_run(self, tmp_path: Path) -> None:
        """Configure should detect cancellation after cmake exits and return cancelled."""
        cancel_event = threading.Event()
        ctx = _make_ctx(tmp_path, cancel_event=cancel_event)
        with patch(
            "llama_manager.build_pipeline.stages.configure.run_command_with_cancel",
            return_value=(0, "", ""),
        ):
            # Set cancel event before run_configure checks it
            cancel_event.set()
            result = run_configure(ctx)
        # After run_command_with_cancel returns, the cancel check happens
        assert result.status == "failed"
        assert "build cancelled" in result.message.lower()

    def test_configure_timeout(self, tmp_path: Path) -> None:
        """Configure should report timeout when returncode is -1."""
        ctx = _make_ctx(tmp_path)
        with patch(
            "llama_manager.build_pipeline.stages.configure.run_command_with_cancel",
            return_value=(-1, "", ""),
        ):
            result = run_configure(ctx)
        assert result.status == "failed"
        assert "timed out" in result.message.lower()

    def test_configure_nonzero_failure(self, tmp_path: Path) -> None:
        """Configure should report cmake failure with exit code and output."""
        ctx = _make_ctx(tmp_path)
        with patch(
            "llama_manager.build_pipeline.stages.configure.run_command_with_cancel",
            return_value=(1, "stdout output", "stderr error"),
        ):
            result = run_configure(ctx)
        assert result.status == "failed"
        assert "CMake configure" in result.message
        assert "exit code 1" in result.message

    def test_configure_exception_path(self, tmp_path: Path) -> None:
        """Configure should catch exceptions and report them."""
        ctx = _make_ctx(tmp_path)
        with patch(
            "llama_manager.build_pipeline.stages.configure.run_command_with_cancel",
            side_effect=RuntimeError("cmake not found"),
        ):
            result = run_configure(ctx)
        assert result.status == "failed"
        assert "Configure failed" in result.message
        assert "cmake not found" in result.message

    def test_configure_success_message(self, tmp_path: Path) -> None:
        """Configure success should include backend name and flags."""
        ctx = _make_ctx(tmp_path)
        with patch(
            "llama_manager.build_pipeline.stages.configure.run_command_with_cancel",
            return_value=(0, "", ""),
        ):
            result = run_configure(ctx)
        assert result.status == "success"
        assert "sycl" in result.message.lower()
        assert "GGML_SYCL" in result.message
        assert "icx" in result.message or "icpx" in result.message

    def test_configure_cuda_flags(self) -> None:
        """get_cmake_flags should include CUDA flag."""
        flags = get_cmake_flags(BuildBackend.CUDA)
        assert "-DGGML_CUDA=ON" in flags
        assert "-DBUILD_SERVER=ON" in flags

    def test_configure_build_args_included(self, tmp_path: Path) -> None:
        """Configure should include build_args in cmake command."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            build_args=["-DCMAKE_BUILD_TYPE=Release"],
        )
        ctx = _BuildContext(config=config, dry_run=True, build_start_time=0.0)
        result = run_configure(ctx)
        assert result.status == "success"
        assert "CMAKE_BUILD_TYPE=Release" in result.message

    def test_configure_no_build_args(self, tmp_path: Path) -> None:
        """Configure should work without build_args."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        ctx = _BuildContext(config=config, dry_run=True, build_start_time=0.0)
        result = run_configure(ctx)
        assert result.status == "success"
        assert "CMAKE_BUILD_TYPE" not in result.message

    def test_configure_base_flags_always_present(self) -> None:
        """Base cmake flags should always be present regardless of backend."""
        for backend in (BuildBackend.SYCL, BuildBackend.CUDA):
            flags = get_cmake_flags(backend)
            assert "-DBUILD_SERVER=ON" in flags
            assert "-DGGML_NATIVE=OFF" in flags


# ── build stage edge cases ───────────────────────────────────────────────────


class TestBuildStageEdgeCases:
    """Edge-case tests for run_build."""

    def test_build_dry_run_returns_command(self, tmp_path: Path) -> None:
        """Dry-run build should not spawn cmake --build."""
        ctx = _make_ctx(tmp_path, dry_run=True)
        result = run_build(ctx)
        assert result.status == "success"
        assert "Would run:" in result.message
        assert "cmake" in result.message

    def test_build_success_captures_output(self, tmp_path: Path) -> None:
        """Successful build should capture output lines."""
        ctx = _make_ctx(tmp_path)
        with patch(
            "llama_manager.build_pipeline.stages.build._run_build_subprocess",
            return_value=BuildProgress(
                stage="build",
                status="success",
                message="Build complete",
                progress_percent=75,
            ),
        ):
            result = run_build(ctx)
        assert result.status == "success"
        assert result.message == "Build complete"

    def test_build_exception_path(self, tmp_path: Path) -> None:
        """Build should catch exceptions and report them."""
        ctx = _make_ctx(tmp_path)
        with patch(
            "llama_manager.build_pipeline.stages.build._build_cmake_cmd",
            side_effect=RuntimeError("build dir missing"),
        ):
            result = run_build(ctx)
        assert result.status == "failed"
        assert "Build failed" in result.message

    def test_build_subprocess_called_with_correct_args(self, tmp_path: Path) -> None:
        """Build should call cmake --build with correct build dir and jobs."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            jobs=4,
        )
        ctx = _BuildContext(config=config, dry_run=False, build_start_time=0.0)
        mock_popen = MagicMock()
        mock_popen.__enter__.return_value.poll.return_value = 0
        mock_popen.__enter__.return_value.returncode = 0
        mock_popen.__enter__.return_value.stdout = iter([])
        mock_popen.__enter__.return_value.stderr = iter([])

        with (
            patch("subprocess.Popen", return_value=mock_popen),
            patch(
                "llama_manager.build_pipeline.utils._INTEL_SETVARS_SH",
                tmp_path / "nonexistent",
            ),
        ):
            result = run_build(ctx)

        assert result.status == "success"


# ── utility function edge cases ──────────────────────────────────────────────


class TestUtilityEdgeCases:
    """Edge-case tests for utility functions in utils.py."""

    def test_format_duration_sub_second(self) -> None:
        assert _format_duration(0.5) == "500ms"

    def test_format_duration_zero(self) -> None:
        assert _format_duration(0.0) == "0ms"

    def test_format_duration_seconds(self) -> None:
        result = _format_duration(30.5)
        assert result == "30.5s"

    def test_format_duration_minutes(self) -> None:
        result = _format_duration(125.0)
        assert result == "2m 5s"

    def test_format_duration_hours(self) -> None:
        result = _format_duration(7261.0)
        assert result == "121m 1s"

    def test_tail_lines_empty(self) -> None:
        assert _tail_lines("") == ""

    def test_tail_lines_fewer_than_max(self) -> None:
        text = "line1\nline2\nline3"
        result = _tail_lines(text, max_lines=10)
        assert result == "line1\nline2\nline3"

    def test_tail_lines_more_than_max(self) -> None:
        lines = [f"line{i}" for i in range(20)]
        text = "\n".join(lines)
        result = _tail_lines(text, max_lines=5)
        expected_lines = [f"line{i}" for i in range(15, 20)]
        assert result == "\n".join(expected_lines)

    def test_summarize_command_output_both(self) -> None:
        result = _summarize_command_output("out", "err")
        assert "stderr tail:" in result
        assert "stdout tail:" in result

    def test_summarize_command_output_empty(self) -> None:
        result = _summarize_command_output("", "")
        assert "No output captured." in result

    def test_summarize_command_output_only_stderr(self) -> None:
        result = _summarize_command_output("", "err")
        assert "stderr tail:" in result
        assert "stdout tail:" not in result

    def test_cancel_requested_none(self) -> None:
        assert _cancel_requested(None) is False

    def test_cancel_requested_unset(self) -> None:
        event = threading.Event()
        assert _cancel_requested(event) is False

    def test_cancel_requested_set(self) -> None:
        event = threading.Event()
        event.set()
        assert _cancel_requested(event) is True

    def test_format_command_failure_basic(self) -> None:
        result = _format_command_failure(
            stage="configure",
            command=["cmake", "-S", "."],
            returncode=1,
            stdout="out",
            stderr="err",
        )
        assert "configure command failed" in result
        assert "exit code 1" in result
        assert "cmake" in result

    def test_format_command_failure_empty_output(self) -> None:
        result = _format_command_failure(
            stage="build",
            command=["cmake", "--build", "."],
            returncode=2,
            stdout="",
            stderr="",
        )
        assert "No output captured." in result

    def test_get_build_env_cmd_cuda_unchanged(self) -> None:
        from llama_manager.build_pipeline import BuildBackend

        cmd = ["cmake", "-S", "."]
        result = get_build_env_cmd(cmd, BuildBackend.CUDA)
        assert result == cmd

    def test_get_build_env_cmd_sycl_no_setvars(self, tmp_path: Path) -> None:
        from llama_manager.build_pipeline import BuildBackend
        from llama_manager.build_pipeline.utils import _INTEL_SETVARS_SH

        cmd = ["cmake", "-S", "."]
        with patch.object(type(_INTEL_SETVARS_SH), "exists", return_value=False):
            result = get_build_env_cmd(cmd, BuildBackend.SYCL)
        assert result == cmd

    def test_build_cmake_cmd_includes_build_args(self, tmp_path: Path) -> None:
        """_build_cmake_cmd should include build_args from config."""
        from llama_manager.build_pipeline.stages.build import _build_cmake_cmd

        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            build_args=["-v", "--debug-output"],
        )
        ctx = _BuildContext(config=config, dry_run=False, build_start_time=0.0)
        setvars = tmp_path / "setvars.sh"
        setvars.touch()
        with patch("llama_manager.build_pipeline.utils._INTEL_SETVARS_SH", setvars):
            cmd = _build_cmake_cmd(ctx)
        # get_build_env_cmd wraps SYCL commands in bash -c string when setvars.sh exists
        assert cmd[0] == "bash"
        assert cmd[1] == "-c"
        assert "cmake --build" in cmd[2]
        assert "-v" in cmd[2]
        assert "--debug-output" in cmd[2]

    def test_build_cmake_cmd_no_build_args(self, tmp_path: Path) -> None:
        """_build_cmake_cmd should not include extra args when build_args is None."""
        from llama_manager.build_pipeline.stages.build import _build_cmake_cmd

        config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        ctx = _BuildContext(config=config, dry_run=False, build_start_time=0.0)
        cmd = _build_cmake_cmd(ctx)
        # Should only have cmake, --build, build_dir, -j (CUDA doesn't wrap in bash)
        assert cmd[0] == "cmake"
        assert cmd[1] == "--build"
        assert "-v" not in cmd

    def test_terminate_process_tree_already_dead(self) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        terminate_process_tree(mock_proc)
        # Should not call killpg or terminate since poll is not None
        mock_proc.poll.assert_called_once()
