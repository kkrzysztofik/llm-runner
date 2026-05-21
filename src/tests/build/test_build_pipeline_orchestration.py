"""Additional BuildPipeline orchestration edge-case tests.

Covers:
- _cancelled_progress / _cancelled_build_result
- _emit_retry_wait / _run_with_retry (success first, success after retry, exhausted, cancel during wait)
- _sleep_until_cancel_or_timeout
- _run_stage_batch (all succeed, first fails, cancel before stage)
- run() (success path, configure failure, build failure, cancel after build, finalize failure, exception)
- run_both_backends (sequential results)
- lock management
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.build_pipeline import BuildBackend, BuildConfig, BuildProgress, BuildResult
from llama_manager.build_pipeline._context import _BuildContext
from llama_manager.build_pipeline.pipeline import BuildPipeline

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pipeline(
    tmp_path: Path, *, dry_run: bool = True, cancel_event=None, **overrides
) -> BuildPipeline:
    """Create a BuildPipeline with dry_run=True to avoid real subprocesses."""
    kwargs: dict = {
        "backend": BuildBackend.SYCL,
        "source_dir": tmp_path / "source",
        "build_dir": tmp_path / "build",
        "output_dir": tmp_path / "output",
        "git_remote_url": "https://github.com/ggerganov/llama.cpp",
        "git_branch": "main",
        "retry_attempts": 3,
        "retry_delay": 0.01,
        "build_timeout_seconds": 30,
    }
    kwargs.update(overrides)
    config = BuildConfig(**kwargs)
    pipeline = BuildPipeline(config, cancel_event=cancel_event)
    if dry_run:
        pipeline.dry_run = True
    return pipeline


# ── _cancelled_progress / _cancelled_build_result ────────────────────────────


class TestCancelledStatus:
    """Tests for _cancelled_progress and _cancelled_build_result helpers."""

    def test_cancelled_progress_basic(self) -> None:
        pipeline = _make_pipeline(Path("/tmp"))
        result = pipeline._cancelled_progress("configure")
        assert result.status == "failed"
        assert result.stage == "configure"
        assert result.progress_percent == 0.0

    def test_cancelled_build_result(self) -> None:
        pipeline = _make_pipeline(Path("/tmp"))
        result = pipeline._cancelled_build_result()
        assert result.success is False
        assert "cancelled" in result.error_message.lower()


# ── _emit_retry_wait ────────────────────────────────────────────────────────


class TestEmitRetryWait:
    """Tests for _emit_retry_wait."""

    def test_emit_retry_wait_no_cancel(self) -> None:
        """Should return None when not cancelled."""
        pipeline = _make_pipeline(Path("/tmp"), retry_delay=0.01)
        result = pipeline._emit_retry_wait("build", 0, 0.01)
        assert result is None

    def test_emit_retry_wait_calls_callback(self) -> None:
        """Should emit retry progress via callback."""
        received: list[BuildProgress] = []
        pipeline = _make_pipeline(Path("/tmp"), retry_delay=0.01)
        pipeline._progress_callback = received.append
        result = pipeline._emit_retry_wait("configure", 0, 0.01)
        assert result is None
        assert len(received) == 1
        assert received[0].status == "retrying"
        assert "retrying" in received[0].message

    def test_emit_retry_wait_cancelled(self) -> None:
        """Should return cancelled progress when cancel is set during wait."""
        cancel_event = threading.Event()
        pipeline = _make_pipeline(Path("/tmp"), retry_delay=0.01, cancel_event=cancel_event)
        cancel_event.set()
        result = pipeline._emit_retry_wait("build", 0, 0.01)
        assert result is not None
        assert result.status == "failed"
        assert "cancelled" in result.message.lower()


# ── _run_with_retry ─────────────────────────────────────────────────────────


class TestRunWithRetry:
    """Tests for _run_with_retry."""

    def test_retry_success_first_attempt(self) -> None:
        """Should return immediately on success."""
        pipeline = _make_pipeline(Path("/tmp"), retry_attempts=3)
        call_count = 0

        def stage_fn():
            nonlocal call_count
            call_count += 1
            return BuildProgress(stage="build", status="success", message="OK", progress_percent=75)

        result = pipeline._run_with_retry(stage_fn, "build")
        assert result.status == "success"
        assert call_count == 1

    def test_retry_success_after_one_failure(self) -> None:
        """Should retry and succeed on second attempt."""
        pipeline = _make_pipeline(Path("/tmp"), retry_attempts=3)
        call_count = 0

        def stage_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return BuildProgress(
                    stage="build", status="failed", message="transient", progress_percent=0
                )
            return BuildProgress(stage="build", status="success", message="OK", progress_percent=75)

        result = pipeline._run_with_retry(stage_fn, "build")
        assert result.status == "success"
        assert call_count == 2

    def test_retry_exhausted_all_attempts(self) -> None:
        """Should return last failure after exhausting retries."""
        pipeline = _make_pipeline(Path("/tmp"), retry_attempts=2)
        call_count = 0

        def stage_fn():
            nonlocal call_count
            call_count += 1
            return BuildProgress(
                stage="build", status="failed", message="always fails", progress_percent=0
            )

        result = pipeline._run_with_retry(stage_fn, "build")
        assert result.status == "failed"
        assert result.message == "always fails"
        assert call_count == 2

    def test_retry_cancel_during_wait(self) -> None:
        """Should stop retrying when cancel set during wait."""
        cancel_event = threading.Event()
        pipeline = _make_pipeline(
            Path("/tmp"), retry_attempts=3, retry_delay=0.01, cancel_event=cancel_event
        )
        call_count = 0

        def stage_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                cancel_event.set()
            return BuildProgress(stage="build", status="failed", message="fail", progress_percent=0)

        result = pipeline._run_with_retry(stage_fn, "build")
        assert result.status == "failed"
        assert "cancelled" in result.message.lower()

    def test_retry_cancel_before_first_attempt(self) -> None:
        """Should return cancelled immediately if cancel already set."""
        cancel_event = threading.Event()
        cancel_event.set()
        pipeline = _make_pipeline(Path("/tmp"), retry_attempts=3, cancel_event=cancel_event)
        call_count = 0

        def stage_fn():
            nonlocal call_count
            call_count += 1
            return BuildProgress(stage="build", status="success", message="OK", progress_percent=75)

        result = pipeline._run_with_retry(stage_fn, "build")
        assert result.status == "failed"
        assert "cancelled" in result.message.lower()
        assert call_count == 0  # stage never called

    def test_retry_skipped_status_returns_immediately(self) -> None:
        """A 'skipped' status should also stop retrying."""
        pipeline = _make_pipeline(Path("/tmp"), retry_attempts=3)

        def stage_fn():
            return BuildProgress(
                stage="configure",
                status="skipped",
                message="Already configured",
                progress_percent=50,
            )

        result = pipeline._run_with_retry(stage_fn, "configure")
        assert result.status == "skipped"

    def test_retry_stops_on_cancelled_message(self) -> None:
        """If stage returns BUILD_CANCELLED_MESSAGE, retry stops."""
        from llama_manager.build_pipeline.pipeline import BUILD_CANCELLED_MESSAGE

        pipeline = _make_pipeline(Path("/tmp"), retry_attempts=3)

        def stage_fn():
            return BuildProgress(
                stage="build",
                status="failed",
                message=BUILD_CANCELLED_MESSAGE,
                progress_percent=0,
            )

        result = pipeline._run_with_retry(stage_fn, "build")
        assert result.status == "failed"
        assert result.message == BUILD_CANCELLED_MESSAGE


# ── _sleep_until_cancel_or_timeout ──────────────────────────────────────────


class TestSleepUntilCancelOrTimeout:
    """Tests for _sleep_until_cancel_or_timeout."""

    def test_sleep_returns_false_when_no_cancel(self) -> None:
        pipeline = _make_pipeline(Path("/tmp"))
        result = pipeline._sleep_until_cancel_or_timeout(0.01)
        assert result is False

    def test_sleep_returns_true_when_cancel_set(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        pipeline = _make_pipeline(Path("/tmp"), cancel_event=cancel_event)
        result = pipeline._sleep_until_cancel_or_timeout(10.0)
        assert result is True


# ── _run_stage_batch ────────────────────────────────────────────────────────


class TestRunStageBatch:
    """Tests for _run_stage_batch."""

    def test_batch_all_succeed(self) -> None:
        """All stages succeed → returns (None, progress)."""
        pipeline = _make_pipeline(Path("/tmp"))
        ctx = MagicMock(spec=_BuildContext)
        stages = [
            (
                "stage1",
                lambda: BuildProgress(
                    stage="stage1", status="success", message="OK", progress_percent=30
                ),
            ),
            (
                "stage2",
                lambda: BuildProgress(
                    stage="stage2", status="success", message="OK", progress_percent=60
                ),
            ),
        ]
        progress = BuildProgress(stage="pipeline", status="pending", message="", progress_percent=0)
        error, result_progress = pipeline._run_stage_batch(
            ctx, stages, write_artifact=True, progress=progress
        )
        assert error is None
        assert result_progress.status == "success"

    def test_batch_first_stage_fails(self) -> None:
        """First stage fails → returns error result, second stage not called."""
        pipeline = _make_pipeline(Path("/tmp"))
        ctx = MagicMock(spec=_BuildContext)
        call_order: list[str] = []

        def stage1():
            call_order.append("stage1")
            return BuildProgress(
                stage="stage1", status="failed", message="boom", progress_percent=0
            )

        def stage2():
            call_order.append("stage2")
            return BuildProgress(
                stage="stage2", status="success", message="OK", progress_percent=60
            )

        stages = [("stage1", stage1), ("stage2", stage2)]
        progress = BuildProgress(stage="pipeline", status="pending", message="", progress_percent=0)
        error, result_progress = pipeline._run_stage_batch(
            ctx, stages, write_artifact=True, progress=progress
        )
        assert error is not None
        assert error.success is False
        assert "Stage1 failed" in error.error_message
        assert "stage2" not in call_order  # second stage not called

    def test_batch_cancel_before_stage(self) -> None:
        """Cancel set → returns cancelled result immediately."""
        cancel_event = threading.Event()
        cancel_event.set()
        pipeline = _make_pipeline(Path("/tmp"), cancel_event=cancel_event)
        ctx = MagicMock(spec=_BuildContext)
        call_count = 0

        def stage_fn():
            nonlocal call_count
            call_count += 1
            return BuildProgress(stage="stage", status="success", message="OK", progress_percent=30)

        stages = [("stage", stage_fn)]
        progress = BuildProgress(stage="pipeline", status="pending", message="", progress_percent=0)
        error, result_progress = pipeline._run_stage_batch(
            ctx, stages, write_artifact=True, progress=progress
        )
        assert error is not None
        assert error.success is False
        assert call_count == 0  # stage not called


# ── run() ───────────────────────────────────────────────────────────────────


class TestPipelineRun:
    """Tests for BuildPipeline.run()."""

    def test_run_both_backend_raises(self, tmp_path: Path) -> None:
        """run() should raise ValueError for BOTH backend."""
        config = BuildConfig(
            backend=BuildBackend.BOTH,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        pipeline = BuildPipeline(config)
        with pytest.raises(ValueError, match="BOTH"):
            pipeline.run()

    def test_run_preflight_failure(self, tmp_path: Path) -> None:
        """Preflight failure returns error result without running clone."""
        with (
            patch(
                "llama_manager.build_pipeline.pipeline.run_preflight",
                return_value=BuildProgress(
                    stage="preflight", status="failed", message="no GPU found", progress_percent=0
                ),
            ),
        ):
            pipeline = _make_pipeline(tmp_path)
            result = pipeline.run()
        assert result.success is False
        assert "Preflight failed" in result.error_message

    def test_run_clone_failure(self, tmp_path: Path) -> None:
        """Clone failure returns error result after preflight succeeds."""
        with (
            patch(
                "llama_manager.build_pipeline.pipeline.run_preflight",
                return_value=BuildProgress(
                    stage="preflight", status="success", message="OK", progress_percent=10
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_clone",
                return_value=BuildProgress(
                    stage="clone", status="failed", message="git error", progress_percent=0
                ),
            ),
        ):
            pipeline = _make_pipeline(tmp_path)
            result = pipeline.run()
        assert result.success is False
        assert "Clone failed" in result.error_message

    def test_run_configure_failure(self, tmp_path: Path) -> None:
        """Configure failure in run() returns error result."""
        with (
            patch(
                "llama_manager.build_pipeline.pipeline.run_preflight",
                return_value=BuildProgress(
                    stage="preflight", status="success", message="OK", progress_percent=10
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_clone",
                return_value=BuildProgress(
                    stage="clone", status="success", message="OK", progress_percent=30
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_configure",
                return_value=BuildProgress(
                    stage="configure", status="failed", message="cmake error", progress_percent=0
                ),
            ),
        ):
            pipeline = _make_pipeline(tmp_path)
            result = pipeline.run()
        assert result.success is False
        assert "Configure failed" in result.error_message

    def test_run_build_failure(self, tmp_path: Path) -> None:
        """Build failure in run() returns error result."""
        with (
            patch(
                "llama_manager.build_pipeline.pipeline.run_preflight",
                return_value=BuildProgress(
                    stage="preflight", status="success", message="OK", progress_percent=10
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_clone",
                return_value=BuildProgress(
                    stage="clone", status="success", message="OK", progress_percent=30
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_configure",
                return_value=BuildProgress(
                    stage="configure", status="success", message="OK", progress_percent=50
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_build",
                return_value=BuildProgress(
                    stage="build", status="failed", message="link error", progress_percent=0
                ),
            ),
        ):
            pipeline = _make_pipeline(tmp_path)
            result = pipeline.run()
        assert result.success is False
        assert "Build failed" in result.error_message

    def test_run_cancel_after_build(self, tmp_path: Path) -> None:
        """Cancel after build succeeds → returns cancelled result."""
        cancel_event = threading.Event()
        with (
            patch(
                "llama_manager.build_pipeline.pipeline.run_preflight",
                return_value=BuildProgress(
                    stage="preflight", status="success", message="OK", progress_percent=10
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_clone",
                return_value=BuildProgress(
                    stage="clone", status="success", message="OK", progress_percent=30
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_configure",
                return_value=BuildProgress(
                    stage="configure", status="success", message="OK", progress_percent=50
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_build",
                return_value=BuildProgress(
                    stage="build", status="success", message="OK", progress_percent=75
                ),
            ),
            patch("llama_manager.build_pipeline.pipeline.run_finalize", return_value=MagicMock()),
        ):
            pipeline = _make_pipeline(tmp_path, cancel_event=cancel_event)
            cancel_event.set()
            result = pipeline.run()
        assert result.success is False
        assert "cancelled" in result.error_message.lower()

    def test_run_finalize_failure(self, tmp_path: Path) -> None:
        """Finalize failure → returns error result."""
        with (
            patch(
                "llama_manager.build_pipeline.pipeline.run_preflight",
                return_value=BuildProgress(
                    stage="preflight", status="success", message="OK", progress_percent=10
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_clone",
                return_value=BuildProgress(
                    stage="clone", status="success", message="OK", progress_percent=30
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_configure",
                return_value=BuildProgress(
                    stage="configure", status="success", message="OK", progress_percent=50
                ),
            ),
            patch(
                "llama_manager.build_pipeline.pipeline.run_build",
                return_value=BuildProgress(
                    stage="build", status="success", message="OK", progress_percent=75
                ),
            ),
            patch("llama_manager.build_pipeline.pipeline.run_finalize", return_value=None),
        ):
            pipeline = _make_pipeline(tmp_path)
            result = pipeline.run()
        assert result.success is False
        assert "Failed to write provenance" in result.error_message

    def test_run_exception_path(self, tmp_path: Path) -> None:
        """Exception during run() → returns error with failure artifact."""
        with patch(
            "llama_manager.build_pipeline.pipeline.run_preflight",
            side_effect=RuntimeError("boom"),
        ):
            pipeline = _make_pipeline(tmp_path)
            result = pipeline.run()
        assert result.success is False
        assert "boom" in result.error_message


# ── run_both_backends ───────────────────────────────────────────────────────


class TestRunBothBackends:
    """Tests for BuildPipeline.run_both_backends()."""

    def test_both_backends_runs_sequentially(self) -> None:
        """Should run SYCL then CUDA sequentially."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=Path("/tmp/source"),
            build_dir=Path("/tmp/build"),
            output_dir=Path("/tmp/output"),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        pipeline = BuildPipeline(config)
        pipeline.dry_run = True
        results = pipeline.run_both_backends()
        assert len(results) == 2
        assert results[0].success is True  # SYCL dry-run succeeds
        assert results[1].success is True  # CUDA dry-run succeeds

    def test_both_backends_first_fails(self) -> None:
        """First backend failure does not prevent second from running."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=Path("/tmp/source"),
            build_dir=Path("/tmp/build"),
            output_dir=Path("/tmp/output"),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        # Mock run to always fail
        with patch.object(
            BuildPipeline, "run", return_value=BuildResult(success=False, error_message="fail")
        ):
            pipeline = BuildPipeline(config)
            pipeline.dry_run = True
            results = pipeline.run_both_backends()
        assert len(results) == 2
        assert results[0].success is False
        assert results[1].success is False


# ── Lock management ─────────────────────────────────────────────────────────


class TestLockManagement:
    """Tests for pipeline lock management methods."""

    def test_acquire_lock_sets_lock_file(self, tmp_path: Path) -> None:
        """Successful lock acquisition sets _lock_file."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        lock_path = tmp_path / "lock.json"
        pipeline = BuildPipeline(config)
        pipeline.dry_run = False
        result = pipeline._acquire_lock(lock_path)
        assert result is True
        assert pipeline._lock_file == lock_path

    def test_acquire_lock_dry_run_no_file(self, tmp_path: Path) -> None:
        """Dry-run acquire_lock returns True but does not set _lock_file."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        lock_path = tmp_path / "lock.json"
        pipeline = BuildPipeline(config)
        pipeline.dry_run = True
        result = pipeline._acquire_lock(lock_path)
        assert result is True
        assert pipeline._lock_file is None

    def test_release_lock_clears_file(self, tmp_path: Path) -> None:
        """release_lock clears _lock_file."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        lock_path = tmp_path / "lock.json"
        pipeline = BuildPipeline(config)
        pipeline._lock_file = lock_path
        pipeline.release_lock()
        assert pipeline._lock_file is None

    def test_is_lock_stale_delegates(self, tmp_path: Path) -> None:
        """_is_lock_stale delegates to is_lock_stale."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        pipeline = BuildPipeline(config)
        lock_path = tmp_path / "lock.json"
        # Missing file → stale
        assert pipeline._is_lock_stale(lock_path) is True

    def test_get_lock_error_message_delegates(self, tmp_path: Path) -> None:
        """_get_lock_error_message delegates to get_lock_error_message."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        pipeline = BuildPipeline(config)
        lock_path = tmp_path / "lock.json"
        msg = pipeline._get_lock_error_message(lock_path)
        assert "could not be read" in msg
