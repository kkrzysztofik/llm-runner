"""Tests for run_build_for_backend orchestration.

Tests:
- Config overrides merging
- Backend-scoped output directory
- Derived field preservation
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.build_pipeline import BuildBackend, BuildConfig, BuildProgress, BuildResult
from llama_manager.build_pipeline.orchestration import (
    _merge_config_overrides,
    run_build_for_backend,
)
from llama_manager.config import BuildPipelineConfig, Config, PathsConfig

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_app_config(tmp_path: Path) -> Config:
    """Create an isolated Config for orchestration tests."""
    return Config(
        paths=PathsConfig(
            llama_cpp_root=str(tmp_path / "llama.cpp"),
            xdg_state_base=str(tmp_path),
            xdg_cache_base=str(tmp_path),
            xdg_data_base=str(tmp_path),
        ),
        build=BuildPipelineConfig(),
    )


# ── _merge_config_overrides tests ────────────────────────────────────────────


class TestMergeConfigOverrides:
    """Tests for _merge_config_overrides helper."""

    def test_config_overrides_applies_git_branch(self, tmp_path: Path) -> None:
        """_merge_config_overrides should apply git_branch from overrides."""
        base = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
        )
        overrides = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "override_source",
            build_dir=tmp_path / "override_build",
            output_dir=tmp_path / "override_output",
            git_remote_url="https://github.com/other/repo.git",
            git_branch="my-branch",
        )

        merged = _merge_config_overrides(base, overrides)

        assert merged.git_branch == "my-branch"
        assert merged.git_remote_url == "https://github.com/other/repo.git"

    def test_config_overrides_preserves_derived_fields(self, tmp_path: Path) -> None:
        """_merge_config_overrides should NOT override derived fields."""
        base = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "base_source",
            build_dir=tmp_path / "base_build",
            output_dir=tmp_path / "base_output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
        )
        overrides = BuildConfig(
            backend=BuildBackend.CUDA,  # Different backend
            source_dir=tmp_path / "override_source",
            build_dir=tmp_path / "override_build",
            output_dir=tmp_path / "override_output",
            git_remote_url="https://github.com/other/repo.git",
            git_branch="other-branch",
        )

        merged = _merge_config_overrides(base, overrides)

        # Derived fields should come from base, NOT overrides
        assert merged.backend == BuildBackend.SYCL
        assert merged.source_dir == tmp_path / "base_source"
        assert merged.build_dir == tmp_path / "base_build"
        assert merged.output_dir == tmp_path / "base_output"

    def test_config_overrides_applies_retry_fields(self, tmp_path: Path) -> None:
        """_merge_config_overrides should apply retry_attempts and retry_delay."""
        base = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            retry_attempts=3,
            retry_delay=5.0,
        )
        overrides = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            retry_attempts=10,
            retry_delay=30.0,
        )

        merged = _merge_config_overrides(base, overrides)

        assert merged.retry_attempts == 10
        assert merged.retry_delay == 30.0

    def test_config_overrides_applies_shallow_clone_and_jobs(self, tmp_path: Path) -> None:
        """_merge_config_overrides should apply shallow_clone and jobs."""
        base = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            shallow_clone=True,
            jobs=4,
        )
        overrides = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            shallow_clone=False,
            jobs=8,
        )

        merged = _merge_config_overrides(base, overrides)

        assert merged.shallow_clone is False
        assert merged.jobs == 8

    def test_config_overrides_applies_git_commit(self, tmp_path: Path) -> None:
        """_merge_config_overrides should apply git_commit override."""
        base = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            git_commit=None,
        )
        overrides = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            git_commit="abcdef1234567890abcdef1234567890abcdef12",
        )

        merged = _merge_config_overrides(base, overrides)

        assert merged.git_commit == "abcdef1234567890abcdef1234567890abcdef12"

    def test_config_overrides_applies_build_timeout(self, tmp_path: Path) -> None:
        """_merge_config_overrides should apply build_timeout_seconds override."""
        base = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            build_timeout_seconds=3600,
        )
        overrides = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            build_timeout_seconds=7200,
        )

        merged = _merge_config_overrides(base, overrides)

        assert merged.build_timeout_seconds == 7200

    def test_config_overrides_applies_update_sources(self, tmp_path: Path) -> None:
        """_merge_config_overrides should apply update_sources override."""
        base = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            update_sources=True,
        )
        overrides = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="master",
            update_sources=False,
        )

        merged = _merge_config_overrides(base, overrides)

        assert merged.update_sources is False

    def test_config_overrides_preserves_flavor_git_remote_when_override_empty(
        self, tmp_path: Path
    ) -> None:
        """TUI overrides use empty git_remote_url to keep flavor-resolved base URL."""
        base = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/Anbeeld/beellama.cpp.git",
            git_branch="main",
        )
        overrides = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "ignored",
            build_dir=tmp_path / "ignored",
            output_dir=tmp_path / "ignored",
            git_remote_url="",
            git_branch="main",
            jobs=8,
        )

        merged = _merge_config_overrides(base, overrides)

        assert merged.git_remote_url == "https://github.com/Anbeeld/beellama.cpp.git"
        assert merged.git_branch == "main"
        assert merged.jobs == 8


# ── run_build_for_backend tests ──────────────────────────────────────────────


class TestRunBuildForBackend:
    """Tests for run_build_for_backend orchestration."""

    def test_output_dir_is_backend_scoped_sycl(self, tmp_path: Path) -> None:
        """run_build_for_backend should set output_dir to builds_dir/sycl for sycl."""
        config = _make_app_config(tmp_path)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        progress_log: list[BuildProgress] = []

        def _capture_progress(p: BuildProgress) -> None:
            progress_log.append(p)

        with (
            patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls,
        ):
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(
                success=True,
                progress=BuildProgress(
                    stage="finalize", status="success", message="Done", progress_percent=100
                ),
            )

            run_build_for_backend(
                "sycl",
                dry_run=True,
                config=config,
                progress_callback=_capture_progress,
            )

        # Verify BuildPipeline was created with correct config
        call_kwargs = mock_pipeline_cls.call_args[0][0]
        assert call_kwargs.output_dir == config.paths.builds_dir / "sycl"
        assert call_kwargs.backend == BuildBackend.SYCL

    def test_output_dir_is_backend_scoped_cuda(self, tmp_path: Path) -> None:
        """run_build_for_backend should set output_dir to builds_dir/cuda for cuda."""
        config = _make_app_config(tmp_path)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        with (
            patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls,
        ):
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(
                success=True,
                progress=BuildProgress(
                    stage="finalize", status="success", message="Done", progress_percent=100
                ),
            )

            run_build_for_backend(
                "cuda",
                dry_run=True,
                config=config,
            )

        call_kwargs = mock_pipeline_cls.call_args[0][0]
        assert call_kwargs.output_dir == config.paths.builds_dir / "cuda"
        assert call_kwargs.backend == BuildBackend.CUDA

    def test_source_flavor_resolves_default_git_remote_and_branch(self, tmp_path: Path) -> None:
        """source_flavor should select its remote and branch when no manual override is set."""
        config = _make_app_config(tmp_path)
        config.build.source_flavor = "beellama"
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend("cuda", dry_run=True, config=config)

        call_kwargs = mock_pipeline_cls.call_args[0][0]
        assert call_kwargs.git_remote_url == "https://github.com/Anbeeld/beellama.cpp.git"
        assert call_kwargs.git_branch == "main"

    def test_build_dir_is_correct_for_backend(self, tmp_path: Path) -> None:
        """run_build_for_backend should set correct build_dir for each backend."""
        config = _make_app_config(tmp_path)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend("sycl", dry_run=True, config=config)

            call_kwargs = mock_cls.call_args[0][0]
            # SYCL build_dir: source_dir/build
            assert call_kwargs.build_dir == Path(config.paths.llama_cpp_root) / "build"

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend("cuda", dry_run=True, config=config)

            call_kwargs = mock_cls.call_args[0][0]
            # CUDA build_dir: source_dir/build_cuda
            assert call_kwargs.build_dir == Path(config.paths.llama_cpp_root) / "build_cuda"

    def test_config_overrides_passed_to_pipeline(self, tmp_path: Path) -> None:
        """run_build_for_backend should merge config_overrides before creating pipeline."""
        config = _make_app_config(tmp_path)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        overrides = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "fake",
            build_dir=tmp_path / "fake",
            output_dir=tmp_path / "fake",
            git_remote_url="https://github.com/ggerganov/llama.cpp.git",
            git_branch="custom-branch",
            retry_attempts=1,
        )

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend(
                "sycl",
                dry_run=True,
                config=config,
                config_overrides=overrides,
            )

        # The merged config should have the overridden git_branch
        call_kwargs = mock_cls.call_args[0][0]
        assert call_kwargs.git_branch == "custom-branch"
        assert call_kwargs.retry_attempts == 1
        # But derived fields should NOT be overridden
        assert call_kwargs.source_dir == Path(config.paths.llama_cpp_root)
        assert call_kwargs.backend == BuildBackend.SYCL

    def test_unsupported_backend_raises(self, tmp_path: Path) -> None:
        """run_build_for_backend should raise ValueError for unsupported backend."""
        config = _make_app_config(tmp_path)

        with pytest.raises(ValueError, match="unsupported backend"):
            run_build_for_backend("invalid", dry_run=True, config=config)

    def test_pipeline_callback_invoked(self, tmp_path: Path) -> None:
        """run_build_for_backend should invoke pipeline_callback with the pipeline."""
        config = _make_app_config(tmp_path)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        received_pipeline: list = []

        def _callback(pipeline: object) -> None:
            received_pipeline.append(pipeline)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend(
                "sycl",
                dry_run=True,
                config=config,
                pipeline_callback=_callback,
            )

        assert len(received_pipeline) == 1
        assert received_pipeline[0] is mock_pipeline

    def test_dry_run_set_on_pipeline(self, tmp_path: Path) -> None:
        """run_build_for_backend should set dry_run on the pipeline."""
        config = _make_app_config(tmp_path)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend("sycl", dry_run=True, config=config)

        assert mock_pipeline.dry_run is True

    def test_non_dry_run_default(self, tmp_path: Path) -> None:
        """run_build_for_backend should default dry_run to False."""
        config = _make_app_config(tmp_path)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_cls:
            mock_pipeline = MagicMock()
            mock_cls.return_value = mock_pipeline
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend("sycl", config=config)

        assert mock_pipeline.dry_run is False


class TestBuildPipelineCancel:
    """Stop/cancel should terminate the active stage subprocess."""

    def test_kill_active_subprocess_terminates_tracked_proc(self, tmp_path: Path) -> None:
        from llama_manager.build_pipeline._context import _BuildContext
        from llama_manager.build_pipeline.pipeline import BuildPipeline

        build_config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        pipeline = BuildPipeline(build_config)
        ctx = _BuildContext(config=build_config, dry_run=False, build_start_time=0.0)
        active_proc = MagicMock()
        ctx.active_proc = active_proc
        pipeline._ctx = ctx

        with patch("llama_manager.build_pipeline.utils.terminate_process_tree") as mock_terminate:
            pipeline.kill_active_subprocess()

        mock_terminate.assert_called_once_with(active_proc, use_process_group=True)
