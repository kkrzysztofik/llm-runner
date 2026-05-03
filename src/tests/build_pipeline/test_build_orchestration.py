"""Characterization tests for build orchestration.

These tests lock in the behavior of ``run_build_for_backend`` by mocking
BuildPipeline.  They verify that the orchestration layer correctly:

* Translates backend strings into BuildBackend values
* Derives build directories from Config
* Passes dry_run and progress callbacks through
* Invokes an optional pipeline_callback before running
* Returns the BuildResult from the pipeline unchanged
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_manager.build_pipeline import BuildBackend, BuildConfig, BuildProgress, BuildResult
from llama_manager.build_pipeline.orchestration import run_build_for_backend


def _make_config(tmp_path: Path) -> MagicMock:
    """Create a mock Config with the attributes needed for build orchestration."""
    config = MagicMock()
    config.llama_cpp_root = str(tmp_path / "llama.cpp")
    config.builds_dir = tmp_path / "builds"
    config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
    config.build_git_branch = "main"
    config.build_retry_attempts = 2
    config.build_retry_delay = 5
    return config


class TestRunBuildForBackend:
    """Characterization tests for run_build_for_backend."""

    def test_success_returns_true_result(self, tmp_path: Path) -> None:
        """Should return the successful BuildResult from the pipeline."""
        config = _make_config(tmp_path)
        mock_result = BuildResult(success=True)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run.return_value = mock_result

            result = run_build_for_backend(
                backend="sycl",
                dry_run=False,
                config=config,
            )

        assert result is mock_result
        assert result.success is True

    def test_failure_returns_false_result(self, tmp_path: Path) -> None:
        """Should return the failed BuildResult from the pipeline."""
        config = _make_config(tmp_path)
        mock_result = BuildResult(success=False, error_message="compile error")

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run.return_value = mock_result

            result = run_build_for_backend(
                backend="sycl",
                dry_run=False,
                config=config,
            )

        assert result is mock_result
        assert result.success is False
        assert result.error_message == "compile error"

    def test_dry_run_set_on_pipeline(self, tmp_path: Path) -> None:
        """Should set dry_run=True on the pipeline instance."""
        config = _make_config(tmp_path)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend(
                backend="sycl",
                dry_run=True,
                config=config,
            )

        assert mock_pipeline.dry_run is True

    def test_sycl_backend_paths(self, tmp_path: Path) -> None:
        """Should use BuildBackend.SYCL and default build directory for sycl."""
        config = _make_config(tmp_path)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend(
                backend="sycl",
                dry_run=False,
                config=config,
            )

        call_args = mock_pipeline_cls.call_args
        build_config = call_args[0][0]
        assert isinstance(build_config, BuildConfig)
        assert build_config.backend == BuildBackend.SYCL
        assert build_config.build_dir.name == "build"
        assert build_config.source_dir == Path(config.llama_cpp_root)
        assert build_config.output_dir == config.builds_dir

    def test_cuda_backend_paths(self, tmp_path: Path) -> None:
        """Should use BuildBackend.CUDA and build_cuda directory for cuda."""
        config = _make_config(tmp_path)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend(
                backend="cuda",
                dry_run=False,
                config=config,
            )

        call_args = mock_pipeline_cls.call_args
        build_config = call_args[0][0]
        assert isinstance(build_config, BuildConfig)
        assert build_config.backend == BuildBackend.CUDA
        assert build_config.build_dir.name == "build_cuda"

    def test_progress_callback_passed(self, tmp_path: Path) -> None:
        """Should forward progress_callback to BuildPipeline."""
        config = _make_config(tmp_path)
        progress_calls: list[BuildProgress] = []

        def callback(progress: BuildProgress) -> None:
            progress_calls.append(progress)

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend(
                backend="sycl",
                dry_run=False,
                config=config,
                progress_callback=callback,
            )

        call_kwargs = mock_pipeline_cls.call_args.kwargs
        assert call_kwargs.get("progress_callback") is callback

    def test_pipeline_callback_invoked_before_run(self, tmp_path: Path) -> None:
        """Should invoke pipeline_callback before calling pipeline.run()."""
        config = _make_config(tmp_path)
        call_order: list[str] = []

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value

            def fake_run() -> BuildResult:
                call_order.append("run")
                return BuildResult(success=True)

            mock_pipeline.run.side_effect = fake_run

            def pipeline_callback(pipeline: object) -> None:
                call_order.append("callback")
                assert pipeline is mock_pipeline
                assert not mock_pipeline.run.called

            run_build_for_backend(
                backend="sycl",
                dry_run=False,
                config=config,
                pipeline_callback=pipeline_callback,
            )

        assert call_order == ["callback", "run"]

    def test_config_values_forwarded(self, tmp_path: Path) -> None:
        """Should forward git remote, branch, retry config into BuildConfig."""
        config = _make_config(tmp_path)
        config.build_git_remote = "https://example.com/repo.git"
        config.build_git_branch = "develop"
        config.build_retry_attempts = 5
        config.build_retry_delay = 10
        config.build_shallow_clone = True

        with patch("llama_manager.build_pipeline.orchestration.BuildPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.run.return_value = BuildResult(success=True)

            run_build_for_backend(
                backend="sycl",
                dry_run=False,
                config=config,
            )

        call_args = mock_pipeline_cls.call_args
        build_config = call_args[0][0]
        assert build_config.git_remote_url == "https://example.com/repo.git"
        assert build_config.git_branch == "develop"
        assert build_config.retry_attempts == 5
        assert build_config.retry_delay == 10
        assert build_config.shallow_clone is True
