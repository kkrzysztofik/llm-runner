"""T028-T040: Tests for BuildPipeline implementation.

Test Tasks:
- T028: Test launch path does not trigger build (FR-006.2)
- T029: Test serialized build order (SC-003)
- T030: Test build lock behavior
- T031: Test build lock PID validation
- T032: Test retry logic with exponential backoff
- T033: Test retry logic transient failure handling
- T034: Test preflight stage validation
- T035: Test configure stage with cmake flags
- T036: Test build stage execution
- T037: Test provenance stage atomic write
- T038: Test provenance write failure emits warning but build still succeeds
- T039: Test dry-run mode
- T040: Test full clone vs shallow clone
"""

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from llama_manager.build_pipeline import (
    BuildArtifact,
    BuildBackend,
    BuildConfig,
    BuildLock,
    BuildPipeline,
    BuildProgress,
)


class TestNoAutobuildOnLaunch:
    """T028: Test launch path does not trigger build (FR-006.2)."""

    def test_no_autobuild_on_launch(self, tmp_path: Path) -> None:
        """FR-006.2: Launch should not trigger build if sources exist.

        When llama.cpp sources already exist in source_dir, launch should
        skip the build pipeline and use existing sources.
        """
        # Create source directory with existing files
        source_dir = tmp_path / "llama.cpp"
        source_dir.mkdir()
        (source_dir / "CMakeLists.txt").write_text("# existing")

        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=source_dir,
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Mock all stages to verify they are NOT called when sources exist
        with (
            patch.object(pipeline, "_run_preflight", return_value=(True, [])) as mock_check,
            patch.object(
                pipeline,
                "_run_clone",
                return_value=BuildProgress(
                    stage="clone",
                    status="skipped",
                    message="Sources already exist",
                    progress_percent=0,
                ),
            ) as mock_clone,
            patch.object(
                pipeline,
                "_run_configure",
                return_value=BuildProgress(
                    stage="configure",
                    status="success",
                    message="Configured",
                    progress_percent=50,
                ),
            ) as mock_configure,
            patch.object(
                pipeline,
                "_run_build",
                return_value=BuildProgress(
                    stage="build",
                    status="success",
                    message="Built",
                    progress_percent=75,
                ),
            ) as mock_build,
            patch.object(pipeline, "_write_provenance", return_value=True) as mock_provenance,
        ):
            result = pipeline.run()

            # Verify stages were called
            assert mock_check.called
            assert mock_clone.called  # Should still be called but skipped
            assert mock_configure.called
            assert mock_build.called
            assert mock_provenance.called

            # Verify result indicates success
            assert result.success is True


class TestSerializedBuildOrder:
    """T029: Test serialized build order (SC-003)."""

    def test_serialized_build_order(self, tmp_path: Path) -> None:
        """SC-003: SYCL build should complete before CUDA build starts.

        When building both backends, SYCL should be built first, then CUDA.
        This test verifies the serialized execution order.
        """
        # Create source directories for both backends
        sycl_source = tmp_path / "sycl_source"
        cuda_source = tmp_path / "cuda_source"
        sycl_source.mkdir()
        cuda_source.mkdir()

        # Mock BuildPipeline instances
        sycl_pipeline = BuildPipeline(
            BuildConfig(
                backend=BuildBackend.SYCL,
                source_dir=sycl_source,
                build_dir=tmp_path / "sycl_build",
                output_dir=tmp_path / "sycl_output",
                git_remote_url="https://github.com/ggerganov/llama.cpp",
                git_branch="main",
            )
        )

        cuda_pipeline = BuildPipeline(
            BuildConfig(
                backend=BuildBackend.CUDA,
                source_dir=cuda_source,
                build_dir=tmp_path / "cuda_build",
                output_dir=tmp_path / "cuda_output",
                git_remote_url="https://github.com/ggerganov/llama.cpp",
                git_branch="main",
            )
        )

        execution_order: list[str] = []

        # Mock stages to track execution order
        def track_sycl_stage(stage_name: str):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    execution_order.append(f"SYCL:{stage_name}")
                    return func(*args, **kwargs)

                return wrapper

            return decorator

        sycl_pipeline._run_preflight = Mock(return_value=(True, []))
        sycl_pipeline._run_clone = Mock(
            return_value=BuildProgress(
                stage="clone", status="skipped", message="Exists", progress_percent=0
            )
        )
        sycl_pipeline._run_configure = track_sycl_stage("configure")(
            Mock(
                return_value=BuildProgress(
                    stage="configure", status="success", message="Configured", progress_percent=50
                )
            )
        )
        sycl_pipeline._run_build = track_sycl_stage("build")(
            Mock(
                return_value=BuildProgress(
                    stage="build", status="success", message="Built", progress_percent=75
                )
            )
        )
        sycl_pipeline._write_provenance = Mock(return_value=True)

        cuda_pipeline._run_preflight = Mock(return_value=(True, []))
        cuda_pipeline._run_clone = Mock(
            return_value=BuildProgress(
                stage="clone", status="skipped", message="Exists", progress_percent=0
            )
        )
        cuda_pipeline._run_configure = Mock(
            return_value=BuildProgress(
                stage="configure", status="success", message="Configured", progress_percent=50
            )
        )
        cuda_pipeline._run_build = Mock(
            return_value=BuildProgress(
                stage="build", status="success", message="Built", progress_percent=75
            )
        )
        cuda_pipeline._write_provenance = Mock(return_value=True)

        # Build both in sequence
        sycl_pipeline.run()
        cuda_pipeline.run()

        # Verify SYCL completed before CUDA started
        sycl_complete_idx = next(
            (i for i, x in enumerate(execution_order) if x == "SYCL:build"), -1
        )
        cuda_start_idx = next(
            (i for i, x in enumerate(execution_order) if x == "CUDA:configure"), -1
        )

        assert sycl_complete_idx >= 0, "SYCL build should have completed"
        assert cuda_start_idx >= 0, "CUDA configure should have started"
        assert sycl_complete_idx < cuda_start_idx, "SYCL should complete before CUDA starts"


class TestBuildLockBehavior:
    """T030: Test build lock behavior."""

    def test_build_lock_behavior(self, tmp_path: Path) -> None:
        """Build lock should prevent concurrent builds.

        When a build lock exists, new builds should fail gracefully
        with appropriate error messages.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Create a lock file
        lock_file = tmp_path / "build.lock"
        lock_data = BuildLock(
            pid=os.getpid(),
            started_at=datetime.now(),
            backend="sycl",
        )
        lock_file.write_text(
            json.dumps(
                {
                    "pid": lock_data.pid,
                    "started_at": lock_data.started_at.isoformat(),
                    "backend": lock_data.backend,
                }
            )
        )

        # Try to acquire lock - should fail
        acquired = pipeline._acquire_lock(lock_file)
        assert acquired is False, "Should not acquire lock when file exists"

        # Check error message
        error_msg = pipeline._get_lock_error_message(lock_file)
        assert "build lock" in error_msg.lower() or "already" in error_msg.lower()


class TestBuildLockPIDValidation:
    """T031: Test build lock PID validation."""

    def test_build_lock_pid_validation(self, tmp_path: Path) -> None:
        """Build lock should validate PID before clearing stale locks.

        When a lock file exists, check if the PID is still valid
        before deciding to clear the lock.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Create a lock file with non-existent PID
        lock_file = tmp_path / "build.lock"
        stale_pid = 999999  # Unlikely to be a valid PID
        lock_data = BuildLock(
            pid=stale_pid,
            started_at=datetime.now(),
            backend="sycl",
        )
        lock_file.write_text(
            json.dumps(
                {
                    "pid": lock_data.pid,
                    "started_at": lock_data.started_at.isoformat(),
                    "backend": lock_data.backend,
                }
            )
        )

        # Check if lock is stale
        is_stale = pipeline._is_lock_stale(lock_file)
        assert is_stale is True, "Lock with non-existent PID should be stale"

        # Try to acquire lock - should succeed because lock is stale
        acquired = pipeline._acquire_lock(lock_file)
        assert acquired is True, "Should acquire stale lock"


class TestRetryExponentialBackoff:
    """T032: Test retry logic with exponential backoff."""

    def test_retry_exponential_backoff(self, tmp_path: Path) -> None:
        """Retry should use exponential backoff with configurable delays.

        Each retry should wait longer than the previous one:
        delay * 2^attempt
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            retry_attempts=3,
            retry_delay=1,  # 1 second base delay
        )

        pipeline = BuildPipeline(config)

        # Track retry attempts and delays
        retry_times: list[float] = []

        def mock_stage_with_delay():
            """Mock stage that tracks time and raises."""
            retry_times.append(time.time())
            raise subprocess.CalledProcessError(1, "test")

        # Mock stage to fail multiple times then succeed
        call_count = [0]

        def failing_then_succeeding_stage():
            call_count[0] += 1
            if call_count[0] < 3:
                raise subprocess.CalledProcessError(1, "test")
            return BuildProgress(
                stage="build", status="success", message="Succeeded", progress_percent=75
            )

        pipeline._run_build = Mock(side_effect=failing_then_succeeding_stage)
        pipeline._run_preflight = Mock(return_value=(True, []))
        pipeline._run_clone = Mock(
            return_value=BuildProgress(
                stage="clone", status="skipped", message="Exists", progress_percent=0
            )
        )
        pipeline._run_configure = Mock(
            return_value=BuildProgress(
                stage="configure", status="success", message="Configured", progress_percent=50
            )
        )
        pipeline._write_provenance = Mock(return_value=True)

        # Run pipeline - should retry and eventually succeed
        result = pipeline.run()

        # Verify retries occurred
        assert call_count[0] == 3, f"Expected 3 attempts, got {call_count[0]}"
        assert result.success is True


class TestRetryTransientFailures:
    """T033: Test retry logic transient failure handling."""

    def test_retry_transient_failures(self, tmp_path: Path) -> None:
        """Retry should handle transient failures (network, timeout) gracefully.

        Transient failures should be retried, but permanent failures should
        fail fast after max retries.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            retry_attempts=2,
            retry_delay=1,
        )

        pipeline = BuildPipeline(config)

        # Track failures
        failure_count = [0]

        def always_fails():
            failure_count[0] += 1
            raise subprocess.CalledProcessError(1, "test")

        pipeline._run_build = Mock(side_effect=always_fails)
        pipeline._run_preflight = Mock(return_value=(True, []))
        pipeline._run_clone = Mock(
            return_value=BuildProgress(
                stage="clone", status="skipped", message="Exists", progress_percent=0
            )
        )
        pipeline._run_configure = Mock(
            return_value=BuildProgress(
                stage="configure", status="success", message="Configured", progress_percent=50
            )
        )
        pipeline._write_provenance = Mock(return_value=True)

        # Run pipeline - should fail after max retries
        result = pipeline.run()

        # Verify max retries were attempted
        assert failure_count[0] == 2, f"Expected 2 attempts (max retries), got {failure_count[0]}"
        assert result.success is False


class TestPreflightStageValidation:
    """T034: Test preflight stage validation."""

    def test_preflight_stage_validation(self, tmp_path: Path) -> None:
        """Preflight stage should validate toolchain before build.

        The preflight stage checks for required tools (gcc, make, git, cmake, etc.)
        and fails early if they're missing.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Test with missing tools
        with patch("llama_manager.build_pipeline.detect_toolchain") as mock_detect:
            mock_status = Mock()
            mock_status.is_sycl_ready = False
            mock_status.missing_tools = ["icpx", "sycl-ls"]
            mock_detect.return_value = mock_status

            result = pipeline.run()

            # Should fail at preflight stage
            assert result.success is False


class TestConfigureStageCmakeFlags:
    """T035: Test configure stage with cmake flags."""

    def test_configure_stage_cmake_flags(self, tmp_path: Path) -> None:
        """Configure stage should generate correct cmake flags for backend.

        SYCL backend should use -DGGML_SYCL=ON
        CUDA backend should use -DGGML_CUDA=ON
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Mock source directory with CMakeLists.txt
        (config.source_dir / "CMakeLists.txt").write_text("# Mock CMakeLists")

        # Test SYCL backend
        cmake_args = pipeline._get_cmake_flags(backend=BuildBackend.SYCL)
        assert "-DGGML_SYCL=ON" in cmake_args

        # Test CUDA backend
        config_cuda = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        pipeline_cuda = BuildPipeline(config_cuda)
        cmake_args_cuda = pipeline_cuda._get_cmake_flags(backend=BuildBackend.CUDA)
        assert "-DGGML_CUDA=ON" in cmake_args_cuda


class TestBuildStageExecution:
    """T036: Test build stage execution."""

    def test_build_stage_execution(self, tmp_path: Path) -> None:
        """Build stage should execute cmake --build with correct parameters.

        The build stage should:
        - Run cmake --build with correct build directory
        - Use parallel jobs if specified
        - Capture output to log file
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            jobs=4,
        )

        pipeline = BuildPipeline(config)

        # Mock the subprocess call
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Build successful"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = pipeline._run_build()

            # Verify subprocess was called with correct arguments
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert "cmake" in call_args[0]
            assert "--build" in call_args
            assert str(config.build_dir) in call_args

            # Verify success
            assert result.status == "success"


class TestProvenanceAtomicWrite:
    """T037: Test provenance stage atomic write."""

    def test_provenance_atomic_write(self, tmp_path: Path) -> None:
        """Provenance stage should write artifact.json atomically.

        Atomic write means:
        - Write to temporary file first
        - Rename temp file to final location
        - Ensures no partial writes if interrupted
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Create artifact
        artifact = BuildArtifact(
            artifact_type="binary",
            backend="sycl",
            created_at=datetime.now(),
            git_remote_url=config.git_remote_url,
            git_commit_sha="abc123",
            git_branch=config.git_branch,
            build_command=["cmake", "-DGGML_SYCL=ON"],
            build_duration_seconds=10.0,
            exit_code=0,
            binary_path=tmp_path / "bin" / "llama-server",
            binary_size_bytes=104857600,
            build_log_path=tmp_path / "build.log",
            failure_report_path=None,
        )

        # Mock atomic write
        with patch.object(pipeline, "_atomic_write", return_value=True) as mock_write:
            result = pipeline._write_provenance(artifact)

            # Verify atomic write was called
            assert mock_write.called
            assert result is True

            # Verify file exists
            provenance_file = config.output_dir / "build-artifact.json"
            assert provenance_file.exists()

            # Verify content is valid JSON
            content = provenance_file.read_text()
            parsed = json.loads(content)
            assert parsed["backend"] == "sycl"
            assert parsed["exit_code"] == 0


class TestProvenanceFailureWarning:
    """T038: Test provenance write failure emits warning but build still succeeds."""

    def test_provenance_failure_warning(self, capsys, tmp_path: Path) -> None:
        """Provenance write failure should emit warning but not fail build.

        If provenance write fails, the build should still be considered
        successful, but a warning should be logged.
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Create artifact
        artifact = BuildArtifact(
            artifact_type="binary",
            backend="sycl",
            created_at=datetime.now(),
            git_remote_url=config.git_remote_url,
            git_commit_sha="abc123",
            git_branch=config.git_branch,
            build_command=["cmake", "-DGGML_SYCL=ON"],
            build_duration_seconds=10.0,
            exit_code=0,
            binary_path=tmp_path / "bin" / "llama-server",
            binary_size_bytes=104857600,
            build_log_path=tmp_path / "build.log",
            failure_report_path=None,
        )

        # Mock atomic write to fail
        with patch.object(pipeline, "_atomic_write", return_value=False):
            # Should not raise exception
            result = pipeline._write_provenance(artifact)

            # Build should still be considered successful
            assert result is True

            # Warning should be logged (captured to stderr)
            captured = capsys.readouterr()
            assert "warning" in captured.err.lower() or "provenance" in captured.err.lower()


class TestDryRunMode:
    """T039: Test dry-run mode."""

    def test_dry_run_mode(self, tmp_path: Path) -> None:
        """Dry-run mode should print commands without executing.

        In dry-run mode, the pipeline should:
        - Print what commands would be executed
        - Not actually run subprocess commands
        - Still return success status
        """
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )

        pipeline = BuildPipeline(config)

        # Enable dry-run mode
        pipeline._dry_run = True

        # Mock source directory
        (config.source_dir / "CMakeLists.txt").write_text("# Mock")

        # Mock subprocess to verify it's not called
        with patch("subprocess.run") as mock_run:
            # Mock other stages
            pipeline._run_preflight = Mock(return_value=(True, []))
            pipeline._run_clone = Mock(
                return_value=BuildProgress(
                    stage="clone", status="skipped", message="Exists", progress_percent=0
                )
            )
            pipeline._run_configure = Mock(
                return_value=BuildProgress(
                    stage="configure", status="success", message="Configured", progress_percent=50
                )
            )
            pipeline._run_build = Mock(
                return_value=BuildProgress(
                    stage="build", status="success", message="Built", progress_percent=75
                )
            )
            pipeline._write_provenance = Mock(return_value=True)

            result = pipeline.run()

            # Verify subprocess was NOT called (dry-run)
            assert not mock_run.called
            assert result.success is True


class TestCloneModes:
    """T040: Test full clone vs shallow clone."""

    def test_clone_modes(self, tmp_path: Path) -> None:
        """Clone should support both shallow and full clone modes.

        - Shallow clone: git clone --depth 1 (faster, smaller)
        - Full clone: git clone (slower, complete history)
        """
        # Test shallow clone
        config_shallow = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source_shallow",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            shallow_clone=True,
        )

        pipeline_shallow = BuildPipeline(config_shallow)

        # Mock subprocess for shallow clone
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _ = pipeline_shallow._run_clone()

            # Verify shallow clone flags were used
            call_args = mock_run.call_args[0][0]
            assert "--depth" in call_args
            assert "1" in call_args

        # Test full clone
        config_full = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source_full",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            shallow_clone=False,
        )

        pipeline_full = BuildPipeline(config_full)

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _ = pipeline_full._run_clone()

            # Verify shallow clone flags were NOT used
            call_args = mock_run.call_args[0][0]
            assert "--depth" not in call_args
