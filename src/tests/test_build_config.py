"""T003-T005, T007: Tests for BuildConfig, BuildArtifact, BuildProgress, BuildLock dataclasses.

Test Tasks:
- T003: BuildConfig dataclass tests
- T004: BuildArtifact dataclass tests
- T005: BuildProgress dataclass tests
- T007: BuildLock dataclass tests
"""

import time
from pathlib import Path

import pytest

from llama_manager.build_pipeline import (
    GGML_CUDA,
    GGML_SYCL,
    BuildArtifact,
    BuildBackend,
    BuildConfig,
    BuildLock,
    BuildProgress,
)


class TestBuildConfig:
    """T003: Tests for BuildConfig dataclass."""

    def test_build_config_all_fields_settable(self, tmp_path: Path) -> None:
        """BuildConfig should have all fields settable and retrievable."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
            retry_attempts=5,
            retry_delay=10,
            shallow_clone=False,
            jobs=8,
        )
        assert config.backend == BuildBackend.SYCL
        assert config.source_dir == tmp_path / "source"
        assert config.build_dir == tmp_path / "build"
        assert config.output_dir == tmp_path / "output"
        assert config.git_remote_url == "https://github.com/ggerganov/llama.cpp"
        assert config.git_branch == "main"
        assert config.retry_attempts == 5
        assert config.retry_delay == 10
        assert config.shallow_clone is False
        assert config.jobs == 8

    def test_build_config_default_values(self, tmp_path: Path) -> None:
        """BuildConfig should have correct default values."""
        config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="develop",
        )
        assert config.backend == BuildBackend.CUDA
        assert config.retry_attempts == 3  # Default
        assert config.retry_delay == 5  # Default
        assert config.shallow_clone is True  # Default
        assert config.jobs is None  # Default

    def test_build_config_class_constants(self) -> None:
        """BuildConfig should have GGML_SYCL and GGML_CUDA constants."""
        assert GGML_SYCL == "sycl"
        assert GGML_CUDA == "cuda"

    def test_build_config_path_conversion(self, tmp_path: Path) -> None:
        """BuildConfig should convert strings to Path objects in __post_init__."""
        config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=tmp_path / "source",
            build_dir=tmp_path / "build",
            output_dir=tmp_path / "output",
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        assert isinstance(config.source_dir, Path)
        assert isinstance(config.build_dir, Path)
        assert isinstance(config.output_dir, Path)

    def test_build_config_backend_enum(self) -> None:
        """BuildConfig should accept BuildBackend enum values."""
        config_sycl = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=Path("/source"),
            build_dir=Path("/build"),
            output_dir=Path("/output"),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        assert config_sycl.backend == BuildBackend.SYCL
        assert config_sycl.backend.value == "sycl"

        config_cuda = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=Path("/source"),
            build_dir=Path("/build"),
            output_dir=Path("/output"),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_branch="main",
        )
        assert config_cuda.backend == BuildBackend.CUDA
        assert config_cuda.backend.value == "cuda"


class TestBuildArtifact:
    """T004: Tests for BuildArtifact dataclass."""

    def test_build_artifact_all_fields_settable(self, tmp_path: Path) -> None:
        """BuildArtifact should have all fields settable and retrievable."""
        artifact = BuildArtifact(
            artifact_type="llama-server",
            backend="sycl",
            created_at=time.time(),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_commit_sha="abc123def456",
            git_branch="main",
            build_command=["cmake", "-G", "Ninja", "-DGGML_SYCL=ON"],
            build_duration_seconds=123.456,
            exit_code=0,
            binary_path=tmp_path / "bin" / "llama-server",
            binary_size_bytes=104857600,  # 100 MB
            build_log_path=tmp_path / "logs" / "build.log",
            failure_report_path=None,
        )
        assert artifact.artifact_type == "llama-server"
        assert artifact.backend == "sycl"
        assert isinstance(artifact.created_at, float)
        assert artifact.git_remote_url == "https://github.com/ggerganov/llama.cpp"
        assert artifact.git_commit_sha == "abc123def456"
        assert artifact.git_branch == "main"
        assert artifact.build_command == ["cmake", "-G", "Ninja", "-DGGML_SYCL=ON"]
        assert artifact.build_duration_seconds == 123.456
        assert artifact.exit_code == 0
        assert artifact.binary_path == tmp_path / "bin" / "llama-server"
        assert artifact.binary_size_bytes == 104857600
        assert artifact.build_log_path == tmp_path / "logs" / "build.log"
        assert artifact.failure_report_path is None

    def test_build_artifact_is_success_true(self) -> None:
        """BuildArtifact.is_success should return True when exit_code == 0."""
        artifact = BuildArtifact(
            artifact_type="llama-server",
            backend="sycl",
            created_at=time.time(),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_commit_sha="abc123",
            git_branch="main",
            build_command=["echo", "test"],
            build_duration_seconds=1.0,
            exit_code=0,
            binary_path=Path("/bin/test"),
            binary_size_bytes=1024,
            build_log_path=None,
            failure_report_path=None,
        )
        assert artifact.is_success is True

    def test_build_artifact_is_success_false(self) -> None:
        """BuildArtifact.is_success should return False when exit_code != 0."""
        artifact = BuildArtifact(
            artifact_type="llama-server",
            backend="sycl",
            created_at=time.time(),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_commit_sha="abc123",
            git_branch="main",
            build_command=["echo", "test"],
            build_duration_seconds=1.0,
            exit_code=1,
            binary_path=None,
            binary_size_bytes=None,
            build_log_path=None,
            failure_report_path=None,
        )
        assert artifact.is_success is False

    def test_build_artifact_binary_size_mb_calculation(self) -> None:
        """BuildArtifact.binary_size_mb should calculate correctly (bytes to MB)."""
        artifact = BuildArtifact(
            artifact_type="llama-server",
            backend="sycl",
            created_at=time.time(),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_commit_sha="abc123",
            git_branch="main",
            build_command=["echo", "test"],
            build_duration_seconds=1.0,
            exit_code=0,
            binary_path=Path("/bin/test"),
            binary_size_bytes=104857600,  # 100 MB in bytes
            build_log_path=None,
            failure_report_path=None,
        )
        assert artifact.binary_size_mb == pytest.approx(100.0)

    def test_build_artifact_binary_size_mb_none(self) -> None:
        """BuildArtifact.binary_size_mb should return None when binary_size_bytes is None."""
        artifact = BuildArtifact(
            artifact_type="llama-server",
            backend="sycl",
            created_at=time.time(),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_commit_sha="abc123",
            git_branch="main",
            build_command=["echo", "test"],
            build_duration_seconds=1.0,
            exit_code=1,
            binary_path=None,
            binary_size_bytes=None,
            build_log_path=None,
            failure_report_path=None,
        )
        assert artifact.binary_size_mb is None

    def test_build_artifact_created_at_default(self, tmp_path: Path) -> None:
        """BuildArtifact.created_at should be a float timestamp."""
        now = time.time()
        artifact = BuildArtifact(
            artifact_type="llama-server",
            backend="sycl",
            created_at=now,
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_commit_sha="abc123",
            git_branch="main",
            build_command=["echo", "test"],
            build_duration_seconds=1.0,
            exit_code=0,
            binary_path=tmp_path / "bin" / "test",
            binary_size_bytes=1024,
            build_log_path=None,
            failure_report_path=None,
        )
        assert artifact.created_at == now
        assert isinstance(artifact.created_at, float)


class TestBuildProgress:
    """T005: Tests for BuildProgress dataclass."""

    def test_build_progress_all_fields_settable(self) -> None:
        """BuildProgress should have all fields settable and retrievable."""
        progress = BuildProgress(
            stage="clone",
            status="in_progress",
            message="Cloning repository...",
            progress_percent=25,
            retries_remaining=2,
        )
        assert progress.stage == "clone"
        assert progress.status == "in_progress"
        assert progress.message == "Cloning repository..."
        assert progress.progress_percent == 25
        assert progress.retries_remaining == 2

    def test_build_progress_is_complete_success(self) -> None:
        """BuildProgress.is_complete should return True for success status."""
        progress = BuildProgress(
            stage="build",
            status="success",
            message="Build completed successfully",
            progress_percent=100,
        )
        assert progress.is_complete is True

    def test_build_progress_is_complete_failed(self) -> None:
        """BuildProgress.is_complete should return True for failed status."""
        progress = BuildProgress(
            stage="build",
            status="failed",
            message="Build failed",
            progress_percent=50,
        )
        assert progress.is_complete is True

    def test_build_progress_is_complete_skipped(self) -> None:
        """BuildProgress.is_complete should return True for skipped status."""
        progress = BuildProgress(
            stage="test",
            status="skipped",
            message="Skipped due to build failure",
            progress_percent=100,
        )
        assert progress.is_complete is True

    def test_build_progress_is_complete_in_progress(self) -> None:
        """BuildProgress.is_complete should return False for in_progress status."""
        progress = BuildProgress(
            stage="build",
            status="in_progress",
            message="Building...",
            progress_percent=50,
        )
        assert progress.is_complete is False

    def test_build_progress_is_retrying_true(self) -> None:
        """BuildProgress.is_retrying should return True when status is retrying with retries_remaining."""
        progress = BuildProgress(
            stage="build",
            status="retrying",
            message="Retrying build...",
            progress_percent=25,
            retries_remaining=2,
        )
        assert progress.is_retrying is True

    def test_build_progress_is_retrying_false_no_retries(self) -> None:
        """BuildProgress.is_retrying should return False when retries_remaining is None."""
        progress = BuildProgress(
            stage="build",
            status="retrying",
            message="Retrying build...",
            progress_percent=25,
            retries_remaining=None,
        )
        assert progress.is_retrying is False

    def test_build_progress_is_retrying_false_not_retrying_status(self) -> None:
        """BuildProgress.is_retrying should return False when status is not retrying."""
        progress = BuildProgress(
            stage="build",
            status="in_progress",
            message="Building...",
            progress_percent=50,
            retries_remaining=2,
        )
        assert progress.is_retrying is False

    def test_build_progress_stage_enum_values(self) -> None:
        """BuildProgress should accept various stage values."""
        stages = ["init", "clone", "configure", "build", "test", "package"]
        for stage in stages:
            progress = BuildProgress(
                stage=stage,
                status="in_progress",
                message=f"Processing {stage}...",
                progress_percent=50,
            )
            assert progress.stage == stage

    def test_build_progress_retry_scenarios(self) -> None:
        """BuildProgress should handle various retry scenarios."""
        # First retry attempt
        progress1 = BuildProgress(
            stage="build",
            status="retrying",
            message="First retry attempt",
            progress_percent=0,
            retries_remaining=3,
        )
        assert progress1.is_retrying is True
        assert progress1.retries_remaining == 3

        # Last retry attempt
        progress2 = BuildProgress(
            stage="build",
            status="retrying",
            message="Last retry attempt",
            progress_percent=0,
            retries_remaining=1,
        )
        assert progress2.is_retrying is True
        assert progress2.retries_remaining == 1

        # No retries remaining
        progress3 = BuildProgress(
            stage="build",
            status="failed",
            message="Build failed after retries",
            progress_percent=100,
            retries_remaining=0,
        )
        assert progress3.is_retrying is False
        assert progress3.is_complete is True


class TestBuildLock:
    """T007: Tests for BuildLock dataclass."""

    def test_build_lock_all_fields_settable(self, tmp_path: Path) -> None:
        """BuildLock should have all fields settable and retrievable."""
        now = time.time()
        lock = BuildLock(
            pid=12345,
            started_at=now,
            backend="sycl",
        )
        assert lock.pid == 12345
        assert lock.started_at == now
        assert lock.backend == "sycl"

    def test_build_lock_pid_is_integer(self) -> None:
        """BuildLock.pid should be an integer."""
        lock = BuildLock(
            pid=12345,
            started_at=time.time(),
            backend="cuda",
        )
        assert isinstance(lock.pid, int)

    def test_build_lock_elapsed_seconds(self) -> None:
        """BuildLock.elapsed_seconds should calculate correctly."""
        # Create lock with started_at 30 seconds ago
        thirty_seconds_ago = time.time() - 30
        lock = BuildLock(
            pid=12345,
            started_at=thirty_seconds_ago,
            backend="sycl",
        )
        elapsed = lock.elapsed_seconds
        # Allow small tolerance for execution time
        assert 29.9 <= elapsed <= 30.1

    def test_build_lock_elapsed_seconds_zero(self) -> None:
        """BuildLock.elapsed_seconds should be ~0 for newly created lock."""
        lock = BuildLock(
            pid=12345,
            started_at=time.time(),
            backend="sycl",
        )
        elapsed = lock.elapsed_seconds
        assert elapsed < 0.1  # Should be nearly zero

    def test_build_lock_is_stale_true(self) -> None:
        """BuildLock.is_stale should return True when elapsed > timeout."""
        # Create lock that started 2 hours ago (default timeout is 1 hour)
        two_hours_ago = time.time() - 7200  # 2 hours in seconds
        lock = BuildLock(
            pid=12345,
            started_at=two_hours_ago,
            backend="sycl",
        )
        assert lock.is_stale() is True

    def test_build_lock_is_stale_false(self) -> None:
        """BuildLock.is_stale should return False when elapsed <= timeout."""
        # Create lock that started 30 minutes ago (default timeout is 1 hour)
        thirty_minutes_ago = time.time() - 1800  # 30 minutes in seconds
        lock = BuildLock(
            pid=12345,
            started_at=thirty_minutes_ago,
            backend="sycl",
        )
        assert lock.is_stale() is False

    def test_build_lock_is_stale_custom_timeout(self) -> None:
        """BuildLock.is_stale should respect custom timeout parameter."""
        # Create lock that started 30 seconds ago
        thirty_seconds_ago = time.time() - 30
        lock = BuildLock(
            pid=12345,
            started_at=thirty_seconds_ago,
            backend="sycl",
        )
        # Should not be stale with default 1 hour timeout
        assert lock.is_stale(3600) is False
        # Should be stale with 10 second timeout
        assert lock.is_stale(10) is True

    def test_build_lock_backend_values(self) -> None:
        """BuildLock should accept various backend values."""
        backends = ["sycl", "cuda", "cpu"]
        for backend in backends:
            lock = BuildLock(
                pid=12345,
                started_at=time.time(),
                backend=backend,
            )
            assert lock.backend == backend

    def test_build_lock_repr(self) -> None:
        """BuildLock should have meaningful string representation."""
        lock = BuildLock(
            pid=12345,
            started_at=time.time(),
            backend="sycl",
        )
        # Should be able to create string representation
        lock_str = str(lock)
        assert "12345" in lock_str
        assert "sycl" in lock_str
