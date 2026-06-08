"""Build pipeline data models for M2."""

import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

BUILD_CANCELLED_MESSAGE: Final[str] = "Build cancelled"

# Source flavor defaults — (remote URL, default branch)
SOURCE_FLAVOR_DEFAULTS: Final[dict[str, tuple[str, str]]] = {
    "upstream": ("https://github.com/ggerganov/llama.cpp.git", "master"),
    "beellama": ("https://github.com/Anbeeld/beellama.cpp.git", "main"),
}


class BuildBackend(StrEnum):
    """Supported build backends"""

    SYCL = "sycl"
    CUDA = "cuda"
    BOTH = "both"


@dataclass
class BuildConfig:
    """Build pipeline configuration for llama.cpp compilation.

    This dataclass holds all configuration needed to build llama.cpp
    for a specific backend (SYCL, CUDA, or both). It defines source locations,
    build directories, and retry behavior.

    Class constants for CMake flags and build backend identifiers:
    - GGML_SYCL: CMake flag for SYCL backend
    - GGML_CUDA: CMake flag for CUDA backend
    - CMAKE_C_COMPILER_SYCL: Intel C++ compiler for SYCL
    - CMAKE_CXX_COMPILER_SYCL: Intel C++ compiler for SYCL
    """

    GGML_SYCL: ClassVar[str] = "GGML_SYCL"
    GGML_CUDA: ClassVar[str] = "GGML_CUDA"
    CMAKE_C_COMPILER_SYCL: ClassVar[str] = "icx"
    CMAKE_CXX_COMPILER_SYCL: ClassVar[str] = "icpx"

    backend: BuildBackend
    source_dir: Path
    build_dir: Path
    output_dir: Path
    git_remote_url: str
    git_branch: str
    retry_attempts: int = 3
    retry_delay: float = 5.0
    shallow_clone: bool = True
    jobs: int | None = None
    update_sources: bool = True
    git_commit: str | None = None
    build_timeout_seconds: int = 3600
    build_args: list[str] | None = None
    clean_cache: bool = False

    def __post_init__(self) -> None:
        """Ensure Path objects are Path instances and validate constraints."""
        self.source_dir = Path(self.source_dir)
        self.build_dir = Path(self.build_dir)
        self.output_dir = Path(self.output_dir)
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be >= 1")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")
        if self.jobs is not None and self.jobs < 1:
            raise ValueError("jobs must be >= 1")
        if self.build_timeout_seconds <= 0:
            raise ValueError("build_timeout_seconds must be > 0")


@dataclass
class BuildArtifact:
    """Represents a build artifact with metadata.

    This dataclass captures all information about a successful or failed
    build attempt, including binary location, size, build command, and
    timing information.
    """

    artifact_type: Literal["llama-server"]
    backend: BuildBackend
    created_at: float
    git_remote_url: str
    git_commit_sha: str
    git_branch: str
    build_command: list[str]
    build_duration_seconds: float
    exit_code: int
    binary_path: Path | None
    binary_size_bytes: int | None
    build_log_path: Path | None
    failure_report_path: Path | None

    @property
    def is_success(self) -> bool:
        """Check if the build was successful."""
        return self.exit_code == 0

    @property
    def binary_size_mb(self) -> float | None:
        """Get binary size in megabytes, if available."""
        if self.binary_size_bytes is None:
            return None
        return self.binary_size_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        """Convert BuildArtifact to a dictionary for JSON serialization.

        Returns:
            Dictionary with all artifact fields, converting Path objects to strings.
        """
        data = asdict(self)
        # Convert Path instances to strings
        return {k: str(v) if isinstance(v, Path) else v for k, v in data.items()}


@dataclass
class BuildProgress:
    """Tracks build progress through stages.

    This dataclass provides real-time status updates during the build
    process, including stage information, retry attempts, and completion percentage.
    """

    stage: str
    status: str
    message: str
    progress_percent: float
    retries_remaining: int | None = None
    output_line: str | None = None

    @property
    def is_complete(self) -> bool:
        """Check if the build stage is complete."""
        return self.status in {"success", "failed", "skipped"}

    @property
    def is_retrying(self) -> bool:
        """Check if currently retrying after a failure."""
        return self.status == "retrying" and self.retries_remaining is not None


@dataclass
class BuildLock:
    """Represents an exclusive build lock to prevent concurrent builds.

    This dataclass tracks the process holding a build lock, when it was
    acquired, and which backend is being built. Used to prevent multiple
    builds from running simultaneously for the same backend.
    """

    pid: int
    started_at: float
    backend: str  # BuildBackend value (str for deserialization compatibility)

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since lock was acquired."""
        return time.time() - self.started_at

    def is_stale(self, timeout_seconds: int = 3600) -> bool:
        """Check if the lock has exceeded the timeout threshold.

        Args:
            timeout_seconds: Maximum allowed build time in seconds.
                           Defaults to 1 hour.

        Returns:
            True if the lock has been held longer than the timeout.
        """
        return self.elapsed_seconds > timeout_seconds


@dataclass
class BuildResult:
    """Result of a build pipeline run.

    Attributes:
        success: Whether the build completed successfully
        artifact: BuildArtifact if successful, None otherwise
        error_message: Error message if failed
        progress: Final BuildProgress status
    """

    success: bool
    artifact: BuildArtifact | None = None
    error_message: str = ""
    progress: BuildProgress | None = None
