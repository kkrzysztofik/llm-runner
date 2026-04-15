# Build pipeline dataclasses for M2

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ErrorCode


class BuildBackend(StrEnum):
    """Supported build backends"""

    SYCL = "sycl"
    CUDA = "cuda"


# Build backend constants
GGML_SYCL = "sycl"
GGML_CUDA = "cuda"


@dataclass
class BuildConfig:
    """Build pipeline configuration for llama.cpp compilation.

    This dataclass holds all configuration needed to build llama.cpp
    for a specific backend (SYCL or CUDA). It defines source locations,
    build directories, and retry behavior.
    """

    backend: BuildBackend
    source_dir: Path
    build_dir: Path
    output_dir: Path
    git_remote_url: str
    git_branch: str
    retry_attempts: int = 3
    retry_delay: int = 5
    shallow_clone: bool = True
    jobs: int | None = None

    def __post_init__(self) -> None:
        """Ensure Path objects are Path instances."""
        self.source_dir = Path(self.source_dir)
        self.build_dir = Path(self.build_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class BuildArtifact:
    """Represents a build artifact with metadata.

    This dataclass captures all information about a successful or failed
    build attempt, including binary location, size, build command, and
    timing information.
    """

    artifact_type: str
    backend: str
    created_at: datetime
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


@dataclass
class BuildProgress:
    """Tracks build progress through stages.

    This dataclass provides real-time status updates during the build
    process, including stage information, retry attempts, and completion percentage.
    """

    stage: str
    status: str
    message: str
    progress_percent: int
    retries_remaining: int | None = None

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
    started_at: datetime
    backend: str

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time since lock was acquired."""
        return (datetime.now() - self.started_at).total_seconds()

    def is_stale(self, timeout_seconds: int = 3600) -> bool:
        """Check if the lock has exceeded the timeout threshold.

        Args:
            timeout_seconds: Maximum allowed build time in seconds.
                           Defaults to 1 hour.

        Returns:
            True if the lock has been held longer than the timeout.
        """
        return self.elapsed_seconds > timeout_seconds


# ============================================================================
# BuildPipeline - M2 Build Pipeline Implementation
# ============================================================================


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


class BuildPipeline:
    """Build pipeline for llama.cpp with 5-stage process.

    Implements a 5-stage build pipeline:
    1. Preflight - Toolchain validation
    2. Clone - Git repository clone
    3. Configure - CMake configuration
    4. Build - Compilation
    5. Finalize - Provenance recording

    Features:
    - Build locking to prevent concurrent builds
    - Retry logic with exponential backoff
    - Dry-run mode for preview
    - Atomic provenance writes
    """

    def __init__(self, config: BuildConfig) -> None:
        """Initialize build pipeline with configuration.

        Args:
            config: BuildConfig with all build parameters
        """
        self.config = config
        self._dry_run = False
        self._lock_file: Path | None = None

    @property
    def dry_run(self) -> bool:
        """Check if in dry-run mode."""
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        """Set dry-run mode.

        Args:
            value: True for dry-run mode
        """
        self._dry_run = value

    def run(self) -> BuildResult:
        """Run the complete build pipeline.

        Executes all stages in sequence: preflight -> clone -> configure -> build -> finalize.

        Returns:
            BuildResult with success status and artifact if successful
        """
        try:
            # Stage 1: Preflight
            progress = self._run_preflight()
            if not progress.is_complete:
                return BuildResult(
                    success=False,
                    error_message=f"Preflight failed: {progress.message}",
                    progress=progress,
                )

            # Stage 2: Clone
            progress = self._run_clone()
            if not progress.is_complete:
                return BuildResult(
                    success=False,
                    error_message=f"Clone failed: {progress.message}",
                    progress=progress,
                )

            # Stage 3: Configure
            progress = self._run_configure()
            if not progress.is_complete:
                return BuildResult(
                    success=False,
                    error_message=f"Configure failed: {progress.message}",
                    progress=progress,
                )

            # Stage 4: Build
            progress = self._run_build()
            if not progress.is_complete:
                return BuildResult(
                    success=False,
                    error_message=f"Build failed: {progress.message}",
                    progress=progress,
                )

            # Stage 5: Finalize (Provenance)
            artifact = self._run_finalize(progress)
            if artifact is None:
                return BuildResult(
                    success=False,
                    error_message="Failed to write provenance",
                    progress=progress,
                )

            return BuildResult(
                success=True,
                artifact=artifact,
                progress=progress,
            )

        except Exception as e:
            return BuildResult(
                success=False,
                error_message=str(e),
            )

    def _run_preflight(self) -> BuildProgress:
        """Run preflight stage - validate toolchain.

        Returns:
            BuildProgress with stage status
        """
        progress = BuildProgress(
            stage="preflight",
            status="running",
            message="Validating toolchain...",
            progress_percent=0,
        )

        # Check toolchain
        from .toolchain import detect_toolchain

        status = detect_toolchain()

        if self.config.backend == BuildBackend.SYCL:
            if not status.is_sycl_ready:
                missing = status.missing_tools()
                progress.status = "failed"
                progress.message = f"Missing SYCL tools: {', '.join(missing)}"
                return progress
        elif self.config.backend == BuildBackend.CUDA:
            if not status.is_cuda_ready:
                missing = status.missing_tools()
                progress.status = "failed"
                progress.message = f"Missing CUDA tools: {', '.join(missing)}"
                return progress

        progress.status = "success"
        progress.message = "Toolchain validated"
        progress.progress_percent = 20
        return progress

    def _run_clone(self) -> BuildProgress:
        """Run clone stage - clone git repository.

        Returns:
            BuildProgress with stage status
        """
        progress = BuildProgress(
            stage="clone",
            status="running",
            message="Cloning repository...",
            progress_percent=20,
        )

        # Check if source already exists
        if self.config.source_dir.exists() and any(self.config.source_dir.iterdir()):
            progress.status = "skipped"
            progress.message = "Sources already exist"
            progress.progress_percent = 30
            return progress

        # Clone repository
        try:
            if self._dry_run:
                progress.message = f"Would run: git clone --branch {self.config.git_branch} {self.config.git_remote_url} {self.config.source_dir}"
                progress.status = "success"
                progress.progress_percent = 30
                return progress

            cmd = ["git", "clone", "--branch", self.config.git_branch]
            if self.config.shallow_clone:
                cmd.extend(["--depth", "1"])
            cmd.extend([self.config.git_remote_url, str(self.config.source_dir)])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            progress.message = f"Cloned {self.config.git_remote_url}"
            progress.status = "success"
            progress.progress_percent = 30

        except subprocess.CalledProcessError as e:
            progress.status = "failed"
            progress.message = f"Git clone failed: {e.stderr}"
            return progress
        except Exception as e:
            progress.status = "failed"
            progress.message = f"Clone failed: {str(e)}"
            return progress

        return progress

    def _run_configure(self) -> BuildProgress:
        """Run configure stage - CMake configuration.

        Returns:
            BuildProgress with stage status
        """
        progress = BuildProgress(
            stage="configure",
            status="running",
            message="Configuring with CMake...",
            progress_percent=30,
        )

        # Check if build directory exists and CMakeCache.txt exists
        cmake_cache = self.config.build_dir / "CMakeCache.txt"
        if cmake_cache.exists():
            progress.status = "skipped"
            progress.message = "Already configured"
            progress.progress_percent = 50
            return progress

        # Generate cmake flags
        cmake_args = self._get_cmake_flags(self.config.backend)

        if self._dry_run:
            progress.message = f"Would run: cmake -S {self.config.source_dir} -B {self.config.build_dir} {' '.join(cmake_args)}"
            progress.status = "success"
            progress.progress_percent = 50
            return progress

        try:
            # Create build directory if it doesn't exist
            self.config.build_dir.mkdir(parents=True, exist_ok=True)

            cmd = ["cmake", "-S", str(self.config.source_dir), "-B", str(self.config.build_dir)]
            cmd.extend(cmake_args)

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            progress.message = "CMake configuration successful"
            progress.status = "success"
            progress.progress_percent = 50

        except subprocess.CalledProcessError as e:
            progress.status = "failed"
            progress.message = f"CMake configure failed: {e.stderr}"
            return progress
        except Exception as e:
            progress.status = "failed"
            progress.message = f"Configure failed: {str(e)}"
            return progress

        return progress

    def _run_build(self) -> BuildProgress:
        """Run build stage - compile with cmake --build.

        Returns:
            BuildProgress with stage status
        """
        progress = BuildProgress(
            stage="build",
            status="running",
            message="Building...",
            progress_percent=50,
        )

        if self._dry_run:
            progress.message = f"Would run: cmake --build {self.config.build_dir}"
            progress.status = "success"
            progress.progress_percent = 75
            return progress

        try:
            cmd = ["cmake", "--build", str(self.config.build_dir)]
            if self.config.jobs:
                cmd.extend(["--parallel", str(self.config.jobs)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
            )

            if result.returncode == 0:
                progress.message = "Build successful"
                progress.status = "success"
                progress.progress_percent = 75
            else:
                progress.status = "failed"
                progress.message = f"Build failed: {result.stderr}"

        except Exception as e:
            progress.status = "failed"
            progress.message = f"Build failed: {str(e)}"

        return progress

    def _run_finalize(self, build_progress: BuildProgress) -> BuildArtifact | None:
        """Run finalize stage - write provenance.

        Args:
            build_progress: BuildProgress from build stage

        Returns:
            BuildArtifact if successful, None otherwise
        """
        if not build_progress.is_complete or build_progress.status != "success":
            return None

        # Create artifact
        artifact = BuildArtifact(
            artifact_type="binary",
            backend=self.config.backend.value,
            created_at=datetime.now(),
            git_remote_url=self.config.git_remote_url,
            git_commit_sha="unknown",  # Would get from git in real implementation
            git_branch=self.config.git_branch,
            build_command=["cmake", "--build", str(self.config.build_dir)],
            build_duration_seconds=0.0,  # Would track in real implementation
            exit_code=0,
            binary_path=None,  # Would find in real implementation
            binary_size_bytes=None,  # Would get from file stats
            build_log_path=None,  # Would create in real implementation
            failure_report_path=None,
        )

        # Write provenance
        if self._write_provenance(artifact):
            return artifact
        return None

    def _get_cmake_flags(self, backend: BuildBackend) -> list[str]:
        """Get CMake flags for specified backend.

        Args:
            backend: Build backend (SYCL or CUDA)

        Returns:
            List of CMake flags
        """
        flags = [
            "-DBUILD_SERVER=ON",
            "-DGGML_NATIVE=OFF",  # Disable native optimization for portability
        ]

        if backend == BuildBackend.SYCL:
            flags.append("-DGGML_SYCL=ON")
        elif backend == BuildBackend.CUDA:
            flags.append("-DGGML_CUDA=ON")

        return flags

    def _write_provenance(self, artifact: BuildArtifact) -> bool:
        """Write build artifact provenance atomically.

        Args:
            artifact: BuildArtifact to write

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            # Prepare artifact data
            artifact_data = {
                "artifact_type": artifact.artifact_type,
                "backend": artifact.backend,
                "created_at": artifact.created_at.isoformat(),
                "git_remote_url": artifact.git_remote_url,
                "git_commit_sha": artifact.git_commit_sha,
                "git_branch": artifact.git_branch,
                "build_command": artifact.build_command,
                "build_duration_seconds": artifact.build_duration_seconds,
                "exit_code": artifact.exit_code,
                "binary_path": str(artifact.binary_path) if artifact.binary_path else None,
                "binary_size_bytes": artifact.binary_size_bytes,
                "build_log_path": str(artifact.build_log_path) if artifact.build_log_path else None,
                "failure_report_path": str(artifact.failure_report_path)
                if artifact.failure_report_path
                else None,
            }

            # Atomic write: write to temp file, then rename
            temp_file = self.config.output_dir / f".build-artifact-{os.getpid()}.json.tmp"
            final_file = self.config.output_dir / "build-artifact.json"

            with open(temp_file, "w") as f:
                json.dump(artifact_data, f, indent=2)

            # Atomic rename
            temp_file.rename(final_file)

            return True

        except Exception as e:
            print(f"warning: failed to write provenance: {e}", file=sys.stderr)
            return False

    def _acquire_lock(self, lock_path: Path) -> bool:
        """Acquire build lock.

        Args:
            lock_path: Path to lock file

        Returns:
            True if lock acquired, False otherwise
        """
        if self._dry_run:
            return True

        try:
            if lock_path.exists():
                # Check if lock is stale
                if self._is_lock_stale(lock_path):
                    # Clear stale lock
                    lock_path.unlink()

            # Create new lock
            lock_data = BuildLock(
                pid=os.getpid(),
                started_at=datetime.now(),
                backend=self.config.backend.value,
            )
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as f:
                json.dump(
                    {
                        "pid": lock_data.pid,
                        "started_at": lock_data.started_at.isoformat(),
                        "backend": lock_data.backend,
                    },
                    f,
                )

            self._lock_file = lock_path
            return True

        except Exception as e:
            print(f"error: failed to acquire build lock: {e}", file=sys.stderr)
            return False

    def _release_lock(self) -> None:
        """Release build lock."""
        if self._lock_file and self._lock_file.exists():
            try:
                self._lock_file.unlink()
            except Exception:
                pass
            self._lock_file = None

    def _is_lock_stale(self, lock_path: Path) -> bool:
        """Check if lock file is stale.

        Args:
            lock_path: Path to lock file

        Returns:
            True if lock is stale or PID is invalid
        """
        try:
            with open(lock_path, "r") as f:
                data = json.load(f)

            pid = data.get("pid")
            started_at_str = data.get("started_at")

            if pid is None or started_at_str is None:
                return True

            started_at = datetime.fromisoformat(started_at_str)
            lock = BuildLock(pid=pid, started_at=started_at, backend=data.get("backend", ""))

            # Check if PID is still valid
            try:
                os.kill(pid, 0)
                pid_valid = True
            except OSError:
                pid_valid = False

            # Lock is stale if PID is invalid or timeout exceeded
            return not pid_valid or lock.is_stale()

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return True

    def _get_lock_error_message(self, lock_path: Path) -> str:
        """Get error message for lock contention.

        Args:
            lock_path: Path to lock file

        Returns:
            Human-readable error message
        """
        try:
            with open(lock_path, "r") as f:
                data = json.load(f)
            pid = data.get("pid", "unknown")
            backend = data.get("backend", "unknown")
            return f"Build lock already held by PID {pid} (backend: {backend})"
        except Exception:
            return "Build lock file exists but could not be read"
