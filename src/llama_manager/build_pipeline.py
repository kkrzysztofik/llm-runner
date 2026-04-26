# Build pipeline dataclasses for M2

import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal

logger = logging.getLogger(__name__)

MSG_SOURCES_ALREADY_EXIST = "Sources already exist"

# Intel oneAPI environment setup script (default install location)
_INTEL_SETVARS_SH = Path("/opt/intel/oneapi/setvars.sh")


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

    def __post_init__(self) -> None:
        """Ensure Path objects are Path instances and validate constraints."""
        self.source_dir = Path(self.source_dir)
        self.build_dir = Path(self.build_dir)
        self.output_dir = Path(self.output_dir)
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be >= 1")


@dataclass
class BuildArtifact:
    """Represents a build artifact with metadata.

    This dataclass captures all information about a successful or failed
    build attempt, including binary location, size, build command, and
    timing information.
    """

    artifact_type: Literal["llama-server"]
    backend: Literal["sycl", "cuda", "both"]
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

    def to_dict(self) -> dict:
        """Convert BuildArtifact to a dictionary for JSON serialization.

        Returns:
            Dictionary with all artifact fields, converting Path objects to strings.
        """
        return {
            "artifact_type": self.artifact_type,
            "backend": self.backend,
            "created_at": self.created_at,
            "git_remote_url": self.git_remote_url,
            "git_commit_sha": self.git_commit_sha,
            "git_branch": self.git_branch,
            "build_command": self.build_command,
            "build_duration_seconds": self.build_duration_seconds,
            "exit_code": self.exit_code,
            "binary_path": str(self.binary_path) if self.binary_path else None,
            "binary_size_bytes": self.binary_size_bytes,
            "build_log_path": str(self.build_log_path) if self.build_log_path else None,
            "failure_report_path": str(self.failure_report_path)
            if self.failure_report_path
            else None,
        }


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
    backend: str

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

    def __init__(
        self,
        config: BuildConfig,
        progress_callback: Callable[[BuildProgress], None] | None = None,
    ) -> None:
        """Initialize build pipeline with configuration.

        Args:
            config: BuildConfig with all build parameters
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self._dry_run = False
        self._lock_file: Path | None = None
        self._progress_callback = progress_callback
        self._build_start_time: float = 0.0
        self._build_output: str = ""

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

    def _run_with_retry(
        self, stage_func: Callable[[], BuildProgress], stage_name: str
    ) -> BuildProgress:
        """Run a stage with retry logic and exponential backoff.

        Args:
            stage_func: The stage function to execute
            stage_name: Name of the stage for logging

        Returns:
            BuildProgress with final stage status
        """
        last_result: BuildProgress = BuildProgress(
            stage=stage_name,
            status="failed",
            message="No attempts made",
            progress_percent=0.0,
        )

        for attempt in range(self.config.retry_attempts):
            result = stage_func()
            last_result = result

            # Emit progress if callback is set
            if self._progress_callback:
                self._progress_callback(result)

            if result.status == "success":
                return result

            # If not last attempt, retry with exponential backoff
            if attempt < self.config.retry_attempts - 1:
                delay = self.config.retry_delay * (2**attempt)
                progress = BuildProgress(
                    stage=stage_name,
                    status="retrying",
                    message=f"Stage failed, retrying in {delay}s (attempt {attempt + 2}/{self.config.retry_attempts})",
                    progress_percent=0.0,
                    retries_remaining=self.config.retry_attempts - attempt - 1,
                )
                if self._progress_callback:
                    self._progress_callback(progress)
                time.sleep(delay)

        return last_result

    def run(self) -> BuildResult:
        """Run the complete build pipeline.

        Executes all stages in sequence: preflight -> clone -> configure -> build -> finalize.

        Returns:
            BuildResult with success status and artifact if successful
        """
        # Handle BOTH backend - delegate to run_both_backends
        if self.config.backend == BuildBackend.BOTH:
            raise ValueError("BuildBackend.BOTH must use run_both_backends() instead")

        # Record start time for build duration calculation
        self._build_start_time = time.time()

        # Acquire build lock
        from .config import Config

        config = Config()
        if not self._acquire_lock(config.build_lock_path):
            return BuildResult(
                success=False,
                error_message=f"Failed to acquire build lock for {self.config.backend}",
            )

        try:
            # Stage 1: Preflight
            progress = self._run_with_retry(self._run_preflight, "preflight")
            if progress.status == "failed":
                return BuildResult(
                    success=False,
                    error_message=f"Preflight failed: {progress.message}",
                    progress=progress,
                )

            # Stage 2: Clone
            progress = self._run_with_retry(self._run_clone, "clone")
            if progress.status == "failed":
                return BuildResult(
                    success=False,
                    error_message=f"Clone failed: {progress.message}",
                    progress=progress,
                )

            # Stage 3: Configure
            progress = self._run_with_retry(self._run_configure, "configure")
            if progress.status == "failed":
                return BuildResult(
                    success=False,
                    error_message=f"Configure failed: {progress.message}",
                    progress=progress,
                )

            # Stage 4: Build
            progress = self._run_with_retry(self._run_build, "build")
            if progress.status == "failed":
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

        finally:
            self._release_lock()

    def run_both_backends(self) -> list[BuildResult]:
        """Run builds for both SYCL and CUDA backends sequentially.

        Builds SYCL first, then CUDA, each with its own BuildLock to prevent
        concurrent builds. Returns a list of BuildResults in order: [sycl_result, cuda_result].

        Returns:
            List of BuildResults: [SYCL build result, CUDA build result]
        """
        # Create backend-specific directories for isolation
        sycl_build_dir = self.config.build_dir / "build_sycl"
        sycl_output_dir = self.config.output_dir / "output_sycl"
        cuda_build_dir = self.config.build_dir / "build_cuda"
        cuda_output_dir = self.config.output_dir / "output_cuda"

        # Build SYCL first
        sycl_config = BuildConfig(
            backend=BuildBackend.SYCL,
            source_dir=self.config.source_dir,
            build_dir=sycl_build_dir,
            output_dir=sycl_output_dir,
            git_remote_url=self.config.git_remote_url,
            git_branch=self.config.git_branch,
            retry_attempts=self.config.retry_attempts,
            retry_delay=self.config.retry_delay,
            shallow_clone=self.config.shallow_clone,
            jobs=self.config.jobs,
        )
        sycl_pipeline = BuildPipeline(sycl_config, self._progress_callback)
        sycl_pipeline.dry_run = self._dry_run

        sycl_result = sycl_pipeline.run()

        # Build CUDA after SYCL completes
        cuda_config = BuildConfig(
            backend=BuildBackend.CUDA,
            source_dir=self.config.source_dir,
            build_dir=cuda_build_dir,
            output_dir=cuda_output_dir,
            git_remote_url=self.config.git_remote_url,
            git_branch=self.config.git_branch,
            retry_attempts=self.config.retry_attempts,
            retry_delay=self.config.retry_delay,
            shallow_clone=self.config.shallow_clone,
            jobs=self.config.jobs,
        )
        cuda_pipeline = BuildPipeline(cuda_config, self._progress_callback)
        cuda_pipeline.dry_run = self._dry_run

        cuda_result = cuda_pipeline.run()

        return [sycl_result, cuda_result]

    def _run_preflight(self) -> BuildProgress:
        """Run preflight stage - validate toolchain.

        In dry-run mode, still validates toolchain availability but skips
        actual build operations. This ensures dry-run provides accurate
        feedback about whether the build environment is properly configured.

        Returns:
            BuildProgress with stage status
        """
        progress = BuildProgress(
            stage="preflight",
            status="running",
            message="Validating toolchain...",
            progress_percent=0.0,
        )

        # Check toolchain (always run, even in dry-run mode)
        from .toolchain import detect_toolchain

        status = detect_toolchain()

        if (self.config.backend == BuildBackend.SYCL and not status.is_sycl_ready) or (
            self.config.backend == BuildBackend.CUDA and not status.is_cuda_ready
        ):
            missing = status.missing_tools(self.config.backend)
            progress.status = "failed"
            backend_name = "SYCL" if self.config.backend == BuildBackend.SYCL else "CUDA"
            progress.message = f"Missing {backend_name} tools: {', '.join(missing)}"
            return progress

        progress.status = "success"
        progress.message = "Toolchain validated"
        progress.progress_percent = 20
        return progress

    def _run_clone(self) -> BuildProgress:
        """Run clone stage - clone git repository.

        Implements offline-continue support: if source directory exists and
        contains files, skip the clone operation and continue with existing
        sources. This allows builds to continue when network is unavailable
        but local clone exists.

        Returns:
            BuildProgress with stage status
        """
        import subprocess

        progress = BuildProgress(
            stage="clone",
            status="running",
            message="Cloning repository...",
            progress_percent=20,
        )

        # Check if source already exists and is non-empty
        # This enables offline-continue: use existing sources when network unavailable
        if self._source_exists():
            progress.status = "skipped"
            progress.message = MSG_SOURCES_ALREADY_EXIST
            progress.progress_percent = 30
            return progress

        # Clone repository
        try:
            if self._dry_run:
                progress.message = (
                    f"Would run: git clone --branch {self.config.git_branch} "
                    f"{self.config.git_remote_url} {self.config.source_dir}"
                )
                progress.status = "success"
                progress.progress_percent = 30
                return progress

            self.config.source_dir.parent.mkdir(parents=True, exist_ok=True)

            cmd = ["git", "clone", "--branch", self.config.git_branch]
            if self.config.shallow_clone:
                cmd.extend(["--depth", "1"])
            cmd.extend([self.config.git_remote_url, str(self.config.source_dir)])

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            progress.message = f"Cloned {self.config.git_remote_url}"
            progress.status = "success"
            progress.progress_percent = 30

        except subprocess.SubprocessError as e:
            # Network/subprocess failure - check if sources exist to enable offline continue
            if self._source_exists():
                progress.status = "skipped"
                progress.message = MSG_SOURCES_ALREADY_EXIST
                progress.progress_percent = 30
            else:
                stderr = getattr(e, "stderr", str(e))
                progress.status = "failed"
                progress.message = f"Git clone failed: {stderr}"
            return progress
        except Exception as e:
            # Other errors (TimeoutExpired, etc.) - check if sources exist
            if self._source_exists():
                progress.status = "skipped"
                progress.message = MSG_SOURCES_ALREADY_EXIST
                progress.progress_percent = 30
            else:
                progress.status = "failed"
                progress.message = f"Clone failed: {str(e)}"
            return progress

        return progress

    def _source_exists(self) -> bool:
        """Check if source directory exists and is non-empty.

        This is used for offline-continue support when network operations fail.

        Returns:
            True if source directory exists and contains files, False otherwise
        """
        if not self.config.source_dir.exists():
            return False
        return any(self.config.source_dir.iterdir())

    def _run_configure(self) -> BuildProgress:
        """Run configure stage - CMake configuration.

        Returns:
            BuildProgress with stage status
        """
        import subprocess

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
            cmd = ["cmake", "-S", str(self.config.source_dir), "-B", str(self.config.build_dir)]
            cmd.extend(cmake_args)
            cmd = self._get_build_env_cmd(cmd)
            progress.message = f"Would run: {' '.join(cmd)}"
            progress.status = "success"
            progress.progress_percent = 50
            return progress

        try:
            # Create build directory if it doesn't exist
            self.config.build_dir.mkdir(parents=True, exist_ok=True)

            cmd = ["cmake", "-S", str(self.config.source_dir), "-B", str(self.config.build_dir)]
            cmd.extend(cmake_args)
            cmd = self._get_build_env_cmd(cmd)

            subprocess.run(cmd, capture_output=True, text=True, check=True)

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
        import subprocess

        progress = BuildProgress(
            stage="build",
            status="running",
            message="Building...",
            progress_percent=50,
        )

        if self._dry_run:
            cmd = ["cmake", "--build", str(self.config.build_dir)]
            if self.config.jobs:
                cmd.extend(["--parallel", str(self.config.jobs)])
            cmd = self._get_build_env_cmd(cmd)
            progress.message = f"Would run: {' '.join(cmd)}"
            progress.status = "success"
            progress.progress_percent = 75
            return progress

        try:
            cmd = ["cmake", "--build", str(self.config.build_dir)]
            if self.config.jobs:
                cmd.extend(["--parallel", str(self.config.jobs)])
            cmd = self._get_build_env_cmd(cmd)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
            )

            # Store build output for log file
            self._build_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

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
        import subprocess

        if not build_progress.is_complete or build_progress.status != "success":
            return None

        # Get git commit SHA (skip in dry-run mode)
        git_commit_sha = "unknown"
        if not self._dry_run:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.config.source_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                git_commit_sha = result.stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        # Find built binary
        binary_path = None
        binary_size_bytes = None
        build_dir_bin = self.config.build_dir / "bin"
        if build_dir_bin.exists():
            server_binary = build_dir_bin / "llama-server"
            if server_binary.exists():
                binary_path = server_binary
                binary_size_bytes = server_binary.stat().st_size

        # Create build log path in reports directory
        reports_dir = Path(self.config.output_dir).parent / "reports"
        timestamp = str(int(time.time()))
        backend_name = (
            self.config.backend.value
            if isinstance(self.config.backend, BuildBackend)
            else str(self.config.backend)
        )
        build_log_path = reports_dir / f"{timestamp}-{backend_name}.log"

        # Ensure reports directory exists and write build log
        if self._build_output:
            reports_dir.mkdir(parents=True, exist_ok=True)
            build_log_path.write_text(self._build_output)

        # Create artifact with computed build duration
        artifact = BuildArtifact(
            artifact_type="llama-server",
            backend=self.config.backend.value,
            created_at=time.time(),
            git_remote_url=self.config.git_remote_url,
            git_commit_sha=git_commit_sha,
            git_branch=self.config.git_branch,
            build_command=["cmake", "--build", str(self.config.build_dir)],
            build_duration_seconds=time.time() - self._build_start_time,
            exit_code=0,
            binary_path=binary_path,
            binary_size_bytes=binary_size_bytes,
            build_log_path=build_log_path,
            failure_report_path=None,
        )

        # Write provenance
        if self._write_provenance(artifact):
            return artifact
        return None

    def _get_cmake_flags(self, backend: BuildBackend) -> list[str]:
        """Get CMake flags for specified backend.

        Args:
            backend: Build backend (SYCL, CUDA, or BOTH)

        Returns:
            List of CMake flags
        """
        flags = [
            "-DBUILD_SERVER=ON",
            "-DGGML_NATIVE=OFF",  # Disable native optimization for portability
        ]

        if backend == BuildBackend.SYCL:
            flags.extend([
                f"-D{BuildConfig.GGML_SYCL}=ON",
                "-DCMAKE_C_COMPILER=icx",
                "-DCMAKE_CXX_COMPILER=icpx",
            ])
        elif backend == BuildBackend.CUDA:
            flags.append(f"-D{BuildConfig.GGML_CUDA}=ON")

        return flags

    def _get_build_env_cmd(self, cmd: list[str]) -> list[str]:
        """Wrap command with Intel oneAPI environment if needed.

        For SYCL builds, sources /opt/intel/oneapi/setvars.sh before running
        the command so that compilers and libraries are on PATH.

        Args:
            cmd: Original command list

        Returns:
            Potentially wrapped command list
        """
        if self.config.backend != BuildBackend.SYCL:
            return cmd
        if not _INTEL_SETVARS_SH.exists():
            return cmd
        # Use bash -c to source setvars.sh then run the command
        cmd_str = " ".join(cmd)
        return [
            "bash",
            "-c",
            f'source "{_INTEL_SETVARS_SH}" && {cmd_str}',
        ]

    def _write_provenance(self, artifact: BuildArtifact) -> bool:
        """Write build artifact provenance atomically.

        The provenance file contains a complete record of the build
        including:
        - Artifact metadata (type, backend, creation time)
        - Git repository information (remote URL, commit SHA, branch)
        - Build configuration (command, duration, exit code)
        - Binary information (path, size)
        - Log file references (build log, failure report)

        This data is suitable for:
        - Reproducibility analysis
        - Build auditing
        - Debugging and troubleshooting
        - Release notes generation

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
                "created_at": artifact.created_at,
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

            # Atomic write: write to temp file with restricted permissions, then rename
            temp_file = self.config.output_dir / f".build-artifact-{os.getpid()}.json.tmp"
            final_file = self.config.output_dir / "build-artifact.json"

            with open(temp_file, "w") as f:
                json.dump(artifact_data, f, indent=2)

            # Set restrictive permissions (owner read/write only)
            os.chmod(temp_file, 0o600)

            # Atomic rename
            temp_file.rename(final_file)

            return True

        except Exception as e:
            logger.warning("failed to write provenance: %s", e)
            return False

    def _acquire_lock(self, lock_path: Path) -> bool:
        """Acquire build lock atomically.

        Uses O_EXCL flag to ensure atomic lock acquisition, preventing TOCTOU race conditions.

        Args:
            lock_path: Path to lock file

        Returns:
            True if lock acquired, False otherwise
        """
        if self._dry_run:
            return True

        try:
            # First, check if there's a stale lock and remove it safely
            if lock_path.exists() and self._is_lock_stale(lock_path):
                with contextlib.suppress(Exception):
                    lock_path.unlink()

            # Ensure parent directory exists
            lock_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic lock acquisition using O_EXCL flag
            # O_CREAT|O_EXCL ensures the file is created only if it doesn't exist
            # This is atomic at the filesystem level
            lock_fd = os.open(
                str(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,  # Owner read/write only
            )

            # Write lock data to the file descriptor
            lock_data = {
                "pid": os.getpid(),
                "started_at": time.time(),
                "backend": self.config.backend.value,
            }
            try:
                os.write(lock_fd, json.dumps(lock_data).encode("utf-8"))
            finally:
                os.close(lock_fd)

            self._lock_file = lock_path
            return True

        except FileExistsError:
            # Another process holds the lock
            logger.error("build lock already held by another process: %s", lock_path)
            return False
        except OSError as e:
            logger.error("failed to acquire build lock: %s", e)
            return False
        except Exception as e:
            logger.error("failed to acquire build lock: %s", e)
            return False

    def release_lock(self) -> None:
        """Release build lock.

        Public method to release the build lock. This should be used
        instead of calling the private _release_lock() method directly.
        """
        self._release_lock()

    def _release_lock(self) -> None:
        """Release build lock."""
        if self._lock_file and self._lock_file.exists():
            with contextlib.suppress(Exception):
                self._lock_file.unlink()
            self._lock_file = None

    def _is_lock_stale(self, lock_path: Path) -> bool:
        """Check if lock file is stale.

        Args:
            lock_path: Path to lock file

        Returns:
            True if lock is stale or PID is invalid
        """
        try:
            with open(lock_path) as f:
                data = json.load(f)

            pid = data.get("pid")
            started_at_str = data.get("started_at")

            if pid is None or started_at_str is None:
                return True

            started_at = float(started_at_str)
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
            with open(lock_path) as f:
                data = json.load(f)
            pid = data.get("pid", "unknown")
            backend = data.get("backend", "unknown")
            return f"Build lock already held by PID {pid} (backend: {backend})"
        except Exception:
            return "Build lock file exists but could not be read"
