# Build pipeline dataclasses for M2

import contextlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
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
_MAX_OUTPUT_SUMMARY_LINES = 12


def _format_command(command: list[str]) -> str:
    """Return a shell-readable command string without executing it."""
    return _redact_build_text(shlex.join(command))


def _redact_build_text(text: str) -> str:
    """Redact secrets from command lines and captured build output."""
    from .reports import redact_sensitive

    redacted = redact_sensitive(text)
    return re.sub(r"(https?://)[^\s/@:]+:[^\s/@]+@", r"\1[REDACTED]@", redacted)


def _format_duration(seconds: float) -> str:
    """Format a duration for human-readable build logs."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _tail_lines(text: str, max_lines: int = _MAX_OUTPUT_SUMMARY_LINES) -> str:
    """Return a concise tail excerpt from command output."""
    lines = [line for line in _redact_build_text(text).strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _summarize_command_output(stdout: str, stderr: str) -> str:
    """Build a compact stdout/stderr excerpt for user-facing error messages."""
    excerpts: list[str] = []
    stdout_excerpt = _tail_lines(stdout)
    stderr_excerpt = _tail_lines(stderr)
    if stderr_excerpt:
        excerpts.append(f"stderr tail:\n{stderr_excerpt}")
    if stdout_excerpt:
        excerpts.append(f"stdout tail:\n{stdout_excerpt}")
    if not excerpts:
        return "No output captured."
    return "\n\n".join(excerpts)


def _format_command_failure(
    *,
    stage: str,
    command: list[str],
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    """Format an actionable command failure summary."""
    output_summary = _summarize_command_output(stdout, stderr)
    return (
        f"{stage} command failed with exit code {returncode}: {_format_command(command)}\n"
        f"{output_summary}"
    )


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
        self._last_build_command: list[str] = []
        self._last_exit_code: int = 1

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

    def _append_command_output(
        self,
        *,
        stage: str,
        command: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Append structured command output to the build report payload."""
        # Store last command and exit code for failure artifact
        self._last_build_command = command
        self._last_exit_code = returncode

        entry = (
            f"## {stage}\n"
            f"COMMAND: {_format_command(command)}\n"
            f"EXIT_CODE: {returncode}\n\n"
            f"STDOUT:\n{_redact_build_text(stdout)}\n\n"
            f"STDERR:\n{_redact_build_text(stderr)}\n"
        )
        self._build_output = f"{self._build_output}\n\n{entry}" if self._build_output else entry

    def _build_reports_dir(self) -> Path:
        """Return the directory used for build logs and failure reports."""
        return Path(self.config.output_dir).parent / "reports"

    def _write_build_log(self) -> Path | None:
        """Persist captured command output, returning the path only when written."""
        if not self._build_output:
            return None

        backend_name = (
            self.config.backend.value
            if isinstance(self.config.backend, BuildBackend)
            else str(self.config.backend)
        )
        reports_dir = self._build_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        build_log_path = reports_dir / f"{int(time.time())}-{backend_name}.log"
        build_log_path.write_text(_redact_build_text(self._build_output))
        return build_log_path

    def _create_artifact(
        self,
        *,
        exit_code: int,
        build_log_path: Path | None,
        failure_report_path: Path | None,
        binary_path: Path | None = None,
        binary_size_bytes: int | None = None,
        git_commit_sha: str = "unknown",
        build_command: list[str] | None = None,
    ) -> BuildArtifact:
        """Create build provenance for success or failed command stages."""
        # Use provided build_command or fall back to stored last command
        cmd = (
            build_command
            or self._last_build_command
            or ["cmake", "--build", str(self.config.build_dir)]
        )
        return BuildArtifact(
            artifact_type="llama-server",
            backend=self.config.backend.value,
            created_at=time.time(),
            git_remote_url=_redact_build_text(self.config.git_remote_url),
            git_commit_sha=git_commit_sha,
            git_branch=self.config.git_branch,
            build_command=cmd,
            build_duration_seconds=time.time() - self._build_start_time,
            exit_code=exit_code,
            binary_path=binary_path,
            binary_size_bytes=binary_size_bytes,
            build_log_path=build_log_path,
            failure_report_path=failure_report_path,
        )

    def _write_failure_artifact(self, progress: BuildProgress) -> BuildArtifact:
        """Write failure diagnostics and return an artifact that points to them."""
        from .reports import write_failure_report

        logger.info("[failure] writing failure diagnostics for stage=%s", progress.stage)

        build_log_path = self._write_build_log()
        artifact = self._create_artifact(
            exit_code=self._last_exit_code,
            build_log_path=build_log_path,
            failure_report_path=None,
            build_command=self._last_build_command,
        )
        report = write_failure_report(
            report_dir=self._build_reports_dir(),
            build_artifact_json=json.dumps(artifact.to_dict(), indent=2),
            build_output=self._build_output,
            error_details=[
                {
                    "stage": progress.stage,
                    "status": progress.status,
                    "message": _redact_build_text(progress.message),
                }
            ],
            metadata={"backend": self.config.backend.value},
        )
        artifact.failure_report_path = report.report_dir
        report.save_to_file()
        logger.info("[failure] report written to %s", report.report_dir)
        return artifact

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

        logger.info(
            "[retry] stage=%s max_attempts=%s delay=%ss",
            stage_name,
            self.config.retry_attempts,
            self.config.retry_delay,
        )

        for attempt in range(self.config.retry_attempts):
            logger.info(
                "[retry] stage=%s attempt=%s/%s",
                stage_name,
                attempt + 1,
                self.config.retry_attempts,
            )
            result = stage_func()
            last_result = result

            # Emit progress if callback is set
            if self._progress_callback:
                self._progress_callback(result)

            if result.status == "success":
                logger.info("[retry] stage=%s succeeded on attempt %s", stage_name, attempt + 1)
                return result

            logger.warning(
                "[retry] stage=%s attempt %s failed: %s",
                stage_name,
                attempt + 1,
                result.message,
            )

            # If not last attempt, retry with exponential backoff
            if attempt < self.config.retry_attempts - 1:
                delay = self.config.retry_delay * (2**attempt)
                logger.info("[retry] stage=%s waiting %ss before retry", stage_name, delay)
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

        logger.error(
            "[retry] stage=%s exhausted all %s attempts",
            stage_name,
            self.config.retry_attempts,
        )
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
        backend_name = self.config.backend.value

        logger.info("[pipeline] starting build for backend=%s", backend_name)
        logger.info(
            "[pipeline] config: source_dir=%s build_dir=%s output_dir=%s",
            self.config.source_dir,
            self.config.build_dir,
            self.config.output_dir,
        )
        logger.info(
            "[pipeline] config: git_remote=%s git_branch=%s shallow_clone=%s jobs=%s update_sources=%s",
            _redact_build_text(self.config.git_remote_url),
            self.config.git_branch,
            self.config.shallow_clone,
            self.config.jobs,
            self.config.update_sources,
        )
        logger.info(
            "[pipeline] config: retry_attempts=%s retry_delay=%ss dry_run=%s",
            self.config.retry_attempts,
            self.config.retry_delay,
            self._dry_run,
        )

        # Acquire build lock
        from .config import Config

        config = Config()
        if not self._acquire_lock(config.build_lock_path):
            logger.error("[pipeline] failed to acquire build lock for %s", backend_name)
            return BuildResult(
                success=False,
                error_message=f"Failed to acquire build lock for {self.config.backend}",
            )

        try:
            # Stage 1: Preflight
            logger.info("[pipeline] stage 1/5: preflight")
            progress = self._run_with_retry(self._run_preflight, "preflight")
            if progress.status == "failed":
                logger.error("[pipeline] preflight failed: %s", progress.message)
                return BuildResult(
                    success=False,
                    error_message=f"Preflight failed: {progress.message}",
                    progress=progress,
                )
            logger.info("[pipeline] preflight completed: %s", progress.message)

            # Stage 2: Clone
            logger.info("[pipeline] stage 2/5: clone")
            progress = self._run_with_retry(self._run_clone, "clone")
            if progress.status == "failed":
                logger.error("[pipeline] clone failed: %s", progress.message)
                return BuildResult(
                    success=False,
                    error_message=f"Clone failed: {progress.message}",
                    progress=progress,
                )
            logger.info("[pipeline] clone completed: %s", progress.message)

            # Stage 3: Configure
            logger.info("[pipeline] stage 3/5: configure")
            progress = self._run_with_retry(self._run_configure, "configure")
            if progress.status == "failed":
                logger.error("[pipeline] configure failed: %s", progress.message)
                artifact = self._write_failure_artifact(progress)
                return BuildResult(
                    success=False,
                    artifact=artifact,
                    error_message=f"Configure failed: {_redact_build_text(progress.message)}",
                    progress=progress,
                )
            logger.info("[pipeline] configure completed: %s", progress.message)

            # Stage 4: Build
            logger.info("[pipeline] stage 4/5: build")
            progress = self._run_with_retry(self._run_build, "build")
            if progress.status == "failed":
                logger.error("[pipeline] build failed: %s", progress.message)
                artifact = self._write_failure_artifact(progress)
                return BuildResult(
                    success=False,
                    artifact=artifact,
                    error_message=f"Build failed: {_redact_build_text(progress.message)}",
                    progress=progress,
                )
            logger.info("[pipeline] build completed: %s", progress.message)

            # Stage 5: Finalize (Provenance)
            logger.info("[pipeline] stage 5/5: finalize")
            artifact = self._run_finalize(progress)
            if artifact is None:
                logger.error("[pipeline] finalize failed: could not write provenance")
                return BuildResult(
                    success=False,
                    error_message="Failed to write provenance",
                    progress=progress,
                )

            total_duration = _format_duration(time.time() - self._build_start_time)
            logger.info(
                "[pipeline] build succeeded for %s in %s (binary=%s size=%s commit=%s)",
                backend_name,
                total_duration,
                artifact.binary_path,
                artifact.binary_size_bytes,
                artifact.git_commit_sha,
            )
            return BuildResult(
                success=True,
                artifact=artifact,
                progress=progress,
            )

        except Exception as e:
            logger.exception("[pipeline] unhandled exception in build pipeline")
            return BuildResult(
                success=False,
                error_message=_redact_build_text(str(e)),
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
        logger.info("[both] starting sequential builds for SYCL then CUDA")

        # Create backend-specific directories for isolation
        sycl_build_dir = self.config.build_dir / "build_sycl"
        sycl_output_dir = self.config.output_dir / "output_sycl"
        cuda_build_dir = self.config.build_dir / "build_cuda"
        cuda_output_dir = self.config.output_dir / "output_cuda"

        # Build SYCL first
        logger.info("[both] starting SYCL build")
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
            update_sources=self.config.update_sources,
            git_commit=self.config.git_commit,
        )
        sycl_pipeline = BuildPipeline(sycl_config, self._progress_callback)
        sycl_pipeline.dry_run = self._dry_run

        sycl_result = sycl_pipeline.run()
        logger.info("[both] SYCL build finished: success=%s", sycl_result.success)

        # Build CUDA after SYCL completes
        logger.info("[both] starting CUDA build")
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
            update_sources=self.config.update_sources,
            git_commit=self.config.git_commit,
        )
        cuda_pipeline = BuildPipeline(cuda_config, self._progress_callback)
        cuda_pipeline.dry_run = self._dry_run

        cuda_result = cuda_pipeline.run()
        logger.info("[both] CUDA build finished: success=%s", cuda_result.success)

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

        logger.info("[preflight] detecting toolchain for backend=%s", self.config.backend.value)

        # Check toolchain (always run, even in dry-run mode)
        from .toolchain import detect_toolchain

        status = detect_toolchain()

        logger.debug(
            "[preflight] sycl_ready=%s cuda_ready=%s", status.is_sycl_ready, status.is_cuda_ready
        )

        if (self.config.backend == BuildBackend.SYCL and not status.is_sycl_ready) or (
            self.config.backend == BuildBackend.CUDA and not status.is_cuda_ready
        ):
            missing = status.missing_tools(self.config.backend)
            progress.status = "failed"
            backend_name = "SYCL" if self.config.backend == BuildBackend.SYCL else "CUDA"
            progress.message = f"Missing {backend_name} tools: {', '.join(missing)}"
            logger.error(
                "[preflight] failed: missing %s tools: %s", backend_name, ", ".join(missing)
            )
            return progress

        progress.status = "success"
        progress.message = "Toolchain validated"
        progress.progress_percent = 20
        logger.info("[preflight] toolchain validated for %s", self.config.backend.value)
        return progress

    def _handle_existing_source(self, progress: BuildProgress) -> BuildProgress | None:
        """Handle case where source directory already exists.

        Returns a BuildProgress if the clone should be skipped, None to proceed.
        """
        if self.config.update_sources and self._is_git_repo():
            logger.info(
                "[clone] source exists and update_sources=True; updating existing clone"
            )
            return self._update_sources(progress)
        if self.config.git_commit and self._is_git_repo():
            logger.info(
                "[clone] source exists; checking out configured git_commit=%s",
                self.config.git_commit,
            )
            return self._checkout_commit(progress)
        logger.info("[clone] source exists; skipping clone")
        progress.status = "skipped"
        progress.message = MSG_SOURCES_ALREADY_EXIST
        progress.progress_percent = 30
        return progress

    def _execute_clone(self, progress: BuildProgress) -> BuildProgress:
        """Execute the git clone operation."""
        try:
            if self._dry_run:
                progress.message = (
                    f"Would run: git clone --branch {self.config.git_branch} "
                    f"{_redact_build_text(self.config.git_remote_url)} {self.config.source_dir}"
                )
                progress.status = "success"
                progress.progress_percent = 30
                logger.info("[clone] dry-run: %s", progress.message)
                return progress

            self.config.source_dir.parent.mkdir(parents=True, exist_ok=True)

            cmd = ["git", "clone", "--branch", self.config.git_branch]
            if self.config.shallow_clone:
                cmd.extend(["--depth", "1"])
            cmd.extend([self.config.git_remote_url, str(self.config.source_dir)])

            logger.debug("[clone] running: %s", _format_command(cmd))
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            progress.message = f"Cloned {_redact_build_text(self.config.git_remote_url)}"
            progress.status = "success"
            progress.progress_percent = 30
            logger.info("[clone] cloned successfully into %s", self.config.source_dir)

            if self.config.git_commit:
                progress = self._checkout_commit(progress)
                if progress.status != "success":
                    return progress

        except subprocess.SubprocessError as e:
            return self._handle_clone_error(progress, e)
        except Exception as e:
            return self._handle_clone_error(progress, e)

        return progress

    def _handle_clone_error(
        self, progress: BuildProgress, error: Exception
    ) -> BuildProgress:
        """Handle clone failure with offline-continue support."""
        if self._source_exists():
            logger.warning("[clone] error but source exists; continuing offline: %s", str(error))
            progress.status = "skipped"
            progress.message = MSG_SOURCES_ALREADY_EXIST
            progress.progress_percent = 30
        else:
            stderr = _redact_build_text(getattr(error, "stderr", None) or str(error))
            logger.error("[clone] clone failed: %s", stderr)
            progress.status = "failed"
            progress.message = f"Clone failed: {stderr}"
        return progress

    def _run_clone(self) -> BuildProgress:
        """Run clone stage - clone git repository.

        Implements offline-continue support: if source directory exists and
        contains files, skip the clone operation and continue with existing
        sources. This allows builds to continue when network is unavailable
        but local clone exists.

        When ``update_sources`` is enabled and the source directory is a valid
        git repository, the pipeline fetches the latest changes and fast-forwards
        to ``origin/<branch>`` instead of skipping.

        Returns:
            BuildProgress with stage status
        """
        progress = BuildProgress(
            stage="clone",
            status="running",
            message="Cloning repository...",
            progress_percent=20,
        )

        logger.info("[clone] checking source_dir=%s", self.config.source_dir)
        logger.debug(
            "[clone] source_exists=%s is_git_repo=%s",
            self._source_exists(),
            self._is_git_repo() if self._source_exists() else False,
        )

        if self._source_exists():
            result = self._handle_existing_source(progress)
            if result is not None:
                return result

        logger.info(
            "[clone] source missing; cloning from %s",
            _redact_build_text(self.config.git_remote_url),
        )
        logger.debug(
            "[clone] branch=%s shallow=%s target=%s",
            self.config.git_branch,
            self.config.shallow_clone,
            self.config.source_dir,
        )

        return self._execute_clone(progress)

    def _is_git_repo(self) -> bool:
        """Check if source directory contains a valid git repository.

        Returns:
            True if ``source_dir/.git`` exists and is a directory, else False.
        """
        git_dir = self.config.source_dir / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def _checkout_commit(self, progress: BuildProgress) -> BuildProgress:
        """Checkout a specific commit hash if configured.

        Args:
            progress: Current progress object to update on failure.

        Returns:
            The progress object (modified in-place on error).
        """
        import subprocess

        if not self.config.git_commit:
            return progress

        commit = self.config.git_commit
        logger.info("[clone] checking out commit %s", commit)

        if self._dry_run:
            logger.info("[clone] dry-run: would checkout commit %s", commit)
            return progress

        try:
            checkout_cmd = ["git", "checkout", commit]
            logger.debug("[clone] running: %s", _format_command(checkout_cmd))
            result = subprocess.run(
                checkout_cmd,
                cwd=self.config.source_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            self._append_command_output(
                stage="clone (checkout)",
                command=checkout_cmd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            if result.returncode != 0:
                logger.warning(
                    "[clone] commit checkout failed (rc=%s): %s",
                    result.returncode,
                    _tail_lines(result.stderr),
                )
                progress.status = "skipped"
                progress.message = (
                    f"Commit checkout failed; continuing with existing sources: "
                    f"{_tail_lines(result.stderr)}"
                )
                progress.progress_percent = 30
            else:
                logger.info("[clone] checked out commit %s", commit)
        except subprocess.SubprocessError as e:
            logger.warning("[clone] commit checkout error: %s", str(e))
            progress.status = "skipped"
            progress.message = f"Commit checkout failed: {str(e)}"
            progress.progress_percent = 30

        return progress

    def _update_sources(self, progress: BuildProgress) -> BuildProgress:
        """Fetch and fast-forward an existing clone to the configured branch.

        On network failure the stage falls back to ``skipped`` so the build can
        continue with the local copy.

        Args:
            progress: Mutable progress object for this stage.

        Returns:
            Updated :class:`BuildProgress`.
        """
        import subprocess

        logger.info("[update-sources] fetching origin in %s", self.config.source_dir)

        if self._dry_run:
            progress.message = (
                f"Would run: git fetch origin && git checkout -B "
                f"{self.config.git_branch} origin/{self.config.git_branch}"
            )
            progress.status = "success"
            progress.progress_percent = 30
            logger.info("[update-sources] dry-run: %s", progress.message)
            return progress

        try:
            # 1. Fetch latest refs from the configured remote
            fetch_cmd = ["git", "fetch", "origin"]
            logger.debug("[update-sources] running: %s", _format_command(fetch_cmd))
            fetch_result = subprocess.run(
                fetch_cmd,
                cwd=self.config.source_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("[update-sources] fetch completed")
            logger.debug(
                "[update-sources] fetch stdout: %s", fetch_result.stdout.strip() or "(empty)"
            )

            # 2. Fast-forward to the latest commit on the configured branch
            checkout_cmd = [
                "git",
                "checkout",
                "-B",
                self.config.git_branch,
                f"origin/{self.config.git_branch}",
            ]
            logger.debug("[update-sources] running: %s", _format_command(checkout_cmd))
            result = subprocess.run(
                checkout_cmd,
                cwd=self.config.source_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            self._append_command_output(
                stage="clone (update)",
                command=checkout_cmd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

            if result.returncode != 0:
                logger.warning(
                    "[update-sources] checkout failed (rc=%s): %s",
                    result.returncode,
                    _tail_lines(result.stderr),
                )
                progress.status = "skipped"
                progress.message = (
                    f"Source update failed; continuing with existing sources: "
                    f"{_tail_lines(result.stderr)}"
                )
                progress.progress_percent = 30
                return progress

            progress.message = f"Updated sources to origin/{self.config.git_branch}"
            progress.status = "success"
            progress.progress_percent = 30
            logger.info("[update-sources] checked out %s", progress.message)

            # Checkout specific commit if requested
            if self.config.git_commit:
                progress = self._checkout_commit(progress)

        except subprocess.SubprocessError as e:
            logger.warning("[update-sources] network error during update: %s", str(e))
            progress.status = "skipped"
            progress.message = "Network unavailable; continuing with existing sources"
            progress.progress_percent = 30

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

        logger.info(
            "[configure] build_dir=%s backend=%s", self.config.build_dir, self.config.backend.value
        )

        # Check if build directory exists and CMakeCache.txt exists.
        # When update_sources is enabled we always re-configure because the
        # source tree may have changed (new files, different CMake flags, etc.).
        cmake_cache = self.config.build_dir / "CMakeCache.txt"
        if cmake_cache.exists() and not self.config.update_sources:
            logger.info("[configure] CMakeCache.txt exists; skipping configure")
            progress.status = "skipped"
            progress.message = "Already configured"
            progress.progress_percent = 50
            return progress

        # Generate cmake flags
        cmake_args = self._get_cmake_flags(self.config.backend)
        logger.debug("[configure] cmake_flags=%s", cmake_args)

        if self._dry_run:
            cmd = ["cmake", "-S", str(self.config.source_dir), "-B", str(self.config.build_dir)]
            cmd.extend(cmake_args)
            cmd = self._get_build_env_cmd(cmd)
            progress.message = f"Would run: {_format_command(cmd)}"
            progress.status = "success"
            progress.progress_percent = 50
            logger.info("[configure] dry-run: %s", progress.message)
            return progress

        try:
            # Create build directory if it doesn't exist
            self.config.build_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("[configure] created build_dir=%s", self.config.build_dir)

            cmd = ["cmake", "-S", str(self.config.source_dir), "-B", str(self.config.build_dir)]
            cmd.extend(cmake_args)
            cmd = self._get_build_env_cmd(cmd)

            logger.info("[configure] running cmake (this may take a while)")
            logger.debug("[configure] command: %s", _format_command(cmd))

            started_at = time.monotonic()
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            duration = _format_duration(time.monotonic() - started_at)

            logger.debug("[configure] cmake exited with rc=%s in %s", result.returncode, duration)

            self._append_command_output(
                stage="configure",
                command=cmd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

            if result.returncode != 0:
                logger.error("[configure] cmake failed (rc=%s)", result.returncode)
                progress.status = "failed"
                progress.message = _format_command_failure(
                    stage="CMake configure",
                    command=cmd,
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                return progress

            flags_str = " ".join(cmake_args)
            progress.message = (
                f"CMake configuration completed for {self.config.backend.value} in {duration} "
                f"(flags: {flags_str})"
            )
            progress.status = "success"
            progress.progress_percent = 50
            logger.info("[configure] %s", progress.message)

        except Exception as e:
            logger.error("[configure] exception: %s", str(e))
            progress.status = "failed"
            progress.message = f"Configure failed: {str(e)}"
            return progress

        return progress

    def _build_cmake_cmd(self) -> list[str]:
        """Construct the cmake --build command."""
        cmd = ["cmake", "--build", str(self.config.build_dir)]
        if self.config.jobs:
            cmd.extend(["-j", str(self.config.jobs)])
        return self._get_build_env_cmd(cmd)

    def _run_build_subprocess(self, cmd: list[str], progress: BuildProgress) -> BuildProgress:
        """Execute cmake --build with real-time output streaming."""
        started_at = time.monotonic()

        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        ) as proc:
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []
            stdout_done = threading.Event()
            stderr_done = threading.Event()

            def read_stdout() -> None:
                if proc.stdout:
                    for line in proc.stdout:
                        stdout_lines.append(line.rstrip("\n"))
                        print(line.rstrip("\n"), file=sys.stderr)
                stdout_done.set()

            def read_stderr() -> None:
                if proc.stderr:
                    for line in proc.stderr:
                        stderr_lines.append(line.rstrip("\n"))
                        print(line.rstrip("\n"), file=sys.stderr)
                stderr_done.set()

            stdout_t = threading.Thread(target=read_stdout)
            stderr_t = threading.Thread(target=read_stderr)
            stdout_t.start()
            stderr_t.start()
            proc.wait()
            stdout_t.join()
            stderr_t.join()

        result_stdout = "\n".join(stdout_lines)
        result_stderr = "\n".join(stderr_lines)
        returncode = proc.returncode
        duration = _format_duration(time.monotonic() - started_at)

        logger.debug("[build] cmake exited with rc=%s in %s", returncode, duration)

        self._append_command_output(
            stage="build",
            command=cmd,
            returncode=returncode,
            stdout=result_stdout,
            stderr=result_stderr,
        )

        if returncode == 0:
            progress.message = f"Build completed for {self.config.backend.value} in {duration}"
            progress.status = "success"
            progress.progress_percent = 75
            logger.info("[build] %s", progress.message)
        else:
            logger.error("[build] compilation failed (rc=%s)", returncode)
            progress.status = "failed"
            progress.message = _format_command_failure(
                stage="Build",
                command=cmd,
                returncode=returncode,
                stdout=result_stdout,
                stderr=result_stderr,
            )

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

        logger.info("[build] starting compilation for %s", self.config.backend.value)

        if self._dry_run:
            cmd = self._build_cmake_cmd()
            progress.message = f"Would run: {_format_command(cmd)}"
            progress.status = "success"
            progress.progress_percent = 75
            logger.info("[build] dry-run: %s", progress.message)
            return progress

        try:
            cmd = self._build_cmake_cmd()
            logger.info("[build] running cmake --build (this may take several minutes)")
            logger.info("[build] command: %s", _format_command(cmd))
            logger.debug("[build] jobs=%s", self.config.jobs)
            return self._run_build_subprocess(cmd, progress)
        except Exception as e:
            logger.error("[build] exception: %s", str(e))
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
            logger.warning("[finalize] build stage incomplete or failed; skipping finalize")
            return None

        logger.info("[finalize] collecting build metadata")

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
                logger.info("[finalize] git commit=%s", git_commit_sha)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("[finalize] could not determine git commit SHA")

        # Find built binary
        binary_path = None
        binary_size_bytes = None
        build_dir_bin = self.config.build_dir / "bin"
        if build_dir_bin.exists():
            server_binary = build_dir_bin / "llama-server"
            if server_binary.exists():
                binary_path = server_binary
                binary_size_bytes = server_binary.stat().st_size
                logger.info(
                    "[finalize] found binary: %s (%s bytes)",
                    binary_path,
                    binary_size_bytes,
                )
            else:
                logger.warning("[finalize] expected binary not found: %s", server_binary)
        else:
            logger.warning("[finalize] build bin/ directory not found: %s", build_dir_bin)

        build_log_path = self._write_build_log()
        if build_log_path:
            logger.info("[finalize] build log written to %s", build_log_path)
        else:
            logger.debug("[finalize] no build log to write")

        artifact = self._create_artifact(
            exit_code=0,
            binary_path=binary_path,
            binary_size_bytes=binary_size_bytes,
            build_log_path=build_log_path,
            failure_report_path=None,
            git_commit_sha=git_commit_sha,
        )

        # Write provenance
        logger.info(
            "[finalize] writing provenance to %s", self.config.output_dir / "build-artifact.json"
        )
        if self._write_provenance(artifact):
            logger.info("[finalize] provenance written successfully")
            return artifact
        logger.error("[finalize] provenance write failed")
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
            flags.extend(
                [
                    f"-D{BuildConfig.GGML_SYCL}=ON",
                    "-DCMAKE_C_COMPILER=icx",
                    "-DCMAKE_CXX_COMPILER=icpx",
                ]
            )
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
        cmd_str = shlex.join(cmd)
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

            logger.debug("[provenance] atomically wrote %s", final_file)
            return True

        except Exception as e:
            logger.warning("[provenance] failed to write: %s", e)
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
            logger.debug("[lock] dry-run: skipping lock acquisition")
            return True

        logger.info("[lock] acquiring lock at %s", lock_path)

        try:
            # First, check if there's a stale lock and remove it safely
            if lock_path.exists() and self._is_lock_stale(lock_path):
                logger.warning("[lock] removing stale lock at %s", lock_path)
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
            logger.info(
                "[lock] acquired for backend=%s pid=%s", self.config.backend.value, os.getpid()
            )
            return True

        except FileExistsError:
            # Another process holds the lock
            logger.error("[lock] already held by another process: %s", lock_path)
            return False
        except OSError as e:
            logger.error("[lock] failed to acquire: %s", e)
            return False
        except Exception as e:
            logger.error("[lock] failed to acquire: %s", e)
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
            logger.info("[lock] releasing %s", self._lock_file)
            with contextlib.suppress(Exception):
                self._lock_file.unlink()
            self._lock_file = None
        else:
            logger.debug("[lock] no active lock to release")

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
