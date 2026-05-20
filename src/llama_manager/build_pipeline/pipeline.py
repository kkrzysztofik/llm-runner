"""BuildPipeline — 5-stage build orchestrator for llama.cpp."""

import threading
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from ._context import _BuildContext
from .lock import acquire_lock, get_lock_error_message, is_lock_stale, release_lock
from .models import (
    BUILD_CANCELLED_MESSAGE,
    BuildBackend,
    BuildConfig,
    BuildProgress,
    BuildResult,
)
from .stages.build import run_build
from .stages.clone import run_clone
from .stages.configure import run_configure
from .stages.finalize import run_finalize
from .stages.preflight import run_preflight
from .utils import _format_duration, _redact_build_text


class BuildPipeline:
    """Build pipeline for llama.cpp implementing 5 stages:
    preflight → clone → configure → build → finalize.
    """

    def __init__(
        self,
        config: BuildConfig,
        progress_callback: Callable[[BuildProgress], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self._dry_run = False
        self._lock_file: Path | None = None
        self._progress_callback = progress_callback
        self._cancel_event = cancel_event
        self._ctx: _BuildContext | None = None

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        self._dry_run = value

    def _cancelled_progress(self, stage_name: str) -> BuildProgress:
        return BuildProgress(
            stage=stage_name,
            status="failed",
            message=BUILD_CANCELLED_MESSAGE,
            progress_percent=0.0,
        )

    def _cancelled_build_result(self) -> BuildResult:
        return BuildResult(success=False, error_message=BUILD_CANCELLED_MESSAGE)

    def _emit_retry_wait(self, stage_name: str, attempt: int, delay: float) -> BuildProgress | None:
        """Sleep between retries; return cancelled progress if the user stopped the build."""
        retry_p = BuildProgress(
            stage=stage_name,
            status="retrying",
            message=f"Stage failed, retrying in {delay}s "
            f"(attempt {attempt + 2}/{self.config.retry_attempts})",
            progress_percent=0.0,
            retries_remaining=self.config.retry_attempts - attempt - 1,
        )
        if self._progress_callback:
            self._progress_callback(retry_p)
        if self._sleep_until_cancel_or_timeout(delay):
            return self._cancelled_progress(stage_name)
        return None

    def _run_with_retry(
        self, stage_func: Callable[[], BuildProgress], stage_name: str
    ) -> BuildProgress:
        """Run a stage function with exponential-backoff retry."""
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
            if self._cancel_requested():
                return self._cancelled_progress(stage_name)
            logger.info(
                "[retry] stage=%s attempt=%s/%s",
                stage_name,
                attempt + 1,
                self.config.retry_attempts,
            )
            result = stage_func()
            last_result = result
            if self._progress_callback:
                self._progress_callback(result)
            if result.status in {"success", "skipped"}:
                logger.info(
                    "[retry] stage=%s completed with status=%s on attempt %s",
                    stage_name,
                    result.status,
                    attempt + 1,
                )
                return result
            if result.message == BUILD_CANCELLED_MESSAGE:
                logger.info("[retry] stage=%s stopped (build cancelled)", stage_name)
                return result
            logger.warning(
                "[retry] stage=%s attempt %s failed: %s",
                stage_name,
                attempt + 1,
                result.message,
            )
            if attempt < self.config.retry_attempts - 1:
                delay = self.config.retry_delay * (2**attempt)
                logger.info("[retry] stage=%s waiting %ss before retry", stage_name, delay)
                cancelled = self._emit_retry_wait(stage_name, attempt, delay)
                if cancelled is not None:
                    return cancelled
        logger.error(
            "[retry] stage=%s exhausted all %s attempts", stage_name, self.config.retry_attempts
        )
        return last_result

    def _cancel_requested(self) -> bool:
        return self._cancel_event is not None and self._cancel_event.is_set()

    def _sleep_until_cancel_or_timeout(self, seconds: float) -> bool:
        """Sleep in short chunks; return True if cancel was requested."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._cancel_requested():
                return True
            time.sleep(min(0.25, deadline - time.monotonic()))
        return False

    def kill_active_subprocess(self) -> None:
        """Terminate the in-flight stage subprocess (cmake compile/configure)."""
        from .utils import terminate_process_tree

        ctx = self._ctx
        if ctx is None or ctx.active_proc is None:
            return
        terminate_process_tree(ctx.active_proc, use_process_group=True)

    def _run_stage_batch(
        self,
        ctx: _BuildContext,
        stages: list[tuple[str, Callable[[], BuildProgress]]],
        *,
        write_artifact: bool,
        progress: BuildProgress,
    ) -> tuple[BuildResult | None, BuildProgress]:
        """Run a group of pipeline stages; return an error result or (None, progress)."""
        for stage_name, stage_fn in stages:
            if self._cancel_requested():
                return self._cancelled_build_result(), progress
            progress = self._run_with_retry(stage_fn, stage_name)
            if progress.status != "failed":
                logger.info("[pipeline] %s completed: %s", stage_name, progress.message)
                continue
            if progress.message == BUILD_CANCELLED_MESSAGE:
                return self._cancelled_build_result(), progress
            logger.error("[pipeline] %s failed: %s", stage_name, progress.message)
            if write_artifact:
                artifact = ctx.write_failure_artifact(progress)
                return (
                    BuildResult(
                        success=False,
                        artifact=artifact,
                        error_message=f"{stage_name.capitalize()} failed: "
                        f"{_redact_build_text(progress.message)}",
                        progress=progress,
                    ),
                    progress,
                )
            return (
                BuildResult(
                    success=False,
                    error_message=f"{stage_name.capitalize()} failed: {progress.message}",
                    progress=progress,
                ),
                progress,
            )
        return None, progress

    def run(self) -> BuildResult:
        """Execute the full 5-stage pipeline and return a BuildResult."""
        if self.config.backend == BuildBackend.BOTH:
            raise ValueError("BuildBackend.BOTH must use run_both_backends() instead")

        build_start_time = time.time()
        ctx = _BuildContext(
            config=self.config,
            dry_run=self._dry_run,
            build_start_time=build_start_time,
            progress_callback=self._progress_callback,
            cancel_event=self._cancel_event,
        )
        self._ctx = ctx

        logger.info("[pipeline] starting build for backend=%s", self.config.backend.value)
        logger.info(
            "[pipeline] config: source_dir=%s build_dir=%s output_dir=%s",
            ctx.config.source_dir,
            ctx.config.build_dir,
            ctx.config.output_dir,
        )
        logger.info(
            "[pipeline] config: git_remote=%s git_branch=%s shallow_clone=%s "
            "jobs=%s update_sources=%s",
            _redact_build_text(ctx.config.git_remote_url),
            ctx.config.git_branch,
            ctx.config.shallow_clone,
            ctx.config.jobs,
            ctx.config.update_sources,
        )
        logger.info(
            "[pipeline] config: retry_attempts=%s retry_delay=%ss dry_run=%s",
            ctx.config.retry_attempts,
            ctx.config.retry_delay,
            self._dry_run,
        )

        from ..config import Config

        if not self._acquire_lock(Config().build_lock_path):
            logger.error(
                "[pipeline] failed to acquire build lock for %s", self.config.backend.value
            )
            return BuildResult(
                success=False,
                error_message=f"Failed to acquire build lock for {self.config.backend}",
            )

        try:
            progress = BuildProgress(
                stage="pipeline", status="pending", message="", progress_percent=0
            )
            early_error, progress = self._run_stage_batch(
                ctx,
                [
                    ("preflight", lambda: run_preflight(ctx)),
                    ("clone", lambda: run_clone(ctx)),
                ],
                write_artifact=False,
                progress=progress,
            )
            if early_error is not None:
                return early_error

            build_error, progress = self._run_stage_batch(
                ctx,
                [
                    ("configure", lambda: run_configure(ctx)),
                    ("build", lambda: run_build(ctx)),
                ],
                write_artifact=True,
                progress=progress,
            )
            if build_error is not None:
                return build_error

            if self._cancel_requested():
                return self._cancelled_build_result()
            artifact = run_finalize(ctx, progress)
            if artifact is None:
                logger.error("[pipeline] finalize failed: could not write provenance")
                return BuildResult(
                    success=False,
                    error_message="Failed to write provenance",
                    progress=progress,
                )

            total_duration = _format_duration(time.time() - build_start_time)
            logger.info(
                "[pipeline] build succeeded for %s in %s (binary=%s size=%s commit=%s)",
                self.config.backend.value,
                total_duration,
                artifact.binary_path,
                artifact.binary_size_bytes,
                artifact.git_commit_sha,
            )
            return BuildResult(success=True, artifact=artifact, progress=progress)

        except Exception as e:
            logger.opt(exception=True).error("[pipeline] unhandled exception in build pipeline")
            failure_progress = BuildProgress(
                stage="pipeline",
                status="failed",
                message=_redact_build_text(str(e)),
                progress_percent=0,
            )
            try:
                artifact = ctx.write_failure_artifact(failure_progress)
            except Exception:
                artifact = None
            return BuildResult(
                success=False,
                artifact=artifact,
                error_message=_redact_build_text(str(e)),
                progress=failure_progress,
            )

        finally:
            self._ctx = None
            self._release_lock()

    def run_both_backends(self) -> list[BuildResult]:
        """Run SYCL then CUDA builds sequentially, each with its own lock."""
        logger.info("[both] starting sequential builds for SYCL then CUDA")
        results: list[BuildResult] = []
        for backend, subdir in [
            (BuildBackend.SYCL, "sycl"),
            (BuildBackend.CUDA, "cuda"),
        ]:
            logger.info("[both] starting %s build", backend.value.upper())
            sub_config = BuildConfig(
                backend=backend,
                source_dir=self.config.source_dir,
                build_dir=self.config.build_dir / f"build_{subdir}",
                output_dir=self.config.output_dir / f"output_{subdir}",
                git_remote_url=self.config.git_remote_url,
                git_branch=self.config.git_branch,
                retry_attempts=self.config.retry_attempts,
                retry_delay=self.config.retry_delay,
                shallow_clone=self.config.shallow_clone,
                jobs=self.config.jobs,
                update_sources=self.config.update_sources,
                git_commit=self.config.git_commit,
            )
            sub_pipeline = BuildPipeline(sub_config, self._progress_callback, self._cancel_event)
            sub_pipeline.dry_run = self._dry_run
            result = sub_pipeline.run()
            logger.info(
                "[both] %s build finished: success=%s", backend.value.upper(), result.success
            )
            results.append(result)
        return results

    # ── Lock management ──────────────────────────────────────────────────────

    def _acquire_lock(self, lock_path: Path) -> bool:
        acquired = acquire_lock(lock_path, self.config.backend.value, dry_run=self._dry_run)
        if acquired and not self._dry_run:
            self._lock_file = lock_path
        return acquired

    def release_lock(self) -> None:
        """Release the build lock (public API)."""
        self._release_lock()

    def _release_lock(self) -> None:
        release_lock(self._lock_file)
        self._lock_file = None

    def _is_lock_stale(self, lock_path: Path) -> bool:
        return is_lock_stale(lock_path)

    def _get_lock_error_message(self, lock_path: Path) -> str:
        return get_lock_error_message(lock_path)
