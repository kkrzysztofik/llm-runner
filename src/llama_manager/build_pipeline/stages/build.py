"""Build stage — cmake --build compilation with real-time output streaming."""

import os
import time

from loguru import logger

from .._context import _BuildContext
from ..models import BUILD_CANCELLED_MESSAGE, BuildProgress
from ..utils import (
    _cancel_requested,
    _format_command,
    _format_command_failure,
    _format_duration,
    get_build_env_cmd,
    run_command_with_cancel,
)


def run_build(ctx: _BuildContext) -> BuildProgress:
    """Compile via cmake --build."""
    progress = BuildProgress(
        stage="build",
        status="running",
        message="Building...",
        progress_percent=50,
    )

    logger.info("[build] starting compilation for {}", ctx.config.backend.value)

    if ctx.dry_run:
        return _run_build_dry_run(ctx, progress)
    return _run_build_real(ctx, progress)


def _run_build_dry_run(ctx: _BuildContext, progress: BuildProgress) -> BuildProgress:
    """Dry-run: print the command without executing."""
    cmd = _build_cmake_cmd(ctx)
    progress.message = f"Would run: {_format_command(cmd)}"
    progress.status = "success"
    progress.progress_percent = 75
    logger.info("[build] dry-run: {}", progress.message)
    return progress


def _run_build_real(ctx: _BuildContext, progress: BuildProgress) -> BuildProgress:
    """Execute cmake --build with real-time output streaming."""
    try:
        cmd = _build_cmake_cmd(ctx)
        logger.info("[build] running cmake --build (this may take several minutes)")
        logger.info("[build] command: {}", _format_command(cmd))
        logger.debug("[build] jobs={} (effective)", _effective_parallel_jobs(ctx))
        return _run_build_subprocess(cmd, ctx, progress)
    except Exception as e:
        logger.error("[build] exception: {}", str(e))
        progress.status = "failed"
        progress.message = f"Build failed: {str(e)}"

    return progress


def _effective_parallel_jobs(ctx: _BuildContext) -> int:
    """Parallel compile jobs: explicit config or all logical CPUs."""
    if ctx.config.jobs is not None:
        return ctx.config.jobs
    return os.cpu_count() or 1


def _build_cmake_cmd(ctx: _BuildContext) -> list[str]:
    """Construct the cmake --build command."""
    cmd = ["cmake", "--build", str(ctx.config.build_dir)]
    cmd.extend(["-j", str(_effective_parallel_jobs(ctx))])
    return get_build_env_cmd(cmd, ctx.config.backend)


def _run_build_subprocess(
    cmd: list[str], ctx: _BuildContext, progress: BuildProgress
) -> BuildProgress:
    """Execute cmake --build with real-time output streaming."""
    started_at = time.monotonic()

    def emit_line(line: str) -> None:
        if ctx.progress_callback:
            line_progress = BuildProgress(
                stage="build",
                status="running",
                message="",
                progress_percent=progress.progress_percent,
                output_line=line.rstrip("\n"),
            )
            ctx.progress_callback(line_progress)

    returncode, result_stdout, result_stderr = run_command_with_cancel(
        cmd,
        cancel_event=ctx.cancel_event,
        set_active_proc=lambda proc: setattr(ctx, "active_proc", proc),
        timeout_seconds=float(ctx.config.build_timeout_seconds),
        line_callback=emit_line,
    )
    duration = _format_duration(time.monotonic() - started_at)

    logger.debug("[build] cmake exited with rc={} in {}", returncode, duration)

    ctx.append_command_output(
        stage="build",
        command=cmd,
        returncode=returncode,
        stdout=result_stdout,
        stderr=result_stderr,
    )

    if _cancel_requested(ctx.cancel_event):
        logger.info("[build] cancelled by user")
        progress.status = "failed"
        progress.message = BUILD_CANCELLED_MESSAGE
        return progress

    if returncode == -1:
        progress.status = "failed"
        progress.message = f"Build timed out after {ctx.config.build_timeout_seconds}s"
        logger.error("[build] {}", progress.message)
        return progress

    if returncode == 0:
        progress.message = f"Build completed for {ctx.config.backend.value} in {duration}"
        progress.status = "success"
        progress.progress_percent = 75
        logger.info("[build] {}", progress.message)
    else:
        logger.error("[build] compilation failed (rc={})", returncode)
        progress.status = "failed"
        progress.message = _format_command_failure(
            stage="Build",
            command=cmd,
            returncode=returncode,
            stdout=result_stdout,
            stderr=result_stderr,
        )

    return progress
