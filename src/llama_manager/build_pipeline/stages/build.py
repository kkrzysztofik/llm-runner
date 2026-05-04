"""Build stage — cmake --build compilation with real-time output streaming."""

import logging
import subprocess
import threading
import time

from .._context import _BuildContext
from ..models import BuildProgress
from ..utils import _format_command, _format_command_failure, _format_duration
from .configure import get_build_env_cmd

logger = logging.getLogger(__name__)


def run_build(ctx: _BuildContext) -> BuildProgress:
    """Compile via cmake --build."""
    progress = BuildProgress(
        stage="build",
        status="running",
        message="Building...",
        progress_percent=50,
    )

    logger.info("[build] starting compilation for %s", ctx.config.backend.value)

    if ctx.dry_run:
        cmd = _build_cmake_cmd(ctx)
        progress.message = f"Would run: {_format_command(cmd)}"
        progress.status = "success"
        progress.progress_percent = 75
        logger.info("[build] dry-run: %s", progress.message)
        return progress

    try:
        cmd = _build_cmake_cmd(ctx)
        logger.info("[build] running cmake --build (this may take several minutes)")
        logger.info("[build] command: %s", _format_command(cmd))
        logger.debug("[build] jobs=%s", ctx.config.jobs)
        return _run_build_subprocess(cmd, ctx, progress)
    except Exception as e:
        logger.error("[build] exception: %s", str(e))
        progress.status = "failed"
        progress.message = f"Build failed: {str(e)}"

    return progress


def _build_cmake_cmd(ctx: _BuildContext) -> list[str]:
    """Construct the cmake --build command."""
    cmd = ["cmake", "--build", str(ctx.config.build_dir)]
    if ctx.config.jobs:
        cmd.extend(["-j", str(ctx.config.jobs)])
    return get_build_env_cmd(cmd, ctx.config.backend)


def _run_build_subprocess(
    cmd: list[str], ctx: _BuildContext, progress: BuildProgress
) -> BuildProgress:
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

        def read_stdout() -> None:
            if proc.stdout:
                for line in proc.stdout:
                    stdout_lines.append(line.rstrip("\n"))
                    logger.debug("[build] stdout: %s", line.rstrip("\n"))

        def read_stderr() -> None:
            if proc.stderr:
                for line in proc.stderr:
                    stderr_lines.append(line.rstrip("\n"))
                    logger.debug("[build] stderr: %s", line.rstrip("\n"))

        stdout_t = threading.Thread(target=read_stdout)
        stderr_t = threading.Thread(target=read_stderr)
        stdout_t.start()
        stderr_t.start()
        try:
            proc.wait(timeout=ctx.config.build_timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            proc.wait()
            stdout_t.join(timeout=5)
            stderr_t.join(timeout=5)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            progress.status = "failed"
            progress.message = f"Build timed out after {ctx.config.build_timeout_seconds}s"
            logger.error("[build] %s", progress.message)
            ctx.append_command_output(
                stage="build",
                command=cmd,
                returncode=-1,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
            )
            return progress
        stdout_t.join()
        stderr_t.join()

    result_stdout = "\n".join(stdout_lines)
    result_stderr = "\n".join(stderr_lines)
    returncode = proc.returncode
    duration = _format_duration(time.monotonic() - started_at)

    logger.debug("[build] cmake exited with rc=%s in %s", returncode, duration)

    ctx.append_command_output(
        stage="build",
        command=cmd,
        returncode=returncode,
        stdout=result_stdout,
        stderr=result_stderr,
    )

    if returncode == 0:
        progress.message = f"Build completed for {ctx.config.backend.value} in {duration}"
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
