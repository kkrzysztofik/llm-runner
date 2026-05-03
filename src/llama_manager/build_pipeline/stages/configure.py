"""Configure stage — CMake configuration, flags, and build environment."""

import logging
import shlex
import subprocess
import time

from .._context import _BuildContext
from ..models import BuildBackend, BuildConfig, BuildProgress
from ..utils import (
    _INTEL_SETVARS_SH,
    _format_command,
    _format_command_failure,
    _format_duration,
)

logger = logging.getLogger(__name__)


def run_configure(ctx: _BuildContext) -> BuildProgress:
    """Run CMake configuration stage."""
    progress = BuildProgress(
        stage="configure",
        status="running",
        message="Configuring with CMake...",
        progress_percent=30,
    )

    logger.info(
        "[configure] build_dir=%s backend=%s", ctx.config.build_dir, ctx.config.backend.value
    )

    cmake_cache = ctx.config.build_dir / "CMakeCache.txt"
    if cmake_cache.exists() and not ctx.config.update_sources:
        logger.info("[configure] CMakeCache.txt exists; skipping configure")
        progress.status = "skipped"
        progress.message = "Already configured"
        progress.progress_percent = 50
        return progress

    cmake_args = get_cmake_flags(ctx.config.backend)
    logger.debug("[configure] cmake_flags=%s", cmake_args)

    if ctx.dry_run:
        cmd = ["cmake", "-S", str(ctx.config.source_dir), "-B", str(ctx.config.build_dir)]
        cmd.extend(cmake_args)
        cmd = get_build_env_cmd(cmd, ctx.config.backend)
        progress.message = f"Would run: {_format_command(cmd)}"
        progress.status = "success"
        progress.progress_percent = 50
        logger.info("[configure] dry-run: %s", progress.message)
        return progress

    try:
        ctx.config.build_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("[configure] created build_dir=%s", ctx.config.build_dir)

        cmd = ["cmake", "-S", str(ctx.config.source_dir), "-B", str(ctx.config.build_dir)]
        cmd.extend(cmake_args)
        cmd = get_build_env_cmd(cmd, ctx.config.backend)

        logger.info("[configure] running cmake (this may take a while)")
        logger.debug("[configure] command: %s", _format_command(cmd))

        started_at = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=ctx.config.build_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            duration = _format_duration(time.monotonic() - started_at)
            logger.error("[configure] cmake timed out after %s", duration)
            progress.status = "failed"
            progress.message = f"Configure timed out after {ctx.config.build_timeout_seconds}s"
            ctx.append_command_output(
                stage="configure",
                command=cmd,
                returncode=-1,
                stdout="",
                stderr=f"Timed out after {ctx.config.build_timeout_seconds}s",
            )
            return progress
        duration = _format_duration(time.monotonic() - started_at)

        logger.debug("[configure] cmake exited with rc=%s in %s", result.returncode, duration)

        ctx.append_command_output(
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
            f"CMake configuration completed for {ctx.config.backend.value} in {duration} "
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


def get_cmake_flags(backend: BuildBackend) -> list[str]:
    """Return CMake flags for the specified backend."""
    flags = [
        "-DBUILD_SERVER=ON",
        "-DGGML_NATIVE=OFF",
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


def get_build_env_cmd(cmd: list[str], backend: BuildBackend) -> list[str]:
    """Wrap a command with the Intel oneAPI environment when building for SYCL.

    Sources ``/opt/intel/oneapi/setvars.sh`` via ``bash -c`` so that Intel
    compilers and libraries are on PATH. Returns the command unchanged for
    non-SYCL backends or when the script is missing.
    """
    if backend != BuildBackend.SYCL:
        return cmd
    if not _INTEL_SETVARS_SH.exists():
        return cmd
    cmd_str = shlex.join(cmd)
    return [
        "bash",
        "-c",
        f'source "{_INTEL_SETVARS_SH}" && {cmd_str}',
    ]
