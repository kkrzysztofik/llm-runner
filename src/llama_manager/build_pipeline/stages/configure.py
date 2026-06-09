"""Configure stage — CMake configuration, flags, and build environment."""

import time

from loguru import logger

from .._context import _BuildContext
from ..models import BUILD_CANCELLED_MESSAGE, BuildBackend, BuildConfig, BuildProgress
from ..utils import (
    _cancel_requested,
    _format_command,
    _format_command_failure,
    _format_duration,
    get_build_env_cmd,
    run_command_with_cancel,
)


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
    if cmake_cache.exists() and ctx.config.clean_cache:
        try:
            cmake_cache.unlink()
            logger.info("[configure] removed stale CMakeCache.txt (clean_cache=True)")
        except OSError as exc:
            logger.error("[configure] failed to remove CMakeCache.txt: %s", exc)
    if cmake_cache.exists() and not ctx.config.update_sources:
        logger.info("[configure] CMakeCache.txt exists; skipping configure")
        progress.status = "skipped"
        progress.message = "Already configured"
        progress.progress_percent = 50
        return progress

    cmake_args = get_cmake_flags(ctx.config.backend, ctx.config.git_remote_url)
    if ctx.config.build_args:
        cmake_args.extend(ctx.config.build_args)
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

        # Fail fast if cancellation was requested before spawning cmake
        if _cancel_requested(ctx.cancel_event):
            logger.info("[configure] cancelled before spawn")
            progress.status = "failed"
            progress.message = BUILD_CANCELLED_MESSAGE
            return progress

        logger.info("[configure] running cmake (this may take a while)")
        logger.debug("[configure] command: %s", _format_command(cmd))

        started_at = time.monotonic()
        returncode, stdout_str, stderr_str = run_command_with_cancel(
            cmd,
            cancel_event=ctx.cancel_event,
            set_active_proc=lambda proc: setattr(ctx, "active_proc", proc),
            timeout_seconds=float(ctx.config.build_timeout_seconds),
        )
        duration = _format_duration(time.monotonic() - started_at)

        logger.debug("[configure] cmake exited with rc=%s in %s", returncode, duration)

        ctx.append_command_output(
            stage="configure",
            command=cmd,
            returncode=returncode,
            stdout=stdout_str,
            stderr=stderr_str,
        )

        if _cancel_requested(ctx.cancel_event):
            logger.info("[configure] cancelled by user")
            progress.status = "failed"
            progress.message = BUILD_CANCELLED_MESSAGE
            return progress

        if returncode == -1:
            logger.error("[configure] cmake timed out after %s", duration)
            progress.status = "failed"
            progress.message = f"Configure timed out after {ctx.config.build_timeout_seconds}s"
            return progress

        if returncode != 0:
            logger.error("[configure] cmake failed (rc=%s)", returncode)
            progress.status = "failed"
            progress.message = _format_command_failure(
                stage="CMake configure",
                command=cmd,
                returncode=returncode,
                stdout=stdout_str,
                stderr=stderr_str,
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


def get_cmake_flags(backend: BuildBackend, git_remote_url: str = "") -> list[str]:
    """Return CMake flags for the specified backend."""
    is_beellama_cuda = backend == BuildBackend.CUDA and "beellama" in git_remote_url.lower()
    flags = [
        "-DBUILD_SERVER=ON",
        "-DGGML_NATIVE=ON" if is_beellama_cuda else "-DGGML_NATIVE=OFF",
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
        if is_beellama_cuda:
            flags.extend(
                [
                    "-DGGML_CUDA_FA=ON",
                    "-DGGML_CUDA_FA_ALL_QUANTS=ON",
                ]
            )
    return flags
