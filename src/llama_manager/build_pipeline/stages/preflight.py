"""Preflight stage — toolchain validation."""

from loguru import logger

from .._context import _BuildContext
from ..models import BuildBackend, BuildProgress


def run_preflight(ctx: _BuildContext) -> BuildProgress:
    """Validate toolchain availability for the configured backend."""
    progress = BuildProgress(
        stage="preflight",
        status="running",
        message="Validating toolchain...",
        progress_percent=0.0,
    )

    logger.info("[preflight] detecting toolchain for backend=%s", ctx.config.backend.value)

    from ...toolchain import detect_toolchain

    status = detect_toolchain()

    logger.debug(
        "[preflight] sycl_ready=%s cuda_ready=%s", status.is_sycl_ready, status.is_cuda_ready
    )

    if (ctx.config.backend == BuildBackend.SYCL and not status.is_sycl_ready) or (
        ctx.config.backend == BuildBackend.CUDA and not status.is_cuda_ready
    ):
        missing = status.missing_tools(ctx.config.backend)
        backend_name = "SYCL" if ctx.config.backend == BuildBackend.SYCL else "CUDA"
        progress.status = "failed"
        progress.message = f"Missing {backend_name} tools: {', '.join(missing)}"
        logger.error("[preflight] failed: missing %s tools: %s", backend_name, ", ".join(missing))
        return progress

    progress.status = "success"
    progress.message = "Toolchain validated"
    progress.progress_percent = 20
    logger.info("[preflight] toolchain validated for %s", ctx.config.backend.value)
    return progress
