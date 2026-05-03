"""High-level build orchestration for llama.cpp.

Provides a thin wrapper around BuildPipeline that translates
application-level Config into a BuildConfig and manages the pipeline
lifecycle without CLI or TUI concerns.
"""

from collections.abc import Callable
from pathlib import Path

from ..config import Config
from .models import BuildBackend, BuildConfig, BuildProgress, BuildResult
from .pipeline import BuildPipeline


def run_build_for_backend(
    backend: str,
    dry_run: bool,
    config: Config,
    progress_callback: Callable[[BuildProgress], None] | None = None,
    pipeline_callback: Callable[[BuildPipeline], None] | None = None,
) -> BuildResult:
    """Create and run a BuildPipeline for a single backend.

    Args:
        backend: Build backend ("sycl" or "cuda").
        dry_run: If True, pipeline prints commands without executing.
        config: A Config instance providing paths and build defaults.
        progress_callback: Optional callback receiving BuildProgress updates.
        pipeline_callback: Optional callback invoked with the BuildPipeline
            instance *before* ``run()`` is called.  This lets the caller
            (e.g. the TUI controller) keep a reference for signal handling.

    Returns:
        BuildResult from the pipeline execution.
    """
    if backend not in ("sycl", "cuda"):
        raise ValueError(f"unsupported backend: {backend!r}; expected one of: sycl, cuda")

    source_dir = Path(config.llama_cpp_root)
    build_dir = source_dir / ("build_cuda" if backend == "cuda" else "build")
    output_dir = config.builds_dir

    build_backend = BuildBackend.SYCL if backend == "sycl" else BuildBackend.CUDA
    build_config = BuildConfig(
        backend=build_backend,
        source_dir=source_dir,
        build_dir=build_dir,
        output_dir=output_dir,
        git_remote_url=config.build_git_remote,
        git_branch=config.build_git_branch,
        shallow_clone=getattr(config, "build_shallow_clone", True),
        retry_attempts=config.build_retry_attempts,
        retry_delay=config.build_retry_delay,
    )

    pipeline = BuildPipeline(build_config, progress_callback=progress_callback)
    pipeline.dry_run = dry_run

    if pipeline_callback is not None:
        pipeline_callback(pipeline)

    return pipeline.run()
