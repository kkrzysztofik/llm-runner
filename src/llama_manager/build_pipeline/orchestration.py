"""High-level build orchestration for llama.cpp.

Provides a thin wrapper around BuildPipeline that translates
application-level Config into a BuildConfig and manages the pipeline
lifecycle without CLI or TUI concerns.
"""

import threading
from collections.abc import Callable
from pathlib import Path

from ..config import Config
from .models import BuildBackend, BuildConfig, BuildProgress, BuildResult
from .pipeline import BuildPipeline


def _merge_config_overrides(base: BuildConfig, overrides: BuildConfig) -> BuildConfig:
    """Merge non-None fields from *overrides* onto *base*.

    Derived fields (backend, source_dir, build_dir, output_dir) are
    **never** overwritten — they are always taken from *base*.  Only
    the following overridable fields are merged:

    - git_remote_url
    - git_branch
    - retry_attempts
    - retry_delay
    - shallow_clone
    - jobs
    - update_sources
    - git_commit
    - build_timeout_seconds
    """
    overridable: list[str] = [
        "git_remote_url",
        "git_branch",
        "retry_attempts",
        "retry_delay",
        "shallow_clone",
        "jobs",
        "update_sources",
        "git_commit",
        "build_timeout_seconds",
    ]
    kwargs: dict = {}
    for field_name in overridable:
        val = getattr(overrides, field_name, None)
        if val is not None:
            kwargs[field_name] = val
    # Start from base fields, then overlay overrides
    base_dict = {
        "backend": base.backend,
        "source_dir": base.source_dir,
        "build_dir": base.build_dir,
        "output_dir": base.output_dir,
    }
    base_dict.update(kwargs)
    return BuildConfig(**base_dict)


def run_build_for_backend(
    backend: str,
    *,
    dry_run: bool = False,
    config: Config,
    progress_callback: Callable[[BuildProgress], None] | None = None,
    pipeline_callback: Callable[[BuildPipeline], None] | None = None,
    config_overrides: BuildConfig | None = None,
    cancel_event: threading.Event | None = None,
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
        config_overrides: Optional BuildConfig whose non-None fields override
            the derived defaults.  Derived fields (backend, source_dir,
            build_dir, output_dir) are **never** overridden — they are
            always computed from *backend* and *config*.
        cancel_event: When set, the pipeline cooperatively stops between
            stages and may terminate an in-flight compilation.

    Returns:
        BuildResult from the pipeline execution.
    """
    if backend not in ("sycl", "cuda"):
        raise ValueError(f"unsupported backend: {backend!r}; expected one of: sycl, cuda")

    source_dir = Path(config.llama_cpp_root)
    build_dir = source_dir / ("build_cuda" if backend == "cuda" else "build")
    # Backend-scoped output dir: builds_dir/<backend>/build-artifact.json
    output_dir = config.builds_dir / backend

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

    if config_overrides is not None:
        build_config = _merge_config_overrides(build_config, config_overrides)

    pipeline = BuildPipeline(
        build_config,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
    pipeline.dry_run = dry_run

    if pipeline_callback is not None:
        pipeline_callback(pipeline)

    return pipeline.run()
