"""High-level build orchestration for llama.cpp.

Provides a thin wrapper around BuildPipeline that translates
application-level Config into a BuildConfig and manages the pipeline
lifecycle without CLI or TUI concerns.
"""

import threading
from collections.abc import Callable
from pathlib import Path

from ..config import Config
from .models import SOURCE_FLAVOR_DEFAULTS, BuildBackend, BuildConfig, BuildProgress, BuildResult
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
    - clean_cache
    - build_args
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
        "clean_cache",
        "build_args",
    ]
    # Start from base fields, then overlay non-empty overrides
    base_dict: dict = {
        "backend": base.backend,
        "source_dir": base.source_dir,
        "build_dir": base.build_dir,
        "output_dir": base.output_dir,
    }
    for field_name in overridable:
        base_dict[field_name] = getattr(base, field_name)
    for field_name in overridable:
        val = getattr(overrides, field_name, None)
        # Skip None and empty strings so resolved flavor values are preserved
        if val is not None and val != "":
            base_dict[field_name] = val
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

    source_dir = Path(config.paths.llama_cpp_root)
    build_dir = source_dir / ("build_cuda" if backend == "cuda" else "build")
    # Backend-scoped output dir: builds_dir/<backend>/build-artifact.json
    output_dir = config.paths.builds_dir / backend

    build_backend = BuildBackend.SYCL if backend == "sycl" else BuildBackend.CUDA

    # Resolve source_flavor to remote URL and branch
    flavor = config.build.source_flavor
    flavor_remote, flavor_branch = SOURCE_FLAVOR_DEFAULTS.get(flavor, ("", ""))
    git_remote_url = config.build.git_remote or flavor_remote
    git_branch = config.build.git_branch or flavor_branch

    build_config = BuildConfig(
        backend=build_backend,
        source_dir=source_dir,
        build_dir=build_dir,
        output_dir=output_dir,
        git_remote_url=git_remote_url,
        git_branch=git_branch,
        shallow_clone=getattr(config, "build_shallow_clone", True),
        retry_attempts=config.build.retry_attempts,
        retry_delay=config.build.retry_delay,
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
