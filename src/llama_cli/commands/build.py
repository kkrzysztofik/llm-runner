"""Build CLI interface for M2 build setup.

This module provides command-line interface for building llama.cpp
using the BuildPipeline.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from llama_cli.commands._output import emit_json, emit_json_str, emit_plain
from llama_cli.ui_output import emit_error, emit_success
from llama_manager.build_pipeline import (
    BuildBackend,
    BuildConfig,
    BuildPipeline,
    BuildResult,
)
from llama_manager.build_pipeline.models import SOURCE_FLAVOR_DEFAULTS
from llama_manager.config import Config


def _format_bytes(size_bytes: int | None) -> str:
    """Format an optional byte count for CLI output."""
    if size_bytes is None:
        return "unknown size"
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def _format_duration(seconds: float) -> str:
    """Format build duration for CLI output."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _progress_summary(result: BuildResult) -> dict[str, object] | None:
    """Return JSON-safe progress metadata for a build result."""
    if result.progress is None:
        return None
    return {
        "stage": result.progress.stage,
        "status": result.progress.status,
        "message": result.progress.message,
        "progress_percent": result.progress.progress_percent,
    }


def parse_build_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse build command arguments.

    Args:
        args: Optional list of command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build llama.cpp for specified backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sycl              Build for Intel SYCL (Intel Arc)
  %(prog)s cuda              Build for NVIDIA CUDA
  %(prog)s both              Build for both backends sequentially
  %(prog)s sycl --dry-run    Preview build without executing
  %(prog)s cuda --jobs 8     Build with 8 parallel jobs
        """,
    )

    parser.add_argument(
        "backend",
        choices=["sycl", "cuda", "both"],
        help="Build backend: sycl (Intel), cuda (NVIDIA), or both",
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Source directory for llama.cpp "
            "(default: $XDG_CACHE_HOME/llm-runner/llama.cpp, override with LLAMA_CPP_ROOT)"
        ),
    )

    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory (default: build or build_cuda under selected source directory)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for artifacts (default: ~/.local/share/llm-runner/builds)",
    )

    parser.add_argument(
        "--source-flavor",
        default="upstream",
        choices=list(SOURCE_FLAVOR_DEFAULTS.keys()),
        help="Source flavor (default: upstream)",
    )

    parser.add_argument(
        "--git-remote",
        default=None,
        help="Git remote URL for llama.cpp (overrides source-flavor; default: from flavor)",
    )

    parser.add_argument(
        "--git-branch",
        default=None,
        help="Git branch to checkout (overrides source-flavor; default: from flavor)",
    )

    parser.add_argument(
        "--no-shallow-clone",
        action="store_true",
        help="Perform full clone instead of shallow clone",
    )

    parser.add_argument(
        "--no-update-sources",
        action="store_true",
        help=(
            "Skip fetching updates for existing llama.cpp sources "
            "(default: update sources to latest)"
        ),
    )

    parser.add_argument(
        "--git-commit",
        default=None,
        help="Specific git commit hash to checkout after clone/update (default: use branch HEAD)",
    )

    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel build jobs (default: auto-detect)",
    )

    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=2,
        help="Number of retry attempts on transient failures (default: 2)",
    )

    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Retry delay in seconds (default: 5)",
    )

    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove stale CMakeCache.txt before configuring",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output build artifact as JSON",
    )

    return parser.parse_args(args)


def _get_backends(backend_arg: str) -> list[BuildBackend]:
    """Determine which backends to build based on user argument.

    Args:
        backend_arg: The backend argument from command line (sycl, cuda, or both)

    Returns:
        List of BuildBackend enums to build
    """
    backend_map = {
        "sycl": [BuildBackend.SYCL],
        "cuda": [BuildBackend.CUDA],
        "both": [BuildBackend.SYCL, BuildBackend.CUDA],
    }
    return backend_map.get(backend_arg, [BuildBackend.SYCL])


def _default_build_dir(source_dir: Path, backend: BuildBackend) -> Path:
    """Return the default build directory for a backend under the source root."""
    if backend is BuildBackend.CUDA:
        return source_dir / "build_cuda"
    return source_dir / "build"


def _create_build_config(
    args: argparse.Namespace,
    backend: BuildBackend,
    source_dir: Path,
    build_dir: Path,
    output_dir: Path,
) -> BuildConfig:
    """Create BuildConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments
        backend: Backend to build for
        source_dir: Source directory path
        build_dir: Build directory path
        output_dir: Output directory path

    Returns:
        Configured BuildConfig instance
    """
    # Resolve source_flavor to remote URL and branch
    flavor_remote, flavor_branch = SOURCE_FLAVOR_DEFAULTS.get(args.source_flavor, ("", ""))
    git_remote_url = args.git_remote or flavor_remote
    git_branch = args.git_branch or flavor_branch

    return BuildConfig(
        backend=backend,
        source_dir=source_dir,
        build_dir=build_dir,
        output_dir=output_dir,
        git_remote_url=git_remote_url,
        git_branch=git_branch,
        shallow_clone=not args.no_shallow_clone,
        jobs=args.jobs,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
        update_sources=not args.no_update_sources,
        git_commit=args.git_commit,
        clean_cache=args.clean_cache,
    )


def _build_single_backend(
    backend: BuildBackend,
    args: argparse.Namespace,
    source_dir: Path,
    build_dir: Path,
    output_dir: Path,
) -> tuple[BuildBackend, BuildResult]:
    """Build for a single backend.

    Args:
        backend: Backend to build for
        args: Command-line arguments
        source_dir: Source directory path
        build_dir: Build directory path
        output_dir: Output directory path

    Returns:
        Tuple of (backend, build_result)
    """
    build_config = _create_build_config(args, backend, source_dir, build_dir, output_dir)
    pipeline = BuildPipeline(build_config)
    pipeline.dry_run = args.dry_run

    emit_plain(f"▶ Building llama.cpp [{backend.value}]", err=True)
    emit_plain(f"  source: {source_dir}", err=True)
    emit_plain(f"  build:  {build_dir}", err=True)
    emit_plain(f"  output: {output_dir}", err=True)
    if args.dry_run:
        emit_plain("  mode:   dry-run (commands will not be executed)", err=True)

    result = pipeline.run()
    return (backend, result)


def _format_success_json(results: list[tuple[BuildBackend, BuildResult]]) -> str:
    """Format successful build results as JSON.

    Args:
        results: List of (backend, result) tuples

    Returns:
        JSON string with artifacts
    """
    artifacts = []
    for _backend, result in results:
        if result.artifact:
            artifact_dict = asdict(result.artifact)
            # Convert all Path-like values to strings generically
            for key, value in artifact_dict.items():
                if isinstance(value, Path | os.PathLike):
                    artifact_dict[key] = str(value)
            artifacts.append(artifact_dict)
    return json.dumps({"success": True, "artifacts": artifacts}, indent=2)


def _format_success_text(results: list[tuple[BuildBackend, BuildResult]]) -> None:
    """Print successful build results as text.

    Args:
        results: List of (backend, result) tuples
    """
    emit_success("✓ Build completed successfully")
    for backend, result in results:
        if result.artifact:
            artifact = result.artifact
            duration = _format_duration(artifact.build_duration_seconds)
            size = _format_bytes(artifact.binary_size_bytes)
            emit_plain(f"\n  [{backend.value}]", err=True)
            emit_plain(f"    binary:   {artifact.binary_path or 'not found'}", err=True)
            emit_plain(f"    size:     {size}", err=True)
            emit_plain(f"    duration: {duration}", err=True)
            emit_plain(f"    commit:   {artifact.git_commit_sha}", err=True)
            if artifact.build_log_path:
                emit_plain(f"    log:      {artifact.build_log_path}", err=True)


def _format_error_json(results: list[tuple[BuildBackend, BuildResult]]) -> str:
    """Format failed build results as JSON.

    Args:
        results: List of (backend, result) tuples

    Returns:
        JSON string with errors
    """
    errors = []
    for backend, result in results:
        if not result.success:
            error: dict[str, object] = {
                "backend": backend.value,
                "error": result.error_message or "Unknown error",
            }
            progress = _progress_summary(result)
            if progress is not None:
                error["progress"] = progress
            if result.artifact and result.artifact.build_log_path:
                error["build_log_path"] = str(result.artifact.build_log_path)
            if result.artifact and result.artifact.failure_report_path:
                error["failure_report_path"] = str(result.artifact.failure_report_path)
            errors.append(error)
    return json.dumps({"success": False, "errors": errors}, indent=2)


def _format_error_text(results: list[tuple[BuildBackend, BuildResult]]) -> None:
    """Print failed build results as text.

    Args:
        results: List of (backend, result) tuples
    """
    emit_error("✗ Build failed")
    for backend, result in results:
        if not result.success:
            emit_plain(f"\n  [{backend.value}]", err=True)
            if result.progress:
                emit_plain(f"    stage:  {result.progress.stage}", err=True)
                emit_plain(f"    status: {result.progress.status}", err=True)
            message = result.error_message or "Unknown error"
            indented_message = message.replace("\n", "\n      ")
            emit_plain(f"    error:  {indented_message}", err=True)
            if result.artifact and result.artifact.build_log_path:
                emit_plain(f"    log:    {result.artifact.build_log_path}", err=True)
            if result.artifact and result.artifact.failure_report_path:
                emit_plain(f"    report: {result.artifact.failure_report_path}", err=True)


def _resolve_backend_paths(
    args: argparse.Namespace,
    backend: BuildBackend,
    source_dir: Path,
    config: Config,
) -> tuple[Path, Path]:
    """Compute backend-scoped build and output directories.

    Args:
        args: Parsed build arguments.
        backend: Target backend (SYCL or CUDA).
        source_dir: llama.cpp source directory.
        config: Application configuration.

    Returns:
        Tuple of (build_dir, output_dir).
    """
    if args.build_dir:
        build_dir = Path(args.build_dir) / backend.value
    else:
        build_dir = _default_build_dir(source_dir, backend)

    if args.output_dir:
        output_dir = Path(args.output_dir) / backend.value
    else:
        output_dir = config.paths.builds_dir / backend.value

    return build_dir, output_dir


def run_build_command(args: argparse.Namespace) -> int:
    """Execute build command.

    Args:
        args: Parsed arguments from parse_build_args

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    backends = _get_backends(args.backend)
    config = Config()
    source_dir = args.source_dir or Path(config.paths.llama_cpp_root)

    results: list[tuple[BuildBackend, BuildResult]] = []
    for backend in backends:
        build_dir, output_dir = _resolve_backend_paths(args, backend, source_dir, config)
        result = _build_single_backend(backend, args, source_dir, build_dir, output_dir)
        results.append(result)

    all_success = all(result.success for _backend, result in results)

    if all_success:
        if args.json:
            emit_json_str(_format_success_json(results))
        else:
            _format_success_text(results)
        return 0
    else:
        if args.json:
            emit_json_str(_format_error_json(results))
        else:
            _format_error_text(results)
        return 1


def main(args: list[str] | None = None) -> int:
    """Main entry point for build command.

    Args:
        args: Optional list of command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code
    """
    from llama_manager.logging_setup import configure_logging

    args_parsed = None
    try:
        args_parsed = parse_build_args(args)
        configure_logging()
        return run_build_command(args_parsed)
    except KeyboardInterrupt:
        emit_plain("\nBuild interrupted by user", err=True)
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        if args_parsed is not None and args_parsed.json:
            emit_json({"success": False, "error": str(e)})
        else:
            emit_error(f"{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
