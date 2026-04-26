"""Build CLI interface for M2 build setup.

This module provides command-line interface for building llama.cpp
using the BuildPipeline.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

from llama_manager.build_pipeline import (
    BuildBackend,
    BuildConfig,
    BuildPipeline,
    BuildResult,
)
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
    return f"{size:.1f} GiB"


def _format_duration(seconds: float) -> str:
    """Format build duration for CLI output."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


# ANSI color codes for log formatting
_COLOR_RESET = "\033[0m"
_COLOR_TIMESTAMP = "\033[36m"  # cyan
_COLOR_INFO = "\033[32m"  # green
_COLOR_WARNING = "\033[33m"  # yellow
_COLOR_ERROR = "\033[31m"  # red
_COLOR_DEBUG = "\033[34m"  # blue
_COLOR_DIM = "\033[2m"  # dim


class _ColoredFormatter(logging.Formatter):
    """Custom formatter that adds timestamps and ANSI colors to log output."""

    _LEVEL_COLORS = {
        logging.DEBUG: _COLOR_DEBUG,
        logging.INFO: _COLOR_INFO,
        logging.WARNING: _COLOR_WARNING,
        logging.ERROR: _COLOR_ERROR,
        logging.CRITICAL: _COLOR_ERROR,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with timestamp and color-coded level."""
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level_color = self._LEVEL_COLORS.get(record.levelno, _COLOR_RESET)
        level_name = record.levelname.lower()
        # Extract the tag from the message if it starts with [tag]
        msg = record.getMessage()
        # Build the formatted line
        return (
            f"{_COLOR_DIM}[{timestamp}]{_COLOR_RESET} "
            f"{level_color}[{level_name}]{_COLOR_RESET} "
            f"{msg}"
        )


def _setup_colored_logging(level: int = logging.INFO) -> None:
    """Configure stderr logging with timestamps and ANSI colors.

    Args:
        level: Minimum log level to display (default: INFO).
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_ColoredFormatter())
    handler.setLevel(level)

    # Target the build pipeline logger specifically
    pipeline_logger = logging.getLogger("llama_manager.build_pipeline")
    pipeline_logger.setLevel(level)
    pipeline_logger.handlers.clear()
    pipeline_logger.addHandler(handler)


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
        "--git-remote",
        default="https://github.com/ggerganov/llama.cpp.git",
        help="Git remote URL for llama.cpp (default: official repo)",
    )

    parser.add_argument(
        "--git-branch",
        default="master",
        help="Git branch to checkout (default: master)",
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
        default=None,
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
    return BuildConfig(
        backend=backend,
        source_dir=source_dir,
        build_dir=build_dir,
        output_dir=output_dir,
        git_remote_url=args.git_remote,
        git_branch=args.git_branch,
        shallow_clone=not args.no_shallow_clone,
        jobs=args.jobs,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
        update_sources=not args.no_update_sources,
        git_commit=args.git_commit,
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

    print(f"▶ Building llama.cpp [{backend.value}]", file=sys.stderr)
    print(f"  source: {source_dir}", file=sys.stderr)
    print(f"  build:  {build_dir}", file=sys.stderr)
    print(f"  output: {output_dir}", file=sys.stderr)
    if args.dry_run:
        print("  mode:   dry-run (commands will not be executed)", file=sys.stderr)

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
    print("✓ Build completed successfully", file=sys.stderr)
    for backend, result in results:
        if result.artifact:
            artifact = result.artifact
            duration = _format_duration(artifact.build_duration_seconds)
            size = _format_bytes(artifact.binary_size_bytes)
            print(f"\n  [{backend.value}]", file=sys.stderr)
            print(f"    binary:   {artifact.binary_path or 'not found'}", file=sys.stderr)
            print(f"    size:     {size}", file=sys.stderr)
            print(f"    duration: {duration}", file=sys.stderr)
            print(f"    commit:   {artifact.git_commit_sha}", file=sys.stderr)
            if artifact.build_log_path:
                print(f"    log:      {artifact.build_log_path}", file=sys.stderr)


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
    print("✗ Build failed", file=sys.stderr)
    for backend, result in results:
        if not result.success:
            print(f"\n  [{backend.value}]", file=sys.stderr)
            if result.progress:
                print(f"    stage:  {result.progress.stage}", file=sys.stderr)
                print(f"    status: {result.progress.status}", file=sys.stderr)
            message = result.error_message or "Unknown error"
            indented_message = message.replace("\n", "\n      ")
            print(f"    error:  {indented_message}", file=sys.stderr)
            if result.artifact and result.artifact.build_log_path:
                print(f"    log:    {result.artifact.build_log_path}", file=sys.stderr)
            if result.artifact and result.artifact.failure_report_path:
                print(f"    report: {result.artifact.failure_report_path}", file=sys.stderr)


def run_build_command(args: argparse.Namespace) -> int:
    """Execute build command.

    Args:
        args: Parsed arguments from parse_build_args

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Determine backend(s) and paths
    backends = _get_backends(args.backend)
    config = Config()
    source_dir = args.source_dir or Path(config.llama_cpp_root)
    output_dir = args.output_dir or config.builds_dir

    # Build each backend sequentially
    results: list[tuple[BuildBackend, BuildResult]] = []
    for backend in backends:
        build_dir = args.build_dir or _default_build_dir(source_dir, backend)
        result = _build_single_backend(backend, args, source_dir, build_dir, output_dir)
        results.append(result)

    # Check if all builds succeeded
    all_success = all(result.success for _backend, result in results)

    # Output results
    if all_success:
        if args.json:
            print(_format_success_json(results))
        else:
            _format_success_text(results)
        return 0
    else:
        if args.json:
            print(_format_error_json(results))
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
    args_parsed = None
    try:
        args_parsed = parse_build_args(args)
        _setup_colored_logging()
        return run_build_command(args_parsed)
    except KeyboardInterrupt:
        print("\nBuild interrupted by user", file=sys.stderr)
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        if args_parsed is not None and args_parsed.json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
