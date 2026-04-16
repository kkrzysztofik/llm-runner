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

from llama_manager.build_pipeline import (
    BuildBackend,
    BuildConfig,
    BuildPipeline,
    BuildResult,
)
from llama_manager.config import Config


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
        help="Source directory for llama.cpp (default: src/llama.cpp)",
    )

    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory (default: src/llama.cpp/build)",
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

    print(f"Building for {backend.value} backend...", file=sys.stderr)
    if args.dry_run:
        print("DRY RUN MODE - commands will not be executed", file=sys.stderr)

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
                if isinstance(value, (Path, os.PathLike)):
                    artifact_dict[key] = str(value)
            artifacts.append(artifact_dict)
    return json.dumps({"success": True, "artifacts": artifacts}, indent=2)


def _format_success_text(results: list[tuple[BuildBackend, BuildResult]]) -> None:
    """Print successful build results as text.

    Args:
        results: List of (backend, result) tuples
    """
    print("Build completed successfully!", file=sys.stderr)
    for backend, result in results:
        if result.artifact:
            print(f"  {backend.value}: {result.artifact.binary_path}", file=sys.stderr)


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
            errors.append(
                {"backend": backend.value, "error": result.error_message or "Unknown error"}
            )
    return json.dumps({"success": False, "errors": errors}, indent=2)


def _format_error_text(results: list[tuple[BuildBackend, BuildResult]]) -> None:
    """Print failed build results as text.

    Args:
        results: List of (backend, result) tuples
    """
    print("Build failed:", file=sys.stderr)
    for backend, result in results:
        if not result.success:
            print(f"  {backend.value}: {result.error_message}", file=sys.stderr)


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
    build_dir = args.build_dir or (source_dir / "build")
    output_dir = args.output_dir or config.builds_dir

    # Build each backend sequentially
    results: list[tuple[BuildBackend, BuildResult]] = []
    for backend in backends:
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


def main() -> int:
    """Main entry point for build command.

    Returns:
        Exit code
    """
    args = None
    try:
        args = parse_build_args()
        return run_build_command(args)
    except KeyboardInterrupt:
        print("\nBuild interrupted by user", file=sys.stderr)
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        if args is not None and args.json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
