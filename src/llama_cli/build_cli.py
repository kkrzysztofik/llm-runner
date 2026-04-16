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


def run_build_command(args: argparse.Namespace) -> int:
    """Execute build command.

    Args:
        args: Parsed arguments from parse_build_args

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Determine backend(s)
    backends = []
    if args.backend == "sycl":
        backends = [BuildBackend.SYCL]
    elif args.backend == "cuda":
        backends = [BuildBackend.CUDA]
    elif args.backend == "both":
        backends = [BuildBackend.SYCL, BuildBackend.CUDA]

    # Determine paths
    config = Config()
    source_dir = args.source_dir or Path(config.llama_cpp_root)
    build_dir = args.build_dir or (source_dir / "build")
    output_dir = args.output_dir or config.builds_dir

    # Build each backend sequentially
    results = []
    all_success = True

    for backend in backends:
        # Create build config for this backend
        build_config = BuildConfig(
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

        # Create and run pipeline
        pipeline = BuildPipeline(build_config)
        pipeline.dry_run = args.dry_run

        print(f"Building for {backend.value} backend...", file=sys.stderr)
        if args.dry_run:
            print("DRY RUN MODE - commands will not be executed", file=sys.stderr)

        result = pipeline.run()
        results.append((backend, result))

        if not result.success:
            all_success = False
            # Continue to next backend if this one fails

    # Output results
    if all_success:
        if args.json:
            # Output JSON array of all artifacts
            artifacts = []
            for _backend, result in results:
                if result.artifact:
                    artifact_dict = asdict(result.artifact)
                    # Convert all Path-like values to strings generically
                    for key, value in list(artifact_dict.items()):
                        if isinstance(value, (Path, os.PathLike)):
                            artifact_dict[key] = str(value)
                    artifacts.append(artifact_dict)
            print(json.dumps({"success": True, "artifacts": artifacts}, indent=2))
        else:
            print("Build completed successfully!", file=sys.stderr)
            for backend, result in results:
                if result.artifact:
                    print(f"  {backend.value}: {result.artifact.binary_path}", file=sys.stderr)
        return 0
    else:
        if args.json:
            # Output JSON error
            errors = []
            for backend, result in results:
                if not result.success:
                    errors.append(
                        {"backend": backend.value, "error": result.error_message or "Unknown error"}
                    )
            print(json.dumps({"success": False, "errors": errors}, indent=2))
        else:
            print("Build failed:", file=sys.stderr)
            for backend, result in results:
                if not result.success:
                    print(f"  {backend.value}: {result.error_message}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for build command.

    Returns:
        Exit code
    """
    try:
        args = parse_build_args()
        return run_build_command(args)
    except KeyboardInterrupt:
        print("\nBuild interrupted by user", file=sys.stderr)
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
