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
  %(prog)s sycl --dry-run    Preview build without executing
  %(prog)s cuda --jobs 8     Build with 8 parallel jobs
        """,
    )

    parser.add_argument(
        "backend",
        choices=["sycl", "cuda"],
        help="Build backend: sycl (Intel) or cuda (NVIDIA)",
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
        help="Git remote URL",
    )

    parser.add_argument(
        "--git-branch",
        default="master",
        help="Git branch to build",
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
        help="Number of parallel build jobs",
    )

    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Number of retry attempts (default: 3)",
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
    # Determine backend
    backend = BuildBackend.SYCL if args.backend == "sycl" else BuildBackend.CUDA

    # Determine paths
    config = Config()
    source_dir = args.source_dir or Path(config.llama_cpp_root)
    build_dir = args.build_dir or (source_dir / "build")
    output_dir = args.output_dir or config.builds_dir

    # Create build config
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

    if result.success:
        if args.json and result.artifact:
            artifact_dict = asdict(result.artifact)
            # Convert all Path-like values to strings generically
            for key, value in list(artifact_dict.items()):
                if isinstance(value, (Path, os.PathLike)):
                    artifact_dict[key] = str(value)
            print(json.dumps(artifact_dict, indent=2))
        else:
            print("Build completed successfully!", file=sys.stderr)
            if result.artifact:
                print(f"Artifact: {result.artifact.binary_path}", file=sys.stderr)
        return 0
    else:
        print(f"Build failed: {result.error_message}", file=sys.stderr)
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
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
