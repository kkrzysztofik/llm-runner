"""Setup CLI for toolchain diagnostics and venv lifecycle management.

This module provides CLI commands for:
- setup check: Toolchain diagnostics with FR-005 actionable errors (FR-005.1)
- setup venv: Virtual environment creation/reuse (FR-005.2)
- setup clean-venv: Virtual environment cleanup (FR-005.3)

All commands support --json output for programmatic access.
"""

import argparse
import json
import shutil
import sys
from typing import Any

from llama_manager.setup_venv import (
    check_venv_integrity,
    create_venv,
    get_venv_path,
)
from llama_manager.toolchain import (
    detect_toolchain,
    get_toolchain_hints,
)


def _print_error(message: str) -> None:
    """Print error message to stderr.

    Args:
        message: Error message to print.
    """
    print(f"error: {message}", file=sys.stderr)


def _print_success(message: str) -> None:
    """Print success message to stdout.

    Args:
        message: Success message to print.
    """
    print(message)


def _print_json(data: dict[str, Any]) -> None:
    """Print JSON data to stdout.

    Args:
        data: Dictionary to serialize to JSON.
    """
    print(json.dumps(data, indent=2, default=str))


def cmd_check(args: argparse.Namespace) -> int:
    """Execute setup --check command.

    Validates toolchain availability and returns actionable error messages
    for missing tools (FR-005). Skips venv integrity check by default.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        # Detect toolchain status
        status = detect_toolchain()

        # Build status report (contract fields only)
        output: dict[str, Any] = {
            "gcc": status.gcc,
            "make": status.make,
            "git": status.git,
            "cmake": status.cmake,
            "sycl_compiler": status.sycl_compiler,
            "cuda_toolkit": status.cuda_toolkit,
            "nvtop": status.nvtop,
        }

        # Get actionable hints for missing tools
        if args.backend == "sycl":
            hints = get_toolchain_hints("sycl")
        elif args.backend == "cuda":
            hints = get_toolchain_hints("cuda")
        else:
            hints = []

        # Print JSON output if requested (contract fields only)
        if args.json:
            _print_json(output)
            return 0 if status.is_complete else 1

        # Print human-readable output
        _print_success("Toolchain Status:")
        _print_success(f"  gcc: {status.gcc or 'MISSING'}")
        _print_success(f"  make: {status.make or 'MISSING'}")
        _print_success(f"  git: {status.git or 'MISSING'}")
        _print_success(f"  cmake: {status.cmake or 'MISSING'}")
        _print_success(f"  sycl_compiler: {status.sycl_compiler or 'MISSING'}")
        _print_success(f"  cuda_toolkit: {status.cuda_toolkit or 'MISSING'}")
        _print_success(f"  nvtop: {status.nvtop or 'MISSING'}")

        _print_success("")
        _print_success(f"SYCL ready: {'YES' if status.is_sycl_ready else 'NO'}")
        _print_success(f"CUDA ready: {'YES' if status.is_cuda_ready else 'NO'}")
        _print_success(f"Complete: {'YES' if status.is_complete else 'NO'}")

        if status.missing_tools():
            _print_success("")
            _print_error(f"Missing tools: {', '.join(status.missing_tools())}")

            if hints:
                _print_success("")
                _print_success("Installation hints:")
                for hint in hints:
                    _print_success(f"  - {hint.how_to_fix}")
                    if hint.docs_ref:
                        _print_success(f"    Docs: {hint.docs_ref}")

            return 1

        _print_success("")
        _print_success("All required tools are available!")
        return 0
    except Exception as e:
        _print_error(f"Toolchain detection failed: {e}")
        return 1


def cmd_venv(args: argparse.Namespace) -> int:
    """Execute setup venv command.

    Creates virtual environment at XDG cache path if it doesn't exist,
    otherwise reuses existing venv. Validates venv integrity.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    try:
        # Get venv path
        venv_path = get_venv_path()

        # Create or reuse venv
        result = create_venv(venv_path)

        # Check integrity if requested
        if args.check_integrity:
            is_valid, error = check_venv_integrity(venv_path)
            if not is_valid:
                _print_error(f"Venv integrity check failed: {error}")
                return 1

        # Print JSON output if requested (VenvResult fields only)
        if args.json:
            _print_json(
                {
                    "venv_path": str(result.venv_path),
                    "created": result.created,
                    "reused": result.reused,
                    "activation_command": result.activation_command,
                }
            )
            return 0

        # Print human-readable output
        if result.was_created:
            _print_success(f"Created virtual environment at: {venv_path}")
        elif result.was_reused:
            _print_success(f"Reused existing virtual environment at: {venv_path}")

        _print_success(f"Activation command: {result.activation_command}")

        return 0
    except Exception as e:
        _print_error(f"Venv creation failed: {e}")
        return 1


def cmd_clean_venv(args: argparse.Namespace) -> int:
    """Execute setup clean-venv command.

    Removes existing virtual environment at XDG cache path.
    Handles permission errors gracefully.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Get venv path
    venv_path = get_venv_path()

    # Check if venv exists
    if not venv_path.exists():
        _print_success(f"Virtual environment does not exist at: {venv_path}")
        return 0

    # Confirm removal if --yes not provided
    if not args.yes:
        _print_error(f"Virtual environment exists at: {venv_path}")
        _print_error("Use --yes to confirm removal, or run without --yes to see this message")
        return 1

    # Remove venv
    try:
        shutil.rmtree(venv_path)
        _print_success(f"Removed virtual environment at: {venv_path}")
        return 0
    except PermissionError as e:
        _print_error(f"Permission denied removing virtual environment: {e}")
        _print_error("Try running with sudo or fix permissions")
        return 1
    except Exception as e:
        _print_error(f"Error removing virtual environment: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for setup commands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="setup",
        description="Toolchain diagnostics and virtual environment management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --check              Check toolchain availability
  %(prog)s --check --backend sycl  Check SYCL toolchain only
  %(prog)s venv                 Create or reuse virtual environment
  %(prog)s clean-venv           Remove virtual environment
  %(prog)s clean-venv --yes     Remove without confirmation

FR-005: Actionable error messages for missing tools
FR-005.1: setup --check command
FR-005.2: setup venv command
FR-005.3: setup clean-venv command
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup --check
    check_parser = subparsers.add_parser(
        "check",
        help="Check toolchain availability (FR-005.1)",
        description="Check toolchain availability with actionable error messages",
    )
    check_parser.add_argument(
        "--backend",
        choices=["sycl", "cuda", "all"],
        default="all",
        help="Backend to check (default: all)",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    check_parser.set_defaults(func=cmd_check)

    # setup venv
    venv_parser = subparsers.add_parser(
        "venv",
        help="Create or reuse virtual environment (FR-005.2)",
        description="Create virtual environment at XDG cache path",
    )
    venv_parser.add_argument(
        "--check-integrity",
        action="store_true",
        help="Check venv integrity after creation",
    )
    venv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    venv_parser.set_defaults(func=cmd_venv)

    # setup clean-venv
    clean_parser = subparsers.add_parser(
        "clean-venv",
        help="Remove virtual environment (FR-005.3)",
        description="Remove virtual environment at XDG cache path",
    )
    clean_parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm removal without prompting",
    )
    clean_parser.set_defaults(func=cmd_clean_venv)

    return parser


def main(args: list[str] | None = None) -> int:
    """Main entry point for setup CLI.

    Args:
        args: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    if parsed is None or not hasattr(parsed, "func"):
        parser.print_help()
        return 1

    return parsed.func(parsed)
