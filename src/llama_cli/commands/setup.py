"""Setup CLI for toolchain diagnostics and venv lifecycle management.

This module provides CLI commands for:
- setup check: Toolchain diagnostics with FR-005 actionable errors (FR-005.1)
- setup venv: Virtual environment creation/reuse (FR-005.2)
- setup clean-venv: Virtual environment cleanup (FR-005.3)

All commands support --json output for programmatic access.
"""

import argparse
import shutil
from pathlib import Path
from typing import Any

from llama_cli.colors import Colors
from llama_cli.commands._output import (
    print_error,
    print_header,
    print_json,
    print_success,
)
from llama_cli.commands._toolchain import (
    filter_optional_tools,
    get_backend_hints,
    resolve_backend_enum,
)
from llama_manager.setup_venv import (
    check_venv_integrity,
    create_venv,
    get_venv_path,
)
from llama_manager.toolchain import detect_toolchain

JSON_OUTPUT_HELP = "Output in JSON format"


def _build_status_output(status: Any) -> dict[str, Any]:
    """Build status report dictionary.

    Args:
        status: ToolchainStatus object

    Returns:
        Dictionary with toolchain status fields
    """
    return {
        "gcc": status.gcc,
        "make": status.make,
        "git": status.git,
        "cmake": status.cmake,
        "sycl_compiler": status.sycl_compiler,
        "cuda_toolkit": status.cuda_toolkit,
        "nvtop": status.nvtop,
    }


def _print_status(status: Any) -> None:
    """Print human-readable toolchain status with colors.

    Args:
        status: ToolchainStatus object
    """
    yes = Colors.bright_green("✓ YES")
    no = Colors.bright_red("✗ NO")
    missing = Colors.bright_red("MISSING")

    print_header("Toolchain Status:")
    tools = [
        ("gcc", status.gcc),
        ("make", status.make),
        ("git", status.git),
        ("cmake", status.cmake),
        ("sycl_compiler", status.sycl_compiler),
        ("cuda_toolkit", status.cuda_toolkit),
        ("nvtop", status.nvtop),
    ]
    for name, value in tools:
        display = Colors.green(value) if value else missing
        print(f"  {Colors.cyan(name)}: {display}")
    print_success("")
    print_success(f"SYCL ready: {yes if status.is_sycl_ready else no}")
    print_success(f"CUDA ready: {yes if status.is_cuda_ready else no}")
    print_success(f"Complete: {yes if status.is_complete else no}")


def _handle_missing_tools(
    status: Any,
    hints: list[Any],
    backend: str | None = None,
) -> int:
    """Handle and display missing tools information with colors.

    Args:
        status: ToolchainStatus object
        hints: List of toolchain hints
        backend: Backend to check tools for (sycl, cuda, or None for all)

    Returns:
        Exit code (1 for failure)
    """
    backend_enum = resolve_backend_enum(backend)

    missing = filter_optional_tools(status.missing_tools(backend_enum), backend, status.is_complete)

    if not missing:
        print_success("")
        print(Colors.bold(Colors.bright_green("All required tools are available!")))
        return 0

    print_success("")
    print_error(f"Missing tools: {', '.join(missing)}")

    if hints:
        print_success("")
        print(Colors.yellow("Installation hints:"))
        for hint in hints:
            print(f"  {Colors.yellow('-')} {hint.how_to_fix}")
            if hint.docs_ref:
                print(Colors.dim(f"    Docs: {hint.docs_ref}"))

    return 1


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
        status = detect_toolchain()
        hints = get_backend_hints(args.backend)

        if args.json:
            print_json(_build_status_output(status))
            # Use backend-aware exit code logic (without printing)
            if args.backend == "sycl":
                return 0 if status.is_sycl_ready else 1
            elif args.backend == "cuda":
                return 0 if status.is_cuda_ready else 1
            else:
                return 0 if status.is_complete else 1

        _print_status(status)
        return _handle_missing_tools(status, hints, args.backend)
    except Exception as e:
        if getattr(args, "json", False):
            print_json({"error": "Toolchain detection failed", "details": str(e)})
        else:
            print_error(f"Toolchain detection failed: {e}")
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
                if getattr(args, "json", False):
                    print_json({"error": f"Venv integrity check failed: {error}", "success": False})
                else:
                    print_error(f"Venv integrity check failed: {error}")
                return 1

        # Print JSON output if requested (VenvResult fields only)
        if args.json:
            print_json(
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
            print_success(f"Created virtual environment at: {venv_path}")
        elif result.was_reused:
            print_success(f"Reused existing virtual environment at: {venv_path}")

        print_success(f"Activation command: {result.activation_command}")

        return 0
    except Exception as e:
        if getattr(args, "json", False):
            print_json({"error": "Venv creation failed", "details": str(e), "success": False})
        else:
            print_error(f"Venv creation failed: {e}")
        return 1


def _handle_venv_not_found(venv_path: Path, json_output: bool) -> int:
    """Handle case when venv does not exist.

    Args:
        venv_path: Path to virtual environment
        json_output: Whether to output JSON

    Returns:
        Exit code (0)
    """
    if json_output:
        print_json({"status": "not_found", "venv_path": str(venv_path)})
    else:
        print_success(f"Virtual environment does not exist at: {venv_path}")
    return 0


def _handle_confirmation_required(venv_path: Path, json_output: bool) -> int:
    """Handle case when confirmation is needed.

    Args:
        venv_path: Path to virtual environment
        json_output: Whether to output JSON

    Returns:
        Exit code (1)
    """
    if json_output:
        print_json(
            {
                "status": "exists",
                "venv_path": str(venv_path),
                "message": "Use --yes to confirm removal",
            }
        )
    else:
        print_error(f"Virtual environment exists at: {venv_path}")
        print_error("Use --yes to confirm removal, or run without --yes to see this message")
    return 1


def _remove_env(venv_path: Path, json_output: bool) -> int:
    """Remove virtual environment with error handling.

    Args:
        venv_path: Path to virtual environment
        json_output: Whether to output JSON

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        shutil.rmtree(venv_path)
        if json_output:
            print_json({"status": "removed", "venv_path": str(venv_path)})
        else:
            print_success(f"Removed virtual environment at: {venv_path}")
        return 0
    except PermissionError as e:
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(f"Permission denied removing virtual environment: {e}")
            print_error("Check file ownership/permissions, consult system documentation")
        return 1
    except Exception as e:
        if json_output:
            print_json({"status": "error", "error": str(e)})
        else:
            print_error(f"Error removing virtual environment: {e}")
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
    venv_path = get_venv_path()

    if not venv_path.exists():
        return _handle_venv_not_found(venv_path, args.json)

    if not args.yes:
        return _handle_confirmation_required(venv_path, args.json)

    return _remove_env(venv_path, args.json)


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
  %(prog)s check                Check toolchain availability
  %(prog)s check --backend sycl  Check SYCL toolchain only
  %(prog)s venv                 Create or reuse virtual environment
  %(prog)s clean-venv           Remove virtual environment
  %(prog)s clean-venv --yes     Remove without confirmation

FR-005: Actionable error messages for missing tools
FR-005.1: setup check command
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
        help=JSON_OUTPUT_HELP,
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
        help=JSON_OUTPUT_HELP,
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
    clean_parser.add_argument(
        "--json",
        action="store_true",
        help=JSON_OUTPUT_HELP,
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

    if not hasattr(parsed, "func"):
        parser.print_help()
        return 1

    return parsed.func(parsed)
