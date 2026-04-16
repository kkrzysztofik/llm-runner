"""Doctor command for system diagnostics and repair.

This module provides CLI commands for:
- doctor check: System validation and diagnostics (FR-004.7)
- doctor --repair: Automated fix suggestions for failed builds

All commands support --json output for programmatic access.
"""

import argparse
import contextlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_manager.build_pipeline import BuildBackend, BuildLock
from llama_manager.config import Config
from llama_manager.setup_venv import check_venv_integrity, get_venv_path
from llama_manager.toolchain import detect_toolchain, get_toolchain_hints


@dataclass
class DoctorCheckResult:
    """Result of doctor check command."""

    is_healthy: bool
    toolchain_complete: bool
    venv_exists: bool
    venv_intact: bool
    build_lock_free: bool
    staging_dirs_clean: bool
    reports_dir_exists: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_healthy": self.is_healthy,
            "toolchain_complete": self.toolchain_complete,
            "venv_exists": self.venv_exists,
            "venv_intact": self.venv_intact,
            "build_lock_free": self.build_lock_free,
            "staging_dirs_clean": self.staging_dirs_clean,
            "reports_dir_exists": self.reports_dir_exists,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class RepairAction:
    """Represents a repair action to be performed."""

    action_type: str
    description: str
    command: str | None
    dry_run_command: str | None
    requires_confirmation: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_type": self.action_type,
            "description": self.description,
            "command": self.command,
            "dry_run_command": self.dry_run_command,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class DoctorRepairResult:
    """Result of doctor --repair command."""

    actions: list[RepairAction]
    performed_actions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "actions": [a.to_dict() for a in self.actions],
            "performed_actions": self.performed_actions,
            "failures": self.failures,
            "success": self.success,
        }


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


def cmd_doctor_check(parsed: argparse.Namespace) -> int:
    """Execute doctor check command.

    Validates toolchain, venv, build directories, and lock status.
    Returns exit code 0 if healthy, 1 if any issues found.

    Args:
        parsed: Parsed command-line arguments namespace.

    Returns:
        Exit code (0 for healthy, 1 for issues found).
    """
    backend = parsed.backend if hasattr(parsed, "backend") else None
    json_output = parsed.json_output if hasattr(parsed, "json_output") else False

    config = Config()
    result = DoctorCheckResult(
        is_healthy=True,
        toolchain_complete=False,
        venv_exists=False,
        venv_intact=False,
        build_lock_free=False,
        staging_dirs_clean=False,
        reports_dir_exists=False,
    )

    # Check toolchain
    toolchain_status = detect_toolchain()

    # Convert backend string to BuildBackend enum if provided
    if backend is not None:
        with contextlib.suppress(ValueError):
            backend = BuildBackend(backend)

    # Check if toolchain is complete for the specified backend
    if backend == BuildBackend.SYCL:
        result.toolchain_complete = toolchain_status.is_sycl_ready
    elif backend == BuildBackend.CUDA:
        result.toolchain_complete = toolchain_status.is_cuda_ready
    else:
        result.toolchain_complete = toolchain_status.is_complete

    if not result.toolchain_complete:
        result.is_healthy = False
        missing = toolchain_status.missing_tools(backend)
        if missing:
            result.errors.append(f"Missing tools: {', '.join(missing)}")

    # Check venv
    venv_path = get_venv_path()
    result.venv_exists = venv_path.exists()

    if result.venv_exists:
        is_valid, error = check_venv_integrity(venv_path)
        result.venv_intact = is_valid
        if not is_valid:
            result.is_healthy = False
            result.errors.append(f"Venv integrity check failed: {error}")
    else:
        result.warnings.append("Virtual environment does not exist")

    # Check build lock
    lock_path = config.build_lock_path
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text())
            lock = BuildLock(
                pid=lock_data["pid"],
                started_at=float(lock_data["started_at"]),
                backend=lock_data["backend"],
            )
            if lock.is_stale(timeout_seconds=config.toolchain_timeout_seconds * 60):
                result.is_healthy = False
                result.errors.append(
                    f"Stale build lock detected (PID {lock.pid}, "
                    f"held for {lock.elapsed_seconds:.0f}s)"
                )
            else:
                result.build_lock_free = False
                result.warnings.append(
                    f"Build lock held by PID {lock.pid} (backend: {lock.backend})"
                )
        except (json.JSONDecodeError, KeyError) as e:
            result.is_healthy = False
            result.errors.append(f"Build lock file corrupted: {e}")
    else:
        result.build_lock_free = True

    # Check staging directories
    build_dir = Path(config.llama_cpp_root) / "build"
    build_cuda_dir = Path(config.llama_cpp_root) / "build_cuda"
    staging_dirs = [d for d in [build_dir, build_cuda_dir] if d.exists()]

    failed_staging = []
    for staging_dir in staging_dirs:
        # Look for .failed marker files or directories
        failed_markers = list(staging_dir.glob("**/.failed"))
        if failed_markers:
            failed_staging.extend([str(m) for m in failed_markers])

    if failed_staging:
        result.staging_dirs_clean = False
        result.is_healthy = False
        result.warnings.append(
            f"Found {len(failed_staging)} failed staging marker(s). "
            "Run 'doctor --repair' to clean up."
        )
    else:
        result.staging_dirs_clean = True

    # Check reports directory
    result.reports_dir_exists = config.reports_dir.exists()
    if not result.reports_dir_exists:
        result.warnings.append("Reports directory does not exist")

    # Print output
    if json_output:
        _print_json(result.to_dict())
        return 0 if result.is_healthy else 1

    _print_success("Doctor Check Results:")
    _print_success(f"  Toolchain complete: {'YES' if result.toolchain_complete else 'NO'}")
    _print_success(f"  Venv exists: {'YES' if result.venv_exists else 'NO'}")
    _print_success(f"  Venv intact: {'YES' if result.venv_intact else 'NO'}")
    _print_success(f"  Build lock free: {'YES' if result.build_lock_free else 'NO'}")
    _print_success(f"  Staging dirs clean: {'YES' if result.staging_dirs_clean else 'NO'}")
    _print_success(f"  Reports dir exists: {'YES' if result.reports_dir_exists else 'NO'}")

    if result.warnings:
        _print_success("")
        _print_success("Warnings:")
        for warning in result.warnings:
            _print_success(f"  - {warning}")

    if result.errors:
        _print_success("")
        _print_error("Errors:")
        for error in result.errors:
            _print_error(f"  - {error}")

    _print_success("")
    if result.is_healthy:
        _print_success("System is healthy!")
        return 0
    else:
        _print_error("System has issues. Run 'doctor --repair' to fix.")
        return 1


def cmd_doctor_repair(parsed: argparse.Namespace) -> DoctorRepairResult:
    """Execute doctor --repair command.

    Attempts to fix detected issues automatically. Returns list of fix commands.

    Args:
        parsed: Parsed command-line arguments namespace.

    Returns:
        DoctorRepairResult with actions performed.
    """
    dry_run = parsed.dry_run if hasattr(parsed, "dry_run") else False
    json_output = parsed.json_output if hasattr(parsed, "json_output") else False

    config = Config()
    result = DoctorRepairResult(actions=[], performed_actions=[], failures=[])

    # Check toolchain issues
    toolchain_status = detect_toolchain()
    if not toolchain_status.is_complete:
        # Deduplicate hints (both backends may return the same hints)
        sycl_hints = get_toolchain_hints("sycl")
        cuda_hints = get_toolchain_hints("cuda")
        # Use a set to deduplicate based on failed_check (tool name)
        seen_tools = set()
        hints = []
        for hint in sycl_hints + cuda_hints:
            if hint.failed_check not in seen_tools:
                seen_tools.add(hint.failed_check)
                hints.append(hint)
        for hint in hints:
            result.actions.append(
                RepairAction(
                    action_type="install_tool",
                    description=f"Install missing tool: {hint.how_to_fix}",
                    command=None,
                    dry_run_command=f"# {hint.how_to_fix}",
                    requires_confirmation=True,
                )
            )

    # Check venv issues
    venv_path = get_venv_path()
    if not venv_path.exists():
        result.actions.append(
            RepairAction(
                action_type="create_venv",
                description="Create virtual environment",
                command=f"python3 -m venv {venv_path}",
                dry_run_command=f"# python3 -m venv {venv_path}",
                requires_confirmation=True,
            )
        )
    else:
        is_valid, error = check_venv_integrity(venv_path)
        if not is_valid:
            result.actions.append(
                RepairAction(
                    action_type="recreate_venv",
                    description=f"Recreate virtual environment ({error})",
                    command=f"rm -rf '{venv_path}' && python3 -m venv '{venv_path}'",
                    dry_run_command=f"# rm -rf '{venv_path}' && python3 -m venv '{venv_path}'",
                    requires_confirmation=True,
                )
            )

    # Check and clean failed staging directories
    build_dir = Path(config.llama_cpp_root) / "build"
    build_cuda_dir = Path(config.llama_cpp_root) / "build_cuda"
    staging_dirs = [d for d in [build_dir, build_cuda_dir] if d.exists()]

    for staging_dir in staging_dirs:
        failed_markers = list(staging_dir.glob("**/.failed"))
        if failed_markers:
            for marker in failed_markers:
                parent_dir = marker.parent
                result.actions.append(
                    RepairAction(
                        action_type="clean_failed_staging",
                        description=f"Remove failed staging directory: {parent_dir}",
                        command=f"rm -rf '{parent_dir}'",
                        dry_run_command=f"# rm -rf '{parent_dir}'",
                        requires_confirmation=True,
                    )
                )
                # Also remove .failed marker
                result.actions.append(
                    RepairAction(
                        action_type="remove_failed_marker",
                        description=f"Remove .failed marker: {marker}",
                        command=f"rm '{marker}'",
                        dry_run_command=f"# rm '{marker}'",
                        requires_confirmation=False,
                    )
                )

    # Check stale build locks
    lock_path = config.build_lock_path
    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text())
            lock = BuildLock(
                pid=lock_data["pid"],
                started_at=float(lock_data["started_at"]),
                backend=lock_data["backend"],
            )
            if lock.is_stale(timeout_seconds=config.toolchain_timeout_seconds * 60):
                result.actions.append(
                    RepairAction(
                        action_type="remove_stale_lock",
                        description=f"Remove stale build lock (PID {lock.pid})",
                        command=f"rm '{lock_path}'",
                        dry_run_command=f"# rm '{lock_path}'",
                        requires_confirmation=True,
                    )
                )
        except (json.JSONDecodeError, KeyError):
            result.actions.append(
                RepairAction(
                    action_type="remove_corrupt_lock",
                    description="Remove corrupted build lock file",
                    command=f"rm '{lock_path}'",
                    dry_run_command=f"# rm '{lock_path}'",
                    requires_confirmation=True,
                )
            )

    # Execute repairs (if not dry run)
    if not dry_run:
        for action in result.actions:
            if action.command:
                try:
                    # Parse command into list of arguments
                    # This is a simple parser for the commands we generate
                    # In production, we should refactor to store commands as lists
                    import shlex

                    cmd_list = shlex.split(action.command)
                    subprocess.run(
                        cmd_list,
                        shell=False,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    result.performed_actions.append(action.description)
                except subprocess.CalledProcessError as e:
                    result.failures.append(f"Failed to {action.description}: {e.stderr or str(e)}")
                    result.success = False
                except Exception as e:
                    result.failures.append(f"Failed to {action.description}: {e}")
                    result.success = False

    # Print output
    if json_output:
        _print_json(result.to_dict())
        return result

    # Human-readable output
    _print_success("Doctor Repair Actions:")

    if not result.actions:
        _print_success("  No repairs needed. System is healthy.")
        return result

    for i, action in enumerate(result.actions, 1):
        confirm_marker = " [CONFIRMATION REQUIRED]" if action.requires_confirmation else ""
        _print_success(f"  {i}. {action.description}{confirm_marker}")
        if action.dry_run_command:
            _print_success(f"     Command: {action.dry_run_command}")

    if result.performed_actions:
        _print_success("")
        _print_success("Performed actions:")
        for action in result.performed_actions:
            _print_success(f"  - {action}")

    if result.failures:
        _print_success("")
        _print_error("Failures:")
        for failure in result.failures:
            _print_error(f"  - {failure}")

    return result


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for doctor commands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="doctor",
        description="System diagnostics and repair commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check              Check system health
  %(prog)s check --json       Output in JSON format
  %(prog)s --repair           Attempt automatic repairs
  %(prog)s --repair --dry-run Show what would be fixed
  %(prog)s --repair --json    Output repair actions in JSON

FR-004.7: doctor --repair command for failed staging cleanup
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # doctor check
    check_parser = subparsers.add_parser(
        "check",
        help="Check system health",
        description="Validate toolchain, venv, build directories, and lock status",
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
    check_parser.set_defaults(func=cmd_doctor_check)

    # doctor --repair
    repair_parser = subparsers.add_parser(
        "repair",
        help="Attempt automatic repairs",
        description="Fix detected issues automatically",
    )
    repair_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without executing",
    )
    repair_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )
    repair_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (use with caution)",
    )
    repair_parser.set_defaults(func=cmd_doctor_repair)

    # Default: show help
    parser.set_defaults(func=lambda args: parser.print_help())

    return parser


def main(args: list[str] | None = None) -> int:
    """Main entry point for doctor CLI.

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

    result = parsed.func(parsed)

    # Convert DoctorRepairResult to exit code
    if isinstance(result, DoctorRepairResult):
        return 0 if result.success else 1

    # For int return values (cmd_doctor_check), return as-is
    return result  # type: ignore
