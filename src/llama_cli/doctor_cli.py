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
from llama_manager.profile_cache import (
    ProfileRecord,
    StalenessReason,
    StalenessResult,
    check_staleness,
)
from llama_manager.setup_venv import check_venv_integrity, get_venv_path
from llama_manager.toolchain import (
    ToolchainStatus,
    detect_toolchain,
    get_toolchain_hints,
)


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
    profiles_total: int = 0
    profiles_stale: int = 0
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
            "profiles_total": self.profiles_total,
            "profiles_stale": self.profiles_stale,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class RepairAction:
    """Represents a repair action to be performed."""

    action_type: str
    description: str
    command: str | list[str] | None
    dry_run_command: str | None
    requires_confirmation: bool = False
    args: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_type": self.action_type,
            "description": self.description,
            "command": self.command,
            "dry_run_command": self.dry_run_command,
            "requires_confirmation": self.requires_confirmation,
            "args": self.args,
        }


@dataclass
class DoctorRepairResult:
    """Result of doctor --repair command."""

    actions: list[RepairAction]
    performed_actions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    success: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "actions": [a.to_dict() for a in self.actions],
            "performed_actions": self.performed_actions,
            "failures": self.failures,
            "success": self.success,
            "warnings": self.warnings,
        }


def _print_error(message: str) -> None:
    """Print error message to stderr."""
    print(f"error: {message}", file=sys.stderr)


def _print_success(message: str) -> None:
    """Print success message to stdout."""
    print(message)


def _print_json(data: dict[str, Any]) -> None:
    """Print JSON data to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _get_backend(backend_str: str | None) -> BuildBackend | None:
    """Convert backend string to BuildBackend enum."""
    if backend_str is None:
        return None
    with contextlib.suppress(ValueError):
        return BuildBackend(backend_str)
    return None


def _check_toolchain(
    result: DoctorCheckResult,
    toolchain_status: ToolchainStatus,
    backend: BuildBackend | None,
) -> None:
    """Check toolchain completeness and update result."""
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


def _check_venv(result: DoctorCheckResult) -> None:
    """Check virtual environment status and update result."""
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


def _check_build_lock(result: DoctorCheckResult, config: Config) -> None:
    """Check build lock status and update result."""
    lock_path = config.build_lock_path
    if not lock_path.exists():
        result.build_lock_free = True
        return

    try:
        lock_data = json.loads(lock_path.read_text())
        lock = BuildLock(
            pid=lock_data["pid"],
            started_at=float(lock_data["started_at"]),
            backend=lock_data["backend"],
        )
        if lock.is_stale():
            result.is_healthy = False
            result.errors.append(
                f"Stale build lock detected (PID {lock.pid}, held for {lock.elapsed_seconds:.0f}s)"
            )
        else:
            result.build_lock_free = False
            result.warnings.append(f"Build lock held by PID {lock.pid} (backend: {lock.backend})")
    except (KeyError, ValueError):
        result.is_healthy = False
        result.errors.append("Build lock file corrupted")


def _check_staging_dirs(result: DoctorCheckResult, config: Config) -> None:
    """Check for failed staging directories and update result."""
    build_dir = Path(config.llama_cpp_root) / "build"
    build_cuda_dir = Path(config.llama_cpp_root) / "build_cuda"
    staging_dirs = [d for d in [build_dir, build_cuda_dir] if d.exists()]

    failed_staging: list[str] = []
    for staging_dir in staging_dirs:
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


def _check_reports_dir(result: DoctorCheckResult, config: Config) -> None:
    """Check reports directory status and update result."""
    result.reports_dir_exists = config.reports_dir.exists()
    if not result.reports_dir_exists:
        result.warnings.append("Reports directory does not exist")


def _check_profiles(
    result: DoctorCheckResult,
    config: Config,
    max_age_days: int = 90,
    current_driver_version: str | None = None,
    current_binary_version: str | None = None,
) -> None:
    """Check cached performance profiles for staleness.

    Lists all profile JSON files in the profiles directory, parses each as a
    ProfileRecord, and checks staleness using the record's own driver/binary
    versions *unless* explicit current versions are provided (which enables
    driver- and binary-change detection).  Stale profiles are added as
    warnings with actionable guidance, and the total/stale counts are
    recorded on *result*.

    Args:
        result: DoctorCheckResult to update with profile findings.
        config: Application config (provides profiles_dir).
        max_age_days: Maximum acceptable profile age in days.
        current_driver_version: Current GPU driver version (enables
            driver-change detection when provided).
        current_binary_version: Current llama-server binary version
            (enables binary-change detection when provided).
    """
    profiles_dir = config.profiles_dir
    if not profiles_dir.exists():
        return

    total = 0
    stale = 0

    for profile_path in sorted(profiles_dir.glob("*.json")):
        try:
            raw = profile_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            record = ProfileRecord.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            # Corrupt or unrecognised file — warn and skip
            result.warnings.append(f"Corrupt profile file skipped: {profile_path.name} ({e!r})")
            continue

        total += 1

        # Use the record's own versions when current versions are not
        # provided (age-only staleness).  When current versions are
        # supplied, driver/binary changes are also detected.
        staleness = check_staleness(
            record,
            current_driver_version=current_driver_version or record.driver_version,
            current_binary_version=current_binary_version or record.server_binary_version,
            staleness_days=max_age_days,
        )

        if staleness.is_stale:
            stale += 1
            reasons = ", ".join(r.value for r in staleness.reasons)
            guidance = _build_profile_guidance(staleness, record, max_age_days)
            result.warnings.append(
                f"Stale profile: {profile_path.name} ({reasons}, "
                f"{staleness.age_days:.0f} days old) — {guidance}"
            )

    result.profiles_total = total
    result.profiles_stale = stale


def _build_profile_guidance(
    staleness: "StalenessResult",
    record: "ProfileRecord",
    max_age_days: int = 90,
) -> str:
    """Build actionable guidance for a stale profile.

    Args:
        staleness: The staleness result with reasons and version info.
        record: The profile record that was found stale.
        max_age_days: Maximum acceptable profile age in days.

    Returns:
        A human-readable guidance string.
    """
    parts: list[str] = []

    if StalenessReason.DRIVER_CHANGED in staleness.reasons:
        parts.append(
            f"Re-profile recommended: driver version changed "
            f"from {record.driver_version} to {staleness.driver_version_display}"
        )

    if StalenessReason.BINARY_CHANGED in staleness.reasons:
        parts.append("Re-profile recommended: llama-server binary was updated")

    if StalenessReason.AGE_EXCEEDED in staleness.reasons:
        parts.append(
            f"Re-profile recommended: profile is {staleness.age_days:.0f} days old "
            f"(threshold: {max_age_days} days)"
        )

    if parts:
        return "; ".join(parts)

    return "Re-profile recommended"


def _print_check_results(result: DoctorCheckResult) -> int:
    """Print doctor check results in human-readable format.

    Returns:
        Exit code (0 if healthy, 1 otherwise)
    """
    _print_success("Doctor Check Results:")
    _print_success(f"  Toolchain complete: {'YES' if result.toolchain_complete else 'NO'}")
    _print_success(f"  Venv exists: {'YES' if result.venv_exists else 'NO'}")
    _print_success(f"  Venv intact: {'YES' if result.venv_intact else 'NO'}")
    _print_success(f"  Build lock free: {'YES' if result.build_lock_free else 'NO'}")
    _print_success(f"  Staging dirs clean: {'YES' if result.staging_dirs_clean else 'NO'}")
    _print_success(f"  Reports dir exists: {'YES' if result.reports_dir_exists else 'NO'}")
    _print_success(f"  Profiles: {result.profiles_total} total, {result.profiles_stale} stale")

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


def cmd_doctor_check(parsed: argparse.Namespace) -> int:
    """Execute doctor check command.

    Validates toolchain, venv, build directories, lock status, and cached
    performance profiles.
    Returns exit code 0 if healthy, 1 if any issues found.
    """
    backend_str = parsed.backend if hasattr(parsed, "backend") else None
    json_output = getattr(parsed, "json", False)
    max_age_days = getattr(parsed, "max_age_days", 90)

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

    backend = _get_backend(backend_str)
    toolchain_status = detect_toolchain()

    _check_toolchain(result, toolchain_status, backend)
    _check_venv(result)
    _check_build_lock(result, config)
    _check_staging_dirs(result, config)
    _check_reports_dir(result, config)
    # Pass None for current versions to preserve age-only staleness
    # detection.  Driver/binary-change detection requires actual GPU
    # driver version queries (nvidia-smi, SYCL device info) which are
    # not yet integrated here.
    _check_profiles(
        result,
        config,
        max_age_days=max_age_days,
        current_driver_version=None,
        current_binary_version=None,
    )

    if json_output:
        _print_json(result.to_dict())
        return 0 if result.is_healthy else 1

    return _print_check_results(result)


def _collect_toolchain_repair_actions(result: DoctorRepairResult) -> None:
    """Collect repair actions for toolchain issues."""
    toolchain_status = detect_toolchain()
    if toolchain_status.is_complete:
        return

    # Deduplicate hints (both backends may return the same hints)
    sycl_hints = get_toolchain_hints("sycl")
    cuda_hints = get_toolchain_hints("cuda")
    seen_tools: set[str] = set()
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


def _collect_venv_repair_actions(result: DoctorRepairResult) -> None:
    """Collect repair actions for venv issues."""
    venv_path = get_venv_path()
    if not venv_path.exists():
        result.actions.append(
            RepairAction(
                action_type="create_venv",
                description="Create virtual environment",
                command=[sys.executable, "-m", "venv", str(venv_path)],
                dry_run_command=f"# {sys.executable} -m venv {venv_path}",
                requires_confirmation=True,
            )
        )
        return

    is_valid, error = check_venv_integrity(venv_path)
    if is_valid:
        return

    result.actions.append(
        RepairAction(
            action_type="remove_venv",
            description=f"Remove broken virtual environment ({error})",
            command="rm",
            args=["-rf", str(venv_path)],
            dry_run_command=f"# rm -rf '{venv_path}'",
            requires_confirmation=True,
        )
    )
    result.actions.append(
        RepairAction(
            action_type="create_venv",
            description="Recreate virtual environment",
            command=[sys.executable, "-m", "venv", str(venv_path)],
            dry_run_command=f"# {sys.executable} -m venv '{venv_path}'",
            requires_confirmation=False,
        )
    )


def _collect_staging_repair_actions(result: DoctorRepairResult, config: Config) -> None:
    """Collect repair actions for failed staging directories."""
    build_dir = Path(config.llama_cpp_root) / "build"
    build_cuda_dir = Path(config.llama_cpp_root) / "build_cuda"
    staging_dirs = [d for d in [build_dir, build_cuda_dir] if d.exists()]

    for staging_dir in staging_dirs:
        failed_markers = list(staging_dir.glob("**/.failed"))
        for marker in failed_markers:
            parent_dir = marker.parent
            result.actions.append(
                RepairAction(
                    action_type="clean_failed_staging",
                    description=f"Remove failed staging directory: {parent_dir}",
                    command="rm",
                    args=["-rf", str(parent_dir)],
                    dry_run_command=f"# rm -rf '{parent_dir}'",
                    requires_confirmation=True,
                )
            )
            result.actions.append(
                RepairAction(
                    action_type="remove_failed_marker",
                    description=f"Remove .failed marker: {marker}",
                    command="rm",
                    args=[str(marker)],
                    dry_run_command=f"# rm '{marker}'",
                    requires_confirmation=False,
                )
            )


def _collect_lock_repair_actions(result: DoctorRepairResult, config: Config) -> None:
    """Collect repair actions for stale/corrupted build locks."""
    lock_path = config.build_lock_path
    if not lock_path.exists():
        return

    try:
        lock_data = json.loads(lock_path.read_text())
        lock = BuildLock(
            pid=lock_data["pid"],
            started_at=float(lock_data["started_at"]),
            backend=lock_data["backend"],
        )
        if lock.is_stale():
            result.actions.append(
                RepairAction(
                    action_type="remove_stale_lock",
                    description=f"Remove stale build lock (PID {lock.pid})",
                    command="rm",
                    args=[str(lock_path)],
                    dry_run_command=f"# rm '{lock_path}'",
                    requires_confirmation=True,
                )
            )
    except (KeyError, TypeError, ValueError) as e:
        result.actions.append(
            RepairAction(
                action_type="remove_corrupt_lock",
                description=f"Remove corrupted build lock file ({e})",
                command="rm",
                args=[str(lock_path)],
                dry_run_command=f"# rm '{lock_path}'",
                requires_confirmation=True,
            )
        )


def _collect_directories_repair_actions(result: DoctorRepairResult, config: Config) -> None:
    """Collect repair actions for missing standard directories.

    Creates the reports, profiles, and builds directories if they do not
    exist, ensuring they are created with restrictive owner-only
    permissions (0o700).

    Args:
        result: DoctorRepairResult to append actions to.
        config: Application config (provides directory paths).
    """
    directories: list[tuple[str, Path]] = [
        ("reports", config.reports_dir),
        ("profiles", config.profiles_dir),
        ("builds", config.builds_dir),
    ]

    for name, dir_path in directories:
        if not dir_path.exists():
            result.actions.append(
                RepairAction(
                    action_type="create_directory",
                    description=f"Create missing {name} directory: {dir_path}",
                    command=["mkdir", "-m", "700", "-p", str(dir_path)],
                    dry_run_command=f"# mkdir -m 700 -p '{dir_path}'",
                    requires_confirmation=False,
                )
            )


def _collect_profile_repair_actions(
    result: DoctorRepairResult,
    config: Config,
    max_age_days: int,
    current_driver_version: str | None = None,
    current_binary_version: str | None = None,
) -> None:
    """Collect repair actions for stale cached performance profiles.

    Scans the profiles directory for JSON files, parses each as a
    ProfileRecord, and checks staleness.  Profiles that are stale beyond
    *max_age_days* are added as deletion actions.

    Args:
        result: DoctorRepairResult to append actions to.
        config: Application config (provides profiles_dir).
        max_age_days: Maximum acceptable profile age in days.
        current_driver_version: Current GPU driver version (enables
            driver-change detection when provided).
        current_binary_version: Current llama-server binary version
            (enables binary-change detection when provided).
    """
    profiles_dir = config.profiles_dir
    if not profiles_dir.exists():
        return

    for profile_path in sorted(profiles_dir.glob("*.json")):
        try:
            raw = profile_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            record = ProfileRecord.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            # Corrupt or unrecognised file — warn and skip
            result.warnings.append(f"Corrupt profile file skipped: {profile_path.name} ({e!r})")
            continue

        staleness = check_staleness(
            record,
            current_driver_version=current_driver_version or record.driver_version,
            current_binary_version=current_binary_version or record.server_binary_version,
            staleness_days=max_age_days,
        )

        if staleness.is_stale:
            reasons = ", ".join(r.value for r in staleness.reasons)
            guidance = _build_profile_guidance(staleness, record, max_age_days)
            result.actions.append(
                RepairAction(
                    action_type="remove_stale_profile",
                    description=(
                        f"Remove stale profile: {profile_path.name} "
                        f"({reasons}, {staleness.age_days:.0f} days old) — {guidance}"
                    ),
                    command="rm",
                    args=[str(profile_path)],
                    dry_run_command=f"# rm '{profile_path}'",
                    requires_confirmation=True,
                )
            )


def _execute_repair_action(action: RepairAction, result: DoctorRepairResult) -> None:
    """Execute a single repair action and update result."""
    if not action.command:
        return

    try:
        # Handle both string commands and list commands
        cmd_list = action.command if isinstance(action.command, list) else [action.command]

        # Add args if present
        if action.args:
            cmd_list = cmd_list + action.args

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


def _execute_repair_actions(result: DoctorRepairResult) -> None:
    """Execute all repair actions."""
    for action in result.actions:
        _execute_repair_action(action, result)


def _print_repair_results(result: DoctorRepairResult) -> None:
    """Print repair results in human-readable format."""
    _print_success("Doctor Repair Actions:")

    if not result.actions:
        _print_success("  No repairs needed. System is healthy.")
        return

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


def cmd_doctor_repair(parsed: argparse.Namespace) -> DoctorRepairResult:
    """Execute doctor --repair command.

    Attempts to fix detected issues automatically. Returns list of fix commands.
    """
    dry_run = parsed.dry_run if hasattr(parsed, "dry_run") else False
    json_output = getattr(parsed, "json", False)
    max_age_days = getattr(parsed, "max_age_days", 90)

    config = Config()
    result = DoctorRepairResult(actions=[], performed_actions=[], failures=[])

    _collect_toolchain_repair_actions(result)
    _collect_venv_repair_actions(result)
    _collect_staging_repair_actions(result, config)
    _collect_lock_repair_actions(result, config)
    _collect_directories_repair_actions(result, config)
    # Pass None for current versions to preserve age-only staleness
    # detection.  See cmd_doctor_check for rationale.
    _collect_profile_repair_actions(
        result,
        config,
        max_age_days=max_age_days,
        current_driver_version=None,
        current_binary_version=None,
    )

    if not dry_run:
        _execute_repair_actions(result)

    if json_output:
        _print_json(result.to_dict())
        return result

    _print_repair_results(result)
    return result


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for doctor commands."""
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
    check_parser.add_argument(
        "--max-age-days",
        type=int,
        default=90,
        help="Maximum profile age in days before considered stale (default: 90)",
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
    repair_parser.add_argument(
        "--max-age-days",
        type=int,
        default=90,
        help="Remove profiles stale beyond this age in days (default: 90)",
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
