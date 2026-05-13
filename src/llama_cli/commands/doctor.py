"""Doctor command for system diagnostics and repair.

This module provides CLI commands for:
- doctor check: System validation and diagnostics (FR-004.7)
- doctor --repair: Automated fix suggestions for failed builds

All commands support --json output for programmatic access.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_cli.colors import Colors
from llama_cli.commands._output import emit_json, emit_plain
from llama_cli.commands._subprocess import run_capture_command
from llama_cli.commands._toolchain import (
    collect_toolchain_repair_actions,
    resolve_backend_enum,
    toolchain_is_ready_for_backend,
)
from llama_cli.ui_output import (
    emit_error,
    emit_heading,
    emit_info,
    emit_success,
    emit_warn,
)
from llama_manager.build_pipeline import BuildBackend, BuildLock
from llama_manager.config import Config
from llama_manager.config.profile_cache import (
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
    command: list[str] | None
    dry_run_command: str | None
    requires_confirmation: bool = False
    prerequisite_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action_type": self.action_type,
            "description": self.description,
            "command": self.command,
            "dry_run_command": self.dry_run_command,
            "requires_confirmation": self.requires_confirmation,
            "prerequisite_index": self.prerequisite_index,
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


def _check_toolchain(
    result: DoctorCheckResult,
    toolchain_status: ToolchainStatus,
    backend: BuildBackend | None,
) -> None:
    """Check toolchain completeness and update result."""
    result.toolchain_complete = toolchain_is_ready_for_backend(toolchain_status, backend)

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


@dataclass
class ProfileScanEntry:
    """A single profile entry returned by the profile scanner."""

    path: Path
    record: ProfileRecord | None
    staleness: StalenessResult | None
    reasons: str
    guidance: str


def _scan_profiles(
    profiles_dir: Path,
    max_age_days: int,
    stale_only: bool,
    current_driver_version: str | None = None,
    current_binary_version: str | None = None,
) -> list[ProfileScanEntry]:
    """Scan profiles directory and return scan entries.

    Single canonical iterator for profile scanning. When stale_only=False,
    returns all profiles (fresh and stale) with their staleness. When
    stale_only=True, returns only stale profiles (or corrupt entries).

    Args:
        profiles_dir: Path to the profiles directory.
        max_age_days: Maximum acceptable profile age in days.
        stale_only: If True, only return stale entries; if False, return all.
        current_driver_version: Current GPU driver version.
        current_binary_version: Current llama-server binary version.

    Returns:
        List of ProfileScanEntry objects.
    """
    if not profiles_dir.exists():
        return []

    results: list[ProfileScanEntry] = []
    for profile_path in sorted(profiles_dir.glob("*.json")):
        try:
            raw = profile_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            record = ProfileRecord.from_dict(data)
        except (KeyError, TypeError, ValueError) as e:
            results.append(
                ProfileScanEntry(
                    path=profile_path,
                    record=None,
                    staleness=None,
                    reasons="",
                    guidance=str(e),
                )
            )
            continue

        staleness = check_staleness(
            record,
            current_driver_version=current_driver_version or record.driver_version,
            current_binary_version=current_binary_version or record.server_binary_version,
            staleness_days=max_age_days,
        )

        if stale_only and not staleness.is_stale:
            continue

        if staleness.is_stale:
            reasons = ", ".join(r.value for r in staleness.reasons)
            guidance = _build_profile_guidance(staleness, record, max_age_days)
            entry_staleness = staleness
        else:
            reasons = ""
            guidance = ""
            entry_staleness = None
        results.append(
            ProfileScanEntry(
                path=profile_path,
                record=record,
                staleness=entry_staleness,
                reasons=reasons,
                guidance=guidance,
            )
        )

    return results


def _iterate_stale_profiles(
    profiles_dir: Path,
    max_age_days: int,
    current_driver_version: str | None = None,
    current_binary_version: str | None = None,
) -> list[tuple[Path, ProfileRecord | None, StalenessResult | None, str, str]]:
    """Iterate over stale profiles in the profiles directory.

    Shared helper used by both _check_profiles (check path) and
    _collect_profile_repair_actions (repair path) to avoid duplication.

    Returns stale profiles only — corrupt profiles are yielded with
    (profile_path, None, None, "", error_msg) tuples so callers can
    report warnings.

    Args:
        profiles_dir: Path to the profiles directory.
        max_age_days: Maximum acceptable profile age in days.
        current_driver_version: Current GPU driver version.
        current_binary_version: Current llama-server binary version.

    Returns:
        List of tuples: (profile_path, record_or_None, staleness_or_None,
        reasons, guidance).
    """
    entries = _scan_profiles(
        profiles_dir,
        max_age_days,
        stale_only=True,
        current_driver_version=current_driver_version,
        current_binary_version=current_binary_version,
    )
    return [(e.path, e.record, e.staleness, e.reasons, e.guidance) for e in entries]


def _iterate_all_profiles(
    profiles_dir: Path,
    max_age_days: int,
    current_driver_version: str | None = None,
    current_binary_version: str | None = None,
) -> list[tuple[Path, ProfileRecord | None, StalenessResult | None, str, str]]:
    """Iterate over ALL profiles in the profiles directory.

    Returns every profile (stale or not) along with its staleness result.
    Used by _check_profiles to count total profiles.

    Args:
        profiles_dir: Path to the profiles directory.
        max_age_days: Maximum acceptable profile age in days.
        current_driver_version: Current GPU driver version.
        current_binary_version: Current llama-server binary version.

    Returns:
        List of tuples: (profile_path, record, staleness_or_None, reasons, guidance).
    """
    entries = _scan_profiles(
        profiles_dir,
        max_age_days,
        stale_only=False,
        current_driver_version=current_driver_version,
        current_binary_version=current_binary_version,
    )
    return [(e.path, e.record, e.staleness, e.reasons, e.guidance) for e in entries]


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
        result.profiles_total = 0
        result.profiles_stale = 0
        return

    all_profiles = _iterate_all_profiles(
        profiles_dir,
        max_age_days=max_age_days,
        current_driver_version=current_driver_version,
        current_binary_version=current_binary_version,
    )

    total = 0
    stale = 0
    for entry in all_profiles:
        profile_path, record, staleness, reasons, guidance = entry
        if record is None:
            # Corrupt profile — add warning and skip
            result.warnings.append(
                f"Corrupt profile file skipped: {profile_path.name} ({guidance!r})"
            )
            continue
        total += 1
        if staleness is not None and staleness.is_stale:
            stale += 1
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
    """Print doctor check results in human-readable format with colors.

    Returns:
        Exit code (0 if healthy, 1 otherwise)
    """
    yes = Colors.bright_green("✓ YES")
    no = Colors.bright_red("✗ NO")
    warn_no = Colors.bright_yellow("⚠ NO")

    emit_heading("Doctor Check Results:")
    emit_success(f"  Toolchain complete: {yes if result.toolchain_complete else no}")
    emit_success(f"  Venv exists: {yes if result.venv_exists else no}")
    emit_success(f"  Venv intact: {yes if result.venv_intact else no}")
    emit_success(f"  Build lock free: {yes if result.build_lock_free else no}")
    emit_success(f"  Staging dirs clean: {yes if result.staging_dirs_clean else no}")
    emit_success(f"  Reports dir exists: {yes if result.reports_dir_exists else warn_no}")
    stale_count = (
        Colors.bright_red(str(result.profiles_stale))
        if result.profiles_stale > 0
        else Colors.bright_green(str(result.profiles_stale))
    )
    emit_success(f"  Profiles: {result.profiles_total} total, {stale_count} stale")

    if result.warnings:
        emit_success("")
        emit_warn("Warnings:")
        for warning in result.warnings:
            emit_info(f"- {warning}")

    if result.errors:
        emit_success("")
        emit_error("Errors:")
        for error in result.errors:
            emit_error(f"  - {error}")

    emit_success("")
    if result.is_healthy:
        emit_success("System is healthy!")
        return 0
    else:
        emit_error("System has issues. Run 'doctor --repair' to fix.")
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

    backend = resolve_backend_enum(backend_str)
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
        emit_json(result.to_dict())
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
    hints = collect_toolchain_repair_actions(sycl_hints + cuda_hints)

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
            command=["rm", "-rf", str(venv_path)],
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
                    command=["rm", "-rf", str(parent_dir)],
                    dry_run_command=f"# rm -rf '{parent_dir}'",
                    requires_confirmation=True,
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
                    command=["rm", str(lock_path)],
                    dry_run_command=f"# rm '{lock_path}'",
                    requires_confirmation=True,
                )
            )
    except (KeyError, TypeError, ValueError) as e:
        result.actions.append(
            RepairAction(
                action_type="remove_corrupt_lock",
                description=f"Remove corrupted build lock file ({e})",
                command=["rm", str(lock_path)],
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
        elif dir_path.exists() and not dir_path.is_dir():
            # Conflict: path exists but is not a directory (file or symlink)
            # Step 1: Remove conflicting file
            remove_index = len(result.actions)
            result.actions.append(
                RepairAction(
                    action_type="remove_file_or_directory",
                    description=f"Remove conflicting {name} file: {dir_path}",
                    command=["rm", "-rf", str(dir_path)],
                    dry_run_command=f"rm -rf '{dir_path}'",
                    requires_confirmation=True,
                )
            )
            # Step 2: Create directory (linked to removal)
            result.actions.append(
                RepairAction(
                    action_type="create_directory",
                    description=f"Create directory: {dir_path}",
                    command=["mkdir", "-m", "700", "-p", str(dir_path)],
                    dry_run_command=f"mkdir -m 700 -p '{dir_path}'",
                    requires_confirmation=False,
                    prerequisite_index=remove_index,
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

    stale_profiles = _iterate_stale_profiles(
        profiles_dir,
        max_age_days=max_age_days,
        current_driver_version=current_driver_version,
        current_binary_version=current_binary_version,
    )

    for entry in stale_profiles:
        profile_path, record, staleness, reasons, guidance = entry
        if record is None:
            # Corrupt profile — add warning and skip
            result.warnings.append(
                f"Corrupt profile file skipped: {profile_path.name} ({guidance!r})"
            )
            continue

        if staleness and staleness.is_stale:
            result.actions.append(
                RepairAction(
                    action_type="remove_stale_profile",
                    description=(
                        f"Remove stale profile: {profile_path.name} "
                        f"({reasons}, {staleness.age_days:.0f} days old) — {guidance}"
                    ),
                    command=["rm", str(profile_path)],
                    dry_run_command=f"# rm '{profile_path}'",
                    requires_confirmation=True,
                )
            )


def _execute_repair_action(action: RepairAction, result: DoctorRepairResult) -> None:
    """Execute a single repair action and update result."""
    if not action.command:
        return

    try:
        run_capture_command(action.command, check=True)
        result.performed_actions.append(action.description)
    except Exception as e:
        stderr = getattr(e, "stderr", None)
        result.failures.append(f"Failed to {action.description}: {stderr or str(e)}")
        result.success = False


def _execute_repair_actions(result: DoctorRepairResult) -> None:
    """Execute all repair actions."""
    for action in result.actions:
        _execute_repair_action(action, result)


def _print_repair_results(result: DoctorRepairResult) -> None:
    """Print repair results in human-readable format with colors."""
    emit_heading("Doctor Repair Actions:")

    if not result.actions:
        emit_success("No repairs needed. System is healthy.")
        return

    for i, action in enumerate(result.actions, 1):
        confirm_marker = (
            Colors.bright_yellow(" [CONFIRMATION REQUIRED]") if action.requires_confirmation else ""
        )
        emit_plain(f"  {Colors.cyan(str(i))}. {action.description}{confirm_marker}")
        if action.dry_run_command:
            emit_info(f"Command: {action.dry_run_command}")

    if result.performed_actions:
        emit_success("")
        emit_success("Performed actions:")
        for action in result.performed_actions:
            emit_success(f"{action}")

    if result.failures:
        emit_success("")
        emit_error("Failures:")
        for failure in result.failures:
            emit_error(f"{failure}")


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
        skip_confirmation = getattr(parsed, "yes", False)
        if skip_confirmation:
            _execute_repair_actions(result)
        else:
            # Prompt interactively for confirmation-required actions
            declined: set[int] = set()
            failed: set[int] = set()
            for idx, action in enumerate(result.actions):
                # Skip actions whose prerequisite was declined or failed
                if action.prerequisite_index is not None and (
                    action.prerequisite_index in declined or action.prerequisite_index in failed
                ):
                    result.warnings.append(
                        f"Skipped '{action.description}' because prerequisite action was declined"
                    )
                    continue
                if action.requires_confirmation:
                    # When emitting JSON, interactive prompts break the output stream;
                    # skip destructive actions rather than prompting.
                    if json_output:
                        result.warnings.append(
                            f"Skipped '{action.description}' (requires confirmation; use --yes to auto-accept)"
                        )
                        declined.add(idx)
                        continue
                    emit_plain(f"\nAction: {action.description}")
                    if action.dry_run_command:
                        emit_info(f"Command: {action.dry_run_command}")
                    try:
                        response = input("Confirm? [y/N]: ").strip().lower()
                    except EOFError:
                        emit_plain(f"Skipping action (no terminal input): {action.description}")
                        declined.add(idx)
                        continue
                    if response != "y":
                        emit_plain(f"Skipping action: {action.description}")
                        declined.add(idx)
                        continue
                failures_before = len(result.failures)
                _execute_repair_action(action, result)
                if len(result.failures) > failures_before:
                    failed.add(idx)

    if json_output:
        emit_json(result.to_dict())
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
  %(prog)s repair             Attempt automatic repairs
  %(prog)s fix                Alias for repair
  %(prog)s repair --dry-run   Show what would be fixed
  %(prog)s repair --json      Output repair actions in JSON

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

    # doctor --repair (alias: fix)
    repair_parser = subparsers.add_parser(
        "repair",
        aliases=["fix"],
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

    # Default: show help and return exit code
    def _show_help(args: argparse.Namespace) -> int:
        parser.print_help()
        return 1

    parser.set_defaults(func=_show_help)

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
