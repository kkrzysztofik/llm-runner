"""Additional doctor tests to cover uncovered branches.

Covers:
  - Backend-specific toolchain checks (SYCL, CUDA)
  - Venv not existing
  - Build lock branches (stale, corrupted)
  - Staging dir branches (failed markers)
  - _iterate_stale_profiles (empty dir)
  - _iterate_all_profiles (corrupt files)
  - _check_profiles corrupt profiles
  - _collect_toolchain_repair_actions
  - _collect_venv_repair_actions (venv missing, broken)
  - _collect_lock_repair_actions (corrupted lock)
  - _collect_directories_repair_actions (file conflicts)
  - _execute_repair_action (failure paths)
  - _print_check_results branches
  - _print_repair_results branches
  - main() exit code conversion
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.colors import Colors
from llama_cli.commands.doctor import (
    DoctorCheckResult,
    DoctorRepairResult,
    RepairAction,
    _check_build_lock,
    _check_profiles,
    _check_reports_dir,
    _check_staging_dirs,
    _check_toolchain,
    _check_venv,
    _collect_directories_repair_actions,
    _collect_lock_repair_actions,
    _collect_toolchain_repair_actions,
    _collect_venv_repair_actions,
    _execute_repair_action,
    _iterate_all_profiles,
    _iterate_stale_profiles,
    _print_check_results,
    _print_repair_results,
)
from llama_cli.commands.doctor import (
    main as doctor_main,
)
from llama_manager.build_pipeline import BuildBackend
from llama_manager.toolchain import ToolchainStatus


@pytest.fixture(autouse=True)
def disable_colors() -> Generator[None, None, None]:
    """Disable ANSI colors for all tests."""
    original = Colors.enabled
    Colors.enabled = False
    yield
    Colors.enabled = original


def _make_ns(**kwargs: object) -> argparse.Namespace:
    """Create an argparse.Namespace."""
    return argparse.Namespace(**kwargs)


class TestCheckToolchainBackendBranches:
    """Tests for _check_toolchain backend-specific branches."""

    def test_check_toolchain_sycl_backend(self, tmp_path) -> None:
        """_check_toolchain with SYCL backend should check SYCL readiness."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=False,
            venv_exists=False,
            venv_intact=False,
            build_lock_free=False,
            staging_dirs_clean=False,
            reports_dir_exists=False,
        )
        toolchain_status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler=None,
            cuda_toolkit="12.2.0",
            nvtop="3.1.0",
        )
        _check_toolchain(result, toolchain_status, BuildBackend.SYCL)
        assert result.toolchain_complete is False
        assert result.is_healthy is False
        assert any("Missing tools" in e for e in result.errors)

    def test_check_toolchain_cuda_backend(self, tmp_path) -> None:
        """_check_toolchain with CUDA backend should check CUDA readiness."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=False,
            venv_exists=False,
            venv_intact=False,
            build_lock_free=False,
            staging_dirs_clean=False,
            reports_dir_exists=False,
        )
        toolchain_status = ToolchainStatus(
            gcc="11.4.0",
            make="4.3",
            git="2.34.1",
            cmake="3.25.0",
            sycl_compiler="2023.1.0",
            cuda_toolkit=None,
            nvtop=None,
        )
        _check_toolchain(result, toolchain_status, BuildBackend.CUDA)
        assert result.toolchain_complete is False
        assert result.is_healthy is False
        assert any("Missing tools" in e for e in result.errors)

    def test_check_toolchain_all_backend(self, tmp_path) -> None:
        """_check_toolchain with 'all' backend should check completeness."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=False,
            venv_exists=False,
            venv_intact=False,
            build_lock_free=False,
            staging_dirs_clean=False,
            reports_dir_exists=False,
        )
        toolchain_status = ToolchainStatus(
            gcc=None,
            make=None,
            git=None,
            cmake=None,
            sycl_compiler=None,
            cuda_toolkit=None,
            nvtop=None,
        )
        _check_toolchain(result, toolchain_status, None)
        assert result.toolchain_complete is False
        assert result.is_healthy is False


class TestCheckVenvNotExisting:
    """Tests for _check_venv when venv does not exist."""

    def test_venv_not_existing_adds_warning(self, tmp_path) -> None:
        """_check_venv should add warning when venv does not exist."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=False,
            venv_intact=False,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
        )
        with patch("llama_cli.commands.doctor.get_venv_path") as mock_path:
            mock_path.return_value = tmp_path / "nonexistent"
            _check_venv(result)

        assert result.venv_exists is False
        assert any("Virtual environment does not exist" in w for w in result.warnings)


class TestCheckBuildLockBranches:
    """Tests for _check_build_lock branches."""

    def test_check_build_lock_stale(self, tmp_path) -> None:
        """_check_build_lock should detect stale locks."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
        )
        lock_path = tmp_path / "lock"
        old_time = time.time() - (100 * 24 * 60 * 60)
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 12345,
                    "started_at": old_time,
                    "backend": "sycl",
                }
            )
        )
        config = MagicMock()
        config.build_lock_path = lock_path
        config.toolchain_timeout_seconds = 1

        _check_build_lock(result, config)

        assert result.is_healthy is False
        assert any("Stale build lock" in e for e in result.errors)

    def test_check_build_lock_not_stale(self, tmp_path) -> None:
        """_check_build_lock should not flag fresh locks."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
        )
        lock_path = tmp_path / "lock"
        lock_path.write_text(
            json.dumps(
                {
                    "pid": 12345,
                    "started_at": time.time(),
                    "backend": "sycl",
                }
            )
        )
        config = MagicMock()
        config.build_lock_path = lock_path
        config.toolchain_timeout_seconds = 3600

        _check_build_lock(result, config)

        assert result.build_lock_free is False
        assert any("Build lock held" in w for w in result.warnings)

    def test_check_build_lock_corrupted(self, tmp_path) -> None:
        """_check_build_lock should detect corrupted lock files."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
        )
        lock_path = tmp_path / "lock"
        lock_path.write_text("not json")
        config = MagicMock()
        config.build_lock_path = lock_path
        config.toolchain_timeout_seconds = 3600

        _check_build_lock(result, config)

        assert result.is_healthy is False
        assert any("corrupted" in e.lower() for e in result.errors)


class TestCheckStagingDirs:
    """Tests for _check_staging_dirs branches."""

    def test_check_staging_dirs_with_failed_marker(self, tmp_path) -> None:
        """_check_staging_dirs should detect failed staging markers."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
        )
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / ".failed").touch()

        config = MagicMock()
        config.llama_cpp_root = str(tmp_path)

        _check_staging_dirs(result, config)

        assert result.staging_dirs_clean is False
        assert any("failed staging" in w.lower() for w in result.warnings)

    def test_check_staging_dirs_clean(self, tmp_path) -> None:
        """_check_staging_dirs should report clean when no markers."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=False,
            reports_dir_exists=True,
        )
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        config = MagicMock()
        config.llama_cpp_root = str(tmp_path)

        _check_staging_dirs(result, config)

        assert result.staging_dirs_clean is True


class TestIterateStaleProfiles:
    """Tests for _iterate_stale_profiles."""

    def _make_profile(self, days_old: int = 200, driver_version: str = "535.104.05") -> dict:
        """Create a profile record dict."""
        import hashlib
        from datetime import UTC, datetime, timedelta

        profiled_at = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
        return {
            "schema_version": "1.0",
            "gpu_identifier": "nvidia-geforce_rtx_3090-00",
            "backend": "cuda",
            "flavor": "balanced",
            "driver_version": driver_version,
            "driver_version_hash": hashlib.sha256(driver_version.encode()).hexdigest()[:16],
            "server_binary_version": "1.0.0",
            "profiled_at": profiled_at,
            "metrics": {"tokens_per_second": 85.5, "avg_latency_ms": 12.3, "peak_vram_mb": 10240.0},
            "parameters": {"threads": 8},
        }

    def test_iterate_stale_profiles_empty_dir(self, tmp_path) -> None:
        """_iterate_stale_profiles should return empty for non-existent dir."""
        result = _iterate_stale_profiles(tmp_path / "nonexistent", 90)
        assert result == []

    def test_iterate_stale_profiles_finds_stale(self, tmp_path) -> None:
        """_iterate_stale_profiles should find stale profiles."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "stale.json").write_text(json.dumps(self._make_profile(200)))

        result = _iterate_stale_profiles(profiles_dir, 90)
        assert len(result) == 1

    def test_iterate_stale_profiles_skips_fresh(self, tmp_path) -> None:
        """_iterate_stale_profiles should skip fresh profiles."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        # Use a very recent profile (1 day old) — age-only staleness with default
        # current_driver_version=None means no driver change detection
        (profiles_dir / "fresh.json").write_text(json.dumps(self._make_profile(1)))

        result = _iterate_stale_profiles(profiles_dir, 90)
        # Fresh profiles should not appear in stale results
        stale_paths = [r[0] for r in result]
        assert all("fresh" not in str(p) for p in stale_paths)

    def test_iterate_stale_profiles_skips_corrupt(self, tmp_path) -> None:
        """_iterate_stale_profiles should yield corrupt files with error."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "corrupt.json").write_text("not json")

        result = _iterate_stale_profiles(profiles_dir, 90)
        assert len(result) == 1
        _, record, _, _, guidance = result[0]
        assert record is None
        # The error message comes from json.loads
        assert "json" in guidance.lower() or "expecting" in guidance.lower()


class TestIterateAllProfiles:
    """Tests for _iterate_all_profiles."""

    def _make_profile(self, days_old: int = 200, driver_version: str = "535.104.05") -> dict:
        """Create a profile record dict."""
        import hashlib
        from datetime import UTC, datetime, timedelta

        profiled_at = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
        return {
            "schema_version": "1.0",
            "gpu_identifier": "nvidia-geforce_rtx_3090-00",
            "backend": "cuda",
            "flavor": "balanced",
            "driver_version": driver_version,
            "driver_version_hash": hashlib.sha256(driver_version.encode()).hexdigest()[:16],
            "server_binary_version": "1.0.0",
            "profiled_at": profiled_at,
            "metrics": {"tokens_per_second": 85.5, "avg_latency_ms": 12.3, "peak_vram_mb": 10240.0},
            "parameters": {"threads": 8},
        }

    def test_iterate_all_profiles_includes_fresh(self, tmp_path) -> None:
        """_iterate_all_profiles should include fresh profiles with None staleness."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        # Use 1-day-old profile — should not be stale by age
        (profiles_dir / "fresh.json").write_text(json.dumps(self._make_profile(1)))

        result = _iterate_all_profiles(profiles_dir, 90)
        assert len(result) == 1
        _, record, staleness, _, _ = result[0]
        assert record is not None
        # Fresh profiles should have None staleness (only stale ones have StalenessResult)
        assert staleness is None

    def test_iterate_all_profiles_includes_stale(self, tmp_path) -> None:
        """_iterate_all_profiles should include stale profiles with staleness."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "stale.json").write_text(json.dumps(self._make_profile(200)))

        result = _iterate_all_profiles(profiles_dir, 90)
        assert len(result) == 1
        _, record, staleness, _, _ = result[0]
        assert record is not None
        assert staleness is not None
        assert staleness.is_stale is True

    def test_iterate_all_profiles_empty_dir(self, tmp_path) -> None:
        """_iterate_all_profiles should return empty for non-existent dir."""
        result = _iterate_all_profiles(tmp_path / "nonexistent", 90)
        assert result == []


class TestCheckProfilesCorrupt:
    """Tests for _check_profiles with corrupt profile files."""

    def test_check_profiles_corrupt_skips_with_warning(self, tmp_path) -> None:
        """_check_profiles should skip corrupt profiles and add warning."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
            profiles_total=0,
            profiles_stale=0,
        )
        config = MagicMock()
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "corrupt.json").write_text("not json")
        config.profiles_dir = profiles_dir

        _check_profiles(result, config)

        assert result.profiles_total == 0
        assert any("Corrupt profile" in w for w in result.warnings)


class TestCollectToolchainRepairActions:
    """Tests for _collect_toolchain_repair_actions."""

    def test_collect_toolchain_actions_when_incomplete(self, tmp_path) -> None:
        """_collect_toolchain_repair_actions should add install actions when toolchain incomplete."""
        result = DoctorRepairResult(actions=[])

        with (
            patch("llama_cli.commands.doctor.detect_toolchain") as mock_detect,
            patch("llama_cli.commands.doctor.get_toolchain_hints") as mock_hints,
        ):
            mock_detect.return_value = ToolchainStatus(
                gcc=None,
                make=None,
                git=None,
                cmake=None,
                sycl_compiler=None,
                cuda_toolkit=None,
                nvtop=None,
            )
            mock_hints.return_value = [
                MagicMock(how_to_fix="Install GCC"),
                MagicMock(how_to_fix="Install CMake"),
            ]
            _collect_toolchain_repair_actions(result)

        assert len(result.actions) == 2
        assert all(a.action_type == "install_tool" for a in result.actions)

    def test_collect_toolchain_no_actions_when_complete(self, tmp_path) -> None:
        """_collect_toolchain_repair_actions should add no actions when toolchain complete."""
        result = DoctorRepairResult(actions=[])

        with patch("llama_cli.commands.doctor.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )
            _collect_toolchain_repair_actions(result)

        assert len(result.actions) == 0


class TestCollectVenvRepairActions:
    """Tests for _collect_venv_repair_actions."""

    def test_collect_venv_actions_when_missing(self, tmp_path) -> None:
        """_collect_venv_repair_actions should add create action when venv missing."""
        result = DoctorRepairResult(actions=[])

        with patch("llama_cli.commands.doctor.get_venv_path") as mock_path:
            mock_path.return_value = tmp_path / "missing-venv"
            _collect_venv_repair_actions(result)

        assert len(result.actions) == 1
        assert result.actions[0].action_type == "create_venv"

    def test_collect_venv_actions_when_broken(self, tmp_path) -> None:
        """_collect_venv_repair_actions should add remove+create actions when venv broken."""
        result = DoctorRepairResult(actions=[])
        venv_path = tmp_path / "broken-venv"
        venv_path.mkdir()

        with (
            patch("llama_cli.commands.doctor.get_venv_path") as mock_path,
            patch("llama_cli.commands.doctor.check_venv_integrity") as mock_integrity,
        ):
            mock_path.return_value = venv_path
            mock_integrity.return_value = (False, "broken")
            _collect_venv_repair_actions(result)

        assert len(result.actions) == 2
        assert result.actions[0].action_type == "remove_venv"
        assert result.actions[1].action_type == "create_venv"

    def test_collect_venv_no_actions_when_ok(self, tmp_path) -> None:
        """_collect_venv_repair_actions should add no actions when venv is OK."""
        result = DoctorRepairResult(actions=[])
        venv_path = tmp_path / "healthy-venv"
        venv_path.mkdir()

        with (
            patch("llama_cli.commands.doctor.get_venv_path") as mock_path,
            patch("llama_cli.commands.doctor.check_venv_integrity") as mock_integrity,
        ):
            mock_path.return_value = venv_path
            mock_integrity.return_value = (True, None)
            _collect_venv_repair_actions(result)

        assert len(result.actions) == 0


class TestCollectLockRepairActions:
    """Tests for _collect_lock_repair_actions."""

    def test_collect_lock_actions_corrupted(self, tmp_path) -> None:
        """_collect_lock_repair_actions should add remove action for corrupted lock."""
        result = DoctorRepairResult(actions=[])
        lock_path = tmp_path / "lock"
        lock_path.write_text("not json")

        config = MagicMock()
        config.build_lock_path = lock_path

        _collect_lock_repair_actions(result, config)

        assert len(result.actions) == 1
        assert result.actions[0].action_type == "remove_corrupt_lock"

    def test_collect_lock_no_actions_when_no_lock(self, tmp_path) -> None:
        """_collect_lock_repair_actions should add no actions when no lock file."""
        result = DoctorRepairResult(actions=[])
        lock_path = tmp_path / "nonexistent-lock"

        config = MagicMock()
        config.build_lock_path = lock_path

        _collect_lock_repair_actions(result, config)

        assert len(result.actions) == 0


class TestCollectDirectoriesRepairActions:
    """Tests for _collect_directories_repair_actions."""

    def test_collect_directories_file_conflict(self, tmp_path) -> None:
        """_collect_directories_repair_actions should handle file conflicts."""
        result = DoctorRepairResult(actions=[])
        reports_dir = tmp_path / "reports"
        # Create a file (not directory) at the expected path
        reports_dir.write_text("I am a file")

        config = MagicMock()
        config.reports_dir = reports_dir
        config.profiles_dir = tmp_path / "profiles"
        config.builds_dir = tmp_path / "builds"

        _collect_directories_repair_actions(result, config)

        action_types = [a.action_type for a in result.actions]
        assert "remove_file_or_directory" in action_types
        assert "create_directory" in action_types

    def test_collect_directories_all_missing(self, tmp_path) -> None:
        """_collect_directories_repair_actions should create all missing dirs."""
        result = DoctorRepairResult(actions=[])

        config = MagicMock()
        config.reports_dir = tmp_path / "reports"
        config.profiles_dir = tmp_path / "profiles"
        config.builds_dir = tmp_path / "builds"

        _collect_directories_repair_actions(result, config)

        action_types = [a.action_type for a in result.actions]
        assert action_types.count("create_directory") == 3


class TestExecuteRepairAction:
    """Tests for _execute_repair_action."""

    def test_execute_action_no_command(self) -> None:
        """_execute_repair_action should skip actions without command."""
        result = DoctorRepairResult(actions=[])
        action = RepairAction(
            action_type="test",
            description="No command",
            command=None,
            dry_run_command=None,
        )
        _execute_repair_action(action, result)
        assert len(result.performed_actions) == 0

    def test_execute_action_success(self) -> None:
        """_execute_repair_action should record success."""
        result = DoctorRepairResult(actions=[])
        action = RepairAction(
            action_type="test",
            description="Success",
            command="echo",
            args=["hello"],
            dry_run_command="# echo hello",
        )

        with patch("llama_cli.commands.doctor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _execute_repair_action(action, result)

        assert len(result.performed_actions) == 1
        assert "Success" in result.performed_actions[0]

    def test_execute_action_failure(self) -> None:
        """_execute_repair_action should record failure."""
        result = DoctorRepairResult(actions=[])
        action = RepairAction(
            action_type="test",
            description="Fail",
            command="false",
            dry_run_command="# false",
        )

        with patch(
            "llama_cli.commands.doctor.subprocess.run",
            side_effect=Exception("command failed"),
        ):
            _execute_repair_action(action, result)

        assert result.success is False
        assert len(result.failures) == 1
        assert "Fail" in result.failures[0]


class TestPrintCheckResults:
    """Tests for _print_check_results."""

    def test_print_check_results_healthy(self, capsys) -> None:
        """_print_check_results should return 0 for healthy system."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
        )
        exit_code = _print_check_results(result)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "healthy" in captured.out.lower()

    def test_print_check_results_unhealthy(self, capsys) -> None:
        """_print_check_results should return 1 for unhealthy system."""
        result = DoctorCheckResult(
            is_healthy=False,
            toolchain_complete=False,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
            errors=["test error"],
        )
        exit_code = _print_check_results(result)
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "issues" in captured.out.lower()

    def test_print_check_results_shows_warnings(self, capsys) -> None:
        """_print_check_results should display warnings."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
            warnings=["test warning"],
        )
        exit_code = _print_check_results(result)
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Warnings" in captured.out or "warning" in captured.out.lower()

    def test_print_check_results_shows_errors(self, capsys) -> None:
        """_print_check_results should display errors (to stderr via print_error)."""
        result = DoctorCheckResult(
            is_healthy=False,
            toolchain_complete=False,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
            errors=["test error"],
        )
        exit_code = _print_check_results(result)
        assert exit_code == 1
        captured = capsys.readouterr()
        # Errors are printed to stderr via print_error
        assert "Errors" in captured.err or "error" in captured.err.lower()


class TestPrintRepairResults:
    """Tests for _print_repair_results."""

    def test_print_repair_results_no_actions(self, capsys) -> None:
        """_print_repair_results should show 'no repairs needed' when no actions."""
        result = DoctorRepairResult(actions=[])
        _print_repair_results(result)
        captured = capsys.readouterr()
        assert "No repairs needed" in captured.out or "healthy" in captured.out.lower()

    def test_print_repair_results_with_actions(self, capsys) -> None:
        """_print_repair_results should list actions."""
        result = DoctorRepairResult(
            actions=[
                RepairAction(
                    action_type="test",
                    description="Test action",
                    command="test",
                    dry_run_command="# test",
                )
            ]
        )
        _print_repair_results(result)
        captured = capsys.readouterr()
        assert "Test action" in captured.out

    def test_print_repair_results_with_performed(self, capsys) -> None:
        """_print_repair_results should show performed actions."""
        result = DoctorRepairResult(
            actions=[
                RepairAction(
                    action_type="test",
                    description="Test",
                    command="test",
                    dry_run_command="# test",
                )
            ],
            performed_actions=["Action 1", "Action 2"],
        )
        _print_repair_results(result)
        captured = capsys.readouterr()
        assert "Performed actions" in captured.out or "Performed" in captured.out

    def test_print_repair_results_with_failures(self, capsys) -> None:
        """_print_repair_results should show failures (to stderr via print_error)."""
        result = DoctorRepairResult(
            actions=[
                RepairAction(
                    action_type="test",
                    description="Test",
                    command="test",
                    dry_run_command="# test",
                )
            ],
            failures=["Failed action 1"],
            success=False,
        )
        _print_repair_results(result)
        captured = capsys.readouterr()
        # Failures are printed to stderr via print_error
        assert "Failures" in captured.err or "Failure" in captured.err


class TestDoctorMainExitCodeConversion:
    """Tests for doctor main() exit code conversion from DoctorRepairResult."""

    def test_main_repair_result_success(self, tmp_path, capsys) -> None:
        """doctor main should return 0 when DoctorRepairResult.success is True."""
        with (
            patch("llama_cli.commands.doctor.detect_toolchain") as mock_detect,
            patch("llama_cli.commands.doctor.get_venv_path") as mock_venv_path,
            patch("llama_cli.commands.doctor.Config") as mock_config,
        ):
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()
            with patch("llama_cli.commands.doctor.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.profiles_dir = tmp_path / "profiles"
                mock_config_instance.builds_dir = tmp_path / "builds"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance
                (tmp_path / "reports").mkdir()
                (tmp_path / "profiles").mkdir()
                (tmp_path / "builds").mkdir()

                exit_code = doctor_main(["repair", "--dry-run"])

            assert exit_code == 0

    def test_main_check_returns_int(self, tmp_path, capsys) -> None:
        """doctor main should return int directly for cmd_doctor_check."""
        with (
            patch("llama_cli.commands.doctor.detect_toolchain") as mock_detect,
            patch("llama_cli.commands.doctor.get_venv_path") as mock_venv_path,
            patch("llama_cli.commands.doctor.Config") as mock_config,
        ):
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()
            with patch("llama_cli.commands.doctor.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                exit_code = doctor_main(["check"])

            assert exit_code == 0


class TestCheckReportsDir:
    """Tests for _check_reports_dir."""

    def test_check_reports_dir_exists(self, tmp_path) -> None:
        """_check_reports_dir should set reports_dir_exists=True when dir exists."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=False,
        )
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        config = MagicMock()
        config.reports_dir = reports_dir

        _check_reports_dir(result, config)

        assert result.reports_dir_exists is True

    def test_check_reports_dir_missing(self, tmp_path) -> None:
        """_check_reports_dir should add warning when reports dir missing."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=False,
        )
        reports_dir = tmp_path / "reports"
        config = MagicMock()
        config.reports_dir = reports_dir

        _check_reports_dir(result, config)

        assert result.reports_dir_exists is False
        assert any("Reports directory does not exist" in w for w in result.warnings)
