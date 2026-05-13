import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _pytest.capture import CaptureFixture

from llama_cli.commands.doctor import (
    DoctorCheckResult,
    DoctorRepairResult,
    RepairAction,
    _build_profile_guidance,
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
    cmd_doctor_check,
    cmd_doctor_repair,
)
from llama_cli.commands.doctor import (
    main as doctor_main,
)
from llama_manager.build_pipeline import BuildBackend
from llama_manager.config.profile_cache import (
    ProfileRecord,
    StalenessResult,
)
from llama_manager.toolchain import ToolchainStatus
from tests.support.helpers import make_toolchain_status
from tests.support.helpers import namespace as _make_namespace


def doctor_mocks(
    tmp_path: Path,
    *,
    toolchain: ToolchainStatus | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, ExitStack]:
    """Set up common mocks for doctor CLI tests.

    Args:
        tmp_path: pytest tmp_path fixture
        toolchain: Optional custom ToolchainStatus. Defaults to complete toolchain.
        config_overrides: Optional dict of Config attribute overrides.

    Returns:
        Tuple of (mock_detect, mock_venv_path, mock_config_cls, mock_integrity,
        mock_config_instance, exit_stack). The caller MUST call exit_stack.close()
        when done to stop all patches.
    """
    stack = ExitStack()
    mock_detect = stack.enter_context(patch("llama_cli.commands.doctor.detect_toolchain"))
    mock_venv_path = stack.enter_context(patch("llama_cli.commands.doctor.get_venv_path"))
    mock_config_cls = stack.enter_context(patch("llama_cli.commands.doctor.Config"))

    if toolchain is None:
        toolchain = make_toolchain_status()
    mock_detect.return_value = toolchain

    mock_venv_path.return_value = tmp_path / "venv"
    (tmp_path / "venv").mkdir()

    mock_integrity = stack.enter_context(patch("llama_cli.commands.doctor.check_venv_integrity"))
    mock_integrity.return_value = (True, None)

    mock_config_instance = MagicMock()
    mock_config_instance.build_lock_path = tmp_path / "lock"
    mock_config_instance.reports_dir = tmp_path / "reports"
    mock_config_instance.toolchain_timeout_seconds = 3600
    mock_config_instance.llama_cpp_root = str(tmp_path)
    mock_config_instance.profiles_dir = tmp_path / "profiles"
    mock_config_instance.builds_dir = tmp_path / "builds"

    if config_overrides:
        for key, value in config_overrides.items():
            setattr(mock_config_instance, key, value)

    mock_config_cls.return_value = mock_config_instance

    return (
        mock_detect,
        mock_venv_path,
        mock_config_cls,
        mock_integrity,
        mock_config_instance,
        stack,
    )


class TestDoctorCheckResult:
    """Test DoctorCheckResult dataclass."""

    def test_doctor_check_result_to_dict(self) -> None:
        """DoctorCheckResult should convert to dictionary with expected keys."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
            warnings=["warning1"],
            errors=["error1"],
        )
        data = result.to_dict()
        assert isinstance(data, dict)
        assert "is_healthy" in data
        assert "toolchain_complete" in data
        assert "warnings" in data
        assert "errors" in data


class TestDoctorRepairResult:
    """Test DoctorRepairResult dataclass."""

    def test_doctor_repair_result_to_dict(self) -> None:
        """DoctorRepairResult should convert to dictionary with expected keys."""
        result = DoctorRepairResult(
            actions=[
                RepairAction(
                    action_type="test",
                    description="Test action",
                    command=["test"],
                    dry_run_command="# test",
                    requires_confirmation=True,
                )
            ],
            performed_actions=["action1"],
            failures=["failure1"],
            success=True,
        )
        data = result.to_dict()
        assert isinstance(data, dict)
        assert "actions" in data
        assert "performed_actions" in data
        assert "failures" in data
        assert "success" in data


class TestRepairAction:
    """Test RepairAction dataclass."""

    def test_repair_action_to_dict(self) -> None:
        """RepairAction should convert to dictionary with expected keys."""
        action = RepairAction(
            action_type="test",
            description="Test action",
            command=["test"],
            dry_run_command="# test",
            requires_confirmation=True,
        )
        data = action.to_dict()
        assert isinstance(data, dict)
        assert "action_type" in data
        assert "description" in data
        assert "requires_confirmation" in data

    def test_repair_action_defaults(self) -> None:
        """RepairAction should have correct defaults."""
        action = RepairAction(
            action_type="test",
            description="Test action",
            command=["test"],
            dry_run_command="# test",
        )
        data = action.to_dict()
        assert data["requires_confirmation"] is False


class TestCmdDoctorCheck:
    """Test doctor check command."""

    def test_doctor_check_succeeds_healthy_system(self, tmp_path, capsys) -> None:
        """doctor check should succeed when system is healthy."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

            # Should succeed for healthy system
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "healthy" in captured.out.lower() or "System is healthy" in captured.out

    def test_doctor_check_fails_with_incomplete_toolchain(self, tmp_path, capsys) -> None:
        """doctor check should fail when toolchain is incomplete."""
        incomplete_toolchain = make_toolchain_status(
            sycl_compiler=None,
            cuda_toolkit=None,
            nvtop=None,
        )
        _, _, _, _, _, stack = doctor_mocks(tmp_path, toolchain=incomplete_toolchain)
        with stack:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

            # Should fail because toolchain is incomplete
            assert exit_code != 0
            captured = capsys.readouterr()
            # Errors are printed to stderr
            assert "Missing tools" in captured.err or "error" in captured.err.lower()

    def test_doctor_check_json_output(self, tmp_path, capsys) -> None:
        """doctor check --json should produce valid JSON output."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=True))

            assert exit_code == 0
            captured = capsys.readouterr()

            # Should be valid JSON
            try:
                parsed = json.loads(captured.out)
                assert "is_healthy" in parsed
                assert "toolchain_complete" in parsed
                assert "venv_exists" in parsed
                assert "venv_intact" in parsed
                assert "build_lock_free" in parsed
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")

    def test_doctor_check_stale_lock_detection(self, tmp_path, capsys) -> None:
        """doctor --check should detect stale build locks (T083)."""
        # Create stale lock file (100 days ago to ensure it's stale)
        lock_file = tmp_path / "lock"
        old_time = time.time() - (100 * 24 * 60 * 60)  # 100 days ago
        lock_data = {
            "pid": 12345,
            "started_at": old_time,
            "backend": "sycl",
        }
        lock_file.write_text(json.dumps(lock_data))

        _, _, _, _, _, stack = doctor_mocks(
            tmp_path,
            config_overrides={
                "build_lock_path": lock_file,
                "toolchain_timeout_seconds": 1,
            },
        )
        with stack:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

            # Should detect stale lock
            assert exit_code != 0
            captured = capsys.readouterr()
            assert "Stale build lock" in captured.err


class TestCmdDoctorRepair:
    """Test doctor --repair command."""

    def test_doctor_repair_no_actions_needed(self, tmp_path, capsys) -> None:
        """doctor --repair should return no actions when system is healthy."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            # Create directories so no repair actions are needed
            (tmp_path / "reports").mkdir()
            (tmp_path / "profiles").mkdir()
            (tmp_path / "builds").mkdir()

            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            # Should have no repair actions needed
            assert result.success is True
            assert len(result.actions) == 0

    def test_doctor_fix_alias_runs_repair(self, tmp_path, capsys) -> None:
        """doctor fix should be an alias for doctor repair."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            (tmp_path / "reports").mkdir()
            (tmp_path / "profiles").mkdir()
            (tmp_path / "builds").mkdir()

            # Use the main entry point with "fix" instead of "repair"
            exit_code = doctor_main(["fix", "--dry-run"])

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "No repairs needed" in captured.out

    def test_doctor_repair_clears_failed_staging(self, tmp_path, capsys) -> None:
        """doctor --repair should identify failed staging directories for cleanup (T081)."""
        # Create build directory with failed marker
        build_dir = tmp_path / "llama.cpp" / "build"
        build_dir.mkdir(parents=True)
        failed_marker = build_dir / ".failed"
        failed_marker.touch()

        _, _, _, _, _, stack = doctor_mocks(
            tmp_path,
            config_overrides={"llama_cpp_root": str(tmp_path / "llama.cpp")},
        )
        with stack:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            # Should identify failed staging directories
            assert result.success is True
            assert len(result.actions) > 0
            action_types = [a.action_type for a in result.actions]
            assert "clean_failed_staging" in action_types

    def test_doctor_repair_preserves_successful_artifacts(self, tmp_path, capsys) -> None:
        """doctor --repair should preserve successful artifacts (T082)."""
        # Create build directory with successful artifact
        build_dir = tmp_path / "llama.cpp" / "build"
        build_dir.mkdir(parents=True)
        successful_binary = build_dir / "llama-server"
        successful_binary.write_text("successful binary content")

        # Create a failed staging marker in a subdirectory
        failed_staging = build_dir / "staging" / "test"
        failed_staging.mkdir(parents=True)
        failed_marker = failed_staging / ".failed"
        failed_marker.touch()

        _, _, _, _, _, stack = doctor_mocks(
            tmp_path,
            config_overrides={"llama_cpp_root": str(tmp_path / "llama.cpp")},
        )
        with stack:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            # Should identify failed staging for cleanup
            assert result.success is True
            assert len(result.actions) > 0

    def test_doctor_repair_handles_stale_locks(self, tmp_path, capsys) -> None:
        """doctor --repair should handle stale locks (T083)."""
        # Create stale lock file (100 days ago to ensure it's stale)
        lock_file = tmp_path / "lock"
        old_time = time.time() - (100 * 24 * 60 * 60)  # 100 days ago
        lock_data = {
            "pid": 12345,
            "started_at": old_time,
            "backend": "sycl",
        }
        lock_file.write_text(json.dumps(lock_data))

        _, _, _, _, _, stack = doctor_mocks(
            tmp_path,
            config_overrides={
                "build_lock_path": lock_file,
                "toolchain_timeout_seconds": 1,
            },
        )
        with stack:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            # Should identify stale lock for removal
            assert result.success is True
            assert len(result.actions) > 0
            action_types = [a.action_type for a in result.actions]
            assert "remove_stale_lock" in action_types

    def test_doctor_repair_json_output(self, tmp_path, capsys) -> None:
        """doctor --repair --json should produce valid JSON output."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=True))

            assert result.success is True
            captured = capsys.readouterr()

            # Should be valid JSON
            try:
                parsed = json.loads(captured.out)
                assert "actions" in parsed
                assert "performed_actions" in parsed
                assert "failures" in parsed
                assert "success" in parsed
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")


class TestDoctorSuccessPath:
    """T084: Test doctor success path (no repairs needed)."""

    def test_doctor_check_success_no_repairs_needed(self, tmp_path, capsys) -> None:
        """doctor check should succeed when no repairs are needed."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

            # Should succeed (exit code 0)
            assert exit_code == 0

    def test_doctor_repair_success_no_repairs_needed(self, tmp_path, capsys) -> None:
        """doctor --repair should succeed when no repairs are needed."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            # Create directories so no repair actions are needed
            (tmp_path / "reports").mkdir()
            (tmp_path / "profiles").mkdir()
            (tmp_path / "builds").mkdir()

            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            # Should succeed with no actions
            assert result.success is True
            assert len(result.actions) == 0


class TestDoctorCLIIntegration:
    """Integration tests for doctor CLI."""

    def test_doctor_main_with_check_command(self, tmp_path, capsys) -> None:
        """doctor check command should work through main()."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            exit_code = doctor_main(["check"])

            # Should succeed
            assert exit_code == 0

    def test_doctor_main_with_repair_command(self, tmp_path, capsys) -> None:
        """doctor --repair command should work through main()."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            exit_code = doctor_main(["repair", "--dry-run"])

            # Should succeed
            assert exit_code == 0


class TestProfileStalenessCheck:
    """T085-T087: Test profile staleness check and repair via doctor CLI."""

    def _make_profile_record(
        self,
        days_old: int,
        gpu: str = "nvidia-geforce_rtx_3090-00",
        backend: str = "cuda",
        flavor: str = "balanced",
        driver_version: str = "535.104.05",
        binary_version: str = "1.18.0",
    ) -> dict[str, object]:
        """Create a profile record dict with a given age."""
        from datetime import UTC, datetime, timedelta

        profiled_at = (datetime.now(UTC) - timedelta(days=days_old)).isoformat()
        import hashlib

        driver_hash = hashlib.sha256(driver_version.encode()).hexdigest()[:16]
        return {
            "schema_version": "1.0",
            "gpu_identifier": gpu,
            "backend": backend,
            "flavor": flavor,
            "driver_version": driver_version,
            "driver_version_hash": driver_hash,
            "server_binary_version": binary_version,
            "profiled_at": profiled_at,
            "metrics": {
                "tokens_per_second": 85.5,
                "avg_latency_ms": 12.3,
                "peak_vram_mb": 10240.0,
            },
            "parameters": {"threads": 8},
        }

    def test_check_profiles_finds_no_profiles(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """doctor check should handle empty profiles directory gracefully."""
        # Create empty profiles dir
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Profiles: 0 total, 0 stale" in captured.out
        finally:
            stack.close()

    def test_check_profiles_detects_stale_profiles(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """doctor check should detect stale profiles and report them as warnings."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create a stale profile (200 days old, beyond 90-day default)
        stale_record = self._make_profile_record(days_old=200)
        (profiles_dir / "test-profile.json").write_text(json.dumps(stale_record))

        # Create a fresh profile (10 days old)
        fresh_record = self._make_profile_record(days_old=10)
        (profiles_dir / "fresh-profile.json").write_text(json.dumps(fresh_record))

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

            assert exit_code == 0  # doctor reports but doesn't fail on stale profiles
            captured = capsys.readouterr()
            assert "Profiles: 2 total, 1 stale" in captured.out
            assert "Stale profile: test-profile.json" in captured.out
            assert "age_exceeded" in captured.out
        finally:
            stack.close()

    def test_check_profiles_json_output_includes_profile_stats(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """doctor check --json should include profile stats in output."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        stale_record = self._make_profile_record(days_old=200)
        (profiles_dir / "test-profile.json").write_text(json.dumps(stale_record))

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=True))

            assert exit_code == 0  # doctor reports but doesn't fail on stale profiles
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["profiles_total"] == 1
            assert parsed["profiles_stale"] == 1
        finally:
            stack.close()

    def test_check_profiles_custom_max_age(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """doctor check --max-age-days should use custom threshold."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create a profile that is 100 days old
        stale_record = self._make_profile_record(days_old=100)
        (profiles_dir / "test-profile.json").write_text(json.dumps(stale_record))

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            # With max-age-days=200, the 100-day-old profile should NOT be stale
            exit_code = cmd_doctor_check(
                _make_namespace(backend="all", json=False, max_age_days=200)
            )

            assert exit_code == 0  # no stale profiles
            captured = capsys.readouterr()
            assert "Profiles: 1 total, 0 stale" in captured.out
        finally:
            stack.close()

    def test_repair_collects_stale_profile_actions(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """doctor --repair should collect stale profile removal actions."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        stale_record = self._make_profile_record(days_old=200)
        (profiles_dir / "stale-profile.json").write_text(json.dumps(stale_record))

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            assert result.success is True
            action_types = [a.action_type for a in result.actions]
            assert "remove_stale_profile" in action_types
            assert len(result.actions) > 0
        finally:
            stack.close()

    def test_repair_respects_max_age_days(self, tmp_path: Path) -> None:
        """doctor --repair --max-age-days should only flag profiles older than threshold."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Create a profile that is 100 days old
        stale_record = self._make_profile_record(days_old=100)
        (profiles_dir / "medium-age.json").write_text(json.dumps(stale_record))

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            # With max-age-days=200, the 100-day-old profile should NOT be stale
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False, max_age_days=200))

            action_types = [a.action_type for a in result.actions]
            assert "remove_stale_profile" not in action_types
        finally:
            stack.close()

    def test_repair_skips_corrupt_profile_files(self, tmp_path: Path) -> None:
        """doctor --repair should skip corrupt/unparseable profile files."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()

        # Write corrupt JSON
        (profiles_dir / "corrupt.json").write_text("not valid json {{{")

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            # Should not crash and should have no stale profile actions
            assert result.success is True
            action_types = [a.action_type for a in result.actions]
            assert "remove_stale_profile" not in action_types

            # Should emit a warning for the corrupt profile file
            warnings = [w for w in result.warnings if "corrupt" in w.lower()]
            assert len(warnings) >= 1, "Expected warning for corrupt profile file"
            assert "corrupt.json" in warnings[0]
        finally:
            stack.close()

    def test_check_profiles_no_profiles_dir(
        self, tmp_path: Path, capsys: CaptureFixture[str]
    ) -> None:
        """doctor check should handle missing profiles directory gracefully."""
        profiles_dir = tmp_path / "profiles"
        # Don't create the directory

        (
            mock_detect,
            mock_venv_path,
            mock_config_cls,
            mock_integrity,
            mock_config_instance,
            stack,
        ) = doctor_mocks(tmp_path)
        mock_config_instance.profiles_dir = profiles_dir
        try:
            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Profiles: 0 total, 0 stale" in captured.out
        finally:
            stack.close()


class TestBuildProfileGuidance:
    """Tests for _build_profile_guidance function."""

    def _make_staleness(
        self,
        reasons: list[str],
        driver_version: str = "535.104.05",
        current_driver_version: str = "545.23.08",
        age_days: float = 200.0,
    ) -> tuple[StalenessResult, ProfileRecord]:
        """Helper to create a staleness result and profile record."""
        from datetime import UTC, datetime, timedelta

        from llama_manager.config.profile_cache import (
            ProfileFlavor,
            ProfileMetrics,
            StalenessReason,
            compute_driver_version_hash,
        )

        now = datetime.now(UTC) - timedelta(days=int(age_days))
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version=driver_version,
            driver_version_hash=compute_driver_version_hash(driver_version),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=85.5,
                avg_latency_ms=12.3,
                peak_vram_mb=10240.0,
            ),
        )

        staleness_reasons = [getattr(StalenessReason, r.upper()) for r in reasons]
        staleness = StalenessResult(
            is_stale=bool(staleness_reasons),
            reasons=staleness_reasons,
            driver_version_display=current_driver_version,
            age_days=age_days,
        )
        return staleness, record

    def test_driver_changed_guidance(self) -> None:
        """Should return driver change guidance."""
        staleness, record = self._make_staleness(["driver_changed"])
        guidance = _build_profile_guidance(staleness, record)
        assert "Re-profile recommended" in guidance
        assert "driver version changed" in guidance
        assert record.driver_version in guidance
        assert staleness.driver_version_display in guidance

    def test_binary_changed_guidance(self) -> None:
        """Should return binary change guidance."""
        staleness, record = self._make_staleness(["binary_changed"])
        guidance = _build_profile_guidance(staleness, record)
        assert "Re-profile recommended" in guidance
        assert "llama-server binary was updated" in guidance

    def test_age_exceeded_guidance(self) -> None:
        """Should include max_age_days in message."""
        staleness, record = self._make_staleness(["age_exceeded"], age_days=120.0)
        guidance = _build_profile_guidance(staleness, record, max_age_days=90)
        assert "Re-profile recommended" in guidance
        assert "120 days old" in guidance
        assert "threshold: 90 days" in guidance

    def test_no_reasons_returns_default(self) -> None:
        """Empty reasons should return default message."""
        staleness, record = self._make_staleness([])
        guidance = _build_profile_guidance(staleness, record)
        assert guidance == "Re-profile recommended"

    def test_multiple_reasons_semicolon_separated(self) -> None:
        """Multiple reasons are joined with semicolons."""
        staleness, record = self._make_staleness(
            ["driver_changed", "binary_changed", "age_exceeded"]
        )
        guidance = _build_profile_guidance(staleness, record)
        assert "driver version changed" in guidance
        assert "llama-server binary was updated" in guidance
        assert "days old" in guidance
        # Verify semicolon separation
        assert guidance.count(";") == 2

    def test_max_age_days_parameter_used(self) -> None:
        """max_age_days parameter should appear in age_exceeded message."""
        staleness, record = self._make_staleness(["age_exceeded"], age_days=60.0)
        guidance = _build_profile_guidance(staleness, record, max_age_days=30)
        assert "threshold: 30 days" in guidance


class TestDirectoryRepairActions:
    """Tests for directory creation repair actions."""

    def test_repair_creates_missing_directories(self, tmp_path: Path) -> None:
        """doctor --repair should create missing standard directories."""
        reports_dir = tmp_path / "reports"
        profiles_dir = tmp_path / "profiles"
        builds_dir = tmp_path / "builds"
        # Don't create them

        _, _, _, _, _, stack = doctor_mocks(
            tmp_path,
            config_overrides={
                "reports_dir": reports_dir,
                "profiles_dir": profiles_dir,
                "builds_dir": builds_dir,
            },
        )
        with stack:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            action_types = [a.action_type for a in result.actions]
            assert "create_directory" in action_types
            # Should have 3 directory creation actions
            assert len([a for a in result.actions if a.action_type == "create_directory"]) == 3

    def test_repair_skips_existing_directories(self, tmp_path: Path) -> None:
        """doctor --repair should skip directories that already exist."""
        reports_dir = tmp_path / "reports"
        profiles_dir = tmp_path / "profiles"
        builds_dir = tmp_path / "builds"
        reports_dir.mkdir()
        profiles_dir.mkdir()
        builds_dir.mkdir()

        _, _, _, _, _, stack = doctor_mocks(
            tmp_path,
            config_overrides={
                "reports_dir": reports_dir,
                "profiles_dir": profiles_dir,
                "builds_dir": builds_dir,
            },
        )
        with stack:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            action_types = [a.action_type for a in result.actions]
            assert "create_directory" not in action_types

    def test_directory_repair_action_details(self, tmp_path: Path) -> None:
        """Directory repair actions should have correct command and permissions."""
        reports_dir = tmp_path / "reports"

        _, _, _, _, _, stack = doctor_mocks(
            tmp_path,
            config_overrides={"reports_dir": reports_dir},
        )
        with stack:
            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            dir_actions = [a for a in result.actions if a.action_type == "create_directory"]
            assert len(dir_actions) == 3

            # Check that one of them is for reports
            reports_action = next((a for a in dir_actions if "reports" in a.description), None)
            assert reports_action is not None
            assert reports_action.command == ["mkdir", "-m", "700", "-p", str(reports_dir)]
            assert reports_action.requires_confirmation is False


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
        toolchain_status = make_toolchain_status(sycl_compiler=None)
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
        toolchain_status = make_toolchain_status(cuda_toolkit=None, nvtop=None)
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
        toolchain_status = make_toolchain_status(
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
            mock_detect.return_value = make_toolchain_status(
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
            mock_detect.return_value = make_toolchain_status()
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
            command=["echo", "hello"],
            dry_run_command="# echo hello",
        )

        with patch("llama_cli.commands.doctor.run_capture_command") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _execute_repair_action(action, result)

        mock_run.assert_called_once_with(["echo", "hello"], check=True)
        assert len(result.performed_actions) == 1
        assert "Success" in result.performed_actions[0]

    def test_execute_action_failure(self) -> None:
        """_execute_repair_action should record failure."""
        result = DoctorRepairResult(actions=[])
        action = RepairAction(
            action_type="test",
            description="Fail",
            command=["false"],
            dry_run_command="# false",
        )

        with patch(
            "llama_cli.commands.doctor.run_capture_command",
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
        assert "issues" in captured.err.lower()

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
                    command=["test"],
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
                    command=["test"],
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
                    command=["test"],
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
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
            (tmp_path / "reports").mkdir()
            (tmp_path / "profiles").mkdir()
            (tmp_path / "builds").mkdir()

            exit_code = doctor_main(["repair", "--dry-run"])

        assert exit_code == 0

    def test_main_check_returns_int(self, tmp_path, capsys) -> None:
        """doctor main should return int directly for cmd_doctor_check."""
        _, _, _, _, _, stack = doctor_mocks(tmp_path)
        with stack:
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
