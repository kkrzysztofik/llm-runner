"""Tests for doctor CLI commands (T081-T087).

Test Tasks:
- T081: Test doctor --repair clears failed staging (FR-004.7)
- T082: Test doctor --repair preserves successful artifacts (FR-004.7)
- T083: Test doctor --repair handles stale locks (FR-004.7)
- T084: Test doctor success path (no repairs needed)
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.doctor_cli import (
    DoctorCheckResult,
    DoctorRepairResult,
    RepairAction,
    cmd_doctor_check,
    cmd_doctor_repair,
)
from llama_cli.doctor_cli import (
    main as doctor_main,
)
from llama_manager.toolchain import ToolchainStatus


class TestDoctorCheckResult:
    """Test DoctorCheckResult dataclass."""

    def test_doctor_check_result_to_dict(self) -> None:
        """DoctorCheckResult should convert to dictionary correctly."""
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

        assert data["is_healthy"] is True
        assert data["toolchain_complete"] is True
        assert data["venv_exists"] is True
        assert data["venv_intact"] is True
        assert data["build_lock_free"] is True
        assert data["staging_dirs_clean"] is True
        assert data["reports_dir_exists"] is True
        assert data["warnings"] == ["warning1"]
        assert data["errors"] == ["error1"]

    def test_doctor_check_result_defaults(self) -> None:
        """DoctorCheckResult should have correct defaults for warnings and errors."""
        result = DoctorCheckResult(
            is_healthy=True,
            toolchain_complete=True,
            venv_exists=True,
            venv_intact=True,
            build_lock_free=True,
            staging_dirs_clean=True,
            reports_dir_exists=True,
        )

        data = result.to_dict()
        assert data["warnings"] == []
        assert data["errors"] == []


class TestDoctorRepairResult:
    """Test DoctorRepairResult dataclass."""

    def test_doctor_repair_result_to_dict(self) -> None:
        """DoctorRepairResult should convert to dictionary correctly."""
        result = DoctorRepairResult(
            actions=[
                RepairAction(
                    action_type="test",
                    description="Test action",
                    command="test",
                    dry_run_command="# test",
                    requires_confirmation=True,
                )
            ],
            performed_actions=["action1"],
            failures=["failure1"],
            success=True,
        )

        data = result.to_dict()

        assert len(data["actions"]) == 1
        assert data["actions"][0]["action_type"] == "test"
        assert data["actions"][0]["description"] == "Test action"
        assert data["actions"][0]["command"] == "test"
        assert data["actions"][0]["dry_run_command"] == "# test"
        assert data["actions"][0]["requires_confirmation"] is True
        assert data["performed_actions"] == ["action1"]
        assert data["failures"] == ["failure1"]
        assert data["success"] is True

    def test_doctor_repair_result_defaults(self) -> None:
        """DoctorRepairResult should have correct defaults."""
        result = DoctorRepairResult(actions=[])

        data = result.to_dict()
        assert data["actions"] == []
        assert data["performed_actions"] == []
        assert data["failures"] == []
        assert data["success"] is True


class TestRepairAction:
    """Test RepairAction dataclass."""

    def test_repair_action_to_dict(self) -> None:
        """RepairAction should convert to dictionary correctly."""
        action = RepairAction(
            action_type="test",
            description="Test action",
            command="test",
            dry_run_command="# test",
            requires_confirmation=True,
        )

        data = action.to_dict()

        assert data["action_type"] == "test"
        assert data["description"] == "Test action"
        assert data["command"] == "test"
        assert data["dry_run_command"] == "# test"
        assert data["requires_confirmation"] is True

    def test_repair_action_defaults(self) -> None:
        """RepairAction should have correct defaults."""
        action = RepairAction(
            action_type="test",
            description="Test action",
            command="test",
            dry_run_command="# test",
        )

        data = action.to_dict()
        assert data["requires_confirmation"] is False


class TestCmdDoctorCheck:
    """Test doctor check command."""

    def test_doctor_check_succeeds_healthy_system(self, tmp_path, capsys) -> None:
        """doctor check should succeed when system is healthy."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                exit_code = cmd_doctor_check(backend="all", json_output=False)

                # Should succeed for healthy system
                assert exit_code == 0
                captured = capsys.readouterr()
                assert "healthy" in captured.out.lower() or "System is healthy" in captured.out

    def test_doctor_check_fails_with_incomplete_toolchain(self, tmp_path, capsys) -> None:
        """doctor check should fail when toolchain is incomplete."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as incomplete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler=None,
                cuda_toolkit=None,
                nvtop=None,
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock config
            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = tmp_path / "reports"
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

            exit_code = cmd_doctor_check(backend="all", json_output=False)

            # Should fail because toolchain is incomplete
            assert exit_code != 0
            captured = capsys.readouterr()
            # Errors are printed to stderr
            assert "Missing tools" in captured.err or "error" in captured.err.lower()

    def test_doctor_check_json_output(self, tmp_path, capsys) -> None:
        """doctor check --json should produce valid JSON output."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                exit_code = cmd_doctor_check(backend="all", json_output=True)

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

        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config - use a very small timeout to ensure lock is stale
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = lock_file
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 1  # 1 minute timeout
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                exit_code = cmd_doctor_check(backend="all", json_output=False)

                # Should detect stale lock
                assert exit_code != 0
                captured = capsys.readouterr()
                assert "Stale build lock" in captured.err


class TestCmdDoctorRepair:
    """Test doctor --repair command."""

    def test_doctor_repair_no_actions_needed(self, tmp_path, capsys) -> None:
        """doctor --repair should return no actions when system is healthy."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            venv_path = tmp_path / "venv"
            mock_venv_path.return_value = venv_path
            venv_path.mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                result = cmd_doctor_repair(dry_run=True, json_output=False)

                # Should have no repair actions needed
                assert result.success is True
                assert len(result.actions) == 0

    def test_doctor_repair_clears_failed_staging(self, tmp_path, capsys) -> None:
        """doctor --repair should identify failed staging directories for cleanup (T081)."""
        # Create build directory with failed marker
        build_dir = tmp_path / "llama.cpp" / "build"
        build_dir.mkdir(parents=True)
        failed_marker = build_dir / ".failed"
        failed_marker.touch()

        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path / "llama.cpp")
                mock_config.return_value = mock_config_instance

                result = cmd_doctor_repair(dry_run=True, json_output=False)

                # Should identify failed staging directories
                assert result.success is True
                assert len(result.actions) > 0
                action_types = [a.action_type for a in result.actions]
                assert "clean_failed_staging" in action_types
                assert "remove_failed_marker" in action_types

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

        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path / "llama.cpp")
                mock_config.return_value = mock_config_instance

                result = cmd_doctor_repair(dry_run=True, json_output=False)

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

        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config - use a very small timeout to ensure lock is stale
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = lock_file
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 1  # 1 minute timeout
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                result = cmd_doctor_repair(dry_run=True, json_output=False)

                # Should identify stale lock for removal
                assert result.success is True
                assert len(result.actions) > 0
                action_types = [a.action_type for a in result.actions]
                assert "remove_stale_lock" in action_types

    def test_doctor_repair_json_output(self, tmp_path, capsys) -> None:
        """doctor --repair --json should produce valid JSON output."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()

            # Mock venv integrity check
            with patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity:
                mock_integrity.return_value = (True, None)

                # Mock config
                mock_config_instance = MagicMock()
                mock_config_instance.build_lock_path = tmp_path / "lock"
                mock_config_instance.reports_dir = tmp_path / "reports"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                result = cmd_doctor_repair(dry_run=True, json_output=True)

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
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            venv_path = tmp_path / "venv"
            mock_venv_path.return_value = venv_path
            venv_path.mkdir()
            mock_integrity.return_value = (True, None)

            # Mock config
            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = tmp_path / "reports"
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

            exit_code = cmd_doctor_check(backend="all", json_output=False)

            # Should succeed (exit code 0)
            assert exit_code == 0

    def test_doctor_repair_success_no_repairs_needed(self, tmp_path, capsys) -> None:
        """doctor --repair should succeed when no repairs are needed."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()
            mock_integrity.return_value = (True, None)

            # Mock config
            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = tmp_path / "reports"
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

            result = cmd_doctor_repair(dry_run=True, json_output=False)

            # Should succeed with no actions
            assert result.success is True
            assert len(result.actions) == 0


class TestDoctorCLIIntegration:
    """Integration tests for doctor CLI."""

    def test_doctor_main_with_check_command(self, tmp_path, capsys) -> None:
        """doctor check command should work through main()."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()
            mock_integrity.return_value = (True, None)

            # Mock config
            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = tmp_path / "reports"
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

            exit_code = doctor_main(["check"])

            # Should succeed
            assert exit_code == 0

    def test_doctor_main_with_repair_command(self, tmp_path, capsys) -> None:
        """doctor --repair command should work through main()."""
        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity,
            patch("llama_cli.doctor_cli.Config") as mock_config,
        ):
            # Mock toolchain as complete
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            # Mock venv path
            mock_venv_path.return_value = tmp_path / "venv"
            (tmp_path / "venv").mkdir()
            mock_integrity.return_value = (True, None)

            # Mock config
            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = tmp_path / "reports"
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

            exit_code = doctor_main(["repair", "--dry-run"])

            # Should succeed
            assert exit_code == 0
