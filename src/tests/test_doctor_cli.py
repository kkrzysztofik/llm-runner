"""Tests for doctor CLI commands (T081-T087).

Test Tasks:
- T081: Test doctor --repair clears failed staging (FR-004.7)
- T082: Test doctor --repair preserves successful artifacts (FR-004.7)
- T083: Test doctor --repair handles stale locks (FR-004.7)
- T084: Test doctor success path (no repairs needed)
- T085-T087: Test profile staleness check and repair via doctor CLI
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from _pytest.capture import CaptureFixture

from llama_cli.doctor_cli import (
    DoctorCheckResult,
    DoctorRepairResult,
    RepairAction,
    _build_profile_guidance,
    cmd_doctor_check,
    cmd_doctor_repair,
)
from llama_cli.doctor_cli import (
    main as doctor_main,
)
from llama_manager.profile_cache import (
    ProfileRecord,
    StalenessResult,
)
from llama_manager.toolchain import ToolchainStatus


@pytest.fixture(autouse=True)
def disable_colors():
    """Disable ANSI colors for all tests to keep assertions simple."""
    from llama_cli.colors import Colors

    original = Colors.enabled
    Colors.enabled = False
    yield
    Colors.enabled = original


def doctor_mocks(
    tmp_path: Path,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, ExitStack]:
    """Set up common mocks for doctor CLI tests.

    Returns:
        Tuple of (mock_detect, mock_venv_path, mock_config_cls, mock_integrity,
        mock_config_instance, exit_stack). The caller MUST call exit_stack.close()
        when done to stop all patches.
    """
    stack = ExitStack()
    mock_detect = stack.enter_context(patch("llama_cli.doctor_cli.detect_toolchain"))
    mock_venv_path = stack.enter_context(patch("llama_cli.doctor_cli.get_venv_path"))
    mock_config_cls = stack.enter_context(patch("llama_cli.doctor_cli.Config"))

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

    mock_integrity = stack.enter_context(patch("llama_cli.doctor_cli.check_venv_integrity"))
    mock_integrity.return_value = (True, None)

    mock_config_instance = MagicMock()
    mock_config_instance.build_lock_path = tmp_path / "lock"
    mock_config_instance.reports_dir = tmp_path / "reports"
    mock_config_instance.toolchain_timeout_seconds = 3600
    mock_config_instance.llama_cpp_root = str(tmp_path)
    mock_config_instance.profiles_dir = tmp_path / "profiles"
    mock_config_cls.return_value = mock_config_instance

    return (
        mock_detect,
        mock_venv_path,
        mock_config_cls,
        mock_integrity,
        mock_config_instance,
        stack,
    )


def _make_namespace(**kwargs: object) -> argparse.Namespace:
    """Create an argparse.Namespace object with the given attributes.

    Args:
        **kwargs: Attributes to set on the namespace.

    Returns:
        argparse.Namespace with the specified attributes.
    """
    return argparse.Namespace(**kwargs)


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

                exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

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

            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

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
                mock_config_instance.toolchain_timeout_seconds = 1  # 1 second timeout
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

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
                mock_config_instance.profiles_dir = tmp_path / "profiles"
                mock_config_instance.builds_dir = tmp_path / "builds"
                mock_config_instance.toolchain_timeout_seconds = 3600
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                # Create directories so no repair actions are needed
                (tmp_path / "reports").mkdir()
                (tmp_path / "profiles").mkdir()
                (tmp_path / "builds").mkdir()

                result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

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

                result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

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
                mock_config_instance.toolchain_timeout_seconds = 1  # 1 second timeout
                mock_config_instance.llama_cpp_root = str(tmp_path)
                mock_config.return_value = mock_config_instance

                result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

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

            exit_code = cmd_doctor_check(_make_namespace(backend="all", json=False))

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
            mock_config_instance.profiles_dir = tmp_path / "profiles"
            mock_config_instance.builds_dir = tmp_path / "builds"
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

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

        from llama_manager.profile_cache import (
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

        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity,
            patch("llama_cli.doctor_cli.Config") as mock_config,
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
            mock_integrity.return_value = (True, None)

            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = reports_dir
            mock_config_instance.profiles_dir = profiles_dir
            mock_config_instance.builds_dir = builds_dir
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

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

        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity,
            patch("llama_cli.doctor_cli.Config") as mock_config,
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
            mock_integrity.return_value = (True, None)

            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = reports_dir
            mock_config_instance.profiles_dir = profiles_dir
            mock_config_instance.builds_dir = builds_dir
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            action_types = [a.action_type for a in result.actions]
            assert "create_directory" not in action_types

    def test_directory_repair_action_details(self, tmp_path: Path) -> None:
        """Directory repair actions should have correct command and permissions."""
        reports_dir = tmp_path / "reports"

        with (
            patch("llama_cli.doctor_cli.detect_toolchain") as mock_detect,
            patch("llama_cli.doctor_cli.get_venv_path") as mock_venv_path,
            patch("llama_cli.doctor_cli.check_venv_integrity") as mock_integrity,
            patch("llama_cli.doctor_cli.Config") as mock_config,
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
            mock_integrity.return_value = (True, None)

            mock_config_instance = MagicMock()
            mock_config_instance.build_lock_path = tmp_path / "lock"
            mock_config_instance.reports_dir = reports_dir
            mock_config_instance.profiles_dir = tmp_path / "profiles"
            mock_config_instance.builds_dir = tmp_path / "builds"
            mock_config_instance.toolchain_timeout_seconds = 3600
            mock_config_instance.llama_cpp_root = str(tmp_path)
            mock_config.return_value = mock_config_instance

            result = cmd_doctor_repair(_make_namespace(dry_run=True, json=False))

            dir_actions = [a for a in result.actions if a.action_type == "create_directory"]
            assert len(dir_actions) == 3

            # Check that one of them is for reports
            reports_action = next(
                (a for a in dir_actions if "reports" in a.description), None
            )
            assert reports_action is not None
            assert reports_action.command == ["mkdir", "-m", "700", "-p", str(reports_dir)]
            assert reports_action.requires_confirmation is False
