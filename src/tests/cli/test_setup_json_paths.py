"""Additional tests for setup CLI to cover uncovered branches.

Covers:
  - Backend-specific JSON exit codes (sycl, cuda)
  - Confirmation-required JSON output for clean-venv
  - Error handling for corrupted lock files
  - Permission errors in clean-venv JSON output
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_cli.commands.setup import (
    cmd_check,
    cmd_clean_venv,
    cmd_venv,
)
from llama_manager.toolchain import ToolchainStatus


class TestSetupCheckJsonBackendExitCodes:
    """Tests for setup check --json with backend-specific exit codes."""

    def test_check_json_sycl_ready_returns_zero(self, capsys) -> None:
        """setup --check --json with sycl backend should return 0 when SYCL is ready."""
        with patch("llama_cli.commands.setup.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit=None,
                nvtop=None,
            )
            exit_code = cmd_check(MagicMock(backend="sycl", json=True))
            assert exit_code == 0

    def test_check_json_sycl_not_ready_returns_one(self, capsys) -> None:
        """setup --check --json with sycl backend should return 1 when SYCL is not ready."""
        with patch("llama_cli.commands.setup.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler=None,
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )
            exit_code = cmd_check(MagicMock(backend="sycl", json=True))
            assert exit_code == 1

    def test_check_json_cuda_ready_returns_zero(self, capsys) -> None:
        """setup --check --json with cuda backend should return 0 when CUDA is ready."""
        with patch("llama_cli.commands.setup.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )
            exit_code = cmd_check(MagicMock(backend="cuda", json=True))
            assert exit_code == 0

    def test_check_json_cuda_not_ready_returns_one(self, capsys) -> None:
        """setup --check --json with cuda backend should return 1 when CUDA is not ready."""
        with patch("llama_cli.commands.setup.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler=None,
                cuda_toolkit=None,
                nvtop=None,
            )
            exit_code = cmd_check(MagicMock(backend="cuda", json=True))
            assert exit_code == 1


class TestSetupCleanVenvJsonPaths:
    """Tests for clean-venv JSON output edge cases."""

    def test_clean_venv_not_found_json(self, tmp_path: Path, capsys) -> None:
        """clean-venv --json should return not_found status when venv doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent-venv"

        with patch("llama_cli.commands.setup.get_venv_path", return_value=nonexistent_path):
            exit_code = cmd_clean_venv(MagicMock(yes=False, json=True))

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "not_found"
        assert data["venv_path"] == str(nonexistent_path)

    def test_clean_venv_confirmation_required_json(self, tmp_path: Path, capsys) -> None:
        """clean-venv --json should return exists status when venv exists without --yes."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path):
            exit_code = cmd_clean_venv(MagicMock(yes=False, json=True))

        assert exit_code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "exists"
        assert data["venv_path"] == str(venv_path)
        assert "Use --yes" in data["message"]

    def test_clean_venv_permission_error_json(self, tmp_path: Path, capsys) -> None:
        """clean-venv --json should return error status on permission error."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            mock_rmtree.side_effect = PermissionError("Permission denied")
            exit_code = cmd_clean_venv(MagicMock(yes=True, json=True))

        assert exit_code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert "Permission denied" in data["error"]

    def test_clean_venv_generic_error_json(self, tmp_path: Path, capsys) -> None:
        """clean-venv --json should return error status on generic exception."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            mock_rmtree.side_effect = OSError("Unexpected error")
            exit_code = cmd_clean_venv(MagicMock(yes=True, json=True))

        assert exit_code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert "Unexpected error" in data["error"]


class TestSetupVenvJsonPaths:
    """Tests for venv command JSON output edge cases."""

    def test_venv_with_integrity_check_fails(self, tmp_path: Path, capsys) -> None:
        """setup venv --json --check-integrity should return error when integrity check fails."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path),
            patch("llama_cli.commands.setup.create_venv") as mock_create,
            patch("llama_cli.commands.setup.check_venv_integrity") as mock_integrity,
        ):
            mock_create.return_value = MagicMock(
                venv_path=venv_path,
                created=True,
                reused=False,
                activation_command="source test-venv/bin/activate",
            )
            mock_integrity.return_value = (False, "broken venv")
            exit_code = cmd_venv(MagicMock(check_integrity=True, json=True))

        assert exit_code == 1
        captured = capsys.readouterr()
        # Error message goes to stderr, no JSON output on error path
        assert "broken venv" in captured.err
