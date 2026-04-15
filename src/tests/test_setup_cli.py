"""T064-T068: Tests for setup CLI commands.

Test Tasks:
- T064: Test setup --check command (FR-005.1)
- T065: Test setup venv command (FR-005.2)
- T066: Test setup clean-venv command (FR-005.3)
- T067: Test setup --json output
- T068: Test setup error handling
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.setup_cli import (
    cmd_check,
    cmd_clean_venv,
    cmd_venv,
    main as setup_main,
)
from llama_manager.setup_venv import VenvResult, check_venv_integrity, create_venv, get_venv_path
from llama_manager.toolchain import ToolchainStatus, detect_toolchain


class TestSetupCheck:
    """T064: Tests for setup --check command."""

    def test_setup_check_succeeds_with_complete_toolchain(self, capsys) -> None:
        """setup --check should succeed when toolchain is complete."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            exit_code = cmd_check(MagicMock(backend="all", json=False))

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Toolchain Status" in captured.out or "complete" in captured.out.lower()

    def test_setup_check_fails_with_incomplete_toolchain(self, capsys) -> None:
        """setup --check should fail when toolchain is incomplete."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler=None,
                cuda_toolkit=None,
                nvtop=None,
            )

            exit_code = cmd_check(MagicMock(backend="all", json=False))

            # Should fail because toolchain is incomplete
            assert exit_code != 0

    def test_setup_check_json_output(self, capsys) -> None:
        """setup --check --json should produce JSON output."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit=None,
                nvtop=None,
            )

            exit_code = cmd_check(MagicMock(backend="all", json=True))

            assert exit_code == 0
            captured = capsys.readouterr()

            # Should be valid JSON
            try:
                parsed = json.loads(captured.out)
                assert "gcc" in parsed
                assert "make" in parsed
                assert "git" in parsed
                assert "cmake" in parsed
                assert "sycl_compiler" in parsed
                assert "cuda_toolkit" in parsed
                assert "nvtop" in parsed
                assert "is_complete" in parsed
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")

    def test_setup_check_sycl_backend(self, capsys) -> None:
        """setup --check sycl should check SYCL backend only."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit=None,
                nvtop=None,
            )

            exit_code = cmd_check(MagicMock(backend="sycl", json=False))

            # Should succeed for SYCL
            assert exit_code == 0

    def test_setup_check_cuda_backend(self, capsys) -> None:
        """setup --check cuda should check CUDA backend only."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler=None,
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            exit_code = cmd_check(MagicMock(backend="cuda", json=False))

            # Should succeed for CUDA
            assert exit_code == 0


class TestSetupVenv:
    """T065: Tests for setup venv command."""

    def test_setup_venv_creates_venv(self, tmp_path: Path, capsys) -> None:
        """setup venv should create venv at expected path."""
        venv_path = tmp_path / "test-venv"

        with patch("llama_manager.setup_venv.venv.create"):
            exit_code = cmd_venv(MagicMock(check_integrity=False, json=False))

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "venv" in captured.out.lower() or "created" in captured.out.lower()

    def test_setup_venv_reuses_existing(self, tmp_path: Path, capsys) -> None:
        """setup venv should reuse existing venv."""
        venv_path = tmp_path / "existing-venv"
        venv_path.mkdir()

        exit_code = cmd_venv(MagicMock(check_integrity=False, json=False))

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "reused" in captured.out.lower() or "exists" in captured.out.lower()

    def test_setup_venv_json_output(self, tmp_path: Path, capsys) -> None:
        """setup venv --json should produce JSON output."""
        venv_path = tmp_path / "test-venv"

        with patch("llama_manager.setup_venv.venv.create"):
            exit_code = cmd_venv(MagicMock(check_integrity=False, json=True))

        assert exit_code == 0
        captured = capsys.readouterr()

        # Should be valid JSON
        try:
            parsed = json.loads(captured.out)
            assert "venv_path" in parsed
            assert "created" in parsed
            assert "reused" in parsed
            assert "activation_command" in parsed
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_setup_venv_with_integrity_check(self, tmp_path: Path, capsys) -> None:
        """setup venv --check-integrity should validate venv after creation."""
        venv_path = tmp_path / "test-venv"

        with patch("llama_manager.setup_venv.venv.create"):
            exit_code = cmd_venv(MagicMock(check_integrity=True, json=False))

        assert exit_code == 0


class TestSetupCleanVenv:
    """T066: Tests for setup clean-venv command."""

    def test_clean_venv_removes_venv(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv should remove existing venv."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        # Create some venv files
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to("/usr/bin/python3")

        # Verify venv exists
        assert venv_path.exists()

        exit_code = cmd_clean_venv(MagicMock(yes=True))

        assert exit_code == 0
        assert not venv_path.exists()

    def test_clean_venv_handles_nonexistent_venv(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv should handle nonexistent venv gracefully."""
        venv_path = tmp_path / "nonexistent-venv"

        # Should not raise error even if venv doesn't exist
        exit_code = cmd_clean_venv(MagicMock(yes=True))

        assert exit_code == 0

    def test_clean_venv_prompts_without_yes(self, capsys) -> None:
        """setup clean-venv without --yes should prompt."""
        # Without --yes flag, should prompt for confirmation
        # This would require interactive input, so we just verify the flag is checked
        with patch("builtins.input", return_value="y"):
            exit_code = cmd_clean_venv(MagicMock(yes=False))

            # Should succeed with user confirmation
            assert exit_code == 0

    def test_clean_venv_json_output(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv --json should produce JSON output."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        exit_code = cmd_clean_venv(MagicMock(yes=True, json=True))

        assert exit_code == 0
        captured = capsys.readouterr()

        # Should be valid JSON
        try:
            parsed = json.loads(captured.out)
            assert "removed" in parsed or "success" in parsed
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")


class TestSetupJsonOutput:
    """T067: Tests for setup --json output."""

    def test_setup_check_json_structure(self, capsys) -> None:
        """setup --check --json should have correct structure."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            exit_code = cmd_check(MagicMock(backend="all", json=True))

            assert exit_code == 0
            captured = capsys.readouterr()

            parsed = json.loads(captured.out)

            # Verify structure
            assert isinstance(parsed, dict)
            assert "gcc" in parsed
            assert "make" in parsed
            assert "git" in parsed
            assert "cmake" in parsed
            assert "sycl_compiler" in parsed
            assert "cuda_toolkit" in parsed
            assert "nvtop" in parsed
            assert "is_complete" in parsed
            assert isinstance(parsed["is_complete"], bool)

    def test_setup_venv_json_structure(self, tmp_path: Path, capsys) -> None:
        """setup venv --json should have correct structure."""
        venv_path = tmp_path / "test-venv"

        with patch("llama_manager.setup_venv.venv.create"):
            exit_code = cmd_venv(MagicMock(check_integrity=False, json=True))

        assert exit_code == 0
        captured = capsys.readouterr()

        parsed = json.loads(captured.out)

        # Verify structure
        assert isinstance(parsed, dict)
        assert "venv_path" in parsed
        assert "created" in parsed
        assert "reused" in parsed
        assert "activation_command" in parsed
        assert isinstance(parsed["created"], bool)
        assert isinstance(parsed["reused"], bool)


class TestSetupErrorHandling:
    """T068: Tests for setup error handling."""

    def test_setup_check_handles_toolchain_errors(self, capsys) -> None:
        """setup --check should handle toolchain detection errors gracefully."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.side_effect = Exception("Toolchain detection failed")

            exit_code = cmd_check(MagicMock(backend="all", json=False))

            # Should handle error gracefully
            assert exit_code != 0

    def test_setup_venv_handles_creation_errors(self, tmp_path: Path, capsys) -> None:
        """setup venv should handle venv creation errors gracefully."""
        venv_path = tmp_path / "test-venv"

        with patch("llama_manager.setup_venv.venv.create") as mock_create:
            mock_create.side_effect = PermissionError("Permission denied")

            exit_code = cmd_venv(MagicMock(check_integrity=False, json=False))

            # Should handle error gracefully
            assert exit_code != 0

    def test_setup_clean_venv_handles_permission_errors(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv should handle permission errors gracefully."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with patch("shutil.rmtree") as mock_rmtree:
            mock_rmtree.side_effect = PermissionError("Permission denied")

            exit_code = cmd_clean_venv(MagicMock(yes=True, json=False))

            # Should handle error gracefully
            assert exit_code != 0

    def test_setup_invalid_backend(self, capsys) -> None:
        """setup --check should handle invalid backend."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus()

            exit_code = cmd_check(MagicMock(backend="invalid", json=False))

            # Should handle invalid backend
            assert exit_code != 0


class TestSetupMain:
    """Tests for the main setup CLI entry point."""

    def test_setup_main_dispatches_to_check(self, capsys) -> None:
        """setup_main should dispatch to cmd_check for --check command."""
        with patch("llama_manager.toolchain.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus(
                gcc="11.4.0",
                make="4.3",
                git="2.34.1",
                cmake="3.25.0",
                sycl_compiler="2023.1.0",
                cuda_toolkit="12.2.0",
                nvtop="3.1.0",
            )

            with patch("sys.argv", ["setup", "check"]):
                exit_code = setup_main()

            assert exit_code == 0

    def test_setup_main_dispatches_to_venv(self, tmp_path: Path, capsys) -> None:
        """setup_main should dispatch to cmd_venv for venv command."""
        with patch("llama_manager.setup_venv.venv.create"):
            with patch("sys.argv", ["setup", "venv"]):
                exit_code = setup_main()

            assert exit_code == 0

    def test_setup_main_dispatches_to_clean_venv(self, tmp_path: Path, capsys) -> None:
        """setup_main should dispatch to cmd_clean_venv for clean-venv command."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with patch("shutil.rmtree"):
            with patch("sys.argv", ["setup", "clean-venv", "--yes"]):
                exit_code = setup_main()

            assert exit_code == 0

    def test_setup_main_invalid_command(self, capsys) -> None:
        """setup_main should handle invalid command."""
        with patch("sys.argv", ["setup", "invalid-command"]):
            exit_code = setup_main()

            # Should handle invalid command gracefully
            assert exit_code != 0
