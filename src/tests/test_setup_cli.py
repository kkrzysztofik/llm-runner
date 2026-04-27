"""T064-T068: Tests for setup CLI commands.

Test Tasks:
- T064: Test setup --check command (FR-005.1)
- T065: Test setup venv command (FR-005.2)
- T066: Test setup clean-venv command (FR-005.3)
- T067: Test setup --json output
- T068: Test setup error handling
"""

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.commands.setup import (
    cmd_check,
    cmd_clean_venv,
    cmd_venv,
)
from llama_cli.commands.setup import (
    main as setup_main,
)
from llama_manager.config import ErrorCode
from llama_manager.setup_venv import VenvResult
from llama_manager.toolchain import ToolchainStatus


@pytest.fixture(autouse=True)
def disable_colors() -> Iterator[None]:
    """Disable ANSI colors for all tests to keep assertions simple."""
    from llama_cli.colors import Colors

    original = Colors.enabled
    Colors.enabled = False
    yield
    Colors.enabled = original


class TestSetupCheck:
    """T064: Tests for setup --check command."""

    def test_setup_check_succeeds_with_complete_toolchain(self, capsys) -> None:
        """setup --check should succeed when toolchain is complete."""
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

            exit_code = cmd_check(MagicMock(backend="all", json=False))

            assert exit_code == 0
            captured = capsys.readouterr()
            assert "Toolchain Status" in captured.out or "complete" in captured.out.lower()

    def test_setup_check_fails_with_incomplete_toolchain(self, capsys) -> None:
        """setup --check should fail when toolchain is incomplete."""
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

            exit_code = cmd_check(MagicMock(backend="all", json=False))

            # Should fail because toolchain is incomplete
            assert exit_code != 0

    def test_setup_check_json_output(self, capsys) -> None:
        """setup --check --json should produce JSON output."""
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

            exit_code = cmd_check(MagicMock(backend="all", json=True))

            assert exit_code == 0
            captured = capsys.readouterr()

        # Should be valid JSON
        try:
            parsed = json.loads(captured.out)
            assert isinstance(parsed, dict)
            # Contract fields only (no is_complete, hints, etc.)
            assert "gcc" in parsed
            assert "make" in parsed
            assert "git" in parsed
            assert "cmake" in parsed
            assert "sycl_compiler" in parsed
            assert "cuda_toolkit" in parsed
            assert "nvtop" in parsed
            # Should NOT have extra fields
            assert "is_complete" not in parsed
            assert "hints" not in parsed
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_setup_check_sycl_backend(self, capsys) -> None:
        """setup --check sycl should check SYCL backend only."""
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

            exit_code = cmd_check(MagicMock(backend="sycl", json=False))

            # Should succeed for SYCL (all tools present)
            assert exit_code == 0

    def test_setup_check_cuda_backend(self, capsys) -> None:
        """setup --check cuda should check CUDA backend only."""
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

            exit_code = cmd_check(MagicMock(backend="cuda", json=False))

            # Should succeed for CUDA (all tools present)
            assert exit_code == 0

    def test_setup_check_deduplicates_sycl_hints(self, capsys) -> None:
        """setup --check should show each unique hint only once for SYCL.

        SYCL has three compiler aliases (dpcpp, icx, icpx) that all map to
        the same installation hint. The output should not repeat it.
        """
        from llama_manager.toolchain import ToolchainErrorDetail

        with (
            patch("llama_cli.commands.setup.detect_toolchain") as mock_detect,
            patch("llama_cli.commands.setup.get_toolchain_hints") as mock_hints,
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
            # Return three hints with identical how_to_fix/docs_ref
            shared_hint = ToolchainErrorDetail(
                error_code=ErrorCode.TOOLCHAIN_MISSING,
                failed_check="dpcpp",
                why_blocked="Required for sycl backend",
                how_to_fix="Install Intel oneAPI",
                docs_ref="https://intel.com/oneapi",
            )
            mock_hints.return_value = [
                shared_hint,
                shared_hint,
                shared_hint,
            ]

            exit_code = cmd_check(MagicMock(backend="sycl", json=False))

            assert exit_code == 1
            captured = capsys.readouterr()
            # Count occurrences of the hint text — should appear once
            hint_count = captured.out.count("Install Intel oneAPI")
            assert hint_count == 1, f"Expected 1 Intel install hint, found {hint_count}"


class TestSetupVenv:
    """T065: Tests for setup venv command."""

    def test_setup_venv_creates_venv(self, tmp_path: Path, capsys) -> None:
        """setup venv should create venv at expected path."""
        _ = tmp_path / "test-venv"

        with patch("llama_cli.commands.setup.create_venv"):
            exit_code = cmd_venv(MagicMock(check_integrity=False, json=False))

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "venv" in captured.out.lower() or "created" in captured.out.lower()

    def test_setup_venv_reuses_existing(self, tmp_path: Path, capsys) -> None:
        """setup venv should reuse existing venv."""
        venv_path = tmp_path / "existing-venv"
        venv_path.mkdir()

        # Create minimal valid venv structure so check_venv_integrity passes
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)

        with patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path):
            exit_code = cmd_venv(MagicMock(check_integrity=False, json=False))

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "reused" in captured.out.lower() or "exists" in captured.out.lower()

    def test_setup_venv_json_output(self, tmp_path: Path, capsys) -> None:
        """setup venv --json should produce JSON output."""
        _ = tmp_path / "test-venv"

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=tmp_path / "test-venv"),
            patch("llama_cli.commands.setup.create_venv") as mock_create,
        ):
            mock_create.return_value = VenvResult(
                venv_path=tmp_path / "test-venv",
                created=True,
                reused=False,
                activation_command="source test-venv/bin/activate",
            )
            exit_code = cmd_venv(MagicMock(check_integrity=False, json=True))

        assert exit_code == 0
        captured = capsys.readouterr()
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
        venv_path.mkdir()
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        (venv_path / "bin").mkdir()
        (venv_path / "bin" / "python").symlink_to(sys.executable)

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path),
            patch("llama_cli.commands.setup.create_venv") as mock_create,
        ):
            mock_create.return_value = VenvResult(
                venv_path=venv_path,
                created=True,
                reused=False,
                activation_command="source test-venv/bin/activate",
            )
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
        (bin_dir / "python").symlink_to(sys.executable)

        # Verify venv exists
        assert venv_path.exists()

        with patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path):
            exit_code = cmd_clean_venv(MagicMock(yes=True, json=False))

        assert exit_code == 0
        assert not venv_path.exists()

    def test_clean_venv_handles_nonexistent_venv(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv should handle nonexistent venv gracefully."""
        # Mock get_venv_path to return a non-existent path
        nonexistent_path = tmp_path / "nonexistent-venv"

        with patch("llama_cli.commands.setup.get_venv_path", return_value=nonexistent_path):
            # Should not raise error even if venv doesn't exist
            exit_code = cmd_clean_venv(MagicMock(yes=False, json=False))

        assert exit_code == 0

    def test_clean_venv_prompts_without_yes(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv without --yes should prompt."""
        # Without --yes flag, should prompt for confirmation
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path):
            exit_code = cmd_clean_venv(MagicMock(yes=False, json=False))

            # Should prompt for confirmation (exit 1 with message)
            assert exit_code == 1

    def test_clean_venv_json_output(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv --json should produce JSON output."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path):
            exit_code = cmd_clean_venv(MagicMock(yes=True, json=True))

        assert exit_code == 0
        captured = capsys.readouterr()

        # Should print JSON output
        data = json.loads(captured.out)
        assert data["status"] == "removed"
        assert data["venv_path"] == str(venv_path)

    def test_clean_venv_human_output(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv should produce human-readable success message."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path):
            exit_code = cmd_clean_venv(MagicMock(yes=True, json=False))

        assert exit_code == 0
        captured = capsys.readouterr()

        # Should print success message
        assert "Removed virtual environment" in captured.out


class TestSetupJsonOutput:
    """T067: Tests for setup --json output."""

    def test_setup_check_json_structure(self, capsys) -> None:
        """setup --check --json should have correct structure."""
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

            exit_code = cmd_check(MagicMock(backend="all", json=True))

            assert exit_code == 0
            captured = capsys.readouterr()

            parsed = json.loads(captured.out)

            # Verify structure (contract fields only)
            assert isinstance(parsed, dict)
            assert "gcc" in parsed
            assert "make" in parsed
            assert "git" in parsed
            assert "cmake" in parsed
            assert "sycl_compiler" in parsed
            assert "cuda_toolkit" in parsed
            assert "nvtop" in parsed
            # Should NOT have extra fields
            assert "is_complete" not in parsed
            assert "hints" not in parsed

    def test_setup_venv_json_structure(self, tmp_path: Path, capsys) -> None:
        """setup venv --json should have correct structure."""
        _ = tmp_path / "test-venv"

        with patch("llama_cli.commands.setup.create_venv") as mock_create:
            mock_create.return_value = VenvResult(
                venv_path=tmp_path / "test-venv",
                created=True,
                reused=False,
                activation_command="source test-venv/bin/activate",
            )
            exit_code = cmd_venv(MagicMock(check_integrity=False, json=True))

        assert exit_code == 0
        captured = capsys.readouterr()
        try:
            parsed = json.loads(captured.out)
            assert "venv_path" in parsed
            assert "created" in parsed
            assert "reused" in parsed
            assert "activation_command" in parsed
            assert isinstance(parsed["created"], bool)
            assert isinstance(parsed["reused"], bool)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")


class TestSetupErrorHandling:
    """T068: Tests for setup error handling."""

    def test_setup_check_handles_toolchain_errors(self, capsys) -> None:
        """setup --check should handle toolchain detection errors gracefully."""
        with patch("llama_cli.commands.setup.detect_toolchain") as mock_detect:
            mock_detect.side_effect = Exception("Toolchain detection failed")

            exit_code = cmd_check(MagicMock(backend="all", json=False))

            # Should handle error gracefully
            assert exit_code != 0

    def test_setup_venv_handles_creation_errors(self, tmp_path: Path, capsys) -> None:
        """setup venv should handle venv creation errors gracefully."""
        _ = tmp_path / "test-venv"

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=tmp_path / "test-venv"),
            patch("llama_cli.commands.setup.create_venv") as mock_create,
        ):
            mock_create.side_effect = PermissionError("Permission denied")

            exit_code = cmd_venv(MagicMock(check_integrity=False, json=False))

            # Should handle error gracefully
            assert exit_code != 0

    def test_setup_clean_venv_handles_permission_errors(self, tmp_path: Path, capsys) -> None:
        """setup clean-venv should handle permission errors gracefully."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path),
            patch("shutil.rmtree") as mock_rmtree,
        ):
            mock_rmtree.side_effect = PermissionError("Permission denied")

            exit_code = cmd_clean_venv(MagicMock(yes=True, json=False))

            # Should handle error gracefully
            assert exit_code != 0

    def test_setup_invalid_backend(self, capsys) -> None:
        """setup --check should handle invalid backend."""
        with patch("llama_cli.commands.setup.detect_toolchain") as mock_detect:
            mock_detect.return_value = ToolchainStatus()

            exit_code = cmd_check(MagicMock(backend="invalid", json=False))

            # Should handle invalid backend
            assert exit_code != 0


class TestSetupMain:
    """Tests for the main setup CLI entry point."""

    def test_setup_main_dispatches_to_check(self, capsys) -> None:
        """setup_main should dispatch to cmd_check for --check command."""
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

            with patch("sys.argv", ["setup", "check"]):
                exit_code = setup_main()

            assert exit_code == 0

    def test_setup_main_dispatches_to_venv(self, tmp_path: Path, capsys) -> None:
        """setup_main should dispatch to cmd_venv for venv command."""
        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=tmp_path / "venv"),
            patch("llama_cli.commands.setup.create_venv"),
            patch("sys.argv", ["setup", "venv"]),
        ):
            exit_code = setup_main()

        assert exit_code == 0

    def test_setup_main_dispatches_to_clean_venv(self, tmp_path: Path, capsys) -> None:
        """setup_main should dispatch to cmd_clean_venv for clean-venv command."""
        venv_path = tmp_path / "test-venv"
        venv_path.mkdir()

        with (
            patch("llama_cli.commands.setup.get_venv_path", return_value=venv_path),
            patch("shutil.rmtree"),
            patch("sys.argv", ["setup", "clean-venv", "--yes"]),
        ):
            exit_code = setup_main()

        assert exit_code == 0

    def test_setup_main_invalid_command(self, capsys) -> None:
        """setup_main should handle invalid command."""
        with patch("sys.argv", ["setup", "invalid-command"]):
            # argparse raises SystemExit for invalid choices
            with pytest.raises(SystemExit) as exc_info:
                setup_main()

            # Should exit with code 2 (argparse standard for invalid arguments)
            assert exc_info.value.code == 2
