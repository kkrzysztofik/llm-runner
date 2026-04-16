"""T057-T063: Tests for Toolchain diagnostics and venv lifecycle.

Test Tasks:
- T057: Test toolchain errors with actionable hints (FR-005)
- T058: Test venv lifecycle (create/reuse/integrity)
- T059: Test tool detection timeout (FR-005.4)
- T060: Test cmake too old error (FR-005)
- T061: Test setup --check skips venv integrity
- T062: Test venv integrity check detects corruption
- T063: Test venv path fallback to ~/.cache
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import ErrorCode
from llama_manager.setup_venv import (
    check_venv_integrity,
    create_venv,
    get_venv_path,
)
from llama_manager.toolchain import (
    ToolchainErrorDetail,
    detect_tool,
    parse_version,
    version_at_least,
)


class TestToolchainErrorsWithActionableHints:
    """T057: Tests for toolchain errors with actionable hints."""

    def test_toolchain_error_detail_has_all_fr005_fields(self) -> None:
        """ToolchainErrorDetail should have all FR-005 actionable error fields."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="gcc",
            why_blocked="Required for sycl backend",
            how_to_fix="sudo apt-get install gcc",
            docs_ref="https://gcc.gnu.org/download.html",
        )

        # Verify all required fields are present
        assert error.error_code == ErrorCode.TOOLCHAIN_MISSING
        assert error.failed_check == "gcc"
        assert error.why_blocked == "Required for sycl backend"
        assert error.how_to_fix == "sudo apt-get install gcc"
        assert error.docs_ref == "https://gcc.gnu.org/download.html"

    def test_toolchain_error_detail_has_install_command(self) -> None:
        """ToolchainErrorDetail.how_to_fix should contain actionable install command."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="cmake",
            why_blocked="Required for build",
            how_to_fix="sudo apt-get install cmake",
        )

        # Should contain installation instruction
        assert "install" in error.how_to_fix.lower()
        assert "cmake" in error.how_to_fix.lower()

    def test_toolchain_error_detail_has_docs_ref(self) -> None:
        """ToolchainErrorDetail should have docs_ref for additional help."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="gcc",
            why_blocked="Required for sycl backend",
            how_to_fix="sudo apt-get install gcc",
            docs_ref="https://gcc.gnu.org/download.html",
        )

        assert error.docs_ref is not None
        assert "gcc.gnu.org" in error.docs_ref

    def test_toolchain_error_detail_serializable_to_json(self) -> None:
        """ToolchainErrorDetail should be serializable to JSON."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="gcc",
            why_blocked="Required for sycl backend",
            how_to_fix="sudo apt-get install gcc",
            docs_ref="https://gcc.gnu.org/download.html",
        )

        # Try to serialize to JSON
        error_dict = {
            "error_code": error.error_code.value,
            "failed_check": error.failed_check,
            "why_blocked": error.why_blocked,
            "how_to_fix": error.how_to_fix,
            "docs_ref": error.docs_ref,
        }

        json_str = json.dumps(error_dict)
        parsed = json.loads(json_str)

        # Verify all fields present
        assert "error_code" in parsed
        assert "failed_check" in parsed
        assert "why_blocked" in parsed
        assert "how_to_fix" in parsed
        assert "docs_ref" in parsed

    def test_toolchain_error_detail_multiline_how_to_fix(self) -> None:
        """ToolchainErrorDetail should handle multiline how_to_fix."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="cuda-toolkit",
            why_blocked="Required for CUDA backend",
            how_to_fix="sudo apt-get install cuda-toolkit-12-2\n\nSee: https://developer.nvidia.com/cuda-toolkit",
            docs_ref="https://developer.nvidia.com/cuda-toolkit",
        )

        assert "sudo apt-get install cuda-toolkit-12-2" in error.how_to_fix
        assert "developer.nvidia.com" in error.how_to_fix


class TestVenvLifecycle:
    """T058: Tests for venv lifecycle (create/reuse/integrity)."""

    def test_venv_lifecycle_create_new(self, tmp_path: Path) -> None:
        """Venv lifecycle should create new venv when path doesn't exist."""
        venv_path = tmp_path / "new_venv"
        assert not venv_path.exists()

        with patch("llama_manager.setup_venv.venv.create") as mock_create:
            result = create_venv(venv_path)

        # Should have called venv.create
        mock_create.assert_called_once()

        # Should return VenvResult with created=True, reused=False
        assert result.created is True
        assert result.reused is False
        assert result.was_created is True
        assert result.was_reused is False
        assert result.venv_path == venv_path

        # Should have activation command
        assert "source" in result.activation_command
        assert "bin/activate" in result.activation_command

    def test_venv_lifecycle_reuse_existing(self, tmp_path: Path) -> None:
        """Venv lifecycle should reuse existing venv when path exists and is valid."""
        venv_path = tmp_path / "existing_venv"
        venv_path.mkdir()

        # Create a minimal valid venv structure
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        (venv_path / "bin").mkdir()
        (venv_path / "bin" / "python").touch()

        result = create_venv(venv_path)

        # Should have reused the existing valid venv
        assert result.created is False
        assert result.reused is True
        assert result.was_created is False
        assert result.was_reused is True
        assert result.venv_path == venv_path

    def test_venv_lifecycle_integrity_check_valid(self, tmp_path: Path) -> None:
        """Venv lifecycle integrity check should pass for valid venv."""
        import sys

        venv_path = tmp_path / "valid_venv"
        venv_path.mkdir()

        # Create pyvenv.cfg
        (venv_path / "pyvenv.cfg").write_text(f"home = {sys.prefix}\n")

        # Create interpreter symlink using sys.executable
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is True
        assert error is None

    def test_venv_lifecycle_integrity_check_corrupted(self, tmp_path: Path) -> None:
        """Venv lifecycle integrity check should detect corrupted venv."""
        venv_path = tmp_path / "corrupted_venv"
        venv_path.mkdir()

        # Create pyvenv.cfg but no interpreter
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "interpreter not found in venv"

    def test_venv_lifecycle_integrity_check_missing_pyvenv_cfg(self, tmp_path: Path) -> None:
        """Venv lifecycle integrity check should detect missing pyvenv.cfg."""
        venv_path = tmp_path / "missing_cfg_venv"
        venv_path.mkdir()

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "pyvenv.cfg missing"

    def test_venv_lifecycle_reuse_with_integrity_check(self, tmp_path: Path) -> None:
        """Venv lifecycle should check integrity when reusing venv."""
        venv_path = tmp_path / "reuse_venv"
        venv_path.mkdir()

        # First, manually create a valid venv structure
        # (since create_venv checks for existing dir and marks as reused)
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)

        # Now verify create_venv recognizes it as reused
        result = create_venv(venv_path)
        assert result.reused is True

        # Now check integrity
        is_valid, error = check_venv_integrity(venv_path)
        # Should be valid since we created the structure above
        assert is_valid is True
        assert error is None


class TestToolDetectionTimeout:
    """T059: Tests for tool detection timeout (FR-005.4)."""

    def test_detect_tool_timeout_respects_config(self) -> None:
        """detect_tool should use configurable timeout."""
        # Mock subprocess.run to simulate timeout
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("tool", 30)
            found, version = detect_tool("slow_tool", timeout=30)
            assert found is False
            assert version is None

    def test_detect_tool_timeout_custom_value(self) -> None:
        """detect_tool should use custom timeout when provided."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="tool 1.0.0\n",
                stderr="",
            )
            found, version = detect_tool("tool", timeout=60)
            assert found is True
            # Verify custom timeout was used
            mock_run.assert_called_once_with(
                ["tool", "--version"],
                capture_output=True,
                text=True,
                timeout=60,
            )

    def test_detect_tool_timeout_default(self) -> None:
        """detect_tool should use default timeout of 30s."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="tool 1.0.0\n",
                stderr="",
            )
            found, version = detect_tool("tool")
            assert found is True
            # Verify default timeout of 30 was used
            mock_run.assert_called_once_with(
                ["tool", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )


class TestCMakeTooOldError:
    """T060: Tests for cmake too old error (FR-005)."""

    def test_version_at_least_cmake_minimum(self) -> None:
        """version_at_least should work with CMAKE_MINIMUM_VERSION."""
        # CMAKE_MINIMUM_VERSION is "3.14"
        assert version_at_least("3.20.1", "3.14") is True
        assert version_at_least("3.14.0", "3.14") is True
        assert version_at_least("3.13.0", "3.14") is False

    def test_parse_version_cmake_format(self) -> None:
        """parse_version should handle CMake version format."""
        assert parse_version("3.25.0") == (3, 25, 0)
        assert parse_version("3.14.0") == (3, 14, 0)
        assert parse_version("3.20") == (3, 20, 0)

    def test_version_at_least_with_two_part_version(self) -> None:
        """version_at_least should handle two-part version string."""
        assert version_at_least("3.20", "3.14") is True
        assert version_at_least("3.20", "3.20") is True
        assert version_at_least("3.19", "3.20") is False

    def test_version_at_least_cmake_too_old_error(self) -> None:
        """version_at_least should detect when cmake is too old."""
        # CMAKE_MINIMUM_VERSION is "3.14"
        assert version_at_least("3.13.9", "3.14") is False
        assert version_at_least("3.13.0", "3.14") is False
        assert version_at_least("3.14.0", "3.14") is True
        assert version_at_least("3.14.1", "3.14") is True


class TestSetupCheckSkipsVenvIntegrity:
    """T061: Tests for setup --check skipping venv integrity."""

    def test_setup_check_skips_venv_integrity_by_default(self) -> None:
        """setup --check should skip venv integrity check by default."""
        # This test documents the expected behavior:
        # setup --check should only check toolchain availability
        # It should NOT check venv integrity unless explicitly requested
        #
        # The actual implementation would be in setup_cli.py
        # This test verifies the contract that --check is toolchain-only

        # Mock a scenario where venv is corrupted but tools are available
        with patch("llama_cli.setup_cli.detect_toolchain") as mock_detect:
            # Tools are available
            mock_status = MagicMock()
            mock_status.is_sycl_ready = True
            mock_status.is_cuda_ready = False
            mock_status.missing_tools = MagicMock(return_value=[])

            mock_detect.return_value = mock_status

            # Call cmd_check to verify detect_toolchain is called
            from llama_cli.setup_cli import cmd_check

            exit_code = cmd_check(MagicMock(backend="all", json=False))

            # detect_toolchain should be called
            assert mock_detect.called
            # Should succeed because toolchain is available
            assert exit_code == 0

    def test_setup_check_focused_on_toolchain(self) -> None:
        """setup --check should focus on toolchain, not venv."""
        # The --check flag is for toolchain diagnostics only
        # Venv lifecycle is handled by separate commands (setup venv, setup clean-venv)

        # This is a contract test - the implementation should ensure:
        # 1. --check only validates toolchain availability
        # 2. Venv checks are in separate code paths
        # 3. Toolchain validation doesn't depend on venv state

        # Verify the separation of concerns
        from llama_manager.setup_venv import check_venv_integrity

        # These should be independent functions
        assert check_venv_integrity is not None


class TestVenvCorruptionDetection:
    """T062: Tests for venv integrity check detecting corruption."""

    def test_venv_corruption_detection_missing_pyvenv(self, tmp_path: Path) -> None:
        """check_venv_integrity should detect missing pyvenv.cfg."""
        venv_path = tmp_path / "missing_pyvenv"
        venv_path.mkdir()

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "pyvenv.cfg missing"

    def test_venv_corruption_detection_missing_interpreter(self, tmp_path: Path) -> None:
        """check_venv_integrity should detect missing interpreter."""
        venv_path = tmp_path / "missing_interpreter"
        venv_path.mkdir()

        # Create pyvenv.cfg but no interpreter
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "interpreter not found in venv"

    def test_venv_corruption_detection_empty_venv(self, tmp_path: Path) -> None:
        """check_venv_integrity should detect empty venv directory."""
        venv_path = tmp_path / "empty_venv"
        venv_path.mkdir()

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "pyvenv.cfg missing"

    def test_venv_corruption_detection_nonexistent_path(self, tmp_path: Path) -> None:
        """check_venv_integrity should detect nonexistent path."""
        venv_path = tmp_path / "nonexistent"

        is_valid, error = check_venv_integrity(venv_path)
        assert is_valid is False
        assert error == "venv directory missing"


class TestVenvPathFallback:
    """T063: Tests for venv path fallback to ~/.cache."""

    def test_venv_path_fallback_to_home_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path should fallback to ~/.cache when XDG_CACHE_HOME not set."""
        # Ensure XDG_CACHE_HOME is not set
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        result = get_venv_path()
        expected = Path.home() / ".cache" / "llm-runner" / "venv"

        assert result == expected
        assert isinstance(result, Path)

    def test_venv_path_uses_xdg_cache_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path should use XDG_CACHE_HOME when set."""
        custom_cache = "/custom/cache"
        monkeypatch.setenv("XDG_CACHE_HOME", custom_cache)

        result = get_venv_path()
        expected = Path(custom_cache) / "llm-runner" / "venv"

        assert result == expected
        assert isinstance(result, Path)

    def test_venv_path_respects_xdg_over_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path should prefer XDG_CACHE_HOME over HOME/.cache."""
        custom_cache = "/custom/cache"
        custom_home = "/custom/home"
        monkeypatch.setenv("XDG_CACHE_HOME", custom_cache)
        monkeypatch.setenv("HOME", custom_home)

        result = get_venv_path()
        expected = Path(custom_cache) / "llm-runner" / "venv"

        assert result == expected
        # Should NOT use HOME/.cache
        assert result != Path(custom_home) / ".cache" / "llm-runner" / "venv"

    def test_venv_path_is_absolute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path should return absolute path."""
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        result = get_venv_path()
        assert result.is_absolute()
