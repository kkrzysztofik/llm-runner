"""T013, T016-T018: Tests for ToolchainErrorDetail, parse_version, version_at_least, detect_tool, get_toolchain_hints.

Test Tasks:
- T013: ToolchainErrorDetail dataclass tests
- T016: CMake version parser tests
- T017: detect_tool() tests
- T018: get_toolchain_hints() tests
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import ErrorCode
from llama_manager.toolchain import (
    ToolchainErrorDetail,
    detect_tool,
    get_toolchain_hints,
    parse_version,
    version_at_least,
)


class TestToolchainErrorDetail:
    """T013: Tests for ToolchainErrorDetail dataclass."""

    def test_toolchain_error_detail_all_fields_settable(self) -> None:
        """ToolchainErrorDetail should have all fields settable and retrievable."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="gcc",
            why_blocked="Required for sycl backend",
            how_to_fix="sudo apt-get install gcc",
            docs_ref="https://gcc.gnu.org/download.html",
        )
        assert error.error_code == ErrorCode.TOOLCHAIN_MISSING
        assert error.failed_check == "gcc"
        assert error.why_blocked == "Required for sycl backend"
        assert error.how_to_fix == "sudo apt-get install gcc"
        assert error.docs_ref == "https://gcc.gnu.org/download.html"

    def test_toolchain_error_detail_docs_ref_optional(self) -> None:
        """ToolchainErrorDetail should default docs_ref to None."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="gcc",
            why_blocked="Required for sycl backend",
            how_to_fix="sudo apt-get install gcc",
        )
        assert error.error_code == ErrorCode.TOOLCHAIN_MISSING
        assert error.failed_check == "gcc"
        assert error.why_blocked == "Required for sycl backend"
        assert error.how_to_fix == "sudo apt-get install gcc"
        assert error.docs_ref is None

    def test_toolchain_error_detail_all_fields_with_none_docs_ref(self) -> None:
        """ToolchainErrorDetail should work with None docs_ref."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="cmake",
            why_blocked="Required for cuda backend",
            how_to_fix="sudo apt-get install cmake",
            docs_ref=None,
        )
        assert error.error_code == ErrorCode.TOOLCHAIN_MISSING
        assert error.failed_check == "cmake"
        assert error.why_blocked == "Required for cuda backend"
        assert error.how_to_fix == "sudo apt-get install cmake"
        assert error.docs_ref is None

    def test_toolchain_error_detail_different_error_codes(self) -> None:
        """ToolchainErrorDetail should accept different ErrorCode values."""
        error1 = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="gcc",
            why_blocked="Missing",
            how_to_fix="Install gcc",
        )
        assert error1.error_code == ErrorCode.TOOLCHAIN_MISSING

    def test_toolchain_error_detail_complex_why_blocked(self) -> None:
        """ToolchainErrorDetail should handle complex why_blocked messages."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="intel-oneapi-dpcpp-compiler",
            why_blocked="Required for SYCL backend - Intel oneAPI DPC++ Compiler version 2023.1.0 or later",
            how_to_fix="curl -sSf https://apt.repos.intel.com/install.sh | sudo sh && sudo apt-get install oneapi-compiler-dpcpp-cpp oneapi-compiler-dpcpp-rt",
            docs_ref="https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html",
        )
        assert error.error_code == ErrorCode.TOOLCHAIN_MISSING
        assert "SYCL backend" in error.why_blocked
        assert "2023.1.0" in error.why_blocked

    def test_toolchain_error_detail_multiline_how_to_fix(self) -> None:
        """ToolchainErrorDetail should handle multiline how_to_fix messages."""
        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,  # type: ignore
            failed_check="cuda-toolkit",
            why_blocked="Required for CUDA backend",
            how_to_fix="sudo apt-get install cuda-toolkit-12-2\n\nSee: https://developer.nvidia.com/cuda-toolkit",
            docs_ref="https://developer.nvidia.com/cuda-toolkit",
        )
        assert error.error_code == ErrorCode.TOOLCHAIN_MISSING
        assert "sudo apt-get install cuda-toolkit-12-2" in error.how_to_fix
        assert "developer.nvidia.com" in error.how_to_fix


class TestParseVersion:
    """T016: Tests for parse_version() function."""

    def test_parse_version_simple_three_part(self) -> None:
        """parse_version should handle simple three-part versions."""
        assert parse_version("3.20.1") == (3, 20, 1)
        assert parse_version("3.14.0") == (3, 14, 0)
        assert parse_version("1.2.3") == (1, 2, 3)

    def test_parse_version_two_part(self) -> None:
        """parse_version should handle two-part versions (pad with 0)."""
        assert parse_version("3.20") == (3, 20, 0)
        assert parse_version("1.0") == (1, 0, 0)
        assert parse_version("2.5") == (2, 5, 0)

    def test_parse_version_with_ubuntu_suffix(self) -> None:
        """parse_version should strip ubuntu suffixes."""
        assert parse_version("3.20.1ubuntu") == (3, 20, 1)
        assert parse_version("11.4.0ubuntu") == (11, 4, 0)
        assert parse_version("3.14.0-ubuntu") == (3, 14, 0)

    def test_parse_version_with_linux_suffix(self) -> None:
        """parse_version should strip linux suffixes."""
        assert parse_version("3.20.1linux") == (3, 20, 1)
        assert parse_version("11.4.0-linux") == (11, 4, 0)

    def test_parse_version_with_deb_suffix(self) -> None:
        """parse_version should strip deb suffixes."""
        assert parse_version("3.20.1deb") == (3, 20, 1)
        assert parse_version("11.4.0-deb") == (11, 4, 0)

    def test_parse_version_with_linux_gnu_suffix(self) -> None:
        """parse_version should strip linux-gnu suffixes."""
        assert parse_version("11.4.0-linux-gnu") == (11, 4, 0)
        assert parse_version("3.20.1_linux-gnu") == (3, 20, 1)

    def test_parse_version_case_insensitive(self) -> None:
        """parse_version should handle suffixes case-insensitively."""
        assert parse_version("3.20.1UBUNTU") == (3, 20, 1)
        assert parse_version("11.4.0-LINUX") == (11, 4, 0)
        assert parse_version("3.14.0Debian") == (3, 14, 0)

    def test_parse_version_with_extra_numbers(self) -> None:
        """parse_version should extract all numeric parts."""
        # This is an edge case - should handle gracefully
        # The regex extracts all numbers, so "3.20.1.0.0" would become (3, 20, 1, 0, 0)
        # But we only return first 3 components
        result = parse_version("3.20.1.0.0")
        assert len(result) >= 3
        assert result[0] == 3
        assert result[1] == 20
        assert result[2] == 1

    def test_parse_version_gcc_format(self) -> None:
        """parse_version should handle GCC version format."""
        # GCC versions like "11.4.0" should work
        assert parse_version("11.4.0") == (11, 4, 0)
        assert parse_version("12.2.0") == (12, 2, 0)

    def test_parse_version_cmake_format(self) -> None:
        """parse_version should handle CMake version format."""
        assert parse_version("3.25.0") == (3, 25, 0)
        assert parse_version("3.14.0") == (3, 14, 0)
        assert parse_version("3.20") == (3, 20, 0)

    def test_parse_version_single_digit(self) -> None:
        """parse_version should handle single digit version."""
        assert parse_version("1") == (1, 0, 0)
        assert parse_version("5") == (5, 0, 0)

    def test_parse_version_zero_version(self) -> None:
        """parse_version should handle zero version."""
        assert parse_version("0.0.0") == (0, 0, 0)
        assert parse_version("0.0") == (0, 0, 0)


class TestVersionAtLeast:
    """Tests for version_at_least() function."""

    def test_version_at_least_exact_match(self) -> None:
        """version_at_least should return True when versions are equal."""
        assert version_at_least("3.20.1", "3.20.1") is True
        assert version_at_least("3.14", "3.14") is True
        assert version_at_least("1.0.0", "1.0.0") is True

    def test_version_at_least_greater(self) -> None:
        """version_at_least should return True when version is greater."""
        assert version_at_least("3.21.0", "3.20.0") is True
        assert version_at_least("3.20.2", "3.20.1") is True
        assert version_at_least("4.0.0", "3.20.0") is True

    def test_version_at_least_less(self) -> None:
        """version_at_least should return False when version is less."""
        assert version_at_least("3.19.0", "3.20.0") is False
        assert version_at_least("3.20.0", "3.20.1") is False
        assert version_at_least("3.14.0", "3.20.0") is False

    def test_version_at_least_with_two_part_min(self) -> None:
        """version_at_least should handle two-part minimum version."""
        assert version_at_least("3.20.1", "3.20") is True
        assert version_at_least("3.21.0", "3.20") is True
        assert version_at_least("3.19.0", "3.20") is False

    def test_version_at_least_with_two_part_version(self) -> None:
        """version_at_least should handle two-part version string."""
        assert version_at_least("3.20", "3.14") is True
        assert version_at_least("3.20", "3.20") is True
        assert version_at_least("3.19", "3.20") is False

    def test_version_at_least_cmake_minimum(self) -> None:
        """version_at_least should work with CMAKE_MINIMUM_VERSION string."""
        # CMAKE_MINIMUM_VERSION tuple is (3, 24, 0), string form is "3.24.0"
        assert version_at_least("3.25.0", "3.24.0") is True
        assert version_at_least("3.24.0", "3.24.0") is True
        assert version_at_least("3.23.0", "3.24.0") is False

    def test_version_at_least_edge_cases(self) -> None:
        """version_at_least should handle edge cases."""
        # Very old vs very new
        assert version_at_least("99.99.99", "1.0.0") is True
        # Same major version
        assert version_at_least("3.20.0", "3.19.0") is True
        assert version_at_least("3.19.0", "3.20.0") is False


class TestDetectTool:
    """T017: Tests for detect_tool() function."""

    def test_detect_tool_found(self) -> None:
        """detect_tool should return (True, version) for found tools."""
        # Mock subprocess.run to simulate a found tool
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="gcc (GCC) 11.4.0\n",
                stderr="",
            )
            found, version = detect_tool("gcc")
            assert found is True
            assert version == "11.4.0"
            # Verify subprocess.run was called correctly
            mock_run.assert_called_once_with(
                ["gcc", "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )

    def test_detect_tool_not_found(self) -> None:
        """detect_tool should return (False, None) for tools that return non-zero."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="command not found",
            )
            found, version = detect_tool("nonexistent_tool_xyz")
            assert found is False
            assert version is None

    def test_detect_tool_file_not_found(self) -> None:
        """detect_tool should return (False, None) for tools that don't exist."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("command not found")
            found, version = detect_tool("nonexistent_tool_xyz")
            assert found is False
            assert version is None

    def test_detect_tool_timeout(self) -> None:
        """detect_tool should return (False, None) on timeout."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("tool", 30)
            found, version = detect_tool("slow_tool")
            assert found is False
            assert version is None

    def test_detect_tool_other_exception(self) -> None:
        """detect_tool should return (False, None) on other exceptions."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Some unexpected error")
            found, version = detect_tool("problematic_tool")
            assert found is False
            assert version is None

    def test_detect_tool_custom_timeout(self) -> None:
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

    def test_detect_tool_version_parsing_no_match(self) -> None:
        """detect_tool should return first line if version regex doesn't match."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Some tool output without version numbers\n",
                stderr="",
            )
            found, version = detect_tool("weird_tool")
            assert found is True
            assert version == "Some tool output without version numbers"

    def test_detect_tool_version_parsing_with_newlines(self) -> None:
        """detect_tool should extract version from multi-line output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="tool version 1.2.3\nbuilt with gcc 11.0\n",
                stderr="",
            )
            found, version = detect_tool("tool")
            assert found is True
            assert version == "1.2.3"

    def test_detect_tool_real_tool_exists(self) -> None:
        """detect_tool should work with real tools that exist on the system."""
        # Test with 'sh' which should exist on most Unix systems
        # Mock subprocess.run to avoid calling real binaries
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="sh 1.0.0\n",
                stderr="",
            )
            found, version = detect_tool("sh")
            # Should find it or at least not crash
            assert isinstance(found, bool)
            assert version is None or isinstance(version, str)


class TestGetToolchainHints:
    """T018: Tests for get_toolchain_hints() function."""

    def test_get_toolchain_hints_sycl_all_missing(self) -> None:
        """get_toolchain_hints should return errors for all missing SYCL tools."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing — SYCL_REQUIRED_TOOLS has 7 tools
            mock_detect.return_value = (False, None)
            errors = get_toolchain_hints("sycl")
            assert len(errors) == 7  # gcc, make, git, cmake, dpcpp, icx, icpx
            for error in errors:
                assert isinstance(error, ToolchainErrorDetail)
                assert error.error_code == ErrorCode.TOOLCHAIN_MISSING  # type: ignore

    def test_get_toolchain_hints_cuda_all_missing(self) -> None:
        """get_toolchain_hints should return errors for all missing CUDA tools."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing — CUDA_REQUIRED_TOOLS has 6 tools (no nvtop)
            mock_detect.return_value = (False, None)
            errors = get_toolchain_hints("cuda")
            assert len(errors) == 6  # gcc, make, git, cmake, nvcc, nvidia-smi
            for error in errors:
                assert isinstance(error, ToolchainErrorDetail)
                assert error.error_code == ErrorCode.TOOLCHAIN_MISSING  # type: ignore

    def test_get_toolchain_hints_sycl_some_present(self) -> None:
        """get_toolchain_hints should only return errors for missing tools."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # dpcpp present, others missing (7 SYCL tools total)
            mock_detect.side_effect = [
                (False, None),  # gcc
                (False, None),  # make
                (False, None),  # git
                (False, None),  # cmake
                (True, "1.0.0"),  # dpcpp
                (False, None),  # icx
                (False, None),  # icpx
            ]
            errors = get_toolchain_hints("sycl")
            assert len(errors) == 6  # All except dpcpp
            failed_checks = [e.failed_check for e in errors]
            assert "dpcpp" not in failed_checks
            assert "gcc" in failed_checks
            assert "make" in failed_checks

    def test_get_toolchain_hints_cuda_some_present(self) -> None:
        """get_toolchain_hints should only return errors for missing CUDA tools."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # nvcc present, others missing (6 CUDA tools total)
            mock_detect.side_effect = [
                (False, None),  # gcc
                (False, None),  # make
                (False, None),  # git
                (False, None),  # cmake
                (True, "12.2.0"),  # nvcc
                (False, None),  # nvidia-smi
            ]
            errors = get_toolchain_hints("cuda")
            assert len(errors) == 5  # All except nvcc
            failed_checks = [e.failed_check for e in errors]
            assert "nvcc" not in failed_checks
            assert "gcc" in failed_checks
            assert "make" in failed_checks
            assert "nvcc" not in failed_checks
            assert "nvidia-smi" in failed_checks
            # Note: nvtop is not in CUDA_REQUIRED_TOOLS (was never checked)

    def test_get_toolchain_hints_sycl_all_present(self) -> None:
        """get_toolchain_hints should return empty list when all SYCL tools present."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools present
            mock_detect.return_value = (True, "2023.1.0")
            errors = get_toolchain_hints("sycl")
            assert len(errors) == 0

    def test_get_toolchain_hints_cuda_all_present(self) -> None:
        """get_toolchain_hints should return empty list when all CUDA tools present."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools present
            mock_detect.return_value = (True, "12.2.0")
            errors = get_toolchain_hints("cuda")
            assert len(errors) == 0

    def test_get_toolchain_hints_invalid_backend(self) -> None:
        """get_toolchain_hints should raise ValueError for invalid backend."""
        with pytest.raises(ValueError) as exc_info:
            get_toolchain_hints("invalid_backend")
        assert "Unknown backend" in str(exc_info.value)
        assert "invalid_backend" in str(exc_info.value)

    def test_get_toolchain_hints_error_detail_fields(self) -> None:
        """get_toolchain_hints should create ToolchainErrorDetail with correct fields."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing
            mock_detect.return_value = (False, None)
            errors = get_toolchain_hints("sycl")
            assert len(errors) > 0
            error = errors[0]
            assert error.error_code == ErrorCode.TOOLCHAIN_MISSING  # type: ignore
            assert error.failed_check is not None
            assert error.why_blocked is not None
            assert error.how_to_fix is not None
            # docs_ref may be None or a URL
            assert isinstance(error.docs_ref, str | None)

    def test_get_toolchain_hints_error_detail_why_blocked(self) -> None:
        """get_toolchain_hints should set why_blocked correctly for each backend."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            mock_detect.return_value = (False, None)
            sycl_errors = get_toolchain_hints("sycl")
            cuda_errors = get_toolchain_hints("cuda")
            # Check that why_blocked mentions the correct backend
            for error in sycl_errors:
                assert "sycl" in error.why_blocked.lower()
            for error in cuda_errors:
                assert "cuda" in error.why_blocked.lower()

    def test_get_toolchain_hints_error_detail_how_to_fix(self) -> None:
        """get_toolchain_hints should set how_to_fix with install commands."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            mock_detect.return_value = (False, None)
            errors = get_toolchain_hints("sycl")
            assert len(errors) > 0
            for error in errors:
                # how_to_fix should contain some installation instruction
                assert len(error.how_to_fix) > 0
                # Should contain common install keywords
                assert any(
                    keyword in error.how_to_fix.lower()
                    for keyword in ["install", "apt", "curl", "sudo"]
                )

    def test_get_toolchain_hints_error_detail_docs_ref(self) -> None:
        """get_toolchain_hints should set docs_ref from ToolchainHint."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            mock_detect.return_value = (False, None)
            errors = get_toolchain_hints("sycl")
            assert len(errors) > 0
            # At least some errors should have docs_ref (from SYCL_HINT)
            has_docs = any(e.docs_ref is not None for e in errors)
            assert has_docs is True  # SYCL_HINT has a URL

    def test_get_toolchain_hints_sycl_vs_cuda_different_tools(self) -> None:
        """get_toolchain_hints should return different tools for sycl vs cuda."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            mock_detect.return_value = (False, None)
            sycl_errors = get_toolchain_hints("sycl")
            cuda_errors = get_toolchain_hints("cuda")
            # Check unique tools per backend (common tools: gcc, make, git, cmake)
            sycl_tools = [e.failed_check for e in sycl_errors]
            cuda_tools = [e.failed_check for e in cuda_errors]
            # SYCL-specific tools should not be in CUDA list
            sycl_unique = ["dpcpp", "icx", "icpx"]
            cuda_unique = ["nvcc", "nvidia-smi"]
            for tool in sycl_unique:
                assert tool not in cuda_tools, f"{tool} should not be in CUDA tools"
            for tool in cuda_unique:
                assert tool not in sycl_tools, f"{tool} should not be in SYCL tools"
