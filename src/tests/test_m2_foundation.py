"""T026: Foundational regression tests for Phase 2.

Integration tests for:
- detect_toolchain()
- get_toolchain_hints()
- create_venv()
- check_venv_integrity()
- write_failure_report()
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_manager.reports import FailureReport, write_failure_report
from llama_manager.setup_venv import VenvResult, check_venv_integrity, create_venv, get_venv_path
from llama_manager.toolchain import (
    ToolchainErrorDetail,
    ToolchainHint,
    ToolchainStatus,
    get_toolchain_hints,
    parse_version,
    version_at_least,
)


class TestDetectToolchainIntegration:
    """Integration tests for detect_toolchain functionality."""

    def test_detect_toolchain_basic(self) -> None:
        """detect_toolchain should return ToolchainStatus with correct structure."""
        # Mock detect_tool to return known values
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # Simulate all tools present
            mock_detect.return_value = (True, "1.0.0")

            # Import here to avoid circular import
            from llama_manager.toolchain import detect_toolchain

            status = detect_toolchain()

            assert isinstance(status, ToolchainStatus)
            # All fields should be set since we mocked detect_tool to return True
            assert status.gcc is not None
            assert status.make is not None
            assert status.git is not None
            assert status.cmake is not None
            assert status.sycl_compiler is not None
            assert status.cuda_toolkit is not None
            assert status.nvtop is not None

    def test_detect_toolchain_partial_tools(self) -> None:
        """detect_toolchain should handle partial tool availability."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # Simulate some tools present, some missing
            mock_detect.side_effect = [
                (True, "11.4.0"),  # gcc
                (True, "4.3"),  # make
                (True, "2.37.0"),  # git
                (True, "3.25.0"),  # cmake
                (False, None),  # sycl_compiler missing
                (True, "12.2.0"),  # cuda_toolkit
                (True, "2.1.0"),  # nvtop
            ]

            from llama_manager.toolchain import detect_toolchain

            status = detect_toolchain()

            assert status.gcc == "11.4.0"
            assert status.sycl_compiler is None  # Missing
            assert status.is_sycl_ready is False
            assert status.is_cuda_ready is True

    def test_detect_toolchain_all_missing(self) -> None:
        """detect_toolchain should handle all tools missing."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing
            mock_detect.return_value = (False, None)

            from llama_manager.toolchain import detect_toolchain

            status = detect_toolchain()

            assert status.gcc is None
            assert status.make is None
            assert status.is_sycl_ready is False
            assert status.is_cuda_ready is False
            assert status.is_complete is False


class TestGetToolchainHintsIntegration:
    """Integration tests for get_toolchain_hints functionality."""

    def test_get_toolchain_hints_sycl_integration(self) -> None:
        """get_toolchain_hints should return list of ToolchainErrorDetail for SYCL."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing
            mock_detect.return_value = (False, None)

            errors = get_toolchain_hints("sycl")

            assert isinstance(errors, list)
            assert len(errors) == 3  # sycl-ls, icpx, syclpp
            for error in errors:
                assert isinstance(error, ToolchainErrorDetail)
                assert error.error_code.value == "TOOLCHAIN_MISSING"
                assert error.failed_check is not None
                assert error.why_blocked is not None
                assert error.how_to_fix is not None

    def test_get_toolchain_hints_cuda_integration(self) -> None:
        """get_toolchain_hints should return list of ToolchainErrorDetail for CUDA."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing
            mock_detect.return_value = (False, None)

            errors = get_toolchain_hints("cuda")

            assert isinstance(errors, list)
            assert len(errors) == 3  # nvcc, nvidia-smi, nvtop
            for error in errors:
                assert isinstance(error, ToolchainErrorDetail)
                assert error.error_code.value == "TOOLCHAIN_MISSING"
                assert error.failed_check is not None
                assert error.why_blocked is not None
                assert error.how_to_fix is not None

    def test_get_toolchain_hints_empty_when_all_present(self) -> None:
        """get_toolchain_hints should return empty list when all tools present."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools present
            mock_detect.return_value = (True, "1.0.0")

            sycl_errors = get_toolchain_hints("sycl")
            cuda_errors = get_toolchain_hints("cuda")

            assert len(sycl_errors) == 0
            assert len(cuda_errors) == 0

    def test_get_toolchain_hints_invalid_backend_raises(self) -> None:
        """get_toolchain_hints should raise ValueError for invalid backend."""
        with pytest.raises(ValueError) as exc_info:
            get_toolchain_hints("invalid")
        assert "Unknown backend" in str(exc_info.value)


class TestCreateVenvIntegration:
    """Integration tests for create_venv functionality."""

    def test_create_venv_integration(self, tmp_path: Path) -> None:
        """create_venv should return VenvResult with correct structure."""
        venv_path = tmp_path / "test_venv"

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

            assert isinstance(result, VenvResult)
            assert result.venv_path == venv_path
            assert result.created is True
            assert result.reused is False
            assert result.was_created is True
            assert result.was_reused is False
            assert "source" in result.activation_command
            assert "bin/activate" in result.activation_command

    def test_create_venv_reuse_existing(self, tmp_path: Path) -> None:
        """create_venv should reuse existing venv."""
        venv_path = tmp_path / "existing_venv"
        venv_path.mkdir()

        result = create_venv(venv_path)

        assert isinstance(result, VenvResult)
        assert result.venv_path == venv_path
        assert result.created is False
        assert result.reused is True
        assert result.was_created is False
        assert result.was_reused is True

    def test_create_venv_activation_command_format(self, tmp_path: Path) -> None:
        """create_venv should generate correct activation command format."""
        venv_path = tmp_path / "test_venv"

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

            # Should be sourceable
            assert result.activation_command.startswith("source ")
            assert result.activation_command.endswith("/activate")


class TestCheckVenvIntegrityIntegration:
    """Integration tests for check_venv_integrity functionality."""

    def test_check_venv_integrity_valid_venv(self, tmp_path: Path) -> None:
        """check_venv_integrity should validate valid venv structure."""
        venv_path = tmp_path / "valid_venv"
        venv_path.mkdir()

        # Create minimal venv structure
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to("/usr/bin/python3")

        is_valid, error = check_venv_integrity(venv_path)

        assert is_valid is True
        assert error is None

    def test_check_venv_integrity_invalid_venv(self, tmp_path: Path) -> None:
        """check_venv_integrity should detect invalid venv structure."""
        venv_path = tmp_path / "invalid_venv"
        venv_path.mkdir()

        # Missing pyvenv.cfg
        is_valid, error = check_venv_integrity(venv_path)

        assert is_valid is False
        assert error == "pyvenv.cfg missing"

    def test_check_venv_integrity_integration_with_mock(self) -> None:
        """check_venv_integrity should work with mocked paths."""
        with patch("pathlib.Path.exists") as mock_exists:
            # Simulate valid venv
            mock_exists.return_value = True

            is_valid, error = check_venv_integrity("/mock/venv")

            assert is_valid is True
            assert error is None


class TestWriteFailureReportIntegration:
    """Integration tests for write_failure_report functionality."""

    def test_write_failure_report_integration(self, tmp_path: Path) -> None:
        """write_failure_report should create proper failure report structure."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json='{"exit_code": 1}',
            build_output="Build failed: compilation error",
            error_details=[{"type": "BuildError", "message": "compilation failed"}],
            metadata={"backend": "sycl"},
        )

        assert isinstance(report, FailureReport)
        assert report.report_dir.exists()
        assert report.report_dir.is_dir()
        assert "20" in report.report_dir.name  # YYYYMMDD_HHMMSS format

        # Check all files created
        assert (report.report_dir / "build-artifact.json").exists()
        assert (report.report_dir / "build-output.log").exists()
        assert (report.report_dir / "error-details.json").exists()

    def test_write_failure_report_permissions(self, tmp_path: Path) -> None:
        """write_failure_report should enforce correct permissions."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=[],
        )

        # Directory should be 0700
        dir_mode = report.report_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700

        # Files should be 0600
        for filename in ["build-artifact.json", "build-output.log", "error-details.json"]:
            file_path = report.report_dir / filename
            file_mode = file_path.stat().st_mode & 0o777
            assert file_mode == 0o600

    def test_write_failure_report_redaction_integration(self, tmp_path: Path) -> None:
        """write_failure_report should redact sensitive data in output."""
        output_with_secrets = "API_KEY=secret123 TOKEN=abc456 Normal build output"

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output=output_with_secrets,
            error_details=[],
        )

        # Read the output file
        output_path = report.report_dir / "build-output.log"
        with open(output_path) as f:
            actual_output = f.read()

        # Should be redacted
        assert "[REDACTED]" in actual_output
        assert "secret123" not in actual_output
        assert "abc456" not in actual_output
        # Non-sensitive content should remain
        assert "Normal build output" in actual_output

    def test_write_failure_report_truncation_integration(self, tmp_path: Path) -> None:
        """write_failure_report should truncate large outputs."""
        # Create very long output
        long_output = "x" * 20000  # More than default 8192 bytes

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output=long_output,
            error_details=[],
        )

        # Read the output file
        output_path = report.report_dir / "build-output.log"
        with open(output_path) as f:
            actual_output = f.read()

        # Should be truncated to 8192 bytes
        assert len(actual_output) <= 8192
        assert len(actual_output) < len(long_output)

    def test_write_failure_report_json_serialization(self, tmp_path: Path) -> None:
        """write_failure_report should properly serialize JSON data."""
        error_details = [
            {"type": "BuildError", "message": "compilation failed"},
            {"type": "Warning", "message": "deprecated flag used"},
        ]

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json='{"exit_code": 1}',
            build_output="",
            error_details=error_details,
        )

        # Read error details file
        errors_path = report.report_dir / "error-details.json"
        with open(errors_path) as f:
            loaded_errors = json.load(f)

        assert len(loaded_errors) == 2
        assert loaded_errors[0]["type"] == "BuildError"
        assert loaded_errors[1]["type"] == "Warning"


class TestPhase2Comprehensive:
    """Comprehensive tests covering all Phase 2 functionality."""

    def test_version_parsing_and_comparison(self) -> None:
        """Test version parsing and comparison for toolchain validation."""
        # Test parse_version
        assert parse_version("3.20.1") == (3, 20, 1)
        assert parse_version("3.20") == (3, 20, 0)
        assert parse_version("3.20.1ubuntu") == (3, 20, 1)

        # Test version_at_least
        assert version_at_least("3.20.1", "3.20.0") is True
        assert version_at_least("3.19.0", "3.20.0") is False
        assert version_at_least("3.14", "3.14") is True

    def test_toolchain_error_detail_structure(self) -> None:
        """Test ToolchainErrorDetail has all required fields."""
        error = ToolchainErrorDetail(
            error_code="TOOLCHAIN_MISSING",  # type: ignore
            failed_check="gcc",
            why_blocked="Required for sycl backend",
            how_to_fix="sudo apt-get install gcc",
            docs_ref="https://gcc.gnu.org/download.html",
        )

        assert error.error_code == "TOOLCHAIN_MISSING"
        assert error.failed_check == "gcc"
        assert error.why_blocked == "Required for sycl backend"
        assert error.how_to_fix == "sudo apt-get install gcc"
        assert error.docs_ref == "https://gcc.gnu.org/download.html"

    def test_toolchain_hint_structure(self) -> None:
        """Test ToolchainHint has all required fields."""
        hint = ToolchainHint(
            tool_name="gcc",
            install_command="sudo apt-get install gcc",
            install_url="https://gcc.gnu.org/download.html",
            required_for=["sycl", "cuda"],
        )

        assert hint.tool_name == "gcc"
        assert hint.install_command == "sudo apt-get install gcc"
        assert hint.install_url == "https://gcc.gnu.org/download.html"
        assert hint.required_for == ["sycl", "cuda"]
        assert hint.is_url_available is True

    def test_venv_path_utility(self) -> None:
        """Test get_venv_path utility function."""
        # Test with default
        with patch.dict(os.environ, {}, clear=True):
            # Ensure XDG_CACHE_HOME is not set
            os.environ.pop("XDG_CACHE_HOME", None)
            result = get_venv_path()
            assert isinstance(result, Path)
            assert "llm-runner" in str(result)
            assert "venv" in str(result)

    def test_sycl_vs_cuda_toolchain_hints(self) -> None:
        """Test that SYCL and CUDA hints return different tools."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            mock_detect.return_value = (False, None)

            sycl_errors = get_toolchain_hints("sycl")
            cuda_errors = get_toolchain_hints("cuda")

            # Should have different tool names
            sycl_tools = {e.failed_check for e in sycl_errors}
            cuda_tools = {e.failed_check for e in cuda_errors}

            # No overlap
            assert sycl_tools.isdisjoint(cuda_tools)

            # Should have expected tools
            assert sycl_tools == {"sycl-ls", "icpx", "syclpp"}
            assert cuda_tools == {"nvcc", "nvidia-smi", "nvtop"}

    def test_report_security(self, tmp_path: Path) -> None:
        """Test that failure reports properly handle sensitive data."""
        sensitive_output = """
        API_KEY=supersecret123
        TOKEN=abc456def
        PASSWORD=mypass
        Normal log line
        AUTH_HEADER=bearer_token
        """

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output=sensitive_output,
            error_details=[],
        )

        # Read all files and verify redaction
        for filename in ["build-output.log", "build-artifact.json", "error-details.json"]:
            file_path = report.report_dir / filename
            with open(file_path) as f:
                content = f.read()

            # Sensitive values should be redacted
            assert "supersecret123" not in content
            assert "abc456def" not in content
            assert "mypass" not in content
            assert "bearer_token" not in content

            # Should have redaction markers
            if content.strip():
                assert "[REDACTED]" in content or "[]" in content or "{}" in content

    def test_venv_result_properties(self, tmp_path: Path) -> None:
        """Test VenvResult properties."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=True,
            reused=False,
            activation_command="source /tmp/venv/bin/activate",
        )

        assert result.was_created is True
        assert result.was_reused is False
        assert result.is_valid is False  # Path doesn't exist

        # Test path methods
        python_path = result.get_python_path()
        assert python_path.name == "python"

        pip_path = result.get_pip_path()
        assert pip_path.name == "pip"
