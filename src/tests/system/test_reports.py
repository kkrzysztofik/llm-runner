"""Tests for the reports module: FailureReport and write_failure_report.

Covers the FailureReport dataclass (including its JSON contract and
save_to_file() directory structure) and the write_failure_report()
function.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_manager.reports import FailureReport, write_failure_report


class TestFailureReport:
    """T008: Tests for FailureReport dataclass."""

    def test_failure_report_all_fields_settable(self, tmp_path: Path) -> None:
        """FailureReport should have all fields settable and retrievable."""
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        report = FailureReport(
            report_dir=tmp_path / "reports" / "2026-04-15T12-30-00",
            timestamp=timestamp,
            build_artifact_json='{"exit_code": 1}',
            build_output_log="Error: build failed",
            error_details_json='{"type": "BuildError", "message": "compilation failed"}',
            metadata={"backend": "sycl", "commit": "abc123"},
        )
        assert report.report_dir == tmp_path / "reports" / "2026-04-15T12-30-00"
        assert report.timestamp == timestamp
        assert report.build_artifact_json == '{"exit_code": 1}'
        assert report.build_output_log == "Error: build failed"
        assert (
            report.error_details_json == '{"type": "BuildError", "message": "compilation failed"}'
        )
        assert report.metadata == {"backend": "sycl", "commit": "abc123"}

    def test_failure_report_default_metadata(self, tmp_path: Path) -> None:
        """FailureReport should default metadata to empty dict."""
        report = FailureReport(
            report_dir=tmp_path / "reports" / "2026-04-15T12-30-00",
            timestamp=datetime.now(),
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )
        assert report.metadata == {}

    def test_failure_report_report_path_property(self, tmp_path: Path) -> None:
        """FailureReport.report_path should return correct Path object."""
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        report = FailureReport(
            report_dir=tmp_path / "reports" / "2026-04-15T12-30-00",
            timestamp=timestamp,
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )
        report_path = report.report_path
        assert isinstance(report_path, Path)
        expected_name = f"failure_{timestamp.isoformat()}.json"
        assert report_path.name == expected_name
        assert report_path.parent == report.report_dir

    def test_failure_report_report_path_different_timestamps(self, tmp_path: Path) -> None:
        """FailureReport.report_path should vary with different timestamps."""
        timestamp1 = datetime(2026, 4, 15, 12, 0, 0)
        timestamp2 = datetime(2026, 4, 15, 12, 30, 0)
        timestamp3 = datetime(2026, 4, 15, 13, 0, 0)

        report1 = FailureReport(
            report_dir=tmp_path / "reports",
            timestamp=timestamp1,
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )
        report2 = FailureReport(
            report_dir=tmp_path / "reports",
            timestamp=timestamp2,
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )
        report3 = FailureReport(
            report_dir=tmp_path / "reports",
            timestamp=timestamp3,
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )

        # All report paths should be unique
        assert report1.report_path != report2.report_path
        assert report2.report_path != report3.report_path
        assert report1.report_path != report3.report_path

    def test_failure_report_save_to_file(self, tmp_path: Path) -> None:
        """FailureReport.save_to_file should write JSON to file."""
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        report_dir = tmp_path / "reports" / "2026-04-15T12-30-00"
        report = FailureReport(
            report_dir=report_dir,
            timestamp=timestamp,
            build_artifact_json='{"exit_code": 1}',
            build_output_log="Error: build failed",
            error_details_json='{"type": "BuildError"}',
            metadata={"backend": "sycl"},
        )

        saved_path = report.save_to_file()

        # Verify file was created
        assert saved_path.exists()
        assert saved_path == report.report_path

        # Verify file content
        with open(saved_path) as f:
            data = json.load(f)
        assert data["report_dir"] == str(report_dir)
        assert data["timestamp"] == timestamp.isoformat()
        assert data["build_artifact"] == '{"exit_code": 1}'
        assert data["build_output_log"] == "Error: build failed"
        assert data["error_details"] == '{"type": "BuildError"}'
        assert data["metadata"] == {"backend": "sycl"}

    def test_failure_report_save_to_file_creates_directory(self, tmp_path: Path) -> None:
        """FailureReport.save_to_file should create report directory if it doesn't exist."""
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        nested_dir = tmp_path / "reports" / "nested" / "path" / "2026-04-15T12-30-00"

        report = FailureReport(
            report_dir=nested_dir,
            timestamp=timestamp,
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )

        saved_path = report.save_to_file()

        # Directory should be created
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert saved_path.exists()

    def test_failure_report_save_to_file_directory_structure(self, tmp_path: Path) -> None:
        """T070: FailureReport.save_to_file() should create correct directory structure.

        Tests that:
        - Nested directories are created with parents=True
        - Directory permissions are set correctly
        - File is created in the correct location
        """
        # Test deeply nested directory structure
        nested_dir = tmp_path / "level1" / "level2" / "level3" / "level4" / "2026-04-15T12-30-00"

        report = FailureReport(
            report_dir=nested_dir,
            timestamp=datetime(2026, 4, 15, 12, 30, 0),
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )

        saved_path = report.save_to_file()

        # All parent directories should exist
        assert nested_dir.exists()
        assert nested_dir.is_dir()

        # Verify each level was created
        assert (tmp_path / "level1").exists()
        assert (tmp_path / "level1" / "level2").exists()
        assert (tmp_path / "level1" / "level2" / "level3").exists()
        assert (tmp_path / "level1" / "level2" / "level3" / "level4").exists()
        assert nested_dir.exists()

        # Report file should be in the correct location
        assert saved_path.exists()
        assert saved_path.parent == nested_dir

        # Verify directory permissions
        dir_mode = nested_dir.stat().st_mode & 0o777
        # Should have at least read/execute for user
        assert dir_mode & 0o700

        # Verify file permissions
        file_mode = saved_path.stat().st_mode & 0o777
        # Should have read/write for user
        assert file_mode & 0o600

    def test_failure_report_save_to_file_json_content(self, tmp_path: Path) -> None:
        """FailureReport.save_to_file should write correct JSON structure."""
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        report = FailureReport(
            report_dir=tmp_path / "reports" / "2026-04-15T12-30-00",
            timestamp=timestamp,
            build_artifact_json='{"exit_code": 1}',
            build_output_log="Build error output",
            error_details_json='{"error": "compilation failed"}',
            metadata={"key": "value"},
        )

        saved_path = report.save_to_file()

        # Verify JSON content
        with open(saved_path) as f:
            data = json.load(f)
        assert "report_dir" in data
        assert "timestamp" in data
        assert "build_artifact" in data
        assert "build_output_log" in data
        assert "error_details" in data
        assert "metadata" in data

    def test_failure_report_report_path_format(self, tmp_path: Path) -> None:
        """FailureReport.report_path should follow expected format."""
        # Test with different timestamp formats
        test_cases = [
            datetime(2026, 4, 15, 12, 30, 0),
            datetime(2026, 1, 1, 0, 0, 0),
            datetime(2026, 12, 31, 23, 59, 59),
        ]

        for ts in test_cases:
            report = FailureReport(
                report_dir=tmp_path / "reports",
                timestamp=ts,
                build_artifact_json="{}",
                build_output_log="",
                error_details_json="{}",
            )
            # Should have 'failure_' prefix and '.json' suffix
            assert report.report_path.name.startswith("failure_")
            assert report.report_path.name.endswith(".json")

    def test_failure_report_metadata_optional(self, tmp_path: Path) -> None:
        """FailureReport should work without metadata."""
        report = FailureReport(
            report_dir=tmp_path / "reports",
            timestamp=datetime.now(),
            build_artifact_json="{}",
            build_output_log="",
            error_details_json="{}",
        )
        assert report.metadata == {}
        # Should still be able to save
        saved_path = report.save_to_file()
        assert saved_path.exists()


class TestFailureReportJSONContract:
    """T069: Tests for FailureReport JSON contract with redaction."""

    def test_failure_report_json_contract_with_sensitive_data(self, tmp_path: Path) -> None:
        """FailureReport.save_to_file() should preserve sensitive data as-is.

        The FailureReport class stores data as-is. Redaction is performed
        by write_failure_report() before creating the FailureReport.
        This test verifies the JSON contract structure, not redaction behavior.
        """
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        report = FailureReport(
            report_dir=tmp_path / "reports" / "2026-04-15T12-30-00",
            timestamp=timestamp,
            build_artifact_json='{"exit_code": 1}',
            build_output_log="API_KEY=secret123 TOKEN=abc456 Normal log",
            error_details_json='{"error": "compilation failed"}',
            metadata={"backend": "sycl"},
        )

        saved_path = report.save_to_file()

        # Verify JSON content
        with open(saved_path) as f:
            data = json.load(f)

        # Build output log is stored as-is (redaction done by write_failure_report)
        assert "API_KEY=secret123" in data["build_output_log"]
        assert "TOKEN=abc456" in data["build_output_log"]
        assert "Normal log" in data["build_output_log"]

        # Verify all required fields are present
        assert data["report_dir"] == str(tmp_path / "reports" / "2026-04-15T12-30-00")
        assert data["timestamp"] == timestamp.isoformat()
        assert data["build_artifact"] == '{"exit_code": 1}'
        assert data["metadata"] == {"backend": "sycl"}

    def test_failure_report_json_contract_structure(self, tmp_path: Path) -> None:
        """FailureReport JSON should have consistent structure across saves.

        All required fields should be present and properly typed.
        """
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        report = FailureReport(
            report_dir=tmp_path / "reports" / "2026-04-15T12-30-00",
            timestamp=timestamp,
            build_artifact_json='{"exit_code": 1}',
            build_output_log="Error output",
            error_details_json='{"error": "test"}',
            metadata={"key": "value"},
        )

        saved_path = report.save_to_file()

        with open(saved_path) as f:
            data = json.load(f)

        # Verify all required fields
        required_fields = [
            "report_dir",
            "timestamp",
            "build_artifact",
            "build_output_log",
            "error_details",
            "metadata",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify types
        assert isinstance(data["report_dir"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["build_artifact"], str)
        assert isinstance(data["build_output_log"], str)
        assert isinstance(data["error_details"], str)
        assert isinstance(data["metadata"], dict)

    def test_failure_report_json_contract_empty_fields(self, tmp_path: Path) -> None:
        """FailureReport JSON should handle empty fields gracefully.

        Empty strings and empty dicts should serialize correctly.
        """
        timestamp = datetime(2026, 4, 15, 12, 30, 0)
        report = FailureReport(
            report_dir=tmp_path / "reports" / "2026-04-15T12-30-00",
            timestamp=timestamp,
            build_artifact_json="",
            build_output_log="",
            error_details_json="",
            metadata={},
        )

        saved_path = report.save_to_file()

        with open(saved_path) as f:
            data = json.load(f)

        # Empty fields should be empty strings
        assert data["build_artifact"] == ""
        assert data["build_output_log"] == ""
        assert data["error_details"] == ""
        assert data["metadata"] == {}


class TestWriteFailureReport:
    """T022: Tests for write_failure_report() function."""

    def test_write_failure_report_creates_directory(self, tmp_path: Path) -> None:
        """write_failure_report should create report directory with correct name format."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json='{"exit_code": 1}',
            build_output="Build failed",
            error_details=[{"type": "BuildError"}],
        )
        # Directory should be created with timestamp format
        assert report.report_dir.exists()
        assert report.report_dir.is_dir()
        # Directory name should be timestamp format YYYYMMDD_HHMMSS
        assert len(report.report_dir.name) == 15  # YYYYMMDD_HHMMSS
        # Format: YYYYMMDD_HHMMSS, check structure
        assert report.report_dir.name[4] in "0123456789"  # Month is numeric
        assert report.report_dir.name[7] in "0123456789"  # Day is numeric
        assert report.report_dir.name[8] == "_"  # Time separator

    def test_write_failure_report_creates_all_files(self, tmp_path: Path) -> None:
        """write_failure_report should create all 3 required files."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json='{"exit_code": 1}',
            build_output="Build failed",
            error_details=[{"type": "BuildError"}],
        )
        # Check all files exist
        assert (report.report_dir / "build-artifact.json").exists()
        assert (report.report_dir / "build-output.log").exists()
        assert (report.report_dir / "error-details.json").exists()

    def test_write_failure_report_directory_permissions(self, tmp_path: Path) -> None:
        """write_failure_report should set directory permissions to 0700."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=[],
        )
        # Check directory permissions
        dir_mode = report.report_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700

    def test_write_failure_report_file_permissions(self, tmp_path: Path) -> None:
        """write_failure_report should set file permissions to 0600."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=[],
        )
        # Check file permissions
        for filename in ["build-artifact.json", "build-output.log", "error-details.json"]:
            file_path = report.report_dir / filename
            assert file_path.exists()
            file_mode = file_path.stat().st_mode & 0o777
            assert file_mode == 0o600

    def test_write_failure_report_truncates_output(self, tmp_path: Path) -> None:
        """write_failure_report should truncate output to Config.build_output_truncate_bytes."""
        # Create very long output (more than default 8192 bytes)
        long_output = "x" * 10000
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

    def test_write_failure_report_redacts_sensitive(self, tmp_path: Path) -> None:
        """write_failure_report should redact sensitive patterns in output."""
        output_with_secrets = "API_KEY=secret123 TOKEN=abc456 Normal log"
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

    def test_write_failure_report_writes_artifact_json(self, tmp_path: Path) -> None:
        """write_failure_report should write build-artifact.json correctly."""
        artifact_json = '{"exit_code": 1, "command": ["gcc", "test.c"]}'
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json=artifact_json,
            build_output="",
            error_details=[],
        )
        # Read the artifact file
        artifact_path = report.report_dir / "build-artifact.json"
        with open(artifact_path) as f:
            content = f.read()
        assert content == artifact_json

    def test_write_failure_report_writes_error_details_json(self, tmp_path: Path) -> None:
        """write_failure_report should write error-details.json correctly."""
        error_details = [{"type": "BuildError", "message": "compilation failed"}]
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=error_details,
        )
        # Read the error details file
        errors_path = report.report_dir / "error-details.json"
        with open(errors_path) as f:
            content = f.read()
        # Should be valid JSON
        errors = json.loads(content)
        assert len(errors) == 1
        assert errors[0]["type"] == "BuildError"

    def test_write_failure_report_default_report_dir(self, tmp_path: Path) -> None:
        """write_failure_report should use Config().reports_dir when report_dir not provided."""
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.paths = MagicMock()
            mock_config_instance.paths.reports_dir = tmp_path
            mock_config.return_value = mock_config_instance
            report = write_failure_report(
                build_artifact_json="{}",
                build_output="",
                error_details=[],
            )
            # report.report_dir should be a subdirectory of tmp_path
            assert tmp_path in report.report_dir.parents

    def test_write_failure_report_with_metadata(self, tmp_path: Path) -> None:
        """write_failure_report should handle metadata parameter."""
        metadata = {"backend": "sycl", "commit": "abc123"}
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=[],
            metadata=metadata,
        )
        assert report.metadata == metadata

    def test_write_failure_report_all_metadata_fields(self, tmp_path: Path) -> None:
        """T072: write_failure_report should handle all metadata fields correctly.

        Tests comprehensive metadata with various types:
        - Backend information
        - Git commit SHA
        - Build configuration
        - Timestamp
        - Custom fields
        """
        metadata = {
            "backend": "cuda",
            "git_commit": "abc123def456",
            "git_branch": "main",
            "build_config": {
                "jobs": 8,
                "shallow_clone": True,
            },
            "hardware": {
                "gpu": "NVIDIA RTX 3090",
                "cuda_version": "12.2",
            },
            "user": "developer",
            "priority": "high",
            "tags": ["build", "cuda", "production"],
        }
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json='{"exit_code": 1}',
            build_output="Build failed",
            error_details=[{"type": "BuildError", "message": "compilation failed"}],
            metadata=metadata,
        )

        # Verify metadata is preserved in report
        assert report.metadata == metadata
        assert report.metadata["backend"] == "cuda"
        assert report.metadata["git_commit"] == "abc123def456"
        assert report.metadata["build_config"]["jobs"] == 8
        assert report.metadata["hardware"]["gpu"] == "NVIDIA RTX 3090"

    def test_write_failure_report_empty_error_details(self, tmp_path: Path) -> None:
        """write_failure_report should handle empty error details list."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=[],
        )
        # Should not raise
        assert report is not None
        # Error details file should contain empty array
        errors_path = report.report_dir / "error-details.json"
        with open(errors_path) as f:
            content = f.read()
        assert content == "[]"

    def test_write_failure_report_none_error_details(self, tmp_path: Path) -> None:
        """write_failure_report should handle None error details."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=None,
        )
        # Should not raise
        assert report is not None
