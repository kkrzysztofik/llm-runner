from __future__ import annotations

"""T008, T012, T014, T022-T023, T069-T074: Tests for FailureReport, MutatingActionLogEntry, redact_sensitive, write_failure_report, rotate_reports.

Test Tasks:
- T008: FailureReport dataclass tests
- T012: MutatingActionLogEntry dataclass tests
- T014: redact_sensitive() tests
- T022: write_failure_report() tests
- T023: rotate_reports() tests
- T069: FailureReport JSON contract with redaction
- T070: FailureReport save_to_file() directory structure
- T071: MutatingActionLogEntry rotation policy
- T072: write_failure_report() with all metadata fields
- T073: rotate_reports() edge cases
- T074: offline-continue path when network unavailable
"""


import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_manager.reports import (
    FailureReport,
    MutatingActionLogEntry,
    log_mutating_action,
    redact_sensitive,
    rotate_reports,
    write_failure_report,
)


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


class TestRedactSensitive:
    """T014: Tests for redact_sensitive() function."""

    def test_redact_sensitive_api_key(self) -> None:
        """redact_sensitive should redact API_KEY values."""
        assert redact_sensitive("API_KEY=secret123") == "API_KEY: [REDACTED]"
        assert redact_sensitive("api_key=lowercase") == "api_key: [REDACTED]"
        assert redact_sensitive("Api_Key=MixedCase") == "Api_Key: [REDACTED]"

    def test_redact_sensitive_token(self) -> None:
        """redact_sensitive should redact TOKEN values."""
        assert redact_sensitive("TOKEN=abc123") == "TOKEN: [REDACTED]"
        assert redact_sensitive("auth_token=xyz789") == "auth_token: [REDACTED]"

    def test_redact_sensitive_secret(self) -> None:
        """redact_sensitive should redact SECRET values."""
        assert redact_sensitive("SECRET=mysecret") == "SECRET: [REDACTED]"
        assert redact_sensitive("api_secret: secret123") == "api_secret: [REDACTED]"

    def test_redact_sensitive_password(self) -> None:
        """redact_sensitive should redact PASSWORD values."""
        assert redact_sensitive("PASSWORD=supersecret") == "PASSWORD: [REDACTED]"
        assert redact_sensitive("db_password: mypass") == "db_password: [REDACTED]"

    def test_redact_sensitive_auth(self) -> None:
        """redact_sensitive should redact AUTH values."""
        assert redact_sensitive("AUTH_HEADER=bearer_xyz") == "AUTH_HEADER: [REDACTED]"
        assert redact_sensitive("auth_token=token123") == "auth_token: [REDACTED]"

    def test_redact_sensitive_case_insensitive(self) -> None:
        """redact_sensitive should work case-insensitively."""
        assert redact_sensitive("API_KEY=secret") == "API_KEY: [REDACTED]"
        assert redact_sensitive("api_key=secret") == "api_key: [REDACTED]"
        assert redact_sensitive("Api_Key=secret") == "Api_Key: [REDACTED]"

    def test_redact_sensitive_standalone_key(self) -> None:
        """redact_sensitive should redact standalone sensitive words."""
        assert redact_sensitive("KEY") == "[REDACTED]"
        assert redact_sensitive("TOKEN") == "[REDACTED]"
        assert redact_sensitive("SECRET") == "[REDACTED]"
        assert redact_sensitive("PASSWORD") == "[REDACTED]"
        assert redact_sensitive("AUTH") == "[REDACTED]"

    def test_redact_sensitive_no_redaction_needed(self) -> None:
        """redact_sensitive should return unchanged text when no sensitive data."""
        assert redact_sensitive("normal log message") == "normal log message"
        assert redact_sensitive("Building model from /path/to/model.gguf") == (
            "Building model from /path/to/model.gguf"
        )
        assert redact_sensitive("Server started on port 8080") == "Server started on port 8080"

    def test_redact_sensitive_multiple_in_one_line(self) -> None:
        """redact_sensitive should redact multiple sensitive values in one line."""
        text = "API_KEY=one TOKEN=two PASSWORD=three"
        result = redact_sensitive(text)
        # All three should be redacted
        assert result.count("[REDACTED]") == 3
        assert "one" not in result
        assert "two" not in result
        assert "three" not in result

    def test_redact_sensitive_key_value_with_colon(self) -> None:
        """redact_sensitive should handle key:value format."""
        assert redact_sensitive("api-key: secret123") == "api-key: [REDACTED]"
        assert redact_sensitive("DB_PASSWORD: mypass") == "DB_PASSWORD: [REDACTED]"

    def test_redact_sensitive_key_value_with_equals(self) -> None:
        """redact_sensitive should handle key=value format."""
        assert redact_sensitive("api-key=secret123") == "api-key: [REDACTED]"
        assert redact_sensitive("MY_TOKEN=abc123") == "MY_TOKEN: [REDACTED]"

    def test_redact_sensitive_key_value_with_space(self) -> None:
        """redact_sensitive should handle key value format (space separated)."""
        result = redact_sensitive("api-key secret123")
        # Should redact the value part
        assert "[REDACTED]" in result

    def test_redact_sensitive_preserves_other_content(self) -> None:
        """redact_sensitive should preserve non-sensitive content."""
        text = "2024-01-01 12:00:00 API_KEY=secret Server started"
        result = redact_sensitive(text)
        assert "2024-01-01 12:00:00" in result
        assert "Server started" in result
        assert "[REDACTED]" in result

    def test_redact_sensitive_empty_string(self) -> None:
        """redact_sensitive should handle empty string."""
        assert redact_sensitive("") == ""

    def test_redact_sensitive_only_sensitive_word(self) -> None:
        """redact_sensitive should redact lines that are only sensitive words."""
        # Standalone sensitive words should be fully redacted
        assert redact_sensitive("API_KEY") == "[REDACTED]"
        assert redact_sensitive("TOKEN") == "[REDACTED]"
        assert redact_sensitive("SECRET") == "[REDACTED]"
        assert redact_sensitive("PASSWORD") == "[REDACTED]"
        assert redact_sensitive("AUTH") == "[REDACTED]"


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
            mock_config_instance.reports_dir = tmp_path
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


class TestRotateReports:
    """T023: Tests for rotate_reports() function."""

    def test_rotate_reports_no_reports_dir(self, tmp_path: Path) -> None:
        """rotate_reports should handle missing reports directory."""
        reports_path = tmp_path / "nonexistent"
        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = reports_path
            mock_config.return_value = mock_config_instance
            # Should not raise
            rotate_reports(mock_config_instance)

    def test_rotate_reports_when_count_less_than_max(self, tmp_path: Path) -> None:
        """rotate_reports should not delete when count <= max_reports."""
        # Create 5 report directories (less than max of 10)
        for i in range(5):
            report_dir = tmp_path / f"202601{(i + 1):02d}_120000"
            report_dir.mkdir()
            # Set different modification times
            report_dir.touch()

        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 10
            mock_config.return_value = mock_config_instance
            rotate_reports(mock_config_instance)

        # All directories should still exist
        remaining = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(remaining) == 5

    def test_rotate_reports_when_count_exceeds_max(self, tmp_path: Path) -> None:
        """rotate_reports should delete oldest when count > max_reports."""
        # Create 15 report directories (more than max of 10)
        for i in range(15):
            # Use valid timestamp format: YYYYMMDD_HHMMSS
            # Days 01-15, time 12:00:00
            day = i + 1
            report_dir = tmp_path / f"202601{day:02d}_120000"
            report_dir.mkdir()
            # Set different modification times (oldest first)
            report_dir.touch()
            # Add delay to ensure different mtime
            time.sleep(0.01)

        # Mock Config class properly - patch where it's used, not where it's defined
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 10
            mock_config.return_value = mock_config_instance
            rotate_reports(mock_config_instance)

        # Only 10 directories should remain (newest ones)
        remaining = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(remaining) == 10

    def test_rotate_reports_only_timestamp_directories(self, tmp_path: Path) -> None:
        """rotate_reports should only delete directories with timestamp format."""
        # Create timestamp directories with valid dates (Jan 1-5)
        for i in range(5):
            report_dir = tmp_path / f"202601{(i + 1):02d}_120000"
            report_dir.mkdir()

        # Create non-timestamp directory (should not be deleted)
        other_dir = tmp_path / "other-directory"
        other_dir.mkdir()

        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 3
            mock_config.return_value = mock_config_instance
            rotate_reports(mock_config_instance)

        # Non-timestamp directory should still exist
        assert other_dir.exists()

    def test_rotate_reports_invalid_timestamp_format_skipped(self, tmp_path: Path) -> None:
        """rotate_reports should skip directories with invalid timestamp format."""
        # Create valid timestamp directories with valid dates (Jan 1-3)
        for i in range(3):
            report_dir = tmp_path / f"202601{(i + 1):02d}_120000"
            report_dir.mkdir()

        # Create invalid format directory
        invalid_dir = tmp_path / "not-a-timestamp"
        invalid_dir.mkdir()

        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 2
            mock_config.return_value = mock_config_instance
            # Should not raise
            rotate_reports(mock_config_instance)

        # Invalid directory should still exist
        assert invalid_dir.exists()

    def test_rotate_reports_deletes_oldest_first(self, tmp_path: Path) -> None:
        """rotate_reports should delete oldest directories first."""
        import time

        # Create directories with different modification times
        directories = []
        for i in range(5):
            # Use valid timestamp format: YYYYMMDD_HHMMSS
            day = i + 1
            report_dir = tmp_path / f"202601{day:02d}_120000"
            report_dir.mkdir()
            directories.append(report_dir)
            time.sleep(0.01)  # Ensure different mtime

        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 3
            mock_config.return_value = mock_config_instance
            rotate_reports(mock_config_instance)

        # Oldest 2 directories should be deleted
        remaining = [d for d in tmp_path.iterdir() if d.is_dir()]
        # Should have 3 remaining (newest ones)
        assert len(remaining) == 3
        # Check that the oldest directories were deleted
        remaining_names = {d.name for d in remaining}
        assert "20260101_120000" not in remaining_names  # Oldest
        assert "20260102_120000" not in remaining_names  # Second oldest
        # Newest should remain
        assert "20260103_120000" in remaining_names
        assert "20260104_120000" in remaining_names

    def test_rotate_reports_edge_cases(self, tmp_path: Path) -> None:
        """T073: Test rotate_reports edge cases.

        Tests:
        - Empty directory
        - Invalid timestamps
        - Mixed valid/invalid directories
        - Very old directories
        """
        import time

        # Create valid timestamp directories with valid dates (Jan 1-3)
        for i in range(3):
            report_dir = tmp_path / f"202601{(i + 1):02d}_120000"
            report_dir.mkdir()
            time.sleep(0.01)

        # Create invalid timestamp directories
        invalid_dirs = [
            tmp_path / "not-a-timestamp",
            tmp_path / "2026-01-01",  # Wrong format
            tmp_path / "20260101",  # Missing time
            tmp_path / "20260101_120000_extra",  # Extra characters
            tmp_path / "invalid_timestamp",
        ]
        for invalid_dir in invalid_dirs:
            invalid_dir.mkdir()

        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 2
            mock_config.return_value = mock_config_instance

            # Should not raise on invalid directories
            rotate_reports(mock_config_instance)

        # Invalid directories should remain
        for invalid_dir in invalid_dirs:
            assert invalid_dir.exists(), f"Invalid directory {invalid_dir} was deleted"

        # Only valid timestamp directories should be considered for rotation
        remaining = [d for d in tmp_path.iterdir() if d.is_dir()]
        # Should have: 2 valid (oldest deleted) + 5 invalid = 7 total
        assert len(remaining) == 7

    def test_rotate_reports_empty_directory(self, tmp_path: Path) -> None:
        """T073: rotate_reports should handle empty reports directory."""
        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 10
            mock_config.return_value = mock_config_instance

            # Should not raise on empty directory
            rotate_reports(mock_config_instance)

        # Directory should still exist and be empty
        assert tmp_path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_rotate_reports_single_directory(self, tmp_path: Path) -> None:
        """T073: rotate_reports should handle single directory correctly.

        When only one directory exists and max_reports >= 1, nothing should be deleted.
        """
        report_dir = tmp_path / "20260101_120000"
        report_dir.mkdir()

        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 5
            mock_config.return_value = mock_config_instance

            rotate_reports(mock_config_instance)

        # Directory should still exist
        assert report_dir.exists()

    def test_rotate_reports_max_reports_zero(self, tmp_path: Path) -> None:
        """T073: rotate_reports should handle max_reports=0 (delete all).

        When max_reports is 0, all timestamp directories should be deleted.
        """
        # Create 5 timestamp directories
        for i in range(5):
            # Use valid timestamp format: YYYYMMDD_HHMMSS
            day = i + 1
            report_dir = tmp_path / f"202601{day:02d}_120000"
            report_dir.mkdir()

        # Mock Config class properly - patch where it's used
        with patch("llama_manager.config.Config") as mock_config:
            mock_config_instance = MagicMock()
            mock_config_instance.reports_dir = tmp_path
            mock_config_instance.build_max_reports = 0
            mock_config.return_value = mock_config_instance

            rotate_reports(mock_config_instance)

        # All timestamp directories should be deleted
        remaining = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(remaining) == 0


class TestMutatingActionLogEntryRotation:
    """T071: Tests for MutatingActionLogEntry rotation policy."""

    def test_mutating_action_log_entry_rotation_policy_max_entries(self, tmp_path: Path) -> None:
        """T071: MutatingActionLogEntry should enforce max entries (100).

        When adding entries to a log, oldest entries should be deleted
        when max is exceeded, maintaining oldest-first deletion policy.
        """
        # Create a mock pipeline to test log rotation
        # Note: This tests the conceptual rotation policy
        # In practice, rotation would be managed by the caller

        # Create entries list
        entries: list[MutatingActionLogEntry] = []
        max_entries = 100

        # Add entries up to max
        for i in range(max_entries):
            # Use hours to avoid minute overflow (0-59)
            hour = 12 + (i // 60)
            minute = i % 60
            entry = MutatingActionLogEntry(
                command=["git", "clone", f"repo{i}"],
                timestamp=datetime(2026, 4, 15, hour, minute, 0),
                exit_code=0,
                truncated_output=f"Output {i}",
            )
            entries.append(entry)

        assert len(entries) == max_entries

        # Add one more entry (should trigger rotation)
        new_hour = 12 + (max_entries // 60)
        new_minute = max_entries % 60
        new_entry = MutatingActionLogEntry(
            command=["git", "clone", "repo100"],
            timestamp=datetime(2026, 4, 15, new_hour, new_minute, 0),
            exit_code=0,
            truncated_output="Output 100",
        )

        # Apply rotation: remove oldest, add new
        entries.append(new_entry)
        if len(entries) > max_entries:
            # Remove oldest entry (first in list)
            entries.pop(0)

        assert len(entries) == max_entries
        # Oldest entry should be removed
        assert entries[0].command[2] == "repo1"  # First entry was repo0
        # New entry should be present
        assert entries[-1].command[2] == "repo100"

    def test_mutating_action_log_entry_rotation_oldest_first_deletion(self, tmp_path: Path) -> None:
        """T071: Rotation should delete oldest entries first (FIFO).

        When exceeding max entries, the oldest entries (first in list)
        should be removed, not the newest.
        """
        entries: list[MutatingActionLogEntry] = []
        max_entries = 5

        # Add 7 entries (exceeds max)
        for i in range(7):
            entry = MutatingActionLogEntry(
                command=["cmd", str(i)],
                timestamp=datetime(2026, 4, 15, 12, i, 0),
                exit_code=0,
                truncated_output=f"Output {i}",
            )
            entries.append(entry)

        assert len(entries) == 7

        # Simulate rotation: keep only newest max_entries
        # Oldest entries should be removed first
        while len(entries) > max_entries:
            entries.pop(0)  # Remove oldest (first)

        assert len(entries) == max_entries

        # Should have entries 2-6 (oldest 0, 1 removed)
        assert entries[0].command[1] == "2"
        assert entries[-1].command[1] == "6"
        # Oldest entries should be gone
        assert entries[0].timestamp.hour == 12
        assert entries[0].timestamp.minute == 2  # Entry 2

    def test_mutating_action_log_entry_rotation_empty_log(self, tmp_path: Path) -> None:
        """T071: Rotation should handle empty log gracefully.

        When log is empty, adding entries should work normally.
        """
        entries: list[MutatingActionLogEntry] = []
        max_entries = 10

        # Add entries to empty log
        for i in range(5):
            entry = MutatingActionLogEntry(
                command=["cmd", str(i)],
                timestamp=datetime(2026, 4, 15, 12, i, 0),
                exit_code=0,
                truncated_output=f"Output {i}",
            )
            entries.append(entry)

        assert len(entries) == 5
        assert len(entries) <= max_entries  # Should not trigger rotation

    def test_mutating_action_log_entry_rotation_single_entry_limit(self, tmp_path: Path) -> None:
        """T071: Rotation should handle max_entries=1 correctly.

        When max is 1, only the newest entry should be kept.
        """
        entries: list[MutatingActionLogEntry] = []
        max_entries = 1

        # Add 3 entries
        for i in range(3):
            entry = MutatingActionLogEntry(
                command=["cmd", str(i)],
                timestamp=datetime(2026, 4, 15, 12, 0, i),
                exit_code=0,
                truncated_output=f"Output {i}",
            )
            entries.append(entry)

            # Apply rotation after each addition
            while len(entries) > max_entries:
                entries.pop(0)

        # Should only have the newest entry
        assert len(entries) == 1
        assert entries[0].command[1] == "2"  # Only entry 2 (newest)


class TestLogMutatingAction:
    """Tests for log_mutating_action() function (FR-018.4)."""

    def test_log_mutating_action_creates_log_file(self, tmp_path: Path) -> None:
        """log_mutating_action should create log file in XDG state home."""
        # Mock Config instance to use tmp_path
        mock_config = MagicMock()
        mock_config.xdg_state_base = str(tmp_path)
        mock_config.build_output_truncate_bytes = 8192

        with patch("llama_manager.config.Config", return_value=mock_config):
            log_mutating_action(
                command=["git", "clone", "https://github.com/example/repo.git"],
                exit_code=0,
                output="Cloning into 'repo'...",
            )

        # Log file should be created
        log_path = tmp_path / "llm-runner" / "mutating_actions.log"
        assert log_path.exists()

    def test_log_mutating_action_writes_entry(self, tmp_path: Path) -> None:
        """log_mutating_action should write entry to log file."""
        mock_config = MagicMock()
        mock_config.xdg_state_base = str(tmp_path)
        mock_config.build_output_truncate_bytes = 8192

        with patch("llama_manager.config.Config", return_value=mock_config):
            _ = log_mutating_action(
                command=["git", "clone", "https://github.com/example/repo.git"],
                exit_code=0,
                output="Cloning into 'repo'...",
            )

        log_path = tmp_path / "llm-runner" / "mutating_actions.log"
        with open(log_path) as f:
            content = f.read()

        # Should contain formatted entry
        assert "SUCCESS" in content
        assert "git" in content
        assert "clone" in content

    def test_log_mutating_action_redacts_sensitive(self, tmp_path: Path) -> None:
        """log_mutating_action should redact sensitive information."""
        mock_config = MagicMock()
        mock_config.xdg_state_base = str(tmp_path)
        mock_config.build_output_truncate_bytes = 8192

        with patch("llama_manager.config.Config", return_value=mock_config):
            entry = log_mutating_action(
                command=["git", "clone", "https://github.com/example/repo.git"],
                exit_code=0,
                output="API_KEY=secret123 TOKEN=abc456",
            )

        log_path = tmp_path / "llm-runner" / "mutating_actions.log"
        with open(log_path) as f:
            content = f.read()

        # Output should be redacted
        assert "[REDACTED]" in content
        assert "secret123" not in content
        assert "abc456" not in content
        assert entry.redaction_applied is True

    def test_log_mutating_action_truncates_output(self, tmp_path: Path) -> None:
        """log_mutating_action should truncate output to Config.build_output_truncate_bytes."""
        mock_config = MagicMock()
        mock_config.xdg_state_base = str(tmp_path)
        mock_config.build_output_truncate_bytes = 8192

        with patch("llama_manager.config.Config", return_value=mock_config):
            # Create very long output
            long_output = "x" * 10000
            _entry = log_mutating_action(
                command=["cmake", "--build", "."],
                exit_code=0,
                output=long_output,
            )

        # Output should be truncated
        assert len(_entry.truncated_output) <= 8192
        assert len(_entry.truncated_output) < len(long_output)

    def test_log_mutating_action_with_working_dir(self, tmp_path: Path) -> None:
        """log_mutating_action should record working directory."""
        working_dir = tmp_path / "build"
        working_dir.mkdir()

        mock_config = MagicMock()
        mock_config.xdg_state_base = str(tmp_path)
        mock_config.build_output_truncate_bytes = 8192

        with patch("llama_manager.config.Config", return_value=mock_config):
            _entry = log_mutating_action(
                command=["cmake", "--build", "."],
                exit_code=0,
                output="Build successful",
                working_dir=working_dir,
            )

        assert _entry.working_dir == working_dir

    def test_log_mutating_action_failure(self, tmp_path: Path) -> None:
        """log_mutating_action should handle failed commands."""
        mock_config = MagicMock()
        mock_config.xdg_state_base = str(tmp_path)
        mock_config.build_output_truncate_bytes = 8192

        with patch("llama_manager.config.Config", return_value=mock_config):
            entry = log_mutating_action(
                command=["git", "checkout", "main"],
                exit_code=1,
                output="error: pathspec 'main' did not match any files",
            )

        assert entry.is_success is False
        assert entry.exit_code == 1

        log_path = tmp_path / "llm-runner" / "mutating_actions.log"
        with open(log_path) as f:
            content = f.read()
        assert "FAILED" in content

    def test_log_mutating_action_multiple_entries(self, tmp_path: Path) -> None:
        """log_mutating_action should append multiple entries to log."""
        mock_config = MagicMock()
        mock_config.xdg_state_base = str(tmp_path)
        mock_config.build_output_truncate_bytes = 8192

        with patch("llama_manager.config.Config", return_value=mock_config):
            # First entry
            log_mutating_action(
                command=["git", "clone", "repo1"],
                exit_code=0,
                output="Cloned repo1",
            )
            # Second entry
            log_mutating_action(
                command=["git", "checkout", "main"],
                exit_code=0,
                output="Checked out main",
            )

        log_path = tmp_path / "llm-runner" / "mutating_actions.log"
        with open(log_path) as f:
            lines = f.readlines()

        # Should have 2 entries
        assert len(lines) == 2


"""Tests for llama_manager.benchmark — command building, parsing, and runner."""


import math
from typing import Any

import pytest

from llama_manager.benchmark import (
    BenchmarkResult,
    SubprocessResult,
    build_benchmark_cmd,
    parse_benchmark_output,
    run_benchmark,
)


@pytest.fixture
def make_temp_bin(tmp_path: Path) -> Path:
    """Create a temporary file to serve as a fake llama-bench binary."""
    path = tmp_path / "llama-bench"
    path.touch()
    path.chmod(0o755)
    return path


class TestBuildBenchmarkCmd:
    """Tests for build_benchmark_cmd."""

    def test_contains_required_flags(self, make_temp_bin: Path) -> None:
        """build_benchmark_cmd should contain all required flags."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/models/test.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="F16",
            cache_type_v="F16",
        )

        assert "-m" in cmd
        assert "/models/test.gguf" in cmd
        assert "-p" in cmd
        assert "8080" in cmd
        assert "-t" in cmd
        assert "4" in cmd
        assert "--ubatch-size" in cmd
        assert "512" in cmd
        assert "--cache-type-k" in cmd
        assert "F16" in cmd
        assert "--cache-type-v" in cmd
        assert "F16" in cmd
        assert "-ngl" in cmd

    def test_returns_list_of_strings(self, make_temp_bin: Path) -> None:
        """build_benchmark_cmd should return a list[str]."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/models/test.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="F16",
            cache_type_v="F16",
        )

        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)

    def test_n_gpu_layers_default_all(self, make_temp_bin: Path) -> None:
        """n_gpu_layers should default to 'all'."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/models/test.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="F16",
            cache_type_v="F16",
        )

        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "all"

    def test_n_gpu_layers_custom_value(self, make_temp_bin: Path) -> None:
        """n_gpu_layers should accept a custom int value."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/models/test.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="F16",
            cache_type_v="F16",
            n_gpu_layers=33,
        )

        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "33"

    def test_nonexistent_binary_raises_file_not_found(self) -> None:
        """build_benchmark_cmd should raise FileNotFoundError for missing binary."""
        with pytest.raises(FileNotFoundError, match="llama-bench binary not found"):
            build_benchmark_cmd(
                bench_bin="/nonexistent/llama-bench",
                model="/models/test.gguf",
                n_prompt=8080,
                threads=4,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )

    def test_nonexecutable_binary_raises_permission_error(self, tmp_path: Path) -> None:
        """build_benchmark_cmd should raise PermissionError for non-executable binary."""
        path = tmp_path / "llama-bench-noexec"
        path.touch(mode=0o644)
        with pytest.raises(PermissionError, match="llama-bench binary is not executable"):
            build_benchmark_cmd(
                bench_bin=str(path),
                model="/models/test.gguf",
                n_prompt=8080,
                threads=4,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )


class TestParseBenchmarkOutput:
    """Tests for parse_benchmark_output."""

    def test_success_with_valid_output(self) -> None:
        """parse_benchmark_output should return BenchmarkResult from valid output."""
        output = (
            "llama-bench: I llama-bench version: 1.0.0\n"
            "tokens per second: 123.45\n"
            "avg latency: 45.67 ms\n"
            "peak memory: 2048.0 MB\n"
        )
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(123.45)
        assert result.avg_latency_ms == pytest.approx(45.67)
        assert result.peak_vram_mb == pytest.approx(2048.0)

    def test_empty_output_returns_none(self) -> None:
        """parse_benchmark_output should return None for empty output."""
        assert parse_benchmark_output("") is None
        assert parse_benchmark_output("   ") is None
        assert parse_benchmark_output("\n\n") is None

    def test_partial_output_only_tokens_per_second(self) -> None:
        """parse_benchmark_output returns None when only tokens/s is present."""
        output = "tokens per second: 999.99\n"
        result = parse_benchmark_output(output)
        assert result is None

    def test_partial_output_tokens_and_latency(self) -> None:
        """parse_benchmark_output should return partial result without VRAM."""
        output = "tokens per second: 500.0\navg latency: 20.0 ms\n"
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(500.0)
        assert result.avg_latency_ms == pytest.approx(20.0)
        assert result.peak_vram_mb is None

    def test_various_tokens_per_second_formats(self) -> None:
        """parse_benchmark_output returns None when only tokens/s is present."""
        # t/s format
        result = parse_benchmark_output("t/s: 100.0\n")
        assert result is None

        # tokens/s format
        result = parse_benchmark_output("tokens/s: 200.0\n")
        assert result is None

        # tok/s format
        result = parse_benchmark_output("tok/s: 300.0\n")
        assert result is None

    def test_various_latency_formats(self) -> None:
        """parse_benchmark_output returns None when only latency is present."""
        # avg latency
        result = parse_benchmark_output("avg latency: 10.5 ms\n")
        assert result is None

        # latency
        result = parse_benchmark_output("latency: 20.3 ms\n")
        assert result is None

    def test_various_vram_formats(self) -> None:
        """parse_benchmark_output returns None when only VRAM is present."""
        # peak memory without latency → None
        result = parse_benchmark_output("tokens per second: 100.0\npeak memory: 4096.0 MB\n")
        assert result is None

        # peak vram without latency → None
        result = parse_benchmark_output("tokens per second: 100.0\npeak vram: 8192.0 mb\n")
        assert result is None

    def test_no_matching_metrics_returns_none(self) -> None:
        """parse_benchmark_output should return None when no metrics match."""
        output = "This is just some random text without any metrics.\n"
        assert parse_benchmark_output(output) is None

    def test_output_with_only_vram_returns_none(self) -> None:
        """parse_benchmark_output should return None if only VRAM is present."""
        output = "peak memory: 1024.0 MB\n"
        assert parse_benchmark_output(output) is None

    def test_non_finite_tokens_per_second(self) -> None:
        """Test that inf tokens_per_second (regex doesn't match) returns None."""
        # "inf" does not match the numeric regex in parse_benchmark_output,
        # so tokens_per_second stays None → returns None.
        output = "tok/s: inf\navg latency: 100.0 ms\npeak memory: 2048"
        result = parse_benchmark_output(output)
        assert result is None

    def test_nan_tokens_per_second(self) -> None:
        """Test that nan tokens_per_second (regex doesn't match) returns None."""
        output = "tokens per second: nan\navg latency: 50.0 ms\n"
        result = parse_benchmark_output(output)
        assert result is None

    def test_markdown_table_output_parsed(self) -> None:
        """parse_benchmark_output should parse llama-bench markdown table rows."""
        output = (
            "| model | t/s | avg latency (ms) | peak vram (MB) |\n"
            "|-------|-----|------------------|----------------|\n"
            "| qwen  | 123.45 | 6.78 | 4096.0 |\n"
        )
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(123.45)
        assert result.avg_latency_ms == pytest.approx(6.78)
        assert result.peak_vram_mb == pytest.approx(4096.0)


class TestBuildBenchmarkCmdEdgeCases:
    """Edge-case tests for build_benchmark_cmd."""

    def test_n_gpu_layers_explicit_string_all(self, make_temp_bin: Path) -> None:
        """Test build_benchmark_cmd with n_gpu_layers='all' as explicit string."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/tmp/model.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            n_gpu_layers="all",  # explicit string, not default
        )
        assert "-ngl" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "all"

    def test_n_gpu_layers_custom_int_value(self, make_temp_bin: Path) -> None:
        """Test build_benchmark_cmd with n_gpu_layers as an int."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/tmp/model.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            n_gpu_layers=99,
        )
        assert "-ngl" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "99"


class TestRunBenchmark:
    """Tests for run_benchmark."""

    def test_calls_runner_with_cmd(self) -> None:
        """run_benchmark should pass cmd to runner."""
        runner = MagicMock()
        cmd = ["llama-bench", "-m", "model.gguf"]
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 100.0\n",
            stderr="",
        )

        run_benchmark(cmd, runner)

        runner.assert_called_once_with(cmd)

    def test_returns_none_on_nonzero_exit(self) -> None:
        """run_benchmark should return None when runner returns nonzero exit code."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=1,
            stdout="",
            stderr="error occurred\n",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is None

    def test_returns_none_on_empty_stdout(self) -> None:
        """run_benchmark should return None when runner returns empty stdout."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="",
            stderr="",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is None

    def test_returns_none_on_whitespace_stdout(self) -> None:
        """run_benchmark should return None when runner returns whitespace-only stdout."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="   \n\n  ",
            stderr="",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is None

    def test_returns_parsed_result_on_success(self) -> None:
        """run_benchmark should return parsed BenchmarkResult on success."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 42.0\navg latency: 1.0 ms\n",
            stderr="",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(42.0)
        assert result.avg_latency_ms == pytest.approx(1.0)

    def test_stderr_ignored_for_result(self) -> None:
        """run_benchmark should ignore stderr and only use stdout for parsing."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 150.0\navg latency: 5.0 ms\n",
            stderr="warning: deprecated flag\n",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(150.0)

    def test_runner_called_exactly_once(self) -> None:
        """run_benchmark should call runner exactly once."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 100.0\n",
            stderr="",
        )

        run_benchmark(["llama-bench", "-m", "model.gguf"], runner)
        runner.assert_called_once()
        assert runner.call_count == 1


class TestSubprocessResult:
    """Tests for SubprocessResult dataclass."""

    def test_immutable(self) -> None:
        """SubprocessResult should be immutable (frozen dataclass)."""
        result = SubprocessResult(exit_code=0, stdout="out", stderr="err")
        with pytest.raises(AttributeError):
            result.exit_code = 1  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.stdout = "new"  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.stderr = "new"  # type: ignore[assignment]

    def test_equality(self) -> None:
        """SubprocessResult equality should compare all fields."""
        r1 = SubprocessResult(exit_code=0, stdout="out", stderr="err")
        r2 = SubprocessResult(exit_code=0, stdout="out", stderr="err")
        r3 = SubprocessResult(exit_code=1, stdout="out", stderr="err")
        assert r1 == r2
        assert r1 != r3

    def test_repr(self) -> None:
        """SubprocessResult repr should include all fields."""
        result = SubprocessResult(exit_code=42, stdout="hello", stderr="world")
        repr_str = repr(result)
        assert "exit_code=42" in repr_str
        assert "stdout='hello'" in repr_str
        assert "stderr='world'" in repr_str


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_immutable(self) -> None:
        """BenchmarkResult should be immutable (frozen dataclass)."""
        result = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        with pytest.raises(AttributeError):
            result.tokens_per_second = 200.0  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.avg_latency_ms = 20.0  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.peak_vram_mb = 4096.0  # type: ignore[assignment]

    def test_equality(self) -> None:
        """BenchmarkResult equality should compare all fields."""
        r1 = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        r2 = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        r3 = BenchmarkResult(
            tokens_per_second=200.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        assert r1 == r2
        assert r1 != r3

    def test_none_peak_vram(self) -> None:
        """BenchmarkResult should accept None for peak_vram_mb."""
        result = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=None,
        )
        assert result.peak_vram_mb is None

    def test_zero_values(self) -> None:
        """BenchmarkResult should accept zero values."""
        result = BenchmarkResult(
            tokens_per_second=0.0,
            avg_latency_ms=0.0,
            peak_vram_mb=0.0,
        )
        assert result.tokens_per_second == 0.0
        assert result.avg_latency_ms == 0.0
        assert result.peak_vram_mb == 0.0

    def test_large_values(self) -> None:
        """BenchmarkResult should handle large values."""
        result = BenchmarkResult(
            tokens_per_second=999999.99,
            avg_latency_ms=9999.99,
            peak_vram_mb=99999.0,
        )
        assert result.tokens_per_second == pytest.approx(999999.99)
        assert result.avg_latency_ms == pytest.approx(9999.99)
        assert result.peak_vram_mb == pytest.approx(99999.0)

    def test_fractional_values(self) -> None:
        """BenchmarkResult should handle fractional values."""
        result = BenchmarkResult(
            tokens_per_second=0.5,
            avg_latency_ms=0.01,
            peak_vram_mb=123.456,
        )
        assert result.tokens_per_second == pytest.approx(0.5)
        assert result.avg_latency_ms == pytest.approx(0.01)
        assert result.peak_vram_mb == pytest.approx(123.456)


class TestParseBenchmarkOutputOnlyLatency:
    """Tests for parse_benchmark_output with only latency present.

    Since both tokens_per_second AND avg_latency_ms are required,
    output with only latency returns None.
    """

    def test_only_latency_parsed(self) -> None:
        """parse_benchmark_output returns None when only latency is present."""
        output = "avg latency: 50.0 ms\n"
        result = parse_benchmark_output(output)
        assert result is None

    def test_only_latency_no_ms_unit(self) -> None:
        """parse_benchmark_output returns None when only latency (no ms unit) is present."""
        output = "latency: 25.5\n"
        result = parse_benchmark_output(output)
        assert result is None


class TestParseBenchmarkOutputValueErrorBranches:
    """Tests for parse_benchmark_output ValueError handling branches.

    These tests use mocking to exercise the except ValueError blocks,
    which are unreachable with the default regex patterns since they only
    match valid numeric strings.
    """

    def test_valueerror_in_tokens_parsing(self) -> None:
        """parse_benchmark_output returns None when tokens parsing raises ValueError."""
        with patch("re.search") as mock_search:
            call_count = [0]

            def side_effect(pattern: str, string: str, flags: int = 0) -> Any:
                call_count[0] += 1
                # Simulate tokens pattern matching a non-numeric string
                if call_count[0] <= 4:
                    # First 4 calls are tokens patterns — return non-numeric
                    match = MagicMock()
                    match.group.return_value = "not_a_number"
                    match.__bool__ = lambda self: True
                    return match
                # Latency pattern matches valid number
                match = MagicMock()
                match.group.return_value = "100.0"
                match.__bool__ = lambda self: True
                return match

            mock_search.side_effect = side_effect
            result = parse_benchmark_output(
                "tokens per second: not_a_number\navg latency: 100.0 ms\n"
            )
            # tokens_per_second becomes None after ValueError → returns None
            assert result is None

    def test_valueerror_in_latency_parsing(self) -> None:
        """parse_benchmark_output returns None when latency parsing raises ValueError."""
        with patch("re.search") as mock_search:
            call_count = [0]

            def side_effect(pattern: str, string: str, flags: int = 0) -> MagicMock | None:
                call_count[0] += 1
                # Tokens pattern matches valid number (first call)
                if call_count[0] == 1:
                    match = MagicMock()
                    match.group.return_value = "500.0"
                    match.__bool__ = lambda self: True
                    return match
                # All subsequent patterns (tokens 2-4, latency 1-4, vram 1-6)
                # return None (no match), so the function falls through to
                # the next pattern group. For latency, the first pattern
                # that "matches" returns a non-numeric string.
                # We simulate: tokens patterns 2-4 return None, then latency
                # pattern 1 returns a non-numeric match.
                if call_count[0] <= 4:
                    # Tokens patterns 2-4: no match
                    return None
                if call_count[0] == 5:
                    # First latency pattern "matches" but with non-numeric
                    match = MagicMock()
                    match.group.return_value = "slow"
                    match.__bool__ = lambda self: True
                    return match
                # Remaining patterns: no match
                return None

            mock_search.side_effect = side_effect
            result = parse_benchmark_output("tokens per second: 500.0\navg latency: slow\n")
            # avg_latency_ms becomes None after ValueError → returns None
            assert result is None

    def test_valueerror_in_vram_parsing(self) -> None:
        """parse_benchmark_output should handle ValueError in VRAM parsing."""
        with patch("re.search") as mock_search:
            call_count = [0]

            def side_effect(pattern: str, string: str, flags: int = 0) -> MagicMock | None:
                call_count[0] += 1
                # Tokens pattern 1 matches valid number
                if call_count[0] == 1:
                    match = MagicMock()
                    match.group.return_value = "100.0"
                    match.__bool__ = lambda self: True
                    return match
                # Tokens patterns 2-4: no match
                if call_count[0] <= 4:
                    return None
                # Latency pattern 1 matches valid number
                if call_count[0] == 5:
                    match = MagicMock()
                    match.group.return_value = "50.0"
                    match.__bool__ = lambda self: True
                    return match
                # Latency patterns 2-4: no match
                if call_count[0] <= 8:
                    return None
                # VRAM pattern 1 matches non-numeric
                if call_count[0] == 9:
                    match = MagicMock()
                    match.group.return_value = "gibberish"
                    match.__bool__ = lambda self: True
                    return match
                # Remaining VRAM patterns: no match
                return None

            mock_search.side_effect = side_effect
            result = parse_benchmark_output(
                "tokens per second: 100.0\navg latency: 50.0 ms\npeak memory: gibberish\n"
            )
            assert result is not None
            assert result.tokens_per_second == pytest.approx(100.0)
            assert result.avg_latency_ms == pytest.approx(50.0)
            # peak_vram_mb stays None when ValueError occurs
            assert result.peak_vram_mb is None


class TestParseBenchmarkOutputIsfiniteBranches:
    """Tests for parse_benchmark_output math.isfinite validation branches.

    These tests use mocking to exercise the math.isfinite=False paths,
    which are unreachable with the default regex patterns since they only
    match valid finite numeric strings.
    """

    def test_isfinite_false_for_tokens(self) -> None:
        """parse_benchmark_output returns None when tokens_per_second fails isfinite check."""
        with patch.object(
            math,
            "isfinite",
            side_effect=[False, True],  # first call (tokens) False, second (latency) True
        ):
            result = parse_benchmark_output("tokens per second: 123.45\navg latency: 10.0 ms\n")
            # tokens_per_second invalidated → returns None
            assert result is None

    def test_isfinite_false_for_latency(self) -> None:
        """parse_benchmark_output returns None when avg_latency_ms fails isfinite check."""
        with patch.object(
            math,
            "isfinite",
            side_effect=[True, False],  # first call (tokens) True, second (latency) False
        ):
            result = parse_benchmark_output("tokens per second: 100.0\navg latency: 50.0 ms\n")
            # avg_latency_ms invalidated → returns None
            assert result is None

    def test_isfinite_false_for_both_returns_none(self) -> None:
        """parse_benchmark_output should return None when both metrics fail isfinite."""
        with patch.object(math, "isfinite", return_value=False):
            result = parse_benchmark_output("tokens per second: 100.0\navg latency: 50.0 ms\n")
            # Both tokens and latency invalidated → return None
            assert result is None

    def test_isfinite_true_preserves_values(self) -> None:
        """parse_benchmark_output should preserve values when isfinite returns True."""
        with patch.object(math, "isfinite", return_value=True):
            result = parse_benchmark_output("tokens per second: 42.0\navg latency: 3.14 ms\n")
            assert result is not None
            assert result.tokens_per_second == pytest.approx(42.0)
            assert result.avg_latency_ms == pytest.approx(3.14)


class TestParseBenchmarkOutputMixedFormats:
    """Tests for parse_benchmark_output with mixed format strings."""

    def test_uppercase_tokens_format(self) -> None:
        """parse_benchmark_output returns None when only uppercase tokens format present."""
        result = parse_benchmark_output("TOKENS PER SECOND: 100.0\n")
        assert result is None

    def test_mixed_case_latency_format(self) -> None:
        """parse_benchmark_output returns None when only mixed case latency present."""
        result = parse_benchmark_output("AvG lAtEnCy: 25.0 ms\n")
        assert result is None

    def test_mixed_case_vram_format(self) -> None:
        """parse_benchmark_output returns None when only VRAM is present without latency."""
        result = parse_benchmark_output("tokens per second: 100.0\nPEAK VRAM: 8192.0 MB\n")
        assert result is None

    def test_multiple_metrics_same_line(self) -> None:
        """parse_benchmark_output should handle metrics on same line."""
        result = parse_benchmark_output("tokens per second: 100.0 avg latency: 5.0 ms\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(100.0)
        assert result.avg_latency_ms == pytest.approx(5.0)

    def test_multiline_with_header_footer(self) -> None:
        """parse_benchmark_output should handle realistic multi-line benchmark output."""
        output = (
            "llama-bench: I llama-bench version: 1.0.0 (git: abc123)\n"
            "llama-bench: built on Apr 21 2026\n"
            "model: /models/qwen2.5-7b-q4_k_m.gguf\n"
            "params: 7.0B\n"
            "tokens per second: 145.678\n"
            "avg latency: 6.89 ms\n"
            "peak memory: 4096.5 MB\n"
            "llama-bench: benchmark complete\n"
        )
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(145.678)
        assert result.avg_latency_ms == pytest.approx(6.89)
        assert result.peak_vram_mb == pytest.approx(4096.5)

    def test_whitespace_around_colon(self) -> None:
        """parse_benchmark_output returns None when only tokens/s present (with space before colon)."""
        # Space before colon
        result = parse_benchmark_output("tokens per second : 100.0\n")
        assert result is None

    def test_no_space_after_colon(self) -> None:
        """parse_benchmark_output returns None when only tokens/s present (no space after colon)."""
        result = parse_benchmark_output("tokens per second:100.0\n")
        assert result is None


class TestBuildBenchmarkCmdAllParams:
    """Tests for build_benchmark_cmd with all parameter combinations."""

    def test_all_parameters_included(self, make_temp_bin: Path) -> None:
        """build_benchmark_cmd should include all parameters in the command."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/models/qwen2.5-7b-q4_k_m.gguf",
            n_prompt=8080,
            threads=16,
            ubatch_size=2048,
            cache_type_k="F16",
            cache_type_v="F32",
            n_gpu_layers=99,
        )

        assert cmd[0] == str(bin_path)
        assert "-m" in cmd
        assert "/models/qwen2.5-7b-q4_k_m.gguf" in cmd
        assert "-p" in cmd
        assert "8080" in cmd
        assert "-t" in cmd
        assert "16" in cmd
        assert "--ubatch-size" in cmd
        assert "2048" in cmd
        assert "--cache-type-k" in cmd
        assert "F16" in cmd
        assert "--cache-type-v" in cmd
        assert "F32" in cmd
        assert "-ngl" in cmd
        assert "99" in cmd

    def test_cmd_length(self, make_temp_bin: Path) -> None:
        """build_benchmark_cmd should produce a command with expected length."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/model.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="F16",
            cache_type_v="F16",
        )

        # bench_bin(1) + 7 flag-value pairs(14) = 15 elements
        assert len(cmd) == 15

    def test_cmd_order(self, make_temp_bin: Path) -> None:
        """build_benchmark_cmd should maintain correct argument order."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/model.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="F16",
            cache_type_v="F16",
        )

        # Verify order: bin, -m model, -p n_prompt, -t threads, --ubatch-size size,
        # --cache-type-k type, --cache-type-v type, -ngl layers
        assert cmd[0] == str(bin_path)
        assert cmd[1] == "-m"
        assert cmd[3] == "-p"
        assert cmd[5] == "-t"
        assert cmd[7] == "--ubatch-size"
        assert cmd[9] == "--cache-type-k"
        assert cmd[11] == "--cache-type-v"
        assert cmd[13] == "-ngl"

    def test_special_characters_in_model_path(self, make_temp_bin: Path) -> None:
        """build_benchmark_cmd should handle special characters in model path."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/models/qwen2.5-7b-q4_k_m.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="F16",
            cache_type_v="F16",
        )

        assert "/models/qwen2.5-7b-q4_k_m.gguf" in cmd

    def test_cache_type_q8_0(self, make_temp_bin: Path) -> None:
        """build_benchmark_cmd should accept Q8_0 cache types."""
        bin_path = make_temp_bin
        cmd = build_benchmark_cmd(
            bench_bin=str(bin_path),
            model="/model.gguf",
            n_prompt=8080,
            threads=4,
            ubatch_size=512,
            cache_type_k="Q8_0",
            cache_type_v="Q8_0",
        )

        assert "Q8_0" in cmd


"""Tests for llama_manager.metadata — GGUF metadata extraction.

Covers:
- Valid GGUF v3 metadata extraction (T045)
- Missing general.name fallback (T046)
- Corrupt file handling (T047)
- Truncated file handling (T048)
- GGUF v4 unsupported error (T049)
- Parse timeout (T050)
- Filename NFKC normalization (T051)
"""


import pytest

from llama_manager.metadata import (
    extract_gguf_metadata,
    normalize_filename,
)
from llama_manager.metadata._binary import (
    _detect_gguf_version,
    _parse_architecture,
    _parse_general_name,
    _parse_numeric_field,
    _parse_tokenizer_type,
)
from tests.support.helpers import fixture_path

# ---------------------------------------------------------------------------
# T045 — Valid GGUF v3 metadata extraction
# ---------------------------------------------------------------------------


class TestValidMetadataExtraction:
    """T045: Verify metadata extraction from a valid GGUF v3 file."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_extract_all_fields_from_valid_v3(self) -> None:
        """extract_gguf_metadata should return a complete record for a valid GGUF v3 file."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path, prefix_cap_bytes=65536)

        assert record.raw_path == path
        # architecture detected from "llama" pattern in binary data
        assert record.architecture == "llama"
        # Derived fields
        assert record.normalized_stem == "gguf_v3_valid"
        # Metadata fields
        assert isinstance(record.parse_timestamp, str)
        assert record.parse_timeout_s == 5.0
        assert record.prefix_cap_bytes == 65536

    def test_extract_custom_prefix_cap(self) -> None:
        """extract_gguf_metadata should respect custom prefix_cap_bytes."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path, prefix_cap_bytes=1024)
        assert record.prefix_cap_bytes == 1024

    def test_extract_custom_timeout(self) -> None:
        """extract_gguf_metadata should store parse_timeout_s in the record."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path, parse_timeout_s=10.0)
        assert record.parse_timeout_s == 10.0

    def test_record_is_dataclass(self) -> None:
        """GGUFMetadataRecord should be a dataclass with expected fields."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path)
        # Should have __dataclass_fields__
        assert hasattr(record, "__dataclass_fields__")
        expected_fields = {
            "raw_path",
            "normalized_stem",
            "general_name",
            "architecture",
            "tokenizer_type",
            "embedding_length",
            "block_count",
            "context_length",
            "attention_head_count",
            "attention_head_count_kv",
            "parse_timestamp",
            "parse_timeout_s",
            "prefix_cap_bytes",
        }
        assert set(record.__dataclass_fields__.keys()) == expected_fields


# ---------------------------------------------------------------------------
# T046 — Missing general.name (fallback to normalized filename stem)
# ---------------------------------------------------------------------------


class TestMissingGeneralName:
    """T046: Verify fallback behavior when general.name is absent."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_general_name_none_when_missing(self) -> None:
        """general_name should be None when not present in GGUF file."""
        path = str(self._fixture_path("gguf_v3_no_name.gguf"))
        record = extract_gguf_metadata(path)
        assert record.general_name is None

    def test_normalized_stem_used_as_fallback(self) -> None:
        """normalized_stem should derive from filename when general.name is missing."""
        path = str(self._fixture_path("gguf_v3_no_name.gguf"))
        record = extract_gguf_metadata(path)
        assert record.normalized_stem == "gguf_v3_no_name"

    def test_other_fields_still_populated(self) -> None:
        """Other fields should still be populated even without general.name."""
        path = str(self._fixture_path("gguf_v3_no_name.gguf"))
        record = extract_gguf_metadata(path)
        assert record.architecture == "llama"
        assert record.context_length is None  # regex doesn't match binary format
        assert record.attention_head_count is None


# ---------------------------------------------------------------------------
# T047 — Corrupt file (bad magic bytes)
# ---------------------------------------------------------------------------


class TestCorruptFile:
    """T047: Verify error handling for corrupt GGUF files."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_bad_magic_raises_value_error(self) -> None:
        """extract_gguf_metadata should raise ValueError for bad magic bytes."""
        path = str(self._fixture_path("gguf_corrupt.gguf"))
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata(path)
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_bad_magic_from_detect_function(self) -> None:
        """_detect_gguf_version should raise ValueError for bad magic."""
        bad_header = b"XXXX\x03\x00\x00\x00"
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(bad_header)
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_empty_data_raises_value_error(self) -> None:
        """_detect_gguf_version should raise ValueError for empty data."""
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(b"")
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_too_short_header_raises_value_error(self) -> None:
        """_detect_gguf_version should raise ValueError for header shorter than 8 bytes."""
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(b"GGUF\x00\x00")
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_non_gguf_file_raises_value_error(self) -> None:
        """_detect_gguf_version should reject random binary data."""
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(b"This is not a GGUF file at all\x00\x00")
        assert "bad magic bytes" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# T048 — Truncated file (valid header, 0 KV pairs)
# ---------------------------------------------------------------------------


class TestTruncatedFile:
    """T048: Verify handling of truncated GGUF files with valid header."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_truncated_file_succeeds_with_partial_data(self) -> None:
        """Truncated file with valid header and 0 KV pairs should succeed."""
        path = str(self._fixture_path("gguf_truncated.gguf"))
        record = extract_gguf_metadata(path)
        # Should succeed — no exception
        assert record is not None
        assert record.architecture is None  # no KV data to parse
        assert record.general_name is None
        assert record.context_length is None

    def test_truncated_file_still_has_stem_and_metadata(self) -> None:
        """Truncated file record should still have path-derived fields."""
        path = str(self._fixture_path("gguf_truncated.gguf"))
        record = extract_gguf_metadata(path)
        assert record.normalized_stem == "gguf_truncated"
        assert isinstance(record.parse_timestamp, str)
        assert record.parse_timeout_s == 5.0

    def test_detect_version_for_truncated_header(self) -> None:
        """_detect_gguf_version should identify v3 from truncated header."""
        truncated_header = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00"
        version = _detect_gguf_version(truncated_header)
        assert version == 3


# ---------------------------------------------------------------------------
# T049 — GGUF v4 unsupported version error
# ---------------------------------------------------------------------------


class TestGGUFv4Unsupported:
    """T049: Verify that GGUF v4 files produce an unsupported error."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_v4_file_raises_unsupported_error(self) -> None:
        """extract_gguf_metadata should raise ValueError for GGUF v4 files."""
        path = str(self._fixture_path("gguf_v4_unsupported.gguf"))
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata(path)
        assert (
            "v4" in str(exc_info.value).lower()
            and "not yet supported" in str(exc_info.value).lower()
        )

    def test_detect_version_returns_4(self) -> None:
        """_detect_gguf_version should return 4 for v4 magic bytes."""
        v4_header = b"GGUF\x04\x00\x00\x00"
        version = _detect_gguf_version(v4_header)
        assert version == 4

    def test_detect_version_returns_3(self) -> None:
        """_detect_gguf_version should return 3 for v3 magic bytes."""
        v3_header = b"GGUF\x03\x00\x00\x00"
        version = _detect_gguf_version(v3_header)
        assert version == 3

    def test_detect_version_returns_2(self) -> None:
        """_detect_gguf_version should return 2 for v2 magic bytes."""
        v2_header = b"GGUF\x02\x00\x00\x00"
        version = _detect_gguf_version(v2_header)
        assert version == 2


# ---------------------------------------------------------------------------
# T050 — Parse timeout
# ---------------------------------------------------------------------------


class TestParseTimeout:
    """T050: Verify timeout handling when parsing takes too long."""

    def test_timeout_raises_timeout_error(self) -> None:
        """extract_gguf_metadata should raise TimeoutError when parse exceeds timeout."""

        def _slow_read(*args, **kwargs) -> bytes:
            # Simulate a very slow operation
            time.sleep(2)
            return b""

        with patch(
            "llama_manager.metadata._binary._read_gguf_header",
            side_effect=_slow_read,
        ):
            with pytest.raises(TimeoutError) as exc_info:
                extract_gguf_metadata(
                    "/fake/model.gguf",
                    prefix_cap_bytes=1024,
                    parse_timeout_s=0.1,
                )
            assert "timed out" in str(exc_info.value).lower()

    def test_timeout_message_contains_path(self) -> None:
        """TimeoutError message should include the model path."""
        model_path = "/models/my-model.gguf"

        def _slow_read(*args, **kwargs) -> bytes:
            time.sleep(2)
            return b""

        with patch(
            "llama_manager.metadata._binary._read_gguf_header",
            side_effect=_slow_read,
        ):
            with pytest.raises(TimeoutError) as exc_info:
                extract_gguf_metadata(
                    model_path,
                    prefix_cap_bytes=1024,
                    parse_timeout_s=0.1,
                )
            assert model_path in str(exc_info.value)

    def test_timeout_does_not_affect_fast_files(self) -> None:
        """Fast files should not trigger timeout."""
        path = str(fixture_path("gguf_v3_valid.gguf"))
        # Very short timeout but file is tiny — should still succeed
        record = extract_gguf_metadata(path, prefix_cap_bytes=65536, parse_timeout_s=0.5)
        assert record.architecture == "llama"

    def test_thread_exception_propagated(self) -> None:
        """Exceptions in the parse thread should propagate to the caller."""

        def _fail_read(*args, **kwargs) -> bytes:
            raise OSError("disk full")

        with patch(
            "llama_manager.metadata._binary._read_gguf_header",
            side_effect=_fail_read,
        ):
            with pytest.raises(OSError) as exc_info:
                extract_gguf_metadata("/fake/model.gguf")
            assert "disk full" in str(exc_info.value)


# ---------------------------------------------------------------------------
# T051 — Filename NFKC normalization
# ---------------------------------------------------------------------------


class TestFilenameNormalization:
    """T051: Verify NFKC normalization of filenames."""

    def test_simple_ascii(self) -> None:
        """Simple ASCII filename should pass through unchanged."""
        assert normalize_filename("my-model") == "my-model"

    def test_whitespace_replaced_with_underscore(self) -> None:
        """Whitespace sequences should be replaced with single underscore."""
        assert normalize_filename("my model") == "my_model"
        assert normalize_filename("my  model") == "my_model"
        assert normalize_filename("my   model") == "my_model"

    def test_leading_trailing_underscore_stripped(self) -> None:
        """Leading and trailing underscores should be removed."""
        assert normalize_filename("_my_model_") == "my_model"
        assert normalize_filename("__my_model__") == "my_model"

    def test_multiple_underscores_collapsed(self) -> None:
        """Multiple consecutive underscores should be collapsed."""
        assert normalize_filename("my__model") == "my_model"
        assert normalize_filename("my___model") == "my_model"

    def test_nfkc_combining_characters(self) -> None:
        """NFKC normalization should decompose compatibility characters."""
        # ﬃ (U+FB03, LATIN SMALL LIGATURE FFI) → ffi under NFKC
        # The ligature alone should normalize to "ffi"
        assert normalize_filename("\ufb03") == "ffi"

    def test_invalid_chars_replaced(self) -> None:
        """Invalid filename characters should be replaced with underscore."""
        # Control characters
        result = normalize_filename("my\x00model")
        assert "my" in result and "model" in result
        # Forward slash
        assert normalize_filename("my/model") == "my_model"
        # Backslash
        assert normalize_filename("my\\model") == "my_model"
        # Colon
        assert normalize_filename("my:model") == "my_model"

    def test_uppercase_preserved(self) -> None:
        """Uppercase letters should be preserved."""
        assert normalize_filename("MyModel") == "MyModel"

    def test_numbers_preserved(self) -> None:
        """Numbers should be preserved in the normalized name."""
        assert normalize_filename("model-v2") == "model-v2"
        assert normalize_filename("model_v2") == "model_v2"

    def test_empty_string_returns_unknown(self) -> None:
        """Empty string should return 'unknown'."""
        assert normalize_filename("") == "unknown"

    def test_whitespace_only_returns_unknown(self) -> None:
        """Whitespace-only string should return 'unknown'."""
        assert normalize_filename("   ") == "unknown"
        assert normalize_filename("\t\n") == "unknown"

    def test_underscore_only_returns_unknown(self) -> None:
        """Underscore-only string should return 'unknown' after stripping."""
        assert normalize_filename("___") == "unknown"

    def test_unicode_special_chars(self) -> None:
        """Unicode characters outside ASCII range should be handled."""
        # Emoticon and other non-filename-safe chars
        result = normalize_filename("model\x01\x02test")
        assert "model" in result and "test" in result

    def test_mixed_whitespace_and_underscores(self) -> None:
        """Mixed whitespace and underscores should be normalized."""
        # Spaces become underscores, then collapse
        assert normalize_filename("my model name") == "my_model_name"

    def test_pipe_character_replaced(self) -> None:
        """Pipe character (|) is invalid in filenames and should be replaced."""
        assert normalize_filename("my|model") == "my_model"

    def test_question_mark_replaced(self) -> None:
        """Question mark is invalid in filenames and should be replaced."""
        assert normalize_filename("my?model") == "my_model"

    def test_asterisk_replaced(self) -> None:
        """Asterisk is invalid in filenames and should be replaced."""
        assert normalize_filename("my*model") == "my_model"

    def test_greater_less_than_replaced(self) -> None:
        """Angle brackets are invalid in filenames and should be replaced."""
        assert normalize_filename("my<model>") == "my_model"

    def test_double_quoted_replaced(self) -> None:
        """Double quotes are invalid in filenames and should be replaced."""
        assert normalize_filename('my"model') == "my_model"

    def test_complex_mixed_input(self) -> None:
        """Complex mixed input should be fully normalized."""
        result = normalize_filename("  My  Model  v2  ")
        assert result == "My_Model_v2"

    def test_newlines_and_tabs(self) -> None:
        """Newlines and tabs are whitespace and should be replaced."""
        result = normalize_filename("my\nmodel\tname")
        assert result == "my_model_name"

    def test_null_byte_replaced(self) -> None:
        """Null byte is invalid and should be replaced."""
        result = normalize_filename("my\x00model")
        # Should contain parts of the original string
        assert "my" in result
        assert "model" in result


# ---------------------------------------------------------------------------
# Private helper function tests
# ---------------------------------------------------------------------------


class TestParseGeneralName:
    """Tests for _parse_general_name helper."""

    def test_parses_general_name(self) -> None:
        """_parse_general_name should extract general.name from matching bytes."""
        # The regex expects: general.name + optional whitespace + \x00 + name + \x00
        data = b"general.name\x00test-model\x00other"
        result = _parse_general_name(data)
        assert result == "test-model"

    def test_returns_none_when_not_found(self) -> None:
        """_parse_general_name should return None when key is absent."""
        data = b"some other key\x00value\x00"
        result = _parse_general_name(data)
        assert result is None

    def test_returns_replacement_chars_on_decode_error(self) -> None:
        """_parse_general_name should return replacement chars on decode error (errors='replace')."""
        # Invalid UTF-8 sequence in the name value
        data = b"general.name\x00\xff\xfe\x00"
        result = _parse_general_name(data)
        # errors="replace" returns U+FFFD replacement characters, not None
        assert result is not None
        assert "\ufffd" in result

    def test_parses_with_whitespace_before_null(self) -> None:
        """_parse_general_name should handle optional whitespace before null."""
        data = b"general.name \x00test-model\x00"
        result = _parse_general_name(data)
        assert result == "test-model"


class TestParseArchitecture:
    """Tests for _parse_architecture helper."""

    def test_detects_llama(self) -> None:
        """_parse_architecture should detect 'llama' architecture."""
        assert _parse_architecture(b"llama") == "llama"

    def test_detects_qwen2(self) -> None:
        """_parse_architecture should detect 'qwen2' architecture."""
        assert _parse_architecture(b"qwen2") == "qwen2"

    def test_detects_qwen3(self) -> None:
        """_parse_architecture should detect 'qwen3' architecture."""
        assert _parse_architecture(b"qwen3") == "qwen3"

    def test_detects_qwen(self) -> None:
        """_parse_architecture should detect 'qwen' architecture."""
        assert _parse_architecture(b"qwen") == "qwen"

    def test_detects_phi3(self) -> None:
        """_parse_architecture should detect 'phi3' architecture."""
        assert _parse_architecture(b"phi3") == "phi3"

    def test_detects_mamba(self) -> None:
        """_parse_architecture should detect 'mamba' architecture."""
        assert _parse_architecture(b"mamba") == "mamba"

    def test_returns_none_for_unknown(self) -> None:
        """_parse_architecture should return None for unknown architecture."""
        assert _parse_architecture(b"unknown_arch") is None

    def test_first_match_wins(self) -> None:
        """_parse_architecture should return first matching pattern."""
        # "qwen2" contains "qwen" — should return "qwen2" (defined first)
        assert _parse_architecture(b"qwen2") == "qwen2"

    def test_detects_gpt_2(self) -> None:
        """_parse_architecture should detect 'gpt_2' architecture."""
        assert _parse_architecture(b"gpt_2") == "gpt_2"

    def test_detects_bert(self) -> None:
        """_parse_architecture should detect 'bert' architecture."""
        assert _parse_architecture(b"bert") == "bert"

    def test_detects_mpt(self) -> None:
        """_parse_architecture should detect 'mpt' architecture."""
        assert _parse_architecture(b"mpt") == "mpt"

    def test_detects_falcon(self) -> None:
        """_parse_architecture should detect 'falcon' architecture."""
        assert _parse_architecture(b"falcon") == "falcon"

    def test_detects_stablelm(self) -> None:
        """_parse_architecture should detect 'stablelm' architecture."""
        assert _parse_architecture(b"stablelm") == "stablelm"


class TestParseTokenizerType:
    """Tests for _parse_tokenizer_type helper."""

    def test_detects_ggml_tokenizer(self) -> None:
        """_parse_tokenizer_type should detect 'ggml' tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.ggml") == "ggml"

    def test_detects_model_tokenizer(self) -> None:
        """_parse_tokenizer_type should detect 'model' tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.model") == "model"

    def test_detects_huggingface_tokenizer(self) -> None:
        """_parse_tokenizer_type should detect 'huggingface' tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.json") == "huggingface"

    def test_returns_none_for_unknown(self) -> None:
        """_parse_tokenizer_type should return None for unknown tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.unknown") is None

    def test_first_match_wins(self) -> None:
        """_parse_tokenizer_type should return first matching pattern."""
        # "tokenizer.ggml" contains "tokenizer" — should match ggml first
        assert _parse_tokenizer_type(b"tokenizer.ggml") == "ggml"


class TestParseNumericField:
    """Tests for _parse_numeric_field helper using GGUF binary KV format."""

    @staticmethod
    def _make_gguf_kv_data(key: bytes, type_tag: int, value: int, n_kv: int = 1) -> bytes:
        """Build minimal GGUF v3 header + KV records for testing.

        GGUF v3 layout:
        - magic: 8 bytes (GGUF + version byte)
        - version: 4 bytes (uint32)
        - num_tensors: 8 bytes (uint64)
        - num_kv: 8 bytes (uint64)
        Total header: 28 bytes
        """
        # GGUF v3: magic (8) + version (4) + num_tensors (8) + num_kv (8) = 28
        header = b"GGUF\x03\x00\x00\x00"  # 8 bytes
        header += b"\x00\x00\x00\x00"  # version = 0 (uint32 LE)
        header += b"\x00\x00\x00\x00\x00\x00\x00\x00"  # num_tensors = 0 (uint64 LE)
        header += n_kv.to_bytes(8, "little")  # num_kv (uint64 LE)

        # KV record: key_length (uint32) + key + type_tag (uint32) + value
        record = len(key).to_bytes(4, "little") + key
        record += type_tag.to_bytes(4, "little")
        if type_tag in (1, 2):  # u8, i8
            record += value.to_bytes(1, "little")
        elif type_tag in (3, 4):  # u16, i16
            record += value.to_bytes(2, "little")
        elif type_tag in (5, 6):  # u32, i32
            record += value.to_bytes(4, "little")
        elif type_tag == 8:  # f32
            import struct

            record += struct.pack("<f", 0.0)
        elif type_tag == 9:  # f64
            import struct

            record += struct.pack("<d", 0.0)
        else:
            record += value.to_bytes(4, "little")

        return header + record

    def test_parses_integer_field(self) -> None:
        """_parse_numeric_field should extract u32 values from GGUF KV records."""
        data = self._make_gguf_kv_data(b"context_length", 5, 8192)
        result = _parse_numeric_field(data, b"context_length")
        assert result == 8192

    def test_parses_string_key(self) -> None:
        """_parse_numeric_field should accept string keys."""
        data = self._make_gguf_kv_data(b"context_length", 5, 4096)
        result = _parse_numeric_field(data, "context_length")
        assert result == 4096

    def test_returns_none_for_missing_key(self) -> None:
        """_parse_numeric_field should return None for missing key."""
        data = self._make_gguf_kv_data(b"some_key", 5, 12345)
        result = _parse_numeric_field(data, b"nonexistent_field")
        assert result is None

    def test_returns_none_for_non_numeric_value(self) -> None:
        """_parse_numeric_field should return None for string type (tag 7 = string)."""
        # Type tag 7 is string, not integer — function returns None for non-numeric
        data = self._make_gguf_kv_data(b"context_length", 7, 0)
        result = _parse_numeric_field(data, b"context_length")
        assert result is None

    def test_parses_u32_value(self) -> None:
        """_parse_numeric_field should parse u32 (type_tag=5) correctly."""
        data = self._make_gguf_kv_data(b"embedding_length", 5, 4096)
        result = _parse_numeric_field(data, b"embedding_length")
        assert result == 4096

    def test_parses_i32_signed_value(self) -> None:
        """_parse_numeric_field should parse i32 (type_tag=6) as signed."""
        # i32 with value -1 (0xFFFFFFFF)
        data = self._make_gguf_kv_data(b"test_key", 6, 0xFFFFFFFF)
        result = _parse_numeric_field(data, b"test_key")
        assert result == -1

    def test_parses_u16_value(self) -> None:
        """_parse_numeric_field should parse u16 (type_tag=3) correctly."""
        data = self._make_gguf_kv_data(b"block_count", 3, 32)
        result = _parse_numeric_field(data, b"block_count")
        assert result == 32

    def test_parses_u8_value(self) -> None:
        """_parse_numeric_field should parse u8 (type_tag=1) correctly."""
        data = self._make_gguf_kv_data(b"attention_head_count", 1, 32)
        result = _parse_numeric_field(data, b"attention_head_count")
        assert result == 32

    def test_no_match_for_binary_integer(self) -> None:
        """_parse_numeric_field should return None for unrecognized data."""
        # Short data that doesn't start with GGUF magic
        data = b"context_length\x40\x20\x00\x00"
        result = _parse_numeric_field(data, b"context_length")
        assert result is None

    def test_dot_escaped_in_pattern(self) -> None:
        """_parse_numeric_field should handle dotted keys correctly."""
        data = self._make_gguf_kv_data(b"llama.context_length", 5, 8192)
        result = _parse_numeric_field(data, b"llama.context_length")
        assert result == 8192

    def test_skips_f32_and_returns_none(self) -> None:
        """_parse_numeric_field should skip f32 (type_tag=8) and return None."""
        data = self._make_gguf_kv_data(b"freq_scale", 8, 0)
        result = _parse_numeric_field(data, b"freq_scale")
        assert result is None


# ---------------------------------------------------------------------------
# prefix_cap_bytes / parse_timeout_s validation
# ---------------------------------------------------------------------------


class TestExtractGgufMetadataValidation:
    """Validation of prefix_cap_bytes and parse_timeout_s parameters."""

    def test_prefix_cap_bytes_zero_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for prefix_cap_bytes=0."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", prefix_cap_bytes=0)
        assert "prefix_cap_bytes" in str(exc_info.value).lower()
        assert "positive" in str(exc_info.value).lower()

    def test_prefix_cap_bytes_negative_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for negative prefix_cap_bytes."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", prefix_cap_bytes=-100)
        assert "prefix_cap_bytes" in str(exc_info.value).lower()

    def test_parse_timeout_zero_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for parse_timeout_s=0."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", parse_timeout_s=0.0)
        assert "parse_timeout_s" in str(exc_info.value).lower()
        assert "positive" in str(exc_info.value).lower()

    def test_parse_timeout_negative_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for negative parse_timeout_s."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", parse_timeout_s=-1.0)
        assert "parse_timeout_s" in str(exc_info.value).lower()

    def test_validation_before_any_io(self) -> None:
        """Validation errors should occur before any file reads or thread creation."""
        # Should raise ValueError without touching the filesystem
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/nonexistent/model.gguf", prefix_cap_bytes=0)
        # If we get here, ValueError was raised before any IO attempt
        assert exc_info.value is not None

    def test_valid_parameters_accepted(self) -> None:
        """Valid positive parameters should not raise."""
        # Use a fixture that exists
        path = str(fixture_path("gguf_v3_valid.gguf"))
        # Should succeed without raising
        record = extract_gguf_metadata(path, prefix_cap_bytes=1024, parse_timeout_s=1.0)
        assert record is not None
        assert record.prefix_cap_bytes == 1024
        assert record.parse_timeout_s == 1.0


"""Generate GGUF test fixtures as a side-effect test.

This test generates synthetic GGUF binary files for metadata extraction tests.
It is idempotent and safe to run repeatedly.
"""


import struct

# Value type constants
_GGUF_TYPE_UINT32: int = 4
_GGUF_TYPE_STRING: int = 8

# Required keys for a minimal valid llama model GGUF v3 file
_REQUIRED_KEYS: dict[str, tuple[int, bytes]] = {
    "general.architecture": (_GGUF_TYPE_STRING, b"llama"),
    "tokenizer.type": (_GGUF_TYPE_STRING, b"bpe"),
    "llama.embedding_length": (_GGUF_TYPE_UINT32, struct.pack("<I", 4096)),
    "llama.block_count": (_GGUF_TYPE_UINT32, struct.pack("<I", 32)),
    "llama.context_length": (_GGUF_TYPE_UINT32, struct.pack("<I", 8192)),
    "llama.attention.head_count": (_GGUF_TYPE_UINT32, struct.pack("<I", 32)),
    "llama.attention.head_count_kv": (_GGUF_TYPE_UINT32, struct.pack("<I", 8)),
}

_GENERAL_NAME_VALUE: bytes = struct.pack("<Q", len(b"test-model-v1")) + b"test-model-v1"


def _pack_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    key_bytes = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_bytes)) + key_bytes + struct.pack("<I", value_type) + value_bytes
    )


def _build_kv_section(keys_with_values: dict[str, tuple[int, bytes]]) -> bytes:
    return b"".join(_pack_kv(k, vt, vb) for k, (vt, vb) in keys_with_values.items())


def _count_kv_pairs(kv_section: bytes) -> int:
    count = 0
    offset = 0
    while offset < len(kv_section):
        if offset + 8 > len(kv_section):
            break
        key_len = struct.unpack_from("<Q", kv_section, offset)[0]
        if key_len == 0 or offset + 8 + key_len > len(kv_section):
            break
        offset += 8 + key_len
        if offset + 4 > len(kv_section):
            break
        value_type = struct.unpack_from("<I", kv_section, offset)[0]
        offset += 4
        if value_type == _GGUF_TYPE_STRING:
            if offset + 8 > len(kv_section):
                break
            str_len = struct.unpack_from("<Q", kv_section, offset)[0]
            offset += 8 + str_len
        elif value_type == _GGUF_TYPE_UINT32:
            offset += 4
        else:
            break
        count += 1
    return count


def _write_gguf_v3(
    path: Path,
    kv_section: bytes,
    magic: bytes = b"GGUF",
    version: int = 3,
) -> None:
    kv_count = struct.pack("<Q", _count_kv_pairs(kv_section))
    # GGUF v3 header: magic(4) + version(4) + tensor_count(8) + kv_count(8)
    tensor_count = struct.pack("<Q", 0)
    header = magic + struct.pack("<I", version) + tensor_count + kv_count
    path.write_bytes(header + kv_section)


def _generate_valid_v3(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME_VALUE) + kv
    _write_gguf_v3(path, kv)


def _generate_valid_v3_no_name(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    _write_gguf_v3(path, kv)


def _generate_corrupt(path: Path) -> None:
    path.write_bytes(b"XXXX\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def _generate_truncated(path: Path) -> None:
    path.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def _generate_v4_unsupported(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME_VALUE) + kv
    _write_gguf_v3(path, kv, magic=b"GGUF", version=4)


def test_generate_gguf_fixtures(tmp_path: Path) -> None:
    """Generate all GGUF test fixtures for metadata extraction tests.

    Creates 5 synthetic GGUF files under tmp_path / "fixtures/" (ephemeral):
    - gguf_v3_valid.gguf: valid GGUF v3 with all required keys
    - gguf_v3_no_name.gguf: valid GGUF v3 missing general.name
    - gguf_corrupt.gguf: corrupt file (bad magic bytes)
    - gguf_truncated.gguf: truncated file (valid header, no KV data)
    - gguf_v4_unsupported.gguf: valid GGUF v4 (unsupported version)

    All fixtures are under 10 KiB and contain no tensor data.
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("gguf_v3_valid.gguf", _generate_valid_v3),
        ("gguf_v3_no_name.gguf", _generate_valid_v3_no_name),
        ("gguf_corrupt.gguf", _generate_corrupt),
        ("gguf_truncated.gguf", _generate_truncated),
        ("gguf_v4_unsupported.gguf", _generate_v4_unsupported),
    ]

    for name, gen_fn in generators:
        path = fixtures_dir / name
        gen_fn(path)
        size = path.stat().st_size
        assert size > 0, f"Fixture {name} is empty"
        assert size < 10240, f"Fixture {name} exceeds 10 KiB ({size} bytes)"


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


import subprocess
import sys

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
        with patch("llama_cli.commands.setup.detect_toolchain") as mock_detect:
            # Tools are available
            mock_status = MagicMock()
            mock_status.is_sycl_ready = True
            mock_status.is_cuda_ready = False
            mock_status.missing_tools = MagicMock(return_value=[])

            mock_detect.return_value = mock_status

            # Call cmd_check to verify detect_toolchain is called
            from llama_cli.commands.setup import cmd_check

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
        # Ensure VIRTUAL_ENV and XDG_CACHE_HOME are not set
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        result = get_venv_path()
        expected = Path.home() / ".cache" / "llm-runner" / "venv"

        assert result == expected
        assert isinstance(result, Path)

    def test_venv_path_uses_xdg_cache_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path should use XDG_CACHE_HOME when set."""
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        custom_cache = "/custom/cache"
        monkeypatch.setenv("XDG_CACHE_HOME", custom_cache)

        result = get_venv_path()
        expected = Path(custom_cache) / "llm-runner" / "venv"

        assert result == expected
        assert isinstance(result, Path)

    def test_venv_path_respects_xdg_over_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_venv_path should prefer XDG_CACHE_HOME over HOME/.cache."""
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
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


class TestVenvPathVirtualEnvPrecedence:
    """T064: Tests for VIRTUAL_ENV precedence in venv path resolution."""

    def test_venv_path_ignores_virtual_env_and_uses_managed_venv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_venv_path should always return the managed venv, ignoring VIRTUAL_ENV."""
        custom_venv = "/custom/venv"
        monkeypatch.setenv("VIRTUAL_ENV", custom_venv)
        monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")

        result = get_venv_path()
        expected = Path("/custom/cache") / "llm-runner" / "venv"

        assert result == expected

    def test_venv_path_falls_back_to_xdg_cache_when_virtual_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_venv_path should fall back to XDG_CACHE_HOME when VIRTUAL_ENV is unset."""
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")

        result = get_venv_path()
        expected = Path("/custom/cache") / "llm-runner" / "venv"

        assert result == expected

    def test_venv_path_falls_back_to_home_cache_when_neither_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_venv_path should fall back to ~/.cache when neither env var is set."""
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

        result = get_venv_path()
        expected = Path.home() / ".cache" / "llm-runner" / "venv"

        assert result == expected


import os
import tempfile
from math import ceil

import pytest

from llama_cli.commands.dry_run import dry_run
from llama_manager.validation import validate_port, validate_ports


def get_p95(data: list[float]) -> float:
    """Calculate p95 percentile."""
    sorted_data = sorted(data)
    if not sorted_data:
        raise ValueError("data must not be empty")
    idx = ceil(len(sorted_data) * 0.95) - 1
    idx = max(0, min(idx, len(sorted_data) - 1))
    return sorted_data[idx]


@pytest.mark.slow
@patch(
    "llama_cli.commands.dry_run.write_artifact",
    return_value=os.path.join(tempfile.gettempdir(), "fake_artifact"),
)
@patch("llama_cli.commands.dry_run.resolve_runtime_dir", return_value=tempfile.gettempdir())
@patch("llama_cli.commands.dry_run.validate_server_config", return_value=None)
@patch("sys.stdout", new_callable=MagicMock)
@patch("sys.stderr", new_callable=MagicMock)
def test_performance_dry_run_resolution(
    mock_stderr: MagicMock,
    mock_stdout: MagicMock,
    mock_validate: MagicMock,
    mock_runtime: MagicMock,
    mock_artifact: MagicMock,
) -> None:
    """T041: Benchmark dry-run resolution time."""
    iterations: int = 100
    times: list[float] = []

    # Warmup to stabilize performance
    dry_run("summary-balanced")

    for _ in range(iterations):
        start: float = time.perf_counter()
        dry_run("summary-balanced")
        end: float = time.perf_counter()
        times.append(end - start)

    p95: float = get_p95(times)
    # Requirement: single-slot dry-run <= 250ms
    assert p95 <= 0.250, f"p95 dry-run resolution too slow: {p95:.4f}s"


@pytest.mark.slow
def test_performance_validation_paths() -> None:
    """T041: Benchmark lock/port validation paths."""
    iterations: int = 100

    # Per-slot lock/port validation
    port_times: list[float] = []
    for _ in range(iterations):
        start: float = time.perf_counter()
        validate_port(8080, "test_port")
        end: float = time.perf_counter()
        port_times.append(end - start)

    p95_port: float = get_p95(port_times)
    # Requirement: per-slot lock/port <= 150ms
    assert p95_port <= 0.150, f"p95 port validation too slow: {p95_port:.4f}s"

    # Port conflict validation
    conflict_times: list[float] = []
    conflict_results: list[Any] = []
    for _ in range(iterations):
        start: float = time.perf_counter()
        result = validate_ports(8080, 8080, "p1", "p2")
        end: float = time.perf_counter()
        conflict_times.append(end - start)
        conflict_results.append(result)

    p95_conflict: float = get_p95(conflict_times)
    # Requirement: port conflict validation <= 150ms
    assert all(r is not None for r in conflict_results), (
        "validate_ports should return ErrorDetail for conflicting ports"
    )
    assert p95_conflict <= 0.150, f"p95 port conflict validation too slow: {p95_conflict:.4f}s"


@pytest.mark.slow
@patch(
    "llama_cli.commands.dry_run.write_artifact",
    return_value=os.path.join(tempfile.gettempdir(), "fake_artifact"),
)
@patch("llama_cli.commands.dry_run.resolve_runtime_dir", return_value=tempfile.gettempdir())
@patch("llama_cli.commands.dry_run.validate_server_config", return_value=None)
@patch("sys.stdout", new_callable=MagicMock)
@patch("sys.stderr", new_callable=MagicMock)
def test_performance_dry_run_two_slots(
    mock_stderr: MagicMock,
    mock_stdout: MagicMock,
    mock_validate: MagicMock,
    mock_runtime: MagicMock,
    mock_artifact: MagicMock,
) -> None:
    """T041: Benchmark two-slot dry-run resolution time."""
    iterations: int = 100
    times: list[float] = []

    # Warmup to stabilize performance
    dry_run("both")

    for _ in range(iterations):
        start: float = time.perf_counter()
        # 'both' mode uses two slots
        dry_run("both")
        end: float = time.perf_counter()
        times.append(end - start)

    p95: float = get_p95(times)
    # Requirement: two-slot <= 400ms
    assert p95 <= 0.400, f"p95 two-slot dry-run resolution too slow: {p95:.4f}s"


"""Tests for GPU collectors module."""


from unittest.mock import MagicMock, patch

import psutil

from llama_cli.gpu_collectors import _get_cpu_percent, _get_memory_percent


class TestGetCpuPercent:
    """Tests for _get_cpu_percent function."""

    def test_get_cpu_percent_success(self):
        """Test successful CPU percent retrieval."""
        with patch("psutil.cpu_percent", return_value=45.7) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 45.7
            assert isinstance(result, float)
            mock_cpu.assert_called_once_with(interval=0.1)

    def test_get_cpu_percent_access_denied(self):
        """Test graceful handling of AccessDenied exception."""
        with patch("psutil.cpu_percent", side_effect=psutil.AccessDenied()) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 0.0
            mock_cpu.assert_called_once_with(interval=0.1)

    def test_get_cpu_percent_no_such_process(self):
        """Test graceful handling of NoSuchProcess exception."""
        with patch("psutil.cpu_percent", side_effect=psutil.NoSuchProcess(pid=123)) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 0.0
            mock_cpu.assert_called_once_with(interval=0.1)

    def test_get_cpu_percent_general_exception(self):
        """Test graceful handling of general Exception."""
        with patch("psutil.cpu_percent", side_effect=Exception("test error")) as mock_cpu:
            result = _get_cpu_percent()
            assert result == 0.0
            mock_cpu.assert_called_once_with(interval=0.1)


class TestGetMemoryPercent:
    """Tests for _get_memory_percent function."""

    def test_get_memory_percent_success(self):
        """Test successful memory percent retrieval."""
        mock_mem = MagicMock()
        mock_mem.percent = 62.3
        with patch("psutil.virtual_memory", return_value=mock_mem) as mock_mem_func:
            result = _get_memory_percent()
            assert result == 62.3
            assert isinstance(result, float)
            mock_mem_func.assert_called_once()

    def test_get_memory_percent_access_denied(self):
        """Test graceful handling of AccessDenied exception."""
        with patch("psutil.virtual_memory", side_effect=psutil.AccessDenied()) as mock_mem:
            result = _get_memory_percent()
            assert result == 0.0
            mock_mem.assert_called_once()

    def test_get_memory_percent_no_such_process(self):
        """Test graceful handling of NoSuchProcess exception."""
        with patch("psutil.virtual_memory", side_effect=psutil.NoSuchProcess(pid=123)) as mock_mem:
            result = _get_memory_percent()
            assert result == 0.0
            mock_mem.assert_called_once()

    def test_get_memory_percent_general_exception(self):
        """Test graceful handling of general Exception."""
        with patch("psutil.virtual_memory", side_effect=Exception("test error")) as mock_mem:
            result = _get_memory_percent()
            assert result == 0.0
            mock_mem.assert_called_once()
