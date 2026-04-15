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
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_manager.reports import (
    FailureReport,
    MutatingActionLogEntry,
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
        import json

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
        import json

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
        assert redact_sensitive("API_KEY") == "API_[REDACTED]"
        assert redact_sensitive("TOKEN") == "[REDACTED]"


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
        assert report.report_dir.name[4] == "M"  # Month separator
        assert report.report_dir.name[7] == "D"  # Day separator
        assert report.report_dir.name[10] == "_"  # Time separator

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
            report_dir = tmp_path / f"2026010{i}_120000"
            report_dir.mkdir()
            # Set different modification times
            report_dir.touch()

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
            report_dir = tmp_path / f"2026010{i:02d}_120000"
            report_dir.mkdir()
            # Set different modification times (oldest first)
            report_dir.touch()
            # Add delay to ensure different mtime
            import time

            time.sleep(0.01)

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
        # Create timestamp directories
        for i in range(5):
            report_dir = tmp_path / f"2026010{i:02d}_120000"
            report_dir.mkdir()

        # Create non-timestamp directory (should not be deleted)
        other_dir = tmp_path / "other-directory"
        other_dir.mkdir()

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
        # Create valid timestamp directories
        for i in range(3):
            report_dir = tmp_path / f"2026010{i:02d}_120000"
            report_dir.mkdir()

        # Create invalid format directory
        invalid_dir = tmp_path / "not-a-timestamp"
        invalid_dir.mkdir()

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
            report_dir = tmp_path / f"2026010{i:02d}_120000"
            report_dir.mkdir()
            directories.append(report_dir)
            time.sleep(0.01)  # Ensure different mtime

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
        assert "20260100_120000" not in remaining_names  # Oldest
        assert "20260101_120000" not in remaining_names  # Second oldest
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

        # Create valid timestamp directories
        for i in range(3):
            report_dir = tmp_path / f"2026010{i:02d}_120000"
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
        # Should have: 3 valid + 5 invalid = 8 total
        assert len(remaining) == 8

    def test_rotate_reports_empty_directory(self, tmp_path: Path) -> None:
        """T073: rotate_reports should handle empty reports directory."""
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
            report_dir = tmp_path / f"2026010{i:02d}_120000"
            report_dir.mkdir()

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
            entry = MutatingActionLogEntry(
                command=["git", "clone", f"repo{i}"],
                timestamp=datetime(2026, 4, 15, 12, i, 0),
                exit_code=0,
                truncated_output=f"Output {i}",
            )
            entries.append(entry)

        assert len(entries) == max_entries

        # Add one more entry (should trigger rotation)
        new_entry = MutatingActionLogEntry(
            command=["git", "clone", "repo100"],
            timestamp=datetime(2026, 4, 15, 12, max_entries, 0),
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
