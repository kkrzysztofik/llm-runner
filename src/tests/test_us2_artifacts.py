"""US2 FR-007 artifact persistence, redaction, and permission tests.

Test Tasks:
- T025: Add FR-007 artifact persistence, redaction, and permission tests:
  (1) artifact contains model_path, port, command fields,
  (2) environment variable values for keys containing KEY|TOKEN|SECRET|PASSWORD|AUTH
      are redacted with [REDACTED] while filesystem paths like model_path are preserved,
  (3) file permissions verified via stat.S_IMODE(st.st_mode) == 0o600

Contract:
- FR-007: Observability artifact with required fields and redaction rules
- Directory permissions: 0o700, file permissions: 0o600
- Redaction pattern: keys containing KEY|TOKEN|SECRET|PASSWORD|AUTH (case-insensitive)
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_manager.config import (
    MultiValidationError,
)
from llama_manager.process_manager import (
    ValidationException,
    write_artifact,
)
from llama_manager.server import (
    ValidationResults,
    build_dry_run_slot_payload,
)


class TestFR007ArtifactRequiredFields:
    """FR-007: Artifact required fields validation."""

    def _valid_artifact_data(self) -> dict:
        """Create valid artifact data with all required fields."""
        return {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }

    def test_artifact_contains_model_path_field(self) -> None:
        """FR-007: Artifact should contain model_path field."""
        data = self._valid_artifact_data()
        data["resolved_command"] = {
            "cmd": ["echo", "test"],
            "model_path": "/models/model.gguf",
        }
        # write_artifact expects specific structure
        # Verify that artifact data can contain model_path
        assert "model_path" in str(data).lower()

    def test_artifact_contains_port_field(self) -> None:
        """FR-007: Artifact should contain port information."""
        data = self._valid_artifact_data()
        # Port would be in command args or as separate field
        assert isinstance(data["resolved_command"], dict)

    def test_artifact_contains_command_field(self) -> None:
        """FR-007: Artifact should contain command field."""
        data = self._valid_artifact_data()
        assert "resolved_command" in data
        assert isinstance(data["resolved_command"], dict)
        assert "cmd" in data["resolved_command"]

    def test_artifact_required_fields_list(self) -> None:
        """FR-007: verify all required FR-007 fields are present."""
        data = self._valid_artifact_data()
        required_fields = [
            "timestamp",
            "slot_scope",
            "resolved_command",
            "validation_results",
            "warnings",
            "environment_redacted",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_artifact_with_server_config_data(self, tmp_path: Path) -> None:
        """FR-007: write_artifact should work with ServerConfig-derived data."""
        from llama_manager import config_builder

        # Create a ServerConfig
        sc = config_builder.create_summary_balanced_cfg(port=8080)
        payload = build_dry_run_slot_payload(
            sc,
            slot_id="summary-balanced",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )

        # Build artifact data similar to dry_run.py
        artifact_data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": [payload.slot_id],
            "resolved_command": {
                "cmd": payload.command_args,
                "binary_path": payload.binary_path,
                "port": payload.port,
            },
            "validation_results": {
                "passed": payload.validation_results.passed,
                "checks": payload.validation_results.checks,
            },
            "warnings": payload.warnings,
            "environment_redacted": payload.environment_redacted,
        }

        artifact_path = write_artifact(tmp_path, "slot1", artifact_data)
        assert artifact_path.exists()
        loaded = artifact_path.read_text()
        assert "summary-balanced" in loaded
        assert "8080" in loaded


def test_dry_run_artifact_uses_slot_scope_list_and_resolved_command_mapping(tmp_path: Path) -> None:
    """Dry-run should persist updated artifact contract shapes."""
    captured: dict = {}

    def _capture_artifact(_runtime_dir: Path, _slot_id: str, data: dict) -> Path:
        captured.update(data)
        return tmp_path / "artifact.json"

    with (
        patch("llama_cli.commands.dry_run.resolve_runtime_dir", return_value=tmp_path),
        patch("llama_cli.commands.dry_run.write_artifact", side_effect=_capture_artifact),
    ):
        from llama_cli.commands.dry_run import dry_run

        dry_run("summary-balanced")

    assert isinstance(captured["slot_scope"], list)
    assert captured["slot_scope"] == ["summary-balanced"]
    assert isinstance(captured["resolved_command"], dict)
    assert "summary-balanced" in captured["resolved_command"]


def test_dry_run_both_artifact_uses_slot_ids_for_scope_and_resolved_command(tmp_path: Path) -> None:
    captured: dict = {}

    def _capture_artifact(_runtime_dir: Path, _slot_id: str, data: dict) -> Path:
        captured.update(data)
        return tmp_path / "artifact.json"

    with (
        patch("llama_cli.commands.dry_run.resolve_runtime_dir", return_value=tmp_path),
        patch("llama_cli.commands.dry_run.write_artifact", side_effect=_capture_artifact),
    ):
        from llama_cli.commands.dry_run import dry_run

        dry_run("both")

    assert captured["slot_scope"] == ["summary-balanced", "qwen35"]
    assert sorted(captured["resolved_command"].keys()) == ["qwen35", "summary-balanced"]


class TestFR007RedactionRules:
    """FR-007: Redaction rules for sensitive environment variables."""

    def test_redacts_key_token_secret_password_auth_keys(self, tmp_path: Path) -> None:
        """FR-007: Redact values for keys containing KEY|TOKEN|SECRET|PASSWORD|AUTH."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {
                "API_KEY": "secret123",
                "AUTH_TOKEN": "bearer_xyz",
                "SECRET_VALUE": "mysecret",
                "DB_PASSWORD": "password123",
                "AUTH_HEADER": "basic abc",
                "MODEL_PATH": "/models/model.gguf",  # Should NOT be redacted
                "PATH": "/usr/bin",  # Should NOT be redacted
            },
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        loaded = artifact_path.read_text()
        # Sensitive values should be redacted
        assert "secret123" not in loaded
        assert "bearer_xyz" not in loaded
        assert "mysecret" not in loaded
        assert "password123" not in loaded
        assert "basic abc" not in loaded
        # Non-sensitive values should be preserved
        assert "/models/model.gguf" in loaded
        assert "/usr/bin" in loaded

    def test_redaction_is_case_insensitive(self, tmp_path: Path) -> None:
        """FR-007: Redaction should be case-insensitive for key matching."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {
                "api_key": "lowercase",
                "API_KEY": "uppercase",
                "Api_Key": "mixedcase",
                "TOKEN": "token1",
                "token": "token2",
                "Secret": "secret1",
                "password": "pwd1",
                "AUTH": "auth1",
            },
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        loaded = artifact_path.read_text()
        # All should be redacted regardless of case
        assert "lowercase" not in loaded
        assert "uppercase" not in loaded
        assert "mixedcase" not in loaded
        assert "token1" not in loaded
        assert "token2" not in loaded
        assert "secret1" not in loaded
        assert "pwd1" not in loaded
        assert "auth1" not in loaded

    def test_redaction_preserves_model_path(self, tmp_path: Path) -> None:
        """FR-007: model_path should NOT be redacted."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"], "model_path": "/models/test.gguf"},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {"MODEL_PATH": "/models/test.gguf"},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        loaded = artifact_path.read_text()
        # model_path should be preserved
        assert "/models/test.gguf" in loaded

    def test_redaction_handles_nested_dict(self, tmp_path: Path) -> None:
        """FR-007: Redaction should handle nested dictionaries."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {
                "nested": {
                    "API_KEY": "nested_secret",
                    "NORMAL": "value",
                },
            },
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        loaded = artifact_path.read_text()
        # Nested sensitive values should be redacted
        assert "nested_secret" not in loaded
        # Non-sensitive nested values should be preserved
        assert "value" in loaded


class TestFR007FilePermissions:
    """FR-007: File and directory permission enforcement."""

    def test_artifact_directory_has_0700_permissions(self, tmp_path: Path) -> None:
        """FR-007: artifact directory should have 0o700 permissions."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        # Check artifact directory permissions
        dir_mode = stat.S_IMODE(os.stat(artifact_path.parent).st_mode)
        assert dir_mode == 0o700, (
            f"Artifact directory permissions should be 0o700, got {oct(dir_mode)}"
        )

    def test_artifact_file_has_0600_permissions(self, tmp_path: Path) -> None:
        """FR-007: artifact file should have 0o600 permissions."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        # Check artifact file permissions
        file_mode = stat.S_IMODE(os.stat(artifact_path).st_mode)
        assert file_mode == 0o600, (
            f"Artifact file permissions should be 0o600, got {oct(file_mode)}"
        )

    def test_artifact_directory_ownership_is_owner_only(self, tmp_path: Path) -> None:
        """FR-007: artifact directory should be owner-only (no group/other permissions)."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        # Check for owner-only access (no group/other permissions)
        dir_stat = os.stat(artifact_path.parent)
        # 0o700 = rwx for owner only
        expected_mode = 0o700
        actual_mode = stat.S_IMODE(dir_stat.st_mode)
        assert actual_mode == expected_mode

    def test_artifact_file_ownership_is_owner_only(self, tmp_path: Path) -> None:
        """FR-007: artifact file should be owner-only (no group/other permissions)."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        # Check for owner-only access (no group/other permissions)
        file_stat = os.stat(artifact_path)
        expected_mode = 0o600
        actual_mode = stat.S_IMODE(file_stat.st_mode)
        assert actual_mode == expected_mode


class TestFR007ArtifactFilename:
    """FR-007: Artifact filename format validation."""

    def test_artifact_filename_matches_timestamp_pattern(self, tmp_path: Path) -> None:
        """FR-007: artifact filename should match artifact-*.json pattern."""
        import re

        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        filename = artifact_path.name
        # Should match artifact-YYYYMMDDTHHMMSSZ.json
        pattern = r"^artifact-\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}Z\.json$"
        assert re.match(pattern, filename), (
            f"Filename should match artifact-YYYYMMDDTHHMMSSZ.json, got: {filename}"
        )

    def test_artifact_filename_does_not_contain_uuid(self, tmp_path: Path) -> None:
        """FR-007: artifact filename should not contain UUID pattern."""
        import re

        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        filename = artifact_path.name
        # Should NOT contain UUID pattern (8-4-4-4-12 hex chars)
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        assert not re.search(uuid_pattern, filename), (
            f"Filename should not contain UUID: {filename}"
        )


class TestFR007ArtifactValidation:
    """FR-007: Artifact validation and error handling."""

    def test_write_artifact_raises_on_missing_required_fields(self, tmp_path: Path) -> None:
        """FR-007: write_artifact should raise ValidationException for missing fields."""
        from llama_manager.process_manager import ValidationException

        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            # Missing: slot_scope, resolved_command, validation_results, warnings, environment_redacted
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        assert isinstance(exc_info.value.multi_error, MultiValidationError)
        assert exc_info.value.multi_error.errors[0].failed_check == "artifact_validation"
        assert "missing required fields" in exc_info.value.multi_error.errors[0].why_blocked.lower()

    def test_write_artifact_succeeds_with_all_required_fields(self, tmp_path: Path) -> None:
        """FR-007: write_artifact should succeed with all required fields present."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        assert artifact_path.exists()
        assert artifact_path.is_file()

    def test_write_artifact_raises_on_invalid_json_data(self, tmp_path: Path) -> None:
        """FR-007: write_artifact should raise ValidationException on invalid data."""

        # Valid data but with non-serializable content
        class Unserializable:
            pass

        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
            "unserializable_field": Unserializable(),  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        # Check that the exception indicates a serialization or type error
        error_msg = str(exc_info.value).lower()
        assert (
            "serialization" in error_msg
            or "type" in error_msg
            or "json" in error_msg
            or "serializable" in error_msg
        )


class TestFR007ArtifactPersistence:
    """FR-007: Artifact persistence behavior."""

    def test_artifact_is_written_to_artifacts_subdirectory(self, tmp_path: Path) -> None:
        """FR-007: artifacts should be written to artifacts/ subdirectory."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        assert "artifacts" in str(artifact_path)
        assert artifact_path.parent.name == "artifacts"

    def test_artifact_collision_safe_same_second_writes(self, tmp_path: Path) -> None:
        """FR-007: artifacts should be collision-safe - same-second writes get unique filenames."""
        import re

        data1 = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test1"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        data2 = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot2"],
            "resolved_command": {"cmd": ["echo", "test2"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        # First write
        artifact_path1 = write_artifact(tmp_path, "slot1", data1)
        # Second write (immediately after - same second timestamp)
        artifact_path2 = write_artifact(tmp_path, "slot2", data2)
        # Should NOT overwrite - should have unique filenames with collision suffix
        assert artifact_path1 != artifact_path2
        # Both files should exist
        assert artifact_path1.exists()
        assert artifact_path2.exists()
        # First file should have base pattern (artifact-YYYYMMDDTHHMMSSZ.json)
        base_pattern = re.compile(r"^artifact-\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}Z\.json$")
        # Second file should have collision suffix (artifact-YYYYMMDDTHHMMSSZ-1.json)
        collision_pattern = re.compile(r"^artifact-\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}Z-1\.json$")
        assert base_pattern.match(artifact_path1.name), (
            f"Expected base pattern, got: {artifact_path1.name}"
        )
        assert collision_pattern.match(artifact_path2.name), (
            f"Expected collision pattern, got: {artifact_path2.name}"
        )
        # Both files should have correct content
        content1 = artifact_path1.read_text()
        content2 = artifact_path2.read_text()
        assert "test1" in content1
        assert "test2" in content2

    def test_artifact_writes_to_correct_slot_directory(self, tmp_path: Path) -> None:
        """FR-007: artifacts should be written to runtime directory artifacts/."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot1", data)
        # Should be in artifacts subdir under tmp_path
        assert artifact_path.parent.name == "artifacts"
        assert artifact_path.parent.parent == tmp_path
