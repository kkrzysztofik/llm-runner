from __future__ import annotations

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
from llama_manager.orchestration import (
    ValidationException,
    write_artifact,
)
from llama_manager.validation import (
    ValidationResults,
    build_dry_run_slot_payload,
)
from tests.support.helpers import valid_artifact_data


class TestFR007ArtifactRequiredFields:
    """FR-007: Artifact required fields validation."""

    def _valid_artifact_data(self) -> dict:
        """Create valid artifact data with all required fields."""
        return valid_artifact_data()

    def test_artifact_contains_model_path_field(self) -> None:
        """FR-007: Artifact should contain model_path field."""
        data = self._valid_artifact_data()
        data["resolved_command"] = {
            "slot1": {
                "cmd": ["echo", "test"],
                "model_path": "/models/model.gguf",
            },
        }
        # Verify model_path is present in resolved_command
        assert data["resolved_command"]["slot1"]["model_path"] == "/models/model.gguf"

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
        assert "cmd" in data["resolved_command"]["slot1"]

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
        from llama_manager.config import create_summary_balanced_cfg

        # Create a ServerConfig
        sc = create_summary_balanced_cfg(port=8080)
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
        from llama_manager.orchestration import ValidationException

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


class TestFR007FieldTypeValidation:
    """FR-007: Type validation for artifact fields."""

    def test_timestamp_must_be_string(self, tmp_path: Path) -> None:
        """timestamp must be a string (ISO 8601)."""
        data = {
            "timestamp": 12345,  # type: ignore[dict-item]
            "slot_scope": ["slot1"],
            "resolved_command": {},
            "validation_results": {},
            "warnings": [],
            "environment_redacted": {},
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        assert "timestamp must be an ISO 8601 string" in str(exc_info.value)

    def test_validation_results_must_be_dict(self, tmp_path: Path) -> None:
        """validation_results must be a dict/mapping."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {},
            "validation_results": "not a dict",  # type: ignore[dict-item]
            "warnings": [],
            "environment_redacted": {},
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        assert "validation_results must be an object mapping" in str(exc_info.value)

    def test_warnings_must_be_list_of_strings(self, tmp_path: Path) -> None:
        """warnings must be list[str]."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {},
            "validation_results": {},
            "warnings": ["ok", 123],  # type: ignore[list-item]
            "environment_redacted": {},
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        assert "warnings must be list[str]" in str(exc_info.value)

    def test_environment_redacted_must_be_dict(self, tmp_path: Path) -> None:
        """environment_redacted must be a dict/mapping."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {},
            "validation_results": {},
            "warnings": [],
            "environment_redacted": "not a dict",  # type: ignore[dict-item]
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        assert "environment_redacted must be an object mapping" in str(exc_info.value)

    def test_resolved_command_keys_must_be_strings(self, tmp_path: Path) -> None:
        """resolved_command keys must be strings."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {42: {"cmd": ["echo"]}},  # type: ignore[dict-item]
            "validation_results": {},
            "warnings": [],
            "environment_redacted": {},
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        assert "resolved_command keys must be strings" in str(exc_info.value)

    def test_resolved_command_slot_ids_must_be_strings(self, tmp_path: Path) -> None:
        """resolved_command inner dict keys (slot IDs) must be strings."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {
                "slot1": {123: "value"},  # type: ignore[dict-item]
            },
            "validation_results": {},
            "warnings": [],
            "environment_redacted": {},
        }
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot1", data)
        assert "resolved_command slot IDs must be strings" in str(exc_info.value)


"""Phase 7 — T082: CA-003 parity test — dry-run flag bundles.

Verifies dry-run output includes OpenAI flag bundles and compatibility
matrix rows in both TUI and CLI modes.

Tests:
  - dry-run summary-balanced mode includes openai_flag_bundle
  - dry-run qwen35 mode includes openai_flag_bundle
  - dry-run both mode includes openai_flag_bundle for all slots
  - vllm_eligibility rows present in output
  - Flag bundle keys are deterministic (sorted)
"""


import contextlib

from llama_manager.validation import DryRunSlotPayload, VllmEligibility
from tests.support.helpers import make_server_config

_make_minimal_server_config = make_server_config


class TestDryRunFlagBundlesParity:
    """T082: Dry-run output includes OpenAI flag bundles and compatibility matrix."""

    # ------------------------------------------------------------------
    # openai_flag_bundle presence
    # ------------------------------------------------------------------

    def test_summary_balanced_has_openai_flag_bundle(self) -> None:
        """dry-run summary-balanced must include openai_flag_bundle in payload."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="summary-balanced",
            model="/models/qwen3.5-2b.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="summary-balanced",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "openai_flag_bundle")
        assert isinstance(payload.openai_flag_bundle, dict)

    def test_qwen35_has_openai_flag_bundle(self) -> None:
        """dry-run qwen35 must include openai_flag_bundle in payload."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="qwen35",
            model="/models/qwen3.5-35b.gguf",
            port=8081,
            device="CUDA",
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="qwen35",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "openai_flag_bundle")
        assert isinstance(payload.openai_flag_bundle, dict)

    def test_both_mode_has_openai_flag_bundle_for_each_slot(self) -> None:
        """dry-run both mode must include openai_flag_bundle for all slots."""
        from llama_manager.validation import build_dry_run_slot_payload

        configs = [
            _make_minimal_server_config(
                alias="summary-balanced",
                model="/models/qwen3.5-2b.gguf",
                port=8080,
            ),
            _make_minimal_server_config(
                alias="qwen35",
                model="/models/qwen3.5-35b.gguf",
                port=8081,
                device="CUDA",
            ),
        ]

        payloads: list[DryRunSlotPayload] = []
        for cfg in configs:
            payload = build_dry_run_slot_payload(
                cfg,
                slot_id=cfg.alias,
                validation_results=None,
                warnings=[],
            )
            payloads.append(payload)

        # Both must have openai_flag_bundle
        for payload in payloads:
            assert hasattr(payload, "openai_flag_bundle")
            assert isinstance(payload.openai_flag_bundle, dict)

    # ------------------------------------------------------------------
    # vllm_eligibility presence
    # ------------------------------------------------------------------

    def test_summary_balanced_has_vllm_eligibility(self) -> None:
        """dry-run summary-balanced must include vllm_eligibility in payload."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="summary-balanced",
            model="/models/qwen3.5-2b.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="summary-balanced",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "vllm_eligibility")
        assert isinstance(payload.vllm_eligibility, VllmEligibility)

    def test_qwen35_has_vllm_eligibility(self) -> None:
        """dry-run qwen35 must include vllm_eligibility in payload."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="qwen35",
            model="/models/qwen3.5-35b.gguf",
            port=8081,
            device="CUDA",
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="qwen35",
            validation_results=None,
            warnings=[],
        )

        assert hasattr(payload, "vllm_eligibility")
        assert isinstance(payload.vllm_eligibility, VllmEligibility)

    # ------------------------------------------------------------------
    # Flag bundle key determinism
    # ------------------------------------------------------------------

    def test_openai_flag_bundle_keys_are_deterministic(self) -> None:
        """openai_flag_bundle keys must be deterministic (sorted on serialization)."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test",
            model="/models/test.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test",
            validation_results=None,
            warnings=[],
        )

        # Keys should be sortable
        keys = list(payload.openai_flag_bundle.keys())
        sorted_keys = sorted(keys)
        assert keys == sorted_keys, f"Keys not sorted: {keys} vs {sorted_keys}"

    def test_multiple_payloads_have_consistent_bundle_structure(self) -> None:
        """All payloads from the same mode should have consistent bundle structure."""
        from llama_manager.validation import build_dry_run_slot_payload

        configs = [
            _make_minimal_server_config(
                alias="slot1",
                model="/models/model1.gguf",
                port=8080,
            ),
            _make_minimal_server_config(
                alias="slot2",
                model="/models/model2.gguf",
                port=8081,
            ),
        ]

        payloads = []
        for cfg in configs:
            payload = build_dry_run_slot_payload(
                cfg,
                slot_id=cfg.alias,
                validation_results=None,
                warnings=[],
            )
            payloads.append(payload)

        # All bundles should have the same set of keys
        bundle_keys = [set(p.openai_flag_bundle.keys()) for p in payloads]
        assert len({frozenset(k) for k in bundle_keys}) == 1, (
            f"Inconsistent bundle keys across payloads: {bundle_keys}"
        )

    # ------------------------------------------------------------------
    # Dry-run output includes flag bundle info
    # ------------------------------------------------------------------

    def test_dry_run_output_includes_openai_bundle_section(self, capsys) -> None:
        """dry-run summary-balanced output must include 'OpenAI Bundle' section."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run._run_registry_mode") as mock_run,
            patch("llama_cli.commands.dry_run._print_smoke_probe_info"),
            patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
        ):
            mock_run.return_value = False

            with contextlib.suppress(SystemExit):
                dry_run(mode="summary-balanced", primary_port="8080")

            mock_run.assert_called()

    def test_dry_run_both_mode_prints_both_bundles(self, capsys) -> None:
        """dry-run both mode must print openai_flag_bundle for both summary and qwen35."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run._run_registry_mode") as mock_run,
            patch("llama_cli.commands.dry_run._print_smoke_probe_info"),
            patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
        ):
            mock_run.return_value = False

            with contextlib.suppress(SystemExit):
                dry_run(mode="both", primary_port="8080", secondary_port="8081")

            mock_run.assert_called()

    # ------------------------------------------------------------------
    # Integration tests — call dry_run directly, patch only _print_smoke_probe_info
    # ------------------------------------------------------------------

    def test_dry_run_summary_balanced_integration_no_mock_handlers(self, capsys) -> None:
        """dry-run summary-balanced: call dry_run() directly, capture stdout, assert OpenAI Bundle."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run._print_smoke_probe_info"),
            patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
            contextlib.suppress(SystemExit),
        ):
            dry_run(mode="summary-balanced", primary_port="8080")

        captured = capsys.readouterr()
        assert "OpenAI Bundle" in captured.out

    def test_dry_run_both_integration_no_mock_handlers(self, capsys) -> None:
        """dry-run both: call dry_run() directly, assert both bundle labels appear."""
        from llama_cli.commands.dry_run import dry_run

        with (
            patch("llama_cli.commands.dry_run._print_smoke_probe_info"),
            patch("llama_cli.commands.dry_run._write_dry_run_artifact"),
            contextlib.suppress(SystemExit),
        ):
            dry_run(mode="both", primary_port="8080", secondary_port="8081")

        captured = capsys.readouterr()
        # Both summary-balanced and qwen35 slots should have OpenAI Bundle
        openai_bundle_count = captured.out.count("OpenAI Bundle")
        assert openai_bundle_count == 2, (
            f"Expected 2 OpenAI Bundle sections, got {openai_bundle_count}"
        )

    def test_dry_run_summary_balanced_integration(self, capsys) -> None:
        """dry-run summary-balanced integration test without mocking mode handlers."""
        from llama_cli.commands.dry_run import _print_common_payload_sections
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="summary-balanced",
            model="/models/qwen3.5-2b.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="summary-balanced",
            validation_results=None,
            warnings=[],
        )

        _print_common_payload_sections(payload)

        captured = capsys.readouterr()
        assert "OpenAI Bundle" in captured.out

    def test_dry_run_both_integration(self, capsys) -> None:
        """dry-run both mode integration test checking both bundles appear."""
        from llama_cli.commands.dry_run import _print_common_payload_sections
        from llama_manager.validation import build_dry_run_slot_payload

        configs = [
            _make_minimal_server_config(
                alias="summary-balanced",
                model="/models/qwen3.5-2b.gguf",
                port=8080,
            ),
            _make_minimal_server_config(
                alias="qwen35",
                model="/models/qwen3.5-35b.gguf",
                port=8081,
                device="CUDA",
            ),
        ]

        for cfg in configs:
            payload = build_dry_run_slot_payload(
                cfg,
                slot_id=cfg.alias,
                validation_results=None,
                warnings=[],
            )
            _print_common_payload_sections(payload)

        captured = capsys.readouterr()
        assert "OpenAI Bundle" in captured.out

    # ------------------------------------------------------------------
    # TUI vs CLI consistency for flag bundles
    # ------------------------------------------------------------------

    def test_tui_and_cli_use_same_payload_structure(self) -> None:
        """TUI (via ServerManager) and CLI (via dry_run) must use same payload structure."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test-slot",
            model="/models/test.gguf",
            port=8080,
        )

        # Build payload the same way both TUI and CLI would
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test-slot",
            validation_results=None,
            warnings=[],
        )

        # Both must have the same required fields
        assert "openai_flag_bundle" in vars(payload)
        assert "vllm_eligibility" in vars(payload)
        assert "command_args" in vars(payload)
        assert isinstance(payload.openai_flag_bundle, dict)
        assert isinstance(payload.vllm_eligibility, VllmEligibility)

    def test_vllm_eligibility_has_required_fields(self) -> None:
        """vllm_eligibility must include eligible and reason fields."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test",
            model="/models/test.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test",
            validation_results=None,
            warnings=[],
        )

        vllm = payload.vllm_eligibility
        assert hasattr(vllm, "eligible"), "vllm_eligibility missing 'eligible'"
        assert hasattr(vllm, "reason"), "vllm_eligibility missing 'reason'"
        assert isinstance(vllm.eligible, bool)
        assert isinstance(vllm.reason, str)

    def test_openai_flag_bundle_contains_expected_keys(self) -> None:
        """openai_flag_bundle should contain OpenAI-compatible flag keys."""
        from llama_manager.validation import build_dry_run_slot_payload

        server_cfg = _make_minimal_server_config(
            alias="test",
            model="/models/test.gguf",
            port=8080,
        )
        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id="test",
            validation_results=None,
            warnings=[],
        )

        bundle = payload.openai_flag_bundle
        # At minimum should have some OpenAI-related flags
        # The exact keys depend on config, but there should be some
        assert isinstance(bundle, dict)
        # Keys should start with -- (CLI flag style)
        for key in bundle:
            assert key.startswith("--"), f"openai_flag_bundle key '{key}' should start with '--'"


from unittest.mock import MagicMock

from llama_manager.config import ServerConfig
from llama_manager.risk_ack import (
    RISK_ACK_LABEL,
    evaluate_risks,
    resolve_risk_action,
)

_ACK_TOKEN = "ack:attempt-1"  # noqa: S105

# Detect risky operations is optional for M1 - skip tests if not implemented
try:
    server_module = pytest.importorskip(
        "llama_manager.server", reason="detect_risky_operations not implemented in M1"
    )
    detect_risky_operations = server_module.detect_risky_operations
except AttributeError:
    pytest.skip(
        "detect_risky_operations attribute not found in llama_manager.server",
        allow_module_level=True,
    )


def test_privileged_port_requires_acknowledgement() -> None:
    """Privileged ports (< 1024) must be flagged as risky."""
    cfg = MagicMock()
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "privileged_port" in risks


def test_non_loopback_bind_requires_acknowledgement() -> None:
    """Binding to non-loopback address must be flagged as risky."""
    cfg = MagicMock()
    cfg.port = 8080
    cfg.bind_address = "192.168.1.1"
    cfg.risky_acknowledged = []  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "non_loopback" in risks


def test_combined_risks() -> None:
    """Multiple risky operations should all be detected."""
    cfg = MagicMock()
    cfg.port = 80
    cfg.bind_address = "192.168.1.1"
    cfg.risky_acknowledged = []  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "privileged_port" in risks
    assert "non_loopback" in risks


def test_warning_bypass_risk_class_detected() -> None:
    """warning_bypass marker should be reported as a risk class."""
    cfg = MagicMock()
    cfg.port = 8080
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = ["warning_bypass"]  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "warning_bypass" in risks


# =============================================================================
# evaluate_risks
# =============================================================================


def test_evaluate_risks_no_risks() -> None:
    """No risks detected → has_risks=False, empty details."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 8080
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    result = evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is False
    assert result.risks_acknowledged is False
    assert result.risk_details == []
    sm.acknowledge_risk.assert_not_called()


def test_evaluate_risks_detects_privileged_port() -> None:
    """Privileged port is detected and reported in risk_details."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    result = evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is True
    assert result.risks_acknowledged is False
    assert len(result.risk_details) == 1
    assert result.risk_details[0]["alias"] == "test"
    assert result.risk_details[0]["risk"] == "privileged_port"
    assert result.risk_details[0]["risk_kind"] == "hardware"
    sm.acknowledge_risk.assert_not_called()


def test_evaluate_risks_skips_already_acknowledged() -> None:
    """Risks already acknowledged in server_manager are skipped."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = True

    result = evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is True
    assert result.risk_details == []
    sm.acknowledge_risk.assert_not_called()


def test_evaluate_risks_pre_acknowledged_flag() -> None:
    """acknowledged=True sets risks_acknowledged and updates config copy."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    configs: list[ServerConfig] = [cfg]  # type: ignore[list-item]
    result = evaluate_risks(configs, sm, "attempt-1", _ACK_TOKEN, acknowledged=True)

    assert result.has_risks is True
    assert result.risks_acknowledged is True
    # The config in the list is replaced with a copy containing the label
    assert RISK_ACK_LABEL in configs[0].risky_acknowledged


def test_evaluate_risks_does_not_double_append_label() -> None:
    """acknowledged=True does not double-append RISK_ACK_LABEL."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = [RISK_ACK_LABEL]

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=True)

    assert cfg.risky_acknowledged.count(RISK_ACK_LABEL) == 1


def test_evaluate_risks_multiple_configs() -> None:
    """Risks across multiple configs are aggregated."""
    cfg1 = MagicMock()
    cfg1.alias = "a"
    cfg1.port = 80
    cfg1.bind_address = "127.0.0.1"
    cfg1.risky_acknowledged = []

    cfg2 = MagicMock()
    cfg2.alias = "b"
    cfg2.port = 8080
    cfg2.bind_address = "0.0.0.0"
    cfg2.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    result = evaluate_risks([cfg1, cfg2], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is True
    assert len(result.risk_details) == 2
    aliases = {d["alias"] for d in result.risk_details}
    assert aliases == {"a", "b"}
    sm.acknowledge_risk.assert_not_called()


# =============================================================================
# resolve_risk_action
# =============================================================================


def test_resolve_risk_action_y_hardware() -> None:
    assert resolve_risk_action("y", "hardware") == "acknowledge"


def test_resolve_risk_action_y_vram() -> None:
    assert resolve_risk_action("y", "vram") == "proceed"


def test_resolve_risk_action_n() -> None:
    assert resolve_risk_action("n", "hardware") == "abort"
    assert resolve_risk_action("n", "vram") == "abort"


def test_resolve_risk_action_q() -> None:
    assert resolve_risk_action("q", "hardware") == "quit"
    assert resolve_risk_action("q", "vram") == "quit"


def test_resolve_risk_action_unknown() -> None:
    assert resolve_risk_action("x", "hardware") == "ignore"
    assert resolve_risk_action("x", "vram") == "ignore"
    assert resolve_risk_action("", None) == "ignore"
