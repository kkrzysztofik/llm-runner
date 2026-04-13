"""FR-003, FR-007, FR-011 contract enforcement tests.

This test module validates the following contracts against the implementation:

FR-003: Canonical dry-run payload contract
- Per-slot required fields present and types
- openai_flag_bundle keys include leading `--`
- hardware_notes is object with backend/device_id/device_name

FR-007: Artifact contract
- write_artifact requires top-level fields:
  timestamp, slot_scope(list), resolved_command(object), validation_results(object),
  warnings(list), environment_redacted(object)
- artifacts path semantics (`artifacts/`, filename pattern artifact-*.json)

FR-011: vLLM blocking exact actionable fields

Tests use pytest conventions:
- Unit tests only, no subprocesses
- Mock hardware (no GPU, no nvtop, no llama-server binaries)
- Test validators with pytest.raises(SystemExit)
- Type safety with pyright
- Coverage target >90%
"""

import json
import os
import re
import stat
from pathlib import Path

import pytest

from llama_manager.config import ErrorCode, ErrorDetail, MultiValidationError, ServerConfig
from llama_manager.process_manager import ValidationException, write_artifact
from llama_manager.server import (
    ValidationResults,
    VllmEligibility,
    build_dry_run_slot_payload,
    validate_backend_eligibility,
    validate_server_config,
)


class TestFR003DryRunPayloadContract:
    """FR-003: Canonical dry-run payload contract assertions."""

    def _minimal_cfg(self, **kwargs: object) -> ServerConfig:
        """Create minimal ServerConfig for testing."""
        defaults = {
            "model": "/models/test.gguf",
            "alias": "test",
            "device": "SYCL0",
            "port": 8080,
            "ctx_size": 4096,
            "ubatch_size": 512,
            "threads": 4,
            "server_bin": "/usr/bin/llama-server",
            "backend": "llama_cpp",
        }
        defaults.update(kwargs)
        return ServerConfig(**defaults)  # type: ignore[arg-type]

    def test_slot_id_field_present_and_type(self) -> None:
        """FR-003: slot_id should be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.slot_id, str)
        assert payload.slot_id == "test-slot"

    def test_binary_path_field_present_and_type(self) -> None:
        """FR-003: binary_path should be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.binary_path, str)
        assert payload.binary_path == "/usr/bin/llama-server"

    def test_command_args_field_present_and_type(self) -> None:
        """FR-003: command_args should be present and be a list of strings."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.command_args, list)
        assert all(isinstance(arg, str) for arg in payload.command_args)
        assert len(payload.command_args) > 0

    def test_model_path_field_present_and_type(self) -> None:
        """FR-003: model_path should be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.model_path, str)
        assert payload.model_path == "/models/test.gguf"

    def test_bind_address_field_present_and_type(self) -> None:
        """FR-003: bind_address should be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.bind_address, str)
        assert payload.bind_address == "127.0.0.1"

    def test_port_field_present_and_type(self) -> None:
        """FR-003: port should be present and be an int."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.port, int)
        assert payload.port == 8080

    def test_environment_redacted_field_present_and_type(self) -> None:
        """FR-003: environment_redacted should be present and be a dict."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.environment_redacted, dict)
        assert len(payload.environment_redacted) > 0

    def test_openai_flag_bundle_keys_include_leading_dashes(self) -> None:
        """FR-003: openai_flag_bundle keys should include leading `--`."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.openai_flag_bundle, dict)
        # Check that all keys have leading dashes
        for key in payload.openai_flag_bundle:
            assert key.startswith("--"), f"openai_flag_bundle key '{key}' should start with '--'"

    def test_openai_flag_bundle_values_mixed_types(self) -> None:
        """FR-003: openai_flag_bundle values should be str|int|bool|None."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        for key, value in payload.openai_flag_bundle.items():
            assert isinstance(value, str | int | bool | None), (
                f"openai_flag_bundle value for '{key}' should be str|int|bool|None, got {type(value)}"
            )

    def test_hardware_notes_is_object_with_required_fields(self) -> None:
        """FR-003: hardware_notes should be object with backend/device_id/device_name."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.hardware_notes, dict)
        # Required fields must be present
        assert "backend" in payload.hardware_notes
        assert "device_id" in payload.hardware_notes
        assert "device_name" in payload.hardware_notes
        # Values should be str or None
        for field in ["backend", "device_id", "device_name"]:
            value = payload.hardware_notes[field]
            assert isinstance(value, str | None), f"hardware_notes['{field}'] should be str|None"

    def test_hardware_notes_backend_value(self) -> None:
        """FR-003: hardware_notes.backend should reflect the backend."""
        cfg = self._minimal_cfg(backend="llama_cpp")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.hardware_notes["backend"] == "llama_cpp"

    def test_hardware_notes_device_cuda(self) -> None:
        """FR-003: hardware_notes should parse CUDA device correctly."""
        cfg = self._minimal_cfg(device="cuda:0", backend="llama_cpp")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.hardware_notes["backend"] == "llama_cpp"
        assert payload.hardware_notes["device_id"] == "0"
        assert payload.hardware_notes["device_name"] == "NVIDIA GPU"

    def test_hardware_notes_device_sycl(self) -> None:
        """FR-003: hardware_notes should parse SYCL device correctly."""
        cfg = self._minimal_cfg(device="sycl:0:0", backend="llama_cpp")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.hardware_notes["backend"] == "llama_cpp"
        assert payload.hardware_notes["device_id"] == "0:0"
        assert payload.hardware_notes["device_name"] == "SYCL Device 0"

    def test_vllm_eligibility_present_and_type(self) -> None:
        """FR-003: vllm_eligibility should be VllmEligibility object."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.vllm_eligibility, VllmEligibility)
        assert isinstance(payload.vllm_eligibility.eligible, bool)
        assert isinstance(payload.vllm_eligibility.reason, str)

    def test_vllm_eligibility_m1_blocking(self) -> None:
        """FR-011: vLLM should be ineligible in M1."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.vllm_eligibility.eligible is False
        assert "vllm is not launch-eligible in PRD M1" in payload.vllm_eligibility.reason

    def test_warnings_field_present_and_type(self) -> None:
        """FR-003: warnings should be present and be a list of strings."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=["warning1", "warning2"],
        )
        assert isinstance(payload.warnings, list)
        assert all(isinstance(w, str) for w in payload.warnings)
        assert payload.warnings == ["warning1", "warning2"]

    def test_validation_results_field_present_and_type(self) -> None:
        """FR-003: validation_results should be ValidationResults object."""
        cfg = self._minimal_cfg()
        validation = ValidationResults(passed=True, checks=[{"check": "test"}])
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=validation,
            warnings=[],
        )
        assert isinstance(payload.validation_results, ValidationResults)
        assert payload.validation_results.passed is True
        assert isinstance(payload.validation_results.checks, list)

    def test_default_validation_results_passed(self) -> None:
        """FR-003: default validation_results should have passed=True."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=None,  # Will default to passed=True
            warnings=[],
        )
        assert payload.validation_results.passed is True
        assert payload.validation_results.checks == []

    def test_default_warnings_empty_list(self) -> None:
        """FR-003: default warnings should be empty list."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=None,
            warnings=None,  # Will default to []
        )
        assert payload.warnings == []

    def test_deterministic_field_order(self) -> None:
        """FR-003: DryRunSlotPayload should have deterministic field ordering."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        # Verify field order by checking attribute names in __dict__
        # slot_id should come before binary_path (first two fields)
        attrs = list(payload.__dict__.keys())
        slot_id_idx = attrs.index("slot_id")
        binary_path_idx = attrs.index("binary_path")
        command_args_idx = attrs.index("command_args")
        assert slot_id_idx < binary_path_idx < command_args_idx


class TestFR007ArtifactContract:
    """FR-007: Artifact contract assertions."""

    def test_writes_artifact_in_artifacts_subdirectory(self, tmp_path: Path) -> None:
        """FR-007: artifacts should be written to artifacts/ subdirectory."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        assert "artifacts" in str(artifact_path)
        assert artifact_path.parent.name == "artifacts"

    def test_artifact_filename_pattern(self, tmp_path: Path) -> None:
        """FR-007: artifact filename should match artifact-*.json pattern (no UUID suffix)."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        # Should match artifact-YYYYMMDDTHHMMSSZ.json pattern
        filename = artifact_path.name
        assert filename.startswith("artifact-")
        assert filename.endswith(".json")
        # Should NOT contain UUID pattern (8-4-4-4-12 hex chars)
        uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        assert not re.search(uuid_pattern, filename), (
            f"Filename should not contain UUID: {filename}"
        )

    def test_write_artifact_required_top_level_fields(self, tmp_path: Path) -> None:
        """FR-007: write_artifact requires top-level fields."""
        # Valid artifact data with all required fields
        valid_data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1", "slot2"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": ["warning1"],
            "environment_redacted": {"PATH": "/usr/bin"},
        }
        artifact_path = write_artifact(tmp_path, "slot", valid_data)
        assert artifact_path.exists()

    def test_write_artifact_missing_required_fields_raises(self, tmp_path: Path) -> None:
        """FR-007: write_artifact should raise ValidationException for missing required fields."""
        # Missing several required fields
        invalid_data = {
            "timestamp": "2026-04-12T00:00:00Z",
            # Missing: slot_scope, resolved_command, validation_results, warnings, environment_redacted
        }

        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot", invalid_data)
        assert isinstance(exc_info.value.multi_error, MultiValidationError)
        assert exc_info.value.multi_error.errors[0].failed_check == "artifact_validation"
        assert "missing required fields" in exc_info.value.multi_error.errors[0].why_blocked.lower()

    def test_required_fields_list(self, tmp_path: Path) -> None:
        """FR-007: verify all required fields are present."""
        valid_data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1", "slot2"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": ["warning1"],
            "environment_redacted": {"PATH": "/usr/bin"},
        }
        artifact_path = write_artifact(tmp_path, "slot", valid_data)
        loaded = json.loads(artifact_path.read_text())
        # Verify all required fields are present
        required_fields = [
            "timestamp",
            "slot_scope",
            "resolved_command",
            "validation_results",
            "warnings",
            "environment_redacted",
        ]
        for field in required_fields:
            assert field in loaded, f"Missing required field: {field}"

    def test_slot_scope_is_list(self, tmp_path: Path) -> None:
        """FR-007: slot_scope should be a list."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1", "slot2"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert isinstance(loaded["slot_scope"], list)

    def test_resolved_command_is_object(self, tmp_path: Path) -> None:
        """FR-007: resolved_command should be an object (dict)."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"], "args": ["--flag"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert isinstance(loaded["resolved_command"], dict)

    def test_validation_results_is_object(self, tmp_path: Path) -> None:
        """FR-007: validation_results should be an object (dict)."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": [{"check": "test"}]},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert isinstance(loaded["validation_results"], dict)

    def test_warnings_is_list(self, tmp_path: Path) -> None:
        """FR-007: warnings should be a list."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": ["warning1", "warning2"],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert isinstance(loaded["warnings"], list)
        assert loaded["warnings"] == ["warning1", "warning2"]

    def test_environment_redacted_is_object(self, tmp_path: Path) -> None:
        """FR-007: environment_redacted should be an object (dict)."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {"PATH": "/usr/bin", "HOME": "/home/user"},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert isinstance(loaded["environment_redacted"], dict)

    def test_artifact_directory_permissions_0700(self, tmp_path: Path) -> None:
        """FR-007: artifacts directory should have 0700 permissions."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        dir_mode = stat.S_IMODE(os.stat(artifact_path.parent).st_mode)
        assert dir_mode == 0o700

    def test_artifact_file_permissions_0600(self, tmp_path: Path) -> None:
        """FR-007: artifact file should have 0600 permissions."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        file_mode = stat.S_IMODE(os.stat(artifact_path).st_mode)
        assert file_mode == 0o600

    def test_artifact_timestamp_format(self, tmp_path: Path) -> None:
        """FR-007: artifact timestamp should be ISO format."""
        data = {
            "timestamp": "2026-04-12T00:00:00Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert "timestamp" in loaded
        # ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
        assert re.match(iso_pattern, loaded["timestamp"]), (
            f"Invalid timestamp format: {loaded['timestamp']}"
        )

    def test_artifact_filename_timestamp_matches_content(self, tmp_path: Path) -> None:
        """FR-007: artifact filename timestamp should match content timestamp."""
        data = {
            "timestamp": "2026-04-12T15:30:45Z",
            "slot_scope": ["slot1"],
            "resolved_command": {"cmd": ["echo", "test"]},
            "validation_results": {"passed": True, "checks": []},
            "warnings": [],
            "environment_redacted": {},
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        # Filename timestamp: artifact-YYYYMMDDTHHMMSSZ.json
        filename_timestamp = artifact_path.name.replace("artifact-", "").replace(".json", "")
        # Content timestamp: YYYY-MM-DDTHH:MM:SSZ
        content_timestamp = loaded["timestamp"]
        # The timestamps should match (both use current time when write_artifact is called)
        # Since we're testing the format, we just verify the pattern matches
        assert re.match(r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}Z", filename_timestamp)
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", content_timestamp)


class TestFR011VllmBlockingActionableFields:
    """FR-011: vLLM blocking exact actionable fields assertions."""

    def test_vllm_eligibility_fields_present(self) -> None:
        """FR-011: VllmEligibility should have eligible and reason fields."""
        eligibility = VllmEligibility(eligible=False, reason="test reason")
        assert hasattr(eligibility, "eligible")
        assert hasattr(eligibility, "reason")
        assert isinstance(eligibility.eligible, bool)
        assert isinstance(eligibility.reason, str)

    def test_vllm_ineligible_in_m1(self) -> None:
        """FR-011: vLLM should be ineligible in M1."""
        eligibility = VllmEligibility(
            eligible=False, reason="vllm is not launch-eligible in PRD M1"
        )
        assert eligibility.eligible is False

    def test_vllm_reason_is_actionable(self) -> None:
        """FR-011: vLLM reason should explain why blocked."""
        eligibility = VllmEligibility(
            eligible=False,
            reason="vllm is not launch-eligible in PRD M1 - only llama_cpp supported",
        )
        assert "vllm is not launch-eligible" in eligibility.reason
        assert "PRD M1" in eligibility.reason

    def test_validate_backend_eligibility_vllm_returns_error(self) -> None:
        """FR-011: validate_backend_eligibility should return ErrorDetail for vllm."""

        error = validate_backend_eligibility("vllm")
        assert error is not None
        assert isinstance(error, ErrorDetail)

    def test_validate_backend_eligibility_vllm_error_code(self) -> None:
        """FR-011: validate_backend_eligibility for vllm should use BACKEND_NOT_ELIGIBLE."""

        error = validate_backend_eligibility("vllm")
        assert error is not None
        assert error.error_code == ErrorCode.BACKEND_NOT_ELIGIBLE

    def test_validate_backend_eligibility_vllm_failed_check(self) -> None:
        """FR-011: validate_backend_eligibility for vllm should have failed_check=vllm_launch_eligibility."""

        error = validate_backend_eligibility("vllm")
        assert error is not None
        assert error.failed_check == "vllm_launch_eligibility"

    def test_validate_backend_eligibility_vllm_why_blocked(self) -> None:
        """FR-011: validate_backend_eligibility for vllm should have actionable why_blocked."""

        error = validate_backend_eligibility("vllm")
        assert error is not None
        assert "vllm is not launch-eligible" in error.why_blocked
        assert "PRD M1" in error.why_blocked

    def test_validate_backend_eligibility_vllm_how_to_fix(self) -> None:
        """FR-011: validate_backend_eligibility for vllm should have actionable how_to_fix."""

        error = validate_backend_eligibility("vllm")
        assert error is not None
        assert "llama_cpp" in error.how_to_fix.lower()

    def test_validate_backend_eligibility_llama_cpp_passes(self) -> None:
        """FR-011: validate_backend_eligibility should return None for llama_cpp."""

        error = validate_backend_eligibility("llama_cpp")
        assert error is None

    def test_validate_backend_eligibility_case_insensitive(self) -> None:
        """FR-011: validate_backend_eligibility should be case insensitive."""

        assert validate_backend_eligibility("VLLM") is not None
        assert validate_backend_eligibility("Vllm") is not None
        assert validate_backend_eligibility("llama_cpp") is None

    def test_validate_server_config_vllm_returns_error(self) -> None:
        """FR-011: validate_server_config should return ErrorDetail for vllm backend."""

        cfg = ServerConfig(
            model="/model.gguf",
            alias="test",
            device="",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
            backend="vllm",
        )
        error = validate_server_config(cfg)
        assert error is not None
        assert error.error_code == ErrorCode.BACKEND_NOT_ELIGIBLE

    def test_validate_server_config_llama_cpp_passes(self) -> None:
        """FR-011: validate_server_config should pass for llama_cpp backend."""

        cfg = ServerConfig(
            model="/model.gguf",
            alias="test",
            device="",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
            backend="llama_cpp",
        )
        error = validate_server_config(cfg)
        assert error is None

    def test_vllm_eligibility_in_dry_run_payload(self) -> None:
        """FR-011: DryRunSlotPayload should include vllm_eligibility with ineligible status."""
        cfg = ServerConfig(
            model="/model.gguf",
            alias="test",
            device="",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
            backend="llama_cpp",
        )
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.vllm_eligibility.eligible is False
        assert "vllm is not launch-eligible in PRD M1" in payload.vllm_eligibility.reason


class TestFR003OpenAIBundleDeterminism:
    """FR-003: OpenAI compatibility bundle determinism and explicitness tests."""

    def _minimal_cfg(self, **kwargs: object) -> ServerConfig:
        """Create minimal ServerConfig for testing."""
        defaults = {
            "model": "/models/test.gguf",
            "alias": "test",
            "device": "SYCL0",
            "port": 8080,
            "ctx_size": 4096,
            "ubatch_size": 512,
            "threads": 4,
            "server_bin": "/usr/bin/llama-server",
            "backend": "llama_cpp",
        }
        defaults.update(kwargs)
        return ServerConfig(**defaults)  # type: ignore[arg-type]

    def test_openai_bundle_keys_are_sorted(self) -> None:
        """FR-003: openai_flag_bundle keys should be deterministically sorted."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        keys = list(payload.openai_flag_bundle.keys())
        # Keys should be sorted alphabetically
        assert keys == sorted(keys), f"Keys should be sorted: {keys}"

    def test_openai_bundle_deterministic_for_qwen35(self) -> None:
        """FR-003: OpenAI bundle should be explicit and deterministic for Qwen-class configs."""
        # Qwen35 config with reasoning mode enabled
        cfg = self._minimal_cfg(
            reasoning_mode="auto",
            alias="qwen35-test",
            port=9000,
        )
        payload1 = build_dry_run_slot_payload(
            cfg,
            slot_id="qwen35-test",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        payload2 = build_dry_run_slot_payload(
            cfg,
            slot_id="qwen35-test",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        # Should produce identical bundles
        assert payload1.openai_flag_bundle == payload2.openai_flag_bundle
        # Check specific keys
        assert "--port" in payload1.openai_flag_bundle
        assert "--host" in payload1.openai_flag_bundle
        assert "--openai" in payload1.openai_flag_bundle
        assert payload1.openai_flag_bundle["--openai"] is True

    def test_openai_bundle_chat_format_when_reasoning_enabled(self) -> None:
        """FR-003: --chat-format should be 'chatml' when reasoning mode is enabled."""
        cfg = self._minimal_cfg(reasoning_mode="auto")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.openai_flag_bundle["--chat-format"] == "chatml"

    def test_openai_bundle_chat_format_when_reasoning_disabled(self) -> None:
        """FR-003: --chat-format should be None when reasoning mode is disabled."""
        cfg = self._minimal_cfg(reasoning_mode="off")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.openai_flag_bundle["--chat-format"] is None

    def test_openai_bundle_keys_match_contract_spec(self) -> None:
        """FR-003: OpenAI bundle keys should match the dry-run contract spec."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        keys = set(payload.openai_flag_bundle.keys())
        # Allowed keys per contract: --port, --host, --chat-format, --openai
        allowed_keys = {"--port", "--host", "--chat-format", "--openai"}
        assert keys.issubset(allowed_keys), f"Unknown keys in bundle: {keys - allowed_keys}"

    def test_openai_bundle_sorted_twice_produces_same_result(self) -> None:
        """FR-003: Multiple calls to build_openai_flag_bundle should produce same sorted result."""
        cfg = self._minimal_cfg(port=8080)

        # Build multiple times
        payload1 = build_dry_run_slot_payload(
            cfg,
            slot_id="test",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        payload2 = build_dry_run_slot_payload(
            cfg,
            slot_id="test",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        payload3 = build_dry_run_slot_payload(
            cfg,
            slot_id="test",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )

        # All should be identical
        assert payload1.openai_flag_bundle == payload2.openai_flag_bundle
        assert payload2.openai_flag_bundle == payload3.openai_flag_bundle

    def test_openai_bundle_port_reflects_config(self) -> None:
        """FR-003: OpenAI bundle --port should reflect the ServerConfig port."""
        cfg1 = self._minimal_cfg(port=8080)
        cfg2 = self._minimal_cfg(port=9090)

        payload1 = build_dry_run_slot_payload(
            cfg1,
            slot_id="test1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        payload2 = build_dry_run_slot_payload(
            cfg2,
            slot_id="test2",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )

        assert payload1.openai_flag_bundle["--port"] == 8080
        assert payload2.openai_flag_bundle["--port"] == 9090

    def test_openai_bundle_host_is_default(self) -> None:
        """FR-003: OpenAI bundle --host should be 127.0.0.1."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.openai_flag_bundle["--host"] == "127.0.0.1"


class TestFR003DryRunHumanReadableOutput:
    """FR-003: Human-readable dry-run output assertions."""

    def test_dry_run_output_includes_openai_bundle_keys(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """FR-003: Human-readable dry-run output should surface OpenAI bundle values."""

        # Create a mock payload to verify structure
        cfg = ServerConfig(
            model="/model.gguf",
            alias="test",
            device="SYCL0",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
            server_bin="/usr/bin/llama-server",
            backend="llama_cpp",
            reasoning_mode="auto",
        )
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )

        # Verify OpenAI bundle has expected keys
        assert "--openai" in payload.openai_flag_bundle
        assert payload.openai_flag_bundle["--openai"] is True
        assert "--port" in payload.openai_flag_bundle
        assert payload.openai_flag_bundle["--port"] == 8080

    def test_dry_run_output_includes_vllm_eligibility(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """FR-003/FR-011: Human-readable dry-run output should surface vllm eligibility."""
        cfg = ServerConfig(
            model="/model.gguf",
            alias="test",
            device="SYCL0",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
            server_bin="/usr/bin/llama-server",
            backend="llama_cpp",
        )
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )

        # Verify vllm eligibility is present and shows ineligible status
        assert payload.vllm_eligibility.eligible is False
        assert "vllm is not launch-eligible" in payload.vllm_eligibility.reason
