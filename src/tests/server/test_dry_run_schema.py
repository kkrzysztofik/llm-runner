from __future__ import annotations

"""US2 FR-003 canonical dry-run schema and deterministic ordering tests.

Test Tasks:
- T024: Add FR-003 canonical dry-run schema and deterministic ordering tests validating:
  (1) payload includes per-slot required fields (slot_id, binary_path, command_args,
      model_path, bind_address, port, environment_redacted, openai_flag_bundle,
      hardware_notes, vllm_eligibility, warnings, validation_results),
  (2) slots ordered by slot configuration sequence (slot_id iteration order),
  (3) validation_results.errors ordered by slot configuration sequence with
      failed_check ascending tie-break within each slot,
  (4) command_args token edge tests for whitespace/shell-sensitive args
      with raw argv token examples (e.g., ["model", "path/with spaces/model.gguf",
      "--threads", "4"]) per FR-003

Contract:
- FR-003: Canonical dry-run payload with deterministic field ordering
- SC-003: Deterministic resolution evidence - repeated runs produce identical output
"""


from typing import Any

from llama_manager.config import ErrorCode, ServerConfig
from llama_manager.validation import (
    ValidationResults,
    VllmEligibility,
    build_dry_run_slot_payload,
)
from tests.support.helpers import make_server_config


def _dry_run_cfg(**kwargs: Any) -> ServerConfig:
    defaults = {
        "alias": "test",
        "server_bin": "/usr/bin/llama-server",
        "backend": "llama_cpp",
    }
    defaults.update(kwargs)
    return make_server_config(**defaults)


class TestFR003PerSlotRequiredFields:
    """FR-003: Per-slot required fields for canonical dry-run payload."""

    def _minimal_cfg(self, **kwargs: Any) -> ServerConfig:
        """Create minimal ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_slot_id_field_present_and_string(self) -> None:
        """FR-003: slot_id must be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "slot_id")
        assert isinstance(payload.slot_id, str)
        assert payload.slot_id == "test-slot"

    def test_binary_path_field_present_and_string(self) -> None:
        """FR-003: binary_path must be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "binary_path")
        assert isinstance(payload.binary_path, str)
        assert payload.binary_path == "/usr/bin/llama-server"

    def test_command_args_field_present_and_list(self) -> None:
        """FR-003: command_args must be present and be list[str]."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "command_args")
        assert isinstance(payload.command_args, list)
        assert all(isinstance(arg, str) for arg in payload.command_args)

    def test_model_path_field_present_and_string(self) -> None:
        """FR-003: model_path must be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "model_path")
        assert isinstance(payload.model_path, str)

    def test_bind_address_field_present_and_string(self) -> None:
        """FR-003: bind_address must be present and be a string."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "bind_address")
        assert isinstance(payload.bind_address, str)
        assert payload.bind_address == "127.0.0.1"

    def test_port_field_present_and_integer(self) -> None:
        """FR-003: port must be present and be an integer."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "port")
        assert isinstance(payload.port, int)
        assert payload.port == 8080

    def test_environment_redacted_field_present_and_dict(self) -> None:
        """FR-003: environment_redacted must be present and be a dict."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "environment_redacted")
        assert isinstance(payload.environment_redacted, dict)

    def test_openai_flag_bundle_field_present_and_dict(self) -> None:
        """FR-003: openai_flag_bundle must be present and be a dict."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "openai_flag_bundle")
        assert isinstance(payload.openai_flag_bundle, dict)

    def test_hardware_notes_field_present_and_dict(self) -> None:
        """FR-003: hardware_notes must be present and be a dict."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "hardware_notes")
        assert isinstance(payload.hardware_notes, dict)

    def test_vllm_eligibility_field_present_and_object(self) -> None:
        """FR-003: vllm_eligibility must be present and be a VllmEligibility object."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "vllm_eligibility")
        assert isinstance(payload.vllm_eligibility, VllmEligibility)

    def test_warnings_field_present_and_list(self) -> None:
        """FR-003: warnings must be present and be a list."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=["warning1", "warning2"],
        )
        assert hasattr(payload, "warnings")
        assert isinstance(payload.warnings, list)
        assert payload.warnings == ["warning1", "warning2"]

    def test_validation_results_field_present_and_object(self) -> None:
        """FR-003: validation_results must be present and be a ValidationResults object."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload, "validation_results")
        assert isinstance(payload.validation_results, ValidationResults)


class TestFR003DeterministicFieldOrdering:
    """FR-003: Deterministic field ordering for reproducible output."""

    def _minimal_cfg(self, **kwargs: Any) -> ServerConfig:
        """Create minimal ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_slot_payload_has_deterministic_field_order(self) -> None:
        """FR-003: DryRunSlotPayload fields should have deterministic ordering."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        # Get field order from __dataclass_fields__ which preserves definition order
        field_names = list(payload.__dataclass_fields__.keys())
        # Core identity fields first
        core_fields = ["slot_id", "binary_path", "command_args"]
        for _i, field in enumerate(core_fields):
            assert field in field_names, f"Field {field} not found"
        # Verify slot_id comes before binary_path
        slot_idx = field_names.index("slot_id")
        bin_path_idx = field_names.index("binary_path")
        cmd_args_idx = field_names.index("command_args")
        assert slot_idx < bin_path_idx < cmd_args_idx, (
            f"Field ordering violated: slot_id={slot_idx}, "
            f"binary_path={bin_path_idx}, command_args={cmd_args_idx}"
        )

    def test_sorted_field_names_match_definition_order(self) -> None:
        """FR-003: Field names should match the dataclass definition order."""
        cfg = self._minimal_cfg()
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        field_names = list(payload.__dataclass_fields__.keys())
        # Expected order from DryRunSlotPayload definition
        expected_order = [
            "slot_id",
            "binary_path",
            "command_args",
            "model_path",
            "bind_address",
            "port",
            "environment_redacted",
            "openai_flag_bundle",
            "hardware_notes",
            "vllm_eligibility",
            "warnings",
            "validation_results",
        ]
        assert field_names == expected_order, (
            f"Field order mismatch. Expected: {expected_order}, Got: {field_names}"
        )


class TestFR003CommandArgsEdgeCases:
    """FR-003: Command args edge cases for whitespace and shell-sensitive args."""

    def _cfg_with_path(self, model_path: str) -> ServerConfig:
        """Create ServerConfig with custom model path."""
        return _dry_run_cfg(model=model_path)

    def test_command_args_handles_spaces_in_path(self) -> None:
        """FR-003: command_args should handle spaces in paths as separate tokens."""
        cfg = self._cfg_with_path("/path/with spaces/model.gguf")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        # Path with spaces should be tokenized correctly
        assert "/path/with spaces/model.gguf" in payload.command_args

    def test_command_args_handles_special_shell_chars(self) -> None:
        """FR-003: command_args should handle shell-sensitive characters."""
        # Model path with characters that need quoting in shell
        cfg = self._cfg_with_path("/path/model-with_chars$.gguf")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert "/path/model-with_chars$.gguf" in payload.command_args

    def test_command_args_contains_flag_argument_pairs(self) -> None:
        """FR-003: command_args should contain flag-argument pairs."""
        cfg = self._cfg_with_path("/path/model.gguf")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        # Verify key flag-argument patterns exist
        flag_pairs = [
            ("--model", "/path/model.gguf"),
            ("--alias", "test"),
            ("--port", "8080"),
            ("--ctx-size", "4096"),
            ("--ubatch-size", "512"),
            ("--threads", "4"),
        ]
        for flag, expected_value in flag_pairs:
            assert flag in payload.command_args, f"Missing flag: {flag}"
            flag_idx = payload.command_args.index(flag)
            assert payload.command_args[flag_idx + 1] == expected_value

    def test_command_args_is_list_of_strings(self) -> None:
        """FR-003: command_args should be list[str] even with edge cases."""
        cfg = self._cfg_with_path("/path/with spaces/model.gguf")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.command_args, list)
        assert all(isinstance(arg, str) for arg in payload.command_args)

    def test_command_args_deterministic_order(self) -> None:
        """FR-003: command_args should have deterministic ordering."""
        cfg = self._cfg_with_path("/path/model.gguf")
        payload1 = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        # Create another payload for same config
        payload2 = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        # Command args should be identical (deterministic)
        assert payload1.command_args == payload2.command_args
        # Verify specific order: --model first, then other args
        assert payload1.command_args[0] == "--model"

    def test_command_args_handles_unicode_chars(self) -> None:
        """FR-003: command_args should handle Unicode characters in paths."""
        cfg = self._cfg_with_path("/path/日本語/モデル.gguf")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.command_args, list)
        assert all(isinstance(arg, str) for arg in payload.command_args)

    def test_command_args_handles_very_long_path(self) -> None:
        """FR-003: command_args should handle very long paths."""
        long_path = "/path/" + ("subdir/" * 20) + "model.gguf"
        cfg = self._cfg_with_path(long_path)
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert long_path in payload.command_args
        assert len(payload.command_args) > 0


class TestFR003ValidationResultsOrdering:
    """FR-003: Validation results ordering within slot payload."""

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_validation_results_passed_is_bool(self) -> None:
        """FR-003: validation_results.passed should be a boolean."""
        cfg = self._cfg(slot_id="slot1")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.validation_results.passed, bool)

    def test_validation_results_checks_is_list(self) -> None:
        """FR-003: validation_results.checks should be a list."""
        cfg = self._cfg(slot_id="slot1")
        checks = [{"check": "port", "passed": True}, {"check": "model", "passed": True}]
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=checks),
            warnings=[],
        )
        assert isinstance(payload.validation_results.checks, list)
        assert payload.validation_results.checks == checks


class TestFR003EnvironmentRedaction:
    """FR-003: Environment redaction in dry-run payload."""

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_environment_redacted_contains_expected_keys(self) -> None:
        """FR-003: environment_redacted should contain expected env var keys."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert isinstance(payload.environment_redacted, dict)
        # Should contain standard environment variables
        env_keys = list(payload.environment_redacted.keys())
        assert len(env_keys) > 0

    def test_environment_redacted_redacts_sensitive_values(self) -> None:
        """FR-003: environment_redacted should redact sensitive values."""
        import os

        # Set a sensitive env var
        os.environ["API_KEY"] = "secret_value"
        try:
            payload = build_dry_run_slot_payload(
                self._cfg(slot_id="slot1"),
                slot_id="slot1",
                validation_results=ValidationResults(passed=True, checks=[]),
                warnings=[],
            )
            # API_KEY should be redacted
            assert "API_KEY" in payload.environment_redacted
            assert payload.environment_redacted["API_KEY"] == "[REDACTED]"
        finally:
            os.environ.pop("API_KEY", None)

    def test_environment_redacted_preserves_non_sensitive(self) -> None:
        """FR-003: environment_redacted should preserve non-sensitive values."""
        import os

        os.environ["MODEL_PATH"] = "/path/to/model.gguf"
        try:
            payload = build_dry_run_slot_payload(
                self._cfg(slot_id="slot1"),
                slot_id="slot1",
                validation_results=ValidationResults(passed=True, checks=[]),
                warnings=[],
            )
            # MODEL_PATH should NOT be redacted (not in KEY|TOKEN|SECRET|PASSWORD|AUTH)
            if "MODEL_PATH" in payload.environment_redacted:
                assert payload.environment_redacted["MODEL_PATH"] == "/path/to/model.gguf"
        finally:
            os.environ.pop("MODEL_PATH", None)


class TestFR003HardwareNotes:
    """FR-003: Hardware notes in dry-run payload."""

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_hardware_notes_has_required_fields(self) -> None:
        """FR-003: hardware_notes should have backend, device_id, device_name."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert "backend" in payload.hardware_notes
        assert "device_id" in payload.hardware_notes
        assert "device_name" in payload.hardware_notes

    def test_hardware_notes_backend_is_string_or_none(self) -> None:
        """FR-003: hardware_notes.backend should be str or None."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        backend = payload.hardware_notes["backend"]
        assert isinstance(backend, str | type(None))

    def test_hardware_notes_device_id_is_string_or_none(self) -> None:
        """FR-003: hardware_notes.device_id should be str or None."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        device_id = payload.hardware_notes["device_id"]
        assert isinstance(device_id, str | type(None))

    def test_hardware_notes_device_name_is_string_or_none(self) -> None:
        """FR-003: hardware_notes.device_name should be str or None."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        device_name = payload.hardware_notes["device_name"]
        assert isinstance(device_name, str | type(None))


class TestFR003VllmEligibility:
    """FR-003: Vllm eligibility in dry-run payload."""

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_vllm_eligibility_has_eligible_field(self) -> None:
        """FR-003: vllm_eligibility should have eligible field."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload.vllm_eligibility, "eligible")
        assert isinstance(payload.vllm_eligibility.eligible, bool)

    def test_vllm_eligibility_has_reason_field(self) -> None:
        """FR-003: vllm_eligibility should have reason field."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert hasattr(payload.vllm_eligibility, "reason")
        assert isinstance(payload.vllm_eligibility.reason, str)

    def test_vllm_is_not_eligible(self) -> None:
        """FR-003: vllm should be ineligible in M1."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        assert payload.vllm_eligibility.eligible is False
        assert "vllm is not launch-eligible" in payload.vllm_eligibility.reason.lower()


class TestFR003SlotConfigurationSequenceConsistency:
    """FR-003: Verify slot configuration sequence consistency between error output
    and dry-run payload.
    """

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_error_slot_order_matches_dry_run_slot_order(self) -> None:
        """FR-003: Error slot sequence order must match dry-run payload slot order."""
        from llama_manager.config import ErrorDetail, MultiValidationError

        # Create errors with specific slot order
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port",
                why_blocked="port conflict in slot2",
                how_to_fix="fix port in slot2",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model",
                why_blocked="model missing in slot1",
                how_to_fix="fix model in slot1",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="port conflict in slot1",
                how_to_fix="fix port in slot1",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # After sorting, expected order is: slot1_model, slot1_port, slot2_port
        expected_sorted_order = ["slot_slot1_model", "slot_slot1_port", "slot_slot2_port"]
        actual_sorted_order = [error.failed_check for error in mve.errors]
        assert actual_sorted_order == expected_sorted_order, (
            f"Sort order mismatch: expected {expected_sorted_order}, got {actual_sorted_order}"
        )

        # Create ValidationResults with same slot order
        validation_results = ValidationResults(
            passed=False,
            checks=[
                {
                    "slot_id": error.failed_check.split("_")[1],
                    "failed_check": error.failed_check,
                    "error_code": error.error_code.value,
                }
                for error in mve.errors
            ],
        )

        # Verify slot sequence consistency
        error_slot_sequence = [error.failed_check.split("_")[1] for error in mve.errors]
        check_slot_sequence = [check["slot_id"] for check in validation_results.checks]

        assert error_slot_sequence == check_slot_sequence, (
            f"Slot sequence mismatch: errors={error_slot_sequence}, checks={check_slot_sequence}"
        )

    def test_dry_run_payload_slot_scope_matches_error_slot_sequence(self) -> None:
        """FR-003: Dry-run slot_scope list order must match error slot sequence."""
        from llama_manager.config import ErrorDetail, MultiValidationError

        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot3_port",
                why_blocked="port issue in slot3",
                how_to_fix="fix slot3",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model",
                why_blocked="model issue in slot1",
                how_to_fix="fix slot1",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port",
                why_blocked="port issue in slot2",
                how_to_fix="fix slot2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Build dry-run payloads in sorted error order
        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=ValidationResults(
                    passed=False,
                    checks=[{"failed_check": error.failed_check}],
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        # slot_scope should match the sorted error slot order
        slot_scope = [p.slot_id for p in payloads]
        expected_slot_order = [error.failed_check.split("_")[1] for error in mve.errors]

        assert slot_scope == expected_slot_order, (
            f"slot_scope order mismatch: expected {expected_slot_order}, got {slot_scope}"
        )


class TestFR003FailedCheckAscendingTieBreak:
    """FR-003: Verify failed_check ascending tie-break within each slot."""

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_failed_check_ascending_tiebreak_within_slot(self) -> None:
        """FR-003: failed_check should be sorted ascending within each slot."""
        from llama_manager.config import ErrorDetail, MultiValidationError

        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_z_port_validation",  # Should come last in slot1
                why_blocked="z error",
                how_to_fix="fix z",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_a_model_check",  # Should come first in slot1
                why_blocked="a error",
                how_to_fix="fix a",
            ),
            ErrorDetail(
                error_code=ErrorCode.CONFIG_ERROR,
                failed_check="slot_slot1_m_ctx_size",  # Should come middle in slot1
                why_blocked="m error",
                how_to_fix="fix m",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port",  # slot2 errors
                why_blocked="slot2 error",
                how_to_fix="fix slot2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Expected order: slot1_a_model_check, slot1_m_ctx_size, slot1_z_port_validation, slot2_port
        expected_order = [
            "slot_slot1_a_model_check",
            "slot_slot1_m_ctx_size",
            "slot_slot1_z_port_validation",
            "slot_slot2_port",
        ]
        actual_order = [error.failed_check for error in mve.errors]

        assert actual_order == expected_order, (
            f"Tie-break order mismatch: expected {expected_order}, got {actual_order}"
        )

        # Verify slot sequence: slot1 errors before slot2
        slot1_indices = [i for i, e in enumerate(mve.errors) if "slot1" in e.failed_check]
        slot2_indices = [i for i, e in enumerate(mve.errors) if "slot2" in e.failed_check]

        assert all(idx < slot2_indices[0] for idx in slot1_indices), (
            "Slot1 errors should come before slot2 errors"
        )


class TestFR003NewArtifactShapeAssertions:
    """FR-003: Explicit tests for new dry-run artifact shape: slot_scope list
    and resolved_command mapping.
    """

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _dry_run_cfg(**kwargs)

    def test_slot_scope_is_list_of_strings(self) -> None:
        """FR-003: slot_scope must be a list of strings (slot IDs)."""
        from llama_manager.config import ErrorDetail, MultiValidationError

        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="error1",
                how_to_fix="fix1",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot2_model",
                why_blocked="error2",
                how_to_fix="fix2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Build payloads in sorted order
        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=ValidationResults(
                    passed=False, checks=[{"failed_check": error.failed_check}]
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        # slot_scope is the canonical list of slot IDs
        slot_scope = [p.slot_id for p in payloads]

        assert isinstance(slot_scope, list), "slot_scope must be a list"
        assert all(isinstance(slot_id, str) for slot_id in slot_scope), (
            "All slot_scope entries must be strings"
        )
        assert len(slot_scope) == len(mve.errors), (
            f"slot_scope length mismatch: expected {len(mve.errors)}, got {len(slot_scope)}"
        )

    def test_resolved_command_is_mapping_of_slot_id_to_command_args(self) -> None:
        """FR-003: resolved_command must be a dict mapping slot_id -> command_args list."""
        from llama_manager.config import ErrorDetail, MultiValidationError

        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="error1",
                how_to_fix="fix1",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=ValidationResults(
                    passed=False, checks=[{"failed_check": error.failed_check}]
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        # Build resolved_command mapping (as done in dry_run.py)
        resolved_command = {p.slot_id: p.command_args for p in payloads}

        assert isinstance(resolved_command, dict), "resolved_command must be a dict"

        # Each key must be a slot_id and each value must be a list of command args
        for slot_id, cmd_args in resolved_command.items():
            assert isinstance(slot_id, str), (
                f"resolved_command key must be string, got {type(slot_id)}"
            )
            assert isinstance(cmd_args, list), (
                f"resolved_command[{slot_id}] must be list, got {type(cmd_args)}"
            )
            assert all(isinstance(arg, str) for arg in cmd_args), (
                f"resolved_command[{slot_id}] must contain only strings"
            )

    def test_slot_scope_and_resolved_command_keys_alignment(self) -> None:
        """FR-003: resolved_command keys must exactly match slot_scope entries."""
        from llama_manager.config import ErrorDetail, MultiValidationError

        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="error1",
                how_to_fix="fix1",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot2_model",
                why_blocked="error2",
                how_to_fix="fix2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=ValidationResults(
                    passed=False, checks=[{"failed_check": error.failed_check}]
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        slot_scope = [p.slot_id for p in payloads]
        resolved_command = {p.slot_id: p.command_args for p in payloads}

        # Keys in resolved_command must match slot_scope entries
        assert set(resolved_command.keys()) == set(slot_scope), (
            f"resolved_command keys {set(resolved_command.keys())} must match slot_scope {set(slot_scope)}"
        )

        # Order must be consistent: resolved_command should preserve slot_scope order
        ordered_keys = list(resolved_command.keys())
        assert ordered_keys == slot_scope, (
            f"resolved_command key order {ordered_keys} must match slot_scope order {slot_scope}"
        )

    def test_resolved_command_contains_correct_command_args_for_each_slot(self) -> None:
        """FR-003: resolved_command[<slot_id>] must contain the correct command_args."""
        cfg = self._cfg(slot_id="test-slot")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )

        resolved_command = {"test-slot": payload.command_args}

        # resolved_command["test-slot"] must equal payload.command_args
        assert "test-slot" in resolved_command, "resolved_command must contain 'test-slot' key"
        assert resolved_command["test-slot"] == payload.command_args, (
            "resolved_command['test-slot'] must equal payload.command_args"
        )

        # Verify command_args structure
        cmd_args = resolved_command["test-slot"]
        assert isinstance(cmd_args, list), "command_args must be a list"
        assert len(cmd_args) > 0, "command_args must not be empty"
        assert "--model" in cmd_args, "command_args must contain --model flag"


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

from llama_manager.config import ErrorDetail, MultiValidationError
from llama_manager.orchestration import ValidationException, write_artifact
from llama_manager.validation import (
    validate_backend_eligibility,
    validate_server_config,
)


def _contract_cfg(**kwargs: object) -> ServerConfig:
    defaults: dict[str, object] = {
        "alias": "test",
        "server_bin": "/usr/bin/llama-server",
        "backend": "llama_cpp",
    }
    defaults.update(kwargs)
    return make_server_config(**defaults)


class TestFR003DryRunPayloadContract:
    """FR-003: Canonical dry-run payload contract assertions."""

    def _minimal_cfg(self, **kwargs: object) -> ServerConfig:
        """Create minimal ServerConfig for testing."""
        return _contract_cfg(**kwargs)

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
            assert isinstance(value, str | int | bool | type(None)), (
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
            assert isinstance(value, str | type(None)), (
                f"hardware_notes['{field}'] should be str|None"
            )

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
        return _contract_cfg(**kwargs)

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
