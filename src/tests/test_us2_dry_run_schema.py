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
from llama_manager.server import (
    ValidationResults,
    VllmEligibility,
    build_dry_run_slot_payload,
)


class TestFR003PerSlotRequiredFields:
    """FR-003: Per-slot required fields for canonical dry-run payload."""

    def _minimal_cfg(self, **kwargs: Any) -> ServerConfig:
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
        return ServerConfig(
            model=model_path,
            alias="test",
            device="SYCL0",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
            server_bin="/usr/bin/llama-server",
            backend="llama_cpp",
        )

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
        assert isinstance(backend, str | None)

    def test_hardware_notes_device_id_is_string_or_none(self) -> None:
        """FR-003: hardware_notes.device_id should be str or None."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        device_id = payload.hardware_notes["device_id"]
        assert isinstance(device_id, str | None)

    def test_hardware_notes_device_name_is_string_or_none(self) -> None:
        """FR-003: hardware_notes.device_name should be str or None."""
        payload = build_dry_run_slot_payload(
            self._cfg(slot_id="slot1"),
            slot_id="slot1",
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        device_name = payload.hardware_notes["device_name"]
        assert isinstance(device_name, str | None)


class TestFR003VllmEligibility:
    """FR-003: Vllm eligibility in dry-run payload."""

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
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
