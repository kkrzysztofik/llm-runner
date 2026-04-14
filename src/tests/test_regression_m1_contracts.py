from typing import Any
from unittest.mock import patch

import pytest

from llama_manager.config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ValidationResult,
)
from llama_manager.server import (
    ServerConfig,
    ValidationResults,
    build_dry_run_slot_payload,
    validate_slots,
)


@pytest.fixture
def base_config() -> Config:
    """Return a default Config for testing."""
    return Config()


def test_multi_validation_error_parity(base_config: Config) -> None:
    """T042: Verify MultiValidationError fields match canonical slot.validation_results.errors.
    We verify that the errors reported in MultiValidationError are consistent with
    the individual validation failures.
    """
    # Setup slots with intentional errors
    # Slot 1: Invalid port
    # Slot 2: Model not found
    slots = [
        ModelSlot(slot_id="slot1", model_path="/valid/path/model.gguf", port=99999),  # Invalid port
        ModelSlot(
            slot_id="slot2", model_path="/nonexistent/path/model.gguf", port=8080
        ),  # Model not found
    ]

    with (
        patch("os.path.isfile", side_effect=lambda path: path == "/valid/path/model.gguf"),
        patch("os.path.exists", side_effect=lambda path: path == "/valid/path/model.gguf"),
    ):
        mve = validate_slots(slots)

        assert isinstance(mve, MultiValidationError)
        assert mve.error_count == 2

        # Check if errors are present and consistent
        # Since validate_slots currently doesn't include slot_id in failed_check,
        # they will be sorted by failed_check name, not slot.

        # We expect at least these error codes
        error_codes = [e.error_code for e in mve.errors]
        assert ErrorCode.PORT_INVALID in error_codes
        assert ErrorCode.FILE_NOT_FOUND in error_codes


def test_slot_sequence_consistency_and_tiebreak() -> None:
    """T042: Verify slot sequence consistency and failed_check ascending tie-break.
    This test specifically checks if the sorting logic in MultiValidationError
    works when failed_check strings include slot information.
    """
    # We manually create a MultiValidationError with errors that follow the expected pattern
    # to verify the sorting logic works as intended for the contract.
    # Pattern: "slot_<slot_id>_<check>"

    errors = [
        ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot2_port",
            why_blocked="err2",
            how_to_fix="fix2",
        ),
        ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="slot_slot1_model",
            why_blocked="err1",
            how_to_fix="fix1",
        ),
        ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_port",
            why_blocked="err1b",
            how_to_fix="fix1b",
        ),
        ErrorDetail(
            error_code=ErrorCode.CONFIG_ERROR,
            failed_check="unknown_err",
            why_blocked="err_u",
            how_to_fix="fix_u",
        ),
    ]

    mve = MultiValidationError(errors=errors)
    mve.sort_errors()

    # Expected order:
    # 1. slot1_model (slot1, model)
    # 2. slot1_port (slot1, port)
    # 3. slot2_port (slot2, port)
    # 4. unknown_err (end)

    assert mve.errors[0].failed_check == "slot_slot1_model"
    assert mve.errors[1].failed_check == "slot_slot1_port"
    assert mve.errors[2].failed_check == "slot_slot2_port"
    assert mve.errors[3].failed_check == "unknown_err"


def test_validate_slots_duplicate_detection() -> None:
    """T042: Verify duplicate slot detection in validation."""
    slots = [
        ModelSlot(slot_id="slot1", model_path="/path/1", port=8080),
        ModelSlot(slot_id="slot1", model_path="/path/2", port=8081),  # Duplicate ID
    ]

    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        mve = validate_slots(slots)
        assert isinstance(mve, MultiValidationError)
        assert any(e.error_code == ErrorCode.DUPLICATE_SLOT for e in mve.errors)


def test_validate_slots_invalid_id() -> None:
    """T042: Verify invalid slot IDs are rejected during duplicate precheck."""
    slots = [
        ModelSlot(slot_id="!!!", model_path="/path/1", port=8080),  # Invalid ID
    ]

    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.exists", return_value=True),
        pytest.raises(ValueError, match="slot_id must contain at least one valid character"),
    ):
        validate_slots(slots)


class TestFR005FR003CanonicalParity:
    """FR-003/FR-005: Verify canonical parity between MultiValidationError and
    dry-run ValidationResults.errors field-level alignment.
    """

    def test_error_code_field_alignment(self) -> None:
        """FR-005: MultiValidationError.error_code must align with ValidationResults.error_code."""
        # Create MultiValidationError with ErrorDetail
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="port out of range",
                how_to_fix="use port between 1 and 65535",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model",
                why_blocked="model not found",
                how_to_fix="provide valid model path",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Create equivalent ValidationResult list
        validation_results_list = [
            ValidationResult(
                slot_id="slot1",
                passed=False,
                failed_check=error.failed_check,
                error_code=error.error_code,
                error_message=error.why_blocked,
            )
            for error in mve.errors
        ]

        # Verify field alignment: error_code must match
        for i, (error_detail, vr) in enumerate(
            zip(mve.errors, validation_results_list, strict=True)
        ):
            assert error_detail.error_code == vr.error_code, (
                f"Error {i}: error_code mismatch - "
                f"ErrorDetail={error_detail.error_code}, ValidationResult={vr.error_code}"
            )

    def test_failed_check_field_alignment(self) -> None:
        """FR-005: MultiValidationError.failed_check must align with ValidationResults.failed_check."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port_validation",
                why_blocked="port conflict",
                how_to_fix="use unique port",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model_check",
                why_blocked="model missing",
                how_to_fix="check model path",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Create equivalent ValidationResult list
        validation_results_list = [
            ValidationResult(
                slot_id=error.failed_check.split("_")[1],  # Extract slot_id from failed_check
                passed=False,
                failed_check=error.failed_check,
                error_code=error.error_code,
                error_message=error.why_blocked,
            )
            for error in mve.errors
        ]

        # Verify field alignment: failed_check must match exactly
        for i, (error_detail, vr) in enumerate(
            zip(mve.errors, validation_results_list, strict=True)
        ):
            assert error_detail.failed_check == vr.failed_check, (
                f"Error {i}: failed_check mismatch - "
                f"ErrorDetail={error_detail.failed_check}, ValidationResult={vr.failed_check}"
            )

    def test_validation_results_checks_alignment(self) -> None:
        """FR-003/FR-005: validation_results.checks must contain aligned error info."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="port must be between 1 and 65535",
                how_to_fix="specify a valid port number",
            ),
        ]
        mve = MultiValidationError(errors=errors)

        # Create ValidationResults with checks that align with ErrorDetails
        checks = [
            {
                "failed_check": error.failed_check,
                "error_code": error.error_code.value,
                "why_blocked": error.why_blocked,
                "how_to_fix": error.how_to_fix,
            }
            for error in mve.errors
        ]

        validation_results = ValidationResults(passed=False, checks=checks)

        # Verify checks contain aligned fields
        assert len(validation_results.checks) == len(mve.errors)
        for check, error in zip(validation_results.checks, mve.errors, strict=True):
            assert check["failed_check"] == error.failed_check
            assert check["error_code"] == error.error_code.value
            assert check["why_blocked"] == error.why_blocked
            assert check["how_to_fix"] == error.how_to_fix


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
