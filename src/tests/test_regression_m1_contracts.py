from unittest.mock import patch

import pytest

from llama_manager.config import Config, ErrorCode, ModelSlot, MultiValidationError
from llama_manager.server import (
    validate_slots,
)


@pytest.fixture
def base_config():
    return Config()


def test_multi_validation_error_parity(base_config):
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


def test_slot_sequence_consistency_and_tiebreak():
    """T042: Verify slot sequence consistency and failed_check ascending tie-break.
    This test specifically checks if the sorting logic in MultiValidationError
    works when failed_check strings include slot information.
    """
    from llama_manager.config import ErrorDetail

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


def test_validate_slots_duplicate_detection():
    """T042: Verify duplicate slot detection in validation."""
    slots = [
        ModelSlot(slot_id="slot1", model_path="/path/1", port=8080),
        ModelSlot(slot_id="slot1", model_path="/path/2", port=8081),  # Duplicate ID
    ]

    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        mve = validate_slots(slots)
        assert isinstance(mve, MultiValidationError)
        assert any(e.error_code == ErrorCode.DUPLICATE_SLOT for e in mve.errors)


def test_validate_slots_invalid_id():
    """T042: Verify invalid slot IDs are rejected during duplicate precheck."""
    slots = [
        ModelSlot(slot_id="!!!", model_path="/path/1", port=8080),  # Invalid ID
    ]

    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        with pytest.raises(ValueError, match="slot_id must contain at least one valid character"):
            validate_slots(slots)
