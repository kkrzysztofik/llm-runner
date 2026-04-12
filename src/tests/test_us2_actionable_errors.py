"""US2 FR-005 multi-error schema and ordering tests.

Test Tasks:
- T023: Add FR-005 single/multi-error schema and ordering tests verifying:
  (1) MultiValidationError has errors: list[ErrorDetail] with error_count,
  (2) ordering by slot configuration sequence (slot_id iteration order);
      when tie-breaking, use failed_check ascending within slot,
  (3) each ErrorDetail has error_code, failed_check, why_blocked, how_to_fix,
      optional docs_ref fields,
  (4) SC-002 denominator counts all errors[n] entries across runs

Contract:
- FR-005: Actionable error schema with error_code, failed_check, why_blocked, how_to_fix
- MultiValidationError: Container for multiple errors with sort_errors() method
- SC-002: Denominator-style counting across error lists
"""

from llama_manager.config import (
    ErrorCode,
    ErrorDetail,
    MultiValidationError,
)


class TestFR005SingleErrorSchema:
    """FR-005: Single ErrorDetail schema assertions."""

    def test_error_detail_required_fields_present(self) -> None:
        """ErrorDetail must have error_code, failed_check, why_blocked, how_to_fix."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="port must be between 1 and 65535",
            how_to_fix="set port to a valid value in range 1-65535",
        )
        assert hasattr(error, "error_code")
        assert hasattr(error, "failed_check")
        assert hasattr(error, "why_blocked")
        assert hasattr(error, "how_to_fix")

    def test_error_detail_optional_docs_ref_field(self) -> None:
        """ErrorDetail should support optional docs_ref field."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="port must be between 1 and 65535",
            how_to_fix="set port to a valid value in range 1-65535",
            docs_ref="https://docs.example.com/port-validation",
        )
        assert hasattr(error, "docs_ref")
        assert error.docs_ref == "https://docs.example.com/port-validation"

    def test_error_detail_with_none_docs_ref(self) -> None:
        """ErrorDetail should work with docs_ref=None."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="port must be between 1 and 65535",
            how_to_fix="set port to a valid value",
            docs_ref=None,
        )
        assert error.docs_ref is None

    def test_error_detail_error_code_is_valid_enum(self) -> None:
        """ErrorDetail.error_code should be a valid ErrorCode enum value."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        assert isinstance(error.error_code, ErrorCode)
        assert error.error_code == ErrorCode.PORT_INVALID

    def test_error_detail_all_fields_populated(self) -> None:
        """ErrorDetail should work with all fields including docs_ref."""
        error = ErrorDetail(
            error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
            failed_check="vllm_launch_eligibility",
            why_blocked="vllm is not launch-eligible in PRD M1",
            how_to_fix="change backend to 'llama_cpp' for M1",
            docs_ref="https://docs.example.com/backend-eligibility",
        )
        assert error.error_code == ErrorCode.BACKEND_NOT_ELIGIBLE
        assert error.failed_check == "vllm_launch_eligibility"
        assert "vllm is not launch-eligible" in error.why_blocked
        assert "llama_cpp" in error.how_to_fix
        assert error.docs_ref == "https://docs.example.com/backend-eligibility"


class TestFR005MultiValidationErrorSchema:
    """FR-005: MultiValidationError container schema assertions."""

    def test_multi_validation_error_has_errors_field(self) -> None:
        """MultiValidationError must have errors field (list[ErrorDetail])."""
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error1])
        assert hasattr(multi, "errors")
        assert isinstance(multi.errors, list)
        assert len(multi.errors) == 1

    def test_multi_validation_error_error_count_property(self) -> None:
        """MultiValidationError should have error_count property."""
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        error2 = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_validation",
            why_blocked="file missing",
            how_to_fix="add file",
        )
        multi = MultiValidationError(errors=[error1, error2])
        assert hasattr(multi, "error_count")
        assert multi.error_count == 2

    def test_multi_validation_error_empty_list(self) -> None:
        """MultiValidationError should handle empty errors list."""
        multi = MultiValidationError(errors=[])
        assert multi.errors == []
        assert multi.error_count == 0

    def test_multi_validation_error_multiple_errors(self) -> None:
        """MultiValidationError should support multiple errors."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="model_validation",
                why_blocked="file missing",
                how_to_fix="add file",
            ),
            ErrorDetail(
                error_code=ErrorCode.DUPLICATE_SLOT,
                failed_check="duplicate_detection",
                why_blocked="duplicate",
                how_to_fix="rename",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        assert len(multi.errors) == 3
        assert multi.error_count == 3


class TestFR005ErrorOrdering:
    """FR-005: Error ordering and sorting semantics."""

    def test_sort_errors_orders_by_slot_id_first(self) -> None:
        """sort_errors should order by slot_id iteration sequence first."""
        # Create errors in non-sequential slot order
        error2 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot2_port_validation",
            why_blocked="slot2 invalid",
            how_to_fix="fix slot2",
        )
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_port_validation",
            why_blocked="slot1 invalid",
            how_to_fix="fix slot1",
        )
        error3 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot3_port_validation",
            why_blocked="slot3 invalid",
            how_to_fix="fix slot3",
        )
        multi = MultiValidationError(errors=[error2, error1, error3])
        multi.sort_errors()
        # After sorting, errors should be in slot_id order: slot1, slot2, slot3
        assert multi.errors[0].failed_check == "slot_slot1_port_validation"
        assert multi.errors[1].failed_check == "slot_slot2_port_validation"
        assert multi.errors[2].failed_check == "slot_slot3_port_validation"

    def test_sort_errors_tie_breaks_by_failed_check_ascending(self) -> None:
        """sort_errors should tie-break by failed_check ascending within same slot."""
        # Same slot, different failed_check values
        error_b = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_b_port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        error_a = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_a_port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error_b, error_a])
        multi.sort_errors()
        # Should tie-break by failed_check ascending: a_port_validation before b_port_validation
        assert multi.errors[0].failed_check == "slot_slot1_a_port_validation"
        assert multi.errors[1].failed_check == "slot_slot1_b_port_validation"

    def test_sort_errors_mixed_slots_and_checks(self) -> None:
        """sort_errors should handle mixed slot ordering with tie-breaking."""
        # Multiple slots, some with multiple errors
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_b_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_a_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_b_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_a_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        multi.sort_errors()
        # Expected order: slot1_a, slot1_b, slot2_a, slot2_b
        assert multi.errors[0].failed_check == "slot_slot1_a_port_validation"
        assert multi.errors[1].failed_check == "slot_slot1_b_port_validation"
        assert multi.errors[2].failed_check == "slot_slot2_a_port_validation"
        assert multi.errors[3].failed_check == "slot_slot2_b_port_validation"

    def test_sort_errors_does_not_modify_original_order_without_sort(self) -> None:
        """Errors should remain in original order until sort_errors is called."""
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        error2 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot2_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error1, error2])
        # Before sorting - original order preserved
        assert multi.errors[0] is error1
        assert multi.errors[1] is error2
        # Sort to reorder
        multi.sort_errors()
        # After sorting - order changed
        assert multi.errors[0] is error1  # slot1 comes first
        assert multi.errors[1] is error2


class TestSC002DenominatorCounting:
    """SC-002: Denominator-style counting across error lists."""

    def test_error_count_as_denominator(self) -> None:
        """error_count should serve as denominator for percentage calculations."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation2",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation3",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        total_errors = multi.error_count
        # Verify we can use error_count as denominator
        assert total_errors == 3
        # Simulate counting successful validations vs total
        failed = 2
        passed = total_errors - failed
        # This is the SC-002 pattern: denominator is error_count
        assert passed / total_errors == 1 / 3

    def test_error_count_increases_with_more_errors(self) -> None:
        """error_count should increase as more errors are added."""
        multi = MultiValidationError(errors=[])
        assert multi.error_count == 0
        error1 = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi.errors.append(error1)
        assert multi.error_count == 1
        error2 = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_validation",
            why_blocked="file missing",
            how_to_fix="add file",
        )
        multi.errors.append(error2)
        assert multi.error_count == 2

    def test_error_count_stable_after_sorting(self) -> None:
        """error_count should remain stable after sort_errors call."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        initial_count = multi.error_count
        multi.sort_errors()
        assert multi.error_count == initial_count
        assert multi.error_count == 2

    def test_error_count_counts_all_errors_n_entries(self) -> None:
        """error_count should count all entries in errors[n] across runs."""
        # Simulate multiple runs
        all_counts = []
        for i in range(5):
            errors = [
                ErrorDetail(
                    error_code=ErrorCode.PORT_INVALID,
                    failed_check=f"port_{i}_validation",
                    why_blocked=f"invalid {i}",
                    how_to_fix=f"fix {i}",
                )
            ]
            multi = MultiValidationError(errors=errors)
            all_counts.append(multi.error_count)
        # Each run should have counted correctly
        assert all(count == 1 for count in all_counts)
        assert len(all_counts) == 5

    def test_error_count_for_empty_validation(self) -> None:
        """error_count should be 0 when no validation failures."""
        multi = MultiValidationError(errors=[])
        assert multi.error_count == 0
        # No errors means denominator would cause division by zero in percentage calc
        # This is expected behavior - caller must handle zero denominator


class TestMultiValidationErrorFieldTypes:
    """FR-005: MultiValidationError field type assertions."""

    def test_errors_is_list_of_error_detail(self) -> None:
        """MultiValidationError.errors should be list[ErrorDetail]."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error])
        assert isinstance(multi.errors, list)
        assert all(isinstance(e, ErrorDetail) for e in multi.errors)

    def test_error_detail_fields_are_strings(self) -> None:
        """ErrorDetail string fields should be strings."""
        error = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked="this is why",
            how_to_fix="this is how",
        )
        assert isinstance(error.error_code, ErrorCode)
        assert isinstance(error.failed_check, str)
        assert isinstance(error.why_blocked, str)
        assert isinstance(error.how_to_fix, str)


class TestFR005DeterministicOrdering:
    """FR-005: Deterministic error ordering for reproducible output."""

    def test_sort_errors_produces_deterministic_result(self) -> None:
        """sort_errors should produce deterministic ordering across runs."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_z_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_a_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_b_port_validation",
                why_blocked="invalid",
                how_to_fix="fix",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        multi.sort_errors()
        # First run
        run1_order = [e.failed_check for e in multi.errors]
        # Create new instance and sort again
        multi2 = MultiValidationError(errors=errors.copy())
        multi2.sort_errors()
        run2_order = [e.failed_check for e in multi2.errors]
        # Orders should match (deterministic)
        assert run1_order == run2_order
        # Expected order: slot_a, slot_b, slot_z
        assert run1_order == [
            "slot_a_port_validation",
            "slot_b_port_validation",
            "slot_z_port_validation",
        ]

    def test_sort_errors_handles_none_slot_id_gracefully(self) -> None:
        """sort_errors should handle ErrorDetail without slot_id pattern gracefully."""
        # Error without slot_id pattern in failed_check
        error_without_slot = ErrorDetail(
            error_code=ErrorCode.CONFIG_ERROR,
            failed_check="config_validation",
            why_blocked="config invalid",
            how_to_fix="fix config",
        )
        error_with_slot = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_port_validation",
            why_blocked="invalid",
            how_to_fix="fix",
        )
        multi = MultiValidationError(errors=[error_without_slot, error_with_slot])
        multi.sort_errors()
        # Errors should be ordered with slot errors first, then others
        # slot1 comes before config (based on slot_id extraction logic)
        assert multi.errors[0].failed_check == "slot_slot1_port_validation"
        assert multi.errors[1].failed_check == "config_validation"
