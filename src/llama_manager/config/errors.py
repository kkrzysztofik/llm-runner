# Validation result and structured error types

from dataclasses import dataclass

from .enums import ErrorCode


@dataclass
class ValidationResult:
    """Result of a validation check with slot identity for T003 deterministic sorting"""

    slot_id: str
    passed: bool
    failed_check: str = ""
    error_code: ErrorCode | None = None
    error_message: str = ""

    @property
    def valid(self) -> bool:
        """Alias for passed to maintain backward compatibility"""
        return self.passed


@dataclass
class ErrorDetail:
    """FR-005 structured actionable error detail"""

    error_code: ErrorCode
    failed_check: str
    why_blocked: str
    how_to_fix: str
    docs_ref: str | None = None


@dataclass
class MultiValidationError:
    """FR-005 container for multiple validation errors with deterministic ordering"""

    errors: list[ErrorDetail]

    @property
    def error_count(self) -> int:
        """Return the number of errors in this multi-error"""
        return len(self.errors)

    def sort_errors(self) -> None:
        """Sort errors in-place by slot configuration sequence, then failed_check ascending.

        This provides stable, deterministic ordering for consistent error output.
        Slots are ordered alphabetically by slot_id; within each slot, failed_check is sorted alphabetically.

        Slot ID extraction:
        - Pattern: failed_check starts with "slot_<slot_id>_<check>"
        - slot_id is the second underscore-separated component (e.g., "slot_slot1_a_check" -> slot1)
        - Normalize by stripping "slot_" prefix (e.g., "slot_slot1" -> "slot1")
        """
        if not self.errors:
            return

        slot_ids = sorted(
            {
                slot_id
                for error in self.errors
                if (slot_id := _extract_slot_id(error.failed_check)) is not None
            }
        )
        slot_order = {slot_id: idx for idx, slot_id in enumerate(slot_ids)}

        self.errors = sorted(
            self.errors,
            key=lambda error: _error_sort_key(error, slot_order, len(slot_ids) + 1),
        )


def _extract_slot_id(failed_check: str) -> str | None:
    if not failed_check.startswith("slot_"):
        return None

    parts = failed_check.split("_", maxsplit=2)
    if len(parts) < 2 or not parts[1]:
        return None
    return parts[1]


def _error_sort_key(
    error: ErrorDetail,
    slot_order: dict[str, int],
    default_index: int,
) -> tuple[int, str]:
    slot_id = _extract_slot_id(error.failed_check)
    if slot_id is None:
        return (default_index, error.failed_check)
    return (slot_order.get(slot_id, default_index), error.failed_check)
