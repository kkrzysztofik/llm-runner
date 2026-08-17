"""Structured validation error types."""

from dataclasses import dataclass

from .enums import ErrorCode


@dataclass
class ErrorDetail:
    """FR-005 structured actionable error detail.

    Also used as a success result when ``passed=True`` (with empty strings
    for the error-specific fields).
    """

    error_code: ErrorCode | None = None
    failed_check: str = ""
    why_blocked: str = ""
    how_to_fix: str = ""
    docs_ref: str | None = None
    slot_id: str = ""
    passed: bool = False


@dataclass
class MultiValidationError:
    """FR-005 container for multiple validation errors with deterministic ordering"""

    errors: list[ErrorDetail]


class ValidationException(Exception):
    """Exception wrapper for MultiValidationError to enable raising as exception."""

    def __init__(self, multi_error: MultiValidationError) -> None:
        self.multi_error = multi_error
        if multi_error.errors:
            details = "; ".join(e.why_blocked for e in multi_error.errors)
            super().__init__(
                f"Validation failed with {len(multi_error.errors)} error(s): {details}"
            )
        else:
            super().__init__(f"Validation failed with {len(multi_error.errors)} error(s)")
