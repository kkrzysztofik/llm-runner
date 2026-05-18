"""Input validation functions for server configuration."""

import os

from ..config import (
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    ValidationResult,
)


def validate_port(port: int, name: str = "port") -> ErrorDetail | None:
    """Validate port number."""
    if not isinstance(port, int) or port < 1 or port > 65535:
        return ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked=f"{name} must be between 1 and 65535, got: {port}",
            how_to_fix=f"ensure {name} is an integer between 1 and 65535",
        )
    return None


def validate_threads(threads: int, name: str = "threads") -> ErrorDetail | None:
    """Validate thread count."""
    if not isinstance(threads, int) or threads < 1:
        return ErrorDetail(
            error_code=ErrorCode.THREADS_INVALID,
            failed_check="thread_validation",
            why_blocked=f"{name} must be greater than 0, got: {threads}",
            how_to_fix=f"ensure {name} is a positive integer",
        )
    return None


def require_model(model_path: str) -> ErrorDetail | None:
    """Check if model file exists."""
    if not os.path.isfile(model_path):
        return ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_path_exists",
            why_blocked=f"model not found: {model_path}",
            how_to_fix="verify model path exists and is accessible",
        )
    return None


def require_executable(bin_path: str, name: str = "binary") -> ErrorDetail | None:
    """Check if executable exists."""
    if not os.access(bin_path, os.X_OK):
        return ErrorDetail(
            error_code=ErrorCode.PERMISSION_DENIED,
            failed_check="executable_exists",
            why_blocked=f"{name} not found or not executable: {bin_path}",
            how_to_fix="verify executable path exists and has execute permissions",
        )
    return None


def validate_backend_eligibility(backend: str) -> ErrorDetail | None:
    """FR-011: Validate backend eligibility for M1 launch."""
    if backend.lower() == "vllm":
        return ErrorDetail(
            error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
            failed_check="vllm_launch_eligibility",
            why_blocked="vllm is not launch-eligible in PRD M1",
            how_to_fix="change backend to 'llama_cpp' for M1",
        )
    return None


def validate_server_config(cfg: ServerConfig) -> ErrorDetail | None:
    """FR-011: Validate ServerConfig for M1 launch eligibility."""
    return validate_backend_eligibility(cfg.backend)


def _validate_duplicate_slots(slots: list[ModelSlot]) -> list[ValidationResult]:
    """Validate for duplicate slot IDs."""
    from ..config import detect_duplicate_slots

    results: list[ValidationResult] = []
    duplicates = detect_duplicate_slots(slots)
    for dup_slot_id in duplicates:
        results.append(
            ValidationResult(
                slot_id=dup_slot_id,
                passed=False,
                failed_check="duplicate_slot_detection",
                error_code=ErrorCode.DUPLICATE_SLOT,
                error_message=f"Duplicate slot_id detected: {dup_slot_id}",
            )
        )
    return results


def _validate_slot(slot: ModelSlot) -> list[ValidationResult]:
    """Validate a single slot configuration."""
    from ..config import normalize_slot_id

    results: list[ValidationResult] = []

    try:
        normalized_id = normalize_slot_id(slot.slot_id)
    except ValueError as e:
        results.append(
            ValidationResult(
                slot_id=slot.slot_id,
                passed=False,
                failed_check="slot_id_validation",
                error_code=ErrorCode.INVALID_SLOT_ID,
                error_message=str(e),
            )
        )
        return results

    if not (1 <= slot.port <= 65535):
        results.append(
            ValidationResult(
                slot_id=normalized_id,
                passed=False,
                failed_check="port_validation",
                error_code=ErrorCode.PORT_INVALID,
                error_message=f"port must be between 1 and 65535, got: {slot.port}",
            )
        )

    if slot.model_path:
        if os.path.isfile(slot.model_path):
            pass
        elif os.path.isdir(slot.model_path):
            results.append(
                ValidationResult(
                    slot_id=normalized_id,
                    passed=False,
                    failed_check="model_path_validation",
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    error_message="model_path must be a file, not a directory",
                )
            )
        else:
            results.append(
                ValidationResult(
                    slot_id=normalized_id,
                    passed=False,
                    failed_check="model_path_validation",
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    error_message=f"model_path does not exist: {slot.model_path}",
                )
            )

    return results


def _convert_results_to_errors(validation_results: list[ValidationResult]) -> MultiValidationError:
    """Convert validation results to MultiValidationError."""
    failed = [r for r in validation_results if not r.passed]
    if not failed:
        return MultiValidationError(errors=[])

    from .commands.builder import sort_validation_errors

    sorted_results = sort_validation_errors(failed)

    error_details: list[ErrorDetail] = []
    for result in sorted_results:
        error_detail = ErrorDetail(
            error_code=result.error_code or ErrorCode.CONFIG_ERROR,
            failed_check=result.failed_check,
            why_blocked=result.error_message,
            how_to_fix=f"Fix {result.failed_check} for slot {result.slot_id}",
        )
        error_details.append(error_detail)

    return MultiValidationError(errors=error_details)


def validate_slots(slots: list[ModelSlot]) -> MultiValidationError | None:
    """Validate ModelSlot configurations and return MultiValidationError if any fail."""
    duplicate_results = _validate_duplicate_slots(slots)

    slot_results: list[ValidationResult] = []
    for slot in slots:
        slot_results.extend(_validate_slot(slot))

    all_results = duplicate_results + slot_results

    if all_results:
        return _convert_results_to_errors(all_results)
    return None


def validate_ports(
    port1: int, port2: int, name1: str = "port1", name2: str = "port2"
) -> ErrorDetail | None:
    """Validate ports are different."""
    if port1 == port2:
        return ErrorDetail(
            error_code=ErrorCode.PORT_CONFLICT,
            failed_check="port_uniqueness",
            why_blocked=f"{name1} and {name2} must be different, got: {port1}",
            how_to_fix="ensure both ports are unique values between 1 and 65535",
        )
    return None


def detect_risky_operations(cfg: ServerConfig) -> list[str]:
    """Detect potentially risky operations in server configuration."""
    risks: list[str] = []

    if cfg.port < 1024:
        risks.append("privileged_port")

    if cfg.bind_address not in ("127.0.0.1", "::1"):
        risks.append("non_loopback")

    if "warning_bypass" in cfg.risky_acknowledged:
        risks.append("warning_bypass")

    return risks
