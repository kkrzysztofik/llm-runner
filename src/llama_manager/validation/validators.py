"""Input validation functions for server configuration."""

import os

from ..common.validators import PORT_MAX, PORT_MIN, validate_port_range
from ..config import (
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
)


def validate_port(port: int, name: str = "port") -> ErrorDetail | None:
    """Validate port number."""
    err = validate_port_range(port)
    if err is not None:
        return ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked=f"{name}: {err}",
            how_to_fix=f"ensure {name} is an integer between {PORT_MIN} and {PORT_MAX}",
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


def _validate_duplicate_slots(slots: list[ModelSlot]) -> list[ErrorDetail]:
    """Validate for duplicate slot IDs."""
    from ..config import detect_duplicate_slots

    results: list[ErrorDetail] = []
    duplicates = detect_duplicate_slots(slots)
    for dup_slot_id in duplicates:
        results.append(
            ErrorDetail(
                error_code=ErrorCode.DUPLICATE_SLOT,
                failed_check="duplicate_slot_detection",
                why_blocked=f"Duplicate slot_id detected: {dup_slot_id}",
                how_to_fix=f"Fix duplicate slot_id for slot {dup_slot_id}",
                slot_id=dup_slot_id,
            )
        )
    return results


def _validate_slot(slot: ModelSlot) -> list[ErrorDetail]:
    """Validate a single slot configuration."""
    from ..config import normalize_slot_id

    results: list[ErrorDetail] = []

    try:
        normalized_id = normalize_slot_id(slot.slot_id)
    except ValueError as e:
        results.append(
            ErrorDetail(
                error_code=ErrorCode.INVALID_SLOT_ID,
                failed_check="slot_id_validation",
                why_blocked=str(e),
                how_to_fix=f"Fix slot_id_validation for slot {slot.slot_id}",
                slot_id=slot.slot_id,
            )
        )
        return results

    port_err = validate_port_range(slot.port)
    if port_err is not None:
        results.append(
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="port_validation",
                why_blocked=port_err,
                how_to_fix=f"Fix port_validation for slot {normalized_id}",
                slot_id=normalized_id,
            )
        )

    if slot.model_path:
        if os.path.isdir(slot.model_path):
            results.append(
                ErrorDetail(
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    failed_check="model_path_validation",
                    why_blocked="model_path must be a file, not a directory",
                    how_to_fix=f"Fix model_path_validation for slot {normalized_id}",
                    slot_id=normalized_id,
                )
            )
        elif not os.path.isfile(slot.model_path):
            results.append(
                ErrorDetail(
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    failed_check="model_path_validation",
                    why_blocked=f"model_path does not exist: {slot.model_path}",
                    how_to_fix=f"Fix model_path_validation for slot {normalized_id}",
                    slot_id=normalized_id,
                )
            )

    return results


def _convert_results_to_errors(validation_results: list[ErrorDetail]) -> MultiValidationError:
    """Convert validation results to MultiValidationError."""
    if not validation_results:
        return MultiValidationError(errors=[])

    from .commands.builder import sort_validation_errors

    return MultiValidationError(errors=sort_validation_errors(validation_results))


def validate_slots(slots: list[ModelSlot]) -> MultiValidationError | None:
    """Validate ModelSlot configurations and return MultiValidationError if any fail."""
    duplicate_results = _validate_duplicate_slots(slots)

    slot_results: list[ErrorDetail] = []
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
