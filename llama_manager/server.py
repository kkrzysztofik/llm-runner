# Server command building and validation functions


import os
import re
import sys

from .config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    ValidationResult,
)


def build_server_cmd(cfg: ServerConfig) -> list[str]:
    """Build llama-server command arguments"""
    cmd = [
        cfg.server_bin or Config().llama_server_bin_intel,
        "--model",
        cfg.model,
        "--alias",
        cfg.alias,
        "--n-gpu-layers",
        str(cfg.n_gpu_layers),
        "--split-mode",
        "layer",
        "--ctx-size",
        str(cfg.ctx_size),
        "--flash-attn",
        "on",
        "--cache-type-k",
        cfg.cache_type_k,
        "--cache-type-v",
        cfg.cache_type_v,
        "--batch-size",
        "2048",
        "--ubatch-size",
        str(cfg.ubatch_size),
        "--threads",
        str(cfg.threads),
        "--poll",
        "50",
        "--mmap",
        "--host",
        Config().host,
        "--port",
        str(cfg.port),
        "--no-webui",
    ]

    if cfg.device:
        cmd.extend(["--device", cfg.device])
    if cfg.reasoning_mode:
        cmd.extend(["--reasoning", cfg.reasoning_mode])
    if cfg.reasoning_format:
        cmd.extend(["--reasoning-format", cfg.reasoning_format])
    if cfg.tensor_split:
        cmd.extend(["--tensor-split", cfg.tensor_split])
    if cfg.chat_template_kwargs:
        cmd.extend(["--chat-template-kwargs", cfg.chat_template_kwargs])
    if cfg.reasoning_budget:
        cmd.extend(["--reasoning-budget", cfg.reasoning_budget])
    if cfg.use_jinja:
        cmd.append("--jinja")

    return cmd


def validate_port(port: int, name: str = "port") -> None:
    """Validate port number"""
    if not isinstance(port, int) or port < 1 or port > 65535:
        print(f"error: {name} must be between 1 and 65535, got: {port}", file=sys.stderr)
        sys.exit(1)


def validate_threads(threads: int, name: str = "threads") -> None:
    """Validate thread count"""
    if not isinstance(threads, int) or threads < 1:
        print(f"error: {name} must be greater than 0, got: {threads}", file=sys.stderr)
        sys.exit(1)


def require_model(model_path: str) -> None:
    """Check if model file exists"""
    if not os.path.isfile(model_path):
        print(f"error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)


def require_executable(bin_path: str, name: str = "binary") -> None:
    """Check if executable exists"""
    if not os.access(bin_path, os.X_OK):
        print(f"error: {name} not found or not executable: {bin_path}", file=sys.stderr)
        sys.exit(1)


def validate_ports(port1: int, port2: int, name1: str = "port1", name2: str = "port2") -> None:
    """Validate ports are different"""
    if port1 == port2:
        print(
            f"error: {name1} and {name2} must be different, got: {port1}",
            file=sys.stderr,
        )
        sys.exit(1)


def sort_validation_errors(
    results: list[ValidationResult],
) -> list[ValidationResult]:
    """Sort validation errors deterministically for T003 stable ordering.

    Sort order:
    1. Primary: slot configuration sequence (first occurrence in input list)
    2. Secondary: failed_check ascending within each slot

    This provides stable, deterministic ordering for consistent validation output.

    Args:
        results: List of ValidationResult objects with slot_id, passed, failed_check,
                 error_code, and error_message fields

    Returns:
        Sorted list with slots in input order, failed_check alphabetically within slots.
    """
    # Build slot order map: slot_id -> order index (first occurrence wins)
    slot_order: dict[str, int] = {}
    for i, r in enumerate(results):
        if r.slot_id not in slot_order:
            slot_order[r.slot_id] = i

    def sort_key(r: ValidationResult) -> tuple[int, str]:
        """Sort key: (slot_sequence_order, failed_check)"""
        slot_idx = slot_order[r.slot_id]
        failed_check = r.failed_check or ""
        return (slot_idx, failed_check)

    return sorted(results, key=sort_key)


def redact_sensitive(env_value: str, env_key: str) -> str:
    """FR-007: Redact sensitive environment variable values.

    Matches environment variable key names (not values) containing KEY|TOKEN|SECRET|PASSWORD|AUTH
    (case-insensitive) and replaces the value with "[REDACTED]".

    Args:
        env_value: The environment variable value to potentially redact
        env_key: The environment variable key/name to check for sensitive patterns

    Returns:
        "[REDACTED]" if the key matches a sensitive pattern, otherwise the original value

    Example:
        >>> redact_sensitive("my_secret_value", "API_KEY")
        "[REDACTED]"
        >>> redact_sensitive("/path/to/model", "MODEL_PATH")
        "/path/to/model"
    """
    sensitive_patterns = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|AUTH)", re.IGNORECASE)
    if sensitive_patterns.search(env_key):
        return "[REDACTED]"
    return env_value


def validate_slots(slots: list["ModelSlot"]) -> MultiValidationError | None:
    """Validate ModelSlot configurations and return MultiValidationError if any fail.

    FR-005 structured actionable error object builder. Validates:
    - slot_id normalization (allowed characters a-z0-9_-)
    - duplicate slot detection
    - model_path existence (if file)
    - port validity (1-65535)

    Args:
        slots: List of ModelSlot objects to validate

    Returns:
        MultiValidationError if validation fails, None if all slots pass validation
    """
    from .config import (
        detect_duplicate_slots,
        normalize_slot_id,
    )

    validation_results: list[ValidationResult] = []

    # Check for duplicate slot IDs first
    duplicates = detect_duplicate_slots(slots)
    for dup_slot_id in duplicates:
        validation_results.append(
            ValidationResult(
                slot_id=dup_slot_id,
                passed=False,
                failed_check="duplicate_slot_detection",
                error_code=ErrorCode.DUPLICATE_SLOT,
                error_message=f"Duplicate slot_id detected: {dup_slot_id}",
            )
        )

    # Validate each slot
    for slot in slots:
        try:
            normalized_id = normalize_slot_id(slot.slot_id)
        except ValueError as e:
            validation_results.append(
                ValidationResult(
                    slot_id=slot.slot_id,
                    passed=False,
                    failed_check="slot_id_validation",
                    error_code=ErrorCode.INVALID_SLOT_ID,
                    error_message=str(e),
                )
            )
            continue

        # Validate port
        if not (1 <= slot.port <= 65535):
            validation_results.append(
                ValidationResult(
                    slot_id=normalized_id,
                    passed=False,
                    failed_check="port_validation",
                    error_code=ErrorCode.PORT_INVALID,
                    error_message=f"port must be between 1 and 65535, got: {slot.port}",
                )
            )

        # Validate model_path exists if it's a file path
        if slot.model_path and os.path.isfile(slot.model_path):
            # Model file exists, validation passes
            pass
        elif slot.model_path and not os.path.exists(slot.model_path):
            validation_results.append(
                ValidationResult(
                    slot_id=normalized_id,
                    passed=False,
                    failed_check="model_path_validation",
                    error_code=ErrorCode.FILE_NOT_FOUND,
                    error_message=f"model path not found: {slot.model_path}",
                )
            )

    # Convert validation results to MultiValidationError if any failed
    failed = [r for r in validation_results if not r.passed]
    if failed:
        # Sort errors for deterministic output
        sorted_results = sort_validation_errors(failed)

        # Convert ValidationResult to ErrorDetail
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

    return None
