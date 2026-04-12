# Server command building and validation functions


import os
import re
import sys
from dataclasses import dataclass
from typing import Any

from .config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    ValidationResult,
)


# FR-003: Canonical dry-run payload types
@dataclass
class VllmEligibility:
    """FR-003: vLLM eligibility status for a slot."""

    eligible: bool
    reason: str


@dataclass
class ValidationResults:
    """FR-003: Aggregated validation results for a slot."""

    passed: bool
    checks: list[dict[str, Any]]


@dataclass
class DryRunSlotPayload:
    """FR-003: Canonical dry-run slot payload with deterministic field ordering.

    Fields are ordered to provide stable serialization:
    1. slot_id, binary_path, command_args (core identity)
    2. model_path, bind_address, port (network/model config)
    3. environment_redacted, openai_flag_bundle (runtime environment)
    4. hardware_notes, vllm_eligibility (hardware/backend info)
    5. warnings, validation_results (diagnostics)
    """

    slot_id: str
    binary_path: str
    command_args: list[str]
    model_path: str
    bind_address: str
    port: int
    environment_redacted: dict[str, str]
    openai_flag_bundle: dict[str, str | int | bool | None]
    hardware_notes: dict[str, str | None]
    vllm_eligibility: VllmEligibility
    warnings: list[str]
    validation_results: ValidationResults


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


def validate_backend_eligibility(backend: str) -> ErrorDetail | None:
    """FR-011: Validate backend eligibility for M1 launch.

    vllm is not launch-eligible in PRD M1 - only llama_cpp is supported.

    Args:
        backend: Backend name to validate (e.g., 'vllm', 'llama_cpp')

    Returns:
        ErrorDetail if backend is not eligible, None if eligible
    """
    if backend.lower() == "vllm":
        return ErrorDetail(
            error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
            failed_check="vllm_launch_eligibility",
            why_blocked="vllm is not launch-eligible in PRD M1",
            how_to_fix="change backend to 'llama_cpp' for M1",
        )
    return None


def validate_server_config(cfg: ServerConfig) -> ErrorDetail | None:
    """FR-011: Validate ServerConfig for M1 launch eligibility.

    Validates backend eligibility (vllm is blocked in M1).

    Args:
        cfg: ServerConfig to validate

    Returns:
        ErrorDetail if validation fails, None if valid
    """
    return validate_backend_eligibility(cfg.backend)


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


def build_dry_run_slot_payload(
    cfg: ServerConfig,
    slot_id: str,
    bind_address: str = "127.0.0.1",
    validation_results: ValidationResults | None = None,
    warnings: list[str] | None = None,
) -> DryRunSlotPayload:
    """FR-003: Build canonical dry-run slot payload from ServerConfig + slot_id.

    Returns a typed DryRunSlotPayload with deterministic field ordering.
    The payload is suitable for dry-run output without serializing in the backend.

    Args:
        cfg: ServerConfig to build payload from
        slot_id: Slot identifier for this payload
        bind_address: Bind address for the server (default: 127.0.0.1)
        validation_results: Optional validation results (defaults to passed=True)
        warnings: Optional list of warning messages (defaults to empty list)

    Returns:
        DryRunSlotPayload with all fields populated deterministically
    """
    # Build command args (excluding binary path which is in binary_path field)
    cmd = build_server_cmd(cfg)
    command_args = cmd[1:]  # Exclude binary path

    # Redact sensitive environment variables
    environment_redacted = _build_environment_redacted()

    # OpenAI flag bundle - indicates API compatibility flags
    openai_flag_bundle = _build_openai_flag_bundle(cfg)

    # Hardware notes - describe backend and hardware characteristics
    hardware_notes = _build_hardware_notes(cfg)  # type: ignore[assignment]

    # vLLM eligibility - M1: vllm not eligible
    vllm_eligibility = VllmEligibility(
        eligible=False,
        reason="vllm is not launch-eligible in PRD M1 - only llama_cpp supported",
    )

    # Validation results - default to passed if not provided
    if validation_results is None:
        validation_results = ValidationResults(
            passed=True,
            checks=[],
        )

    # Warnings - default to empty list if not provided
    if warnings is None:
        warnings = []

    return DryRunSlotPayload(
        slot_id=slot_id,
        binary_path=cmd[0],
        command_args=command_args,
        model_path=cfg.model,
        bind_address=bind_address,
        port=cfg.port,
        environment_redacted=environment_redacted,
        openai_flag_bundle=openai_flag_bundle,
        hardware_notes=hardware_notes,
        vllm_eligibility=vllm_eligibility,
        warnings=warnings,
        validation_results=validation_results,
    )


def _error_detail_to_stderr(error_detail: ErrorDetail) -> None:
    """FR-005: Print structured error detail to stderr.

    Outputs consistent fields for easy parsing:
    error_code=<code> failed_check=<check> why_blocked=<message> how_to_fix=<fix>

    Args:
        error_detail: ErrorDetail with error_code, failed_check, why_blocked, how_to_fix
    """
    error_code = (
        error_detail.error_code.value
        if isinstance(error_detail.error_code, ErrorCode)
        else str(error_detail.error_code)
    )
    print(
        f"error: error_code={error_code} failed_check={error_detail.failed_check} "
        f"why_blocked={error_detail.why_blocked} how_to_fix={error_detail.how_to_fix}",
        file=sys.stderr,
    )


def validate_port(port: int, name: str = "port") -> None:
    """Validate port number"""
    if not isinstance(port, int) or port < 1 or port > 65535:
        error_detail = ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked=f"{name} must be between 1 and 65535, got: {port}",
            how_to_fix=f"ensure {name} is an integer between 1 and 65535",
        )
        _error_detail_to_stderr(error_detail)
        sys.exit(1)


def validate_threads(threads: int, name: str = "threads") -> None:
    """Validate thread count"""
    if not isinstance(threads, int) or threads < 1:
        error_detail = ErrorDetail(
            error_code=ErrorCode.THREADS_INVALID,
            failed_check="thread_validation",
            why_blocked=f"{name} must be greater than 0, got: {threads}",
            how_to_fix=f"ensure {name} is a positive integer",
        )
        _error_detail_to_stderr(error_detail)
        sys.exit(1)


def require_model(model_path: str) -> None:
    """Check if model file exists"""
    if not os.path.isfile(model_path):
        error_detail = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_path_exists",
            why_blocked=f"model not found: {model_path}",
            how_to_fix="verify model path exists and is accessible",
        )
        _error_detail_to_stderr(error_detail)
        sys.exit(1)


def require_executable(bin_path: str, name: str = "binary") -> None:
    """Check if executable exists"""
    if not os.access(bin_path, os.X_OK):
        error_detail = ErrorDetail(
            error_code=ErrorCode.PERMISSION_DENIED,
            failed_check="executable_exists",
            why_blocked=f"{name} not found or not executable: {bin_path}",
            how_to_fix="verify executable path exists and has execute permissions",
        )
        _error_detail_to_stderr(error_detail)
        sys.exit(1)


def validate_ports(port1: int, port2: int, name1: str = "port1", name2: str = "port2") -> None:
    """Validate ports are different"""
    if port1 == port2:
        error_detail = ErrorDetail(
            error_code=ErrorCode.PORT_CONFLICT,
            failed_check="port_uniqueness",
            why_blocked=f"{name1} and {name2} must be different, got: {port1}",
            how_to_fix="ensure both ports are unique values between 1 and 65535",
        )
        _error_detail_to_stderr(error_detail)
        sys.exit(1)


def _build_environment_redacted() -> dict[str, str]:
    """FR-007: Build environment variable map with sensitive values redacted.

    Returns a dict with environment variable keys and redacted values.
    Includes all environment variables from os.environ with sensitive values redacted.
    """
    env_vars_to_check = [
        "PATH",
        "HOME",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "SYCL_DEVICE_SELECTOR",
        "HF_HOME",
        "HF_HUB_CACHE",
    ]

    result: dict[str, str] = {}

    # First add the standard environment variables
    for key in env_vars_to_check:
        value = os.environ.get(key, "")
        result[key] = redact_sensitive(value, key)

    # Also include any additional environment variables from os.environ
    # that aren't already in the result, with sensitive values redacted
    for key, value in os.environ.items():
        if key not in result:
            result[key] = redact_sensitive(value, key)

    return result


def _build_openai_flag_bundle(cfg: ServerConfig) -> dict[str, str | int | bool | None]:
    """Build OpenAI API compatibility flag bundle.

    Args:
        cfg: ServerConfig to derive flags from

    Returns:
        Dict with OpenAI API compatibility flags (keys with leading dashes, mixed value types)
    """
    # Determine if chat completion is supported based on reasoning mode
    chat_completion_supported = cfg.reasoning_mode in ("auto", "enabled")

    return {
        "--port": cfg.port,
        "--host": "127.0.0.1",
        "--chat-format": "chatml" if chat_completion_supported else None,
        "--openai": True,
    }


def _build_hardware_notes(cfg: ServerConfig) -> dict[str, str | None]:
    """Build hardware notes dict describing backend and hardware.

    Args:
        cfg: ServerConfig to derive hardware info from

    Returns:
        Dict with required fields: backend, device_id, device_name;
        optional fields: driver_version, runtime_version (may be None)
    """
    backend = cfg.backend if cfg.backend else "llama_cpp"
    device = cfg.device if cfg.device else "auto"

    # Parse device string to extract device_id and device_name
    # Format can be "cuda:X" or "sycl:X:Y" or just "auto"
    device_id: str | None = None
    device_name: str = device

    if device != "auto":
        if device.startswith("cuda:"):
            try:
                device_id = device.split(":")[1]
                device_name = "NVIDIA GPU"
            except (IndexError, ValueError):
                device_id = None
                device_name = device
        elif device.startswith("sycl:"):
            parts = device.split(":")
            if len(parts) >= 3:
                device_id = f"{parts[1]}:{parts[2]}"
                device_name = f"SYCL Device {parts[1]}"
            else:
                device_id = ":".join(parts[1:]) if len(parts) > 1 else None

    return {
        "backend": backend,
        "device_id": device_id,
        "device_name": device_name,
        "driver_version": None,
        "runtime_version": None,
    }
