# Server command building and validation functions


import json
import os
import re
from dataclasses import dataclass
from typing import Any, Final

from .config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    ValidationResult,
)

# Precompiled regex pattern for sensitive key detection (Finding 172)
_SENSITIVE_KEY_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|AUTH)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Doctor diagnostics (T069)
# ---------------------------------------------------------------------------


@dataclass
class DoctorCheckResult:
    """Result of a single doctor diagnostic check.

    Attributes:
        name: Check identifier (e.g. 'sycl_device', 'cuda_memory').
        status: One of 'pass', 'warn', 'fail'.
        message: Human-readable description of the result.
    """

    name: str
    status: str  # "pass", "warn", "fail"
    message: str = ""


@dataclass
class DoctorReport:
    """Aggregated doctor diagnostic report.

    Attributes:
        checks: List of individual check results.
    """

    checks: list[DoctorCheckResult]

    def to_json(self) -> str:
        """Return JSON string representation.

        Returns:
            JSON string with check results.
        """
        return json.dumps(
            {
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status,
                        "message": c.message,
                    }
                    for c in self.checks
                ]
            },
            indent=2,
        )

    def to_text(self) -> str:
        """Return human-readable text representation.

        Returns:
            Formatted text with pass/warn/fail indicators.
        """
        lines: list[str] = ["=== DOCTOR DIAGNOSTIC REPORT ==="]
        for check in self.checks:
            icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(check.status, "?")
            lines.append(f"  [{icon}] {check.name}: {check.message}")
        return "\n".join(lines)


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


def build_server_cmd(cfg: ServerConfig, default_bin: str | None = None) -> list[str]:
    """Build llama-server command arguments.

    Args:
        cfg: ServerConfig to build command from.
        default_bin: Optional default binary path if cfg.server_bin is empty.
                    If not provided, falls back to Config().llama_server_bin_intel.

    Returns:
        List of command arguments for subprocess.

    """
    # Get binary path - prefer cfg.server_bin, then default_bin, then Config()
    if cfg.server_bin:
        server_bin = cfg.server_bin
    elif default_bin:
        server_bin = default_bin
    else:
        server_bin = Config().llama_server_bin_intel

    cmd = [
        server_bin,
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
    if _SENSITIVE_KEY_PATTERN.search(env_key):
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


def _validate_duplicate_slots(slots: list["ModelSlot"]) -> list[ValidationResult]:
    """Validate for duplicate slot IDs.

    Args:
        slots: List of ModelSlot objects to check.

    Returns:
        List of validation results for duplicate slot errors.
    """
    from .config import detect_duplicate_slots

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


def _validate_slot(slot: "ModelSlot") -> list[ValidationResult]:
    """Validate a single slot configuration.

    Args:
        slot: ModelSlot to validate.

    Returns:
        List of validation results (empty if all checks pass).
    """
    from .config import normalize_slot_id

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

    # Validate port
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

    # Validate model_path exists if specified
    if slot.model_path:
        if os.path.isfile(slot.model_path):
            # Model file exists, validation passes
            pass
        elif os.path.isdir(slot.model_path):
            # Directory not allowed
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
            # Path doesn't exist
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
    """Convert validation results to MultiValidationError.

    Args:
        validation_results: List of validation results to convert.

    Returns:
        MultiValidationError with sorted error details.
    """
    failed = [r for r in validation_results if not r.passed]
    if not failed:
        return MultiValidationError(errors=[])

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
    # Check for duplicate slot IDs first
    duplicate_results = _validate_duplicate_slots(slots)

    # Validate each slot individually
    slot_results: list[ValidationResult] = []
    for slot in slots:
        slot_results.extend(_validate_slot(slot))

    # Combine all validation results
    all_results = duplicate_results + slot_results

    # Return None if all passed, otherwise convert to error
    if all_results:
        return _convert_results_to_errors(all_results)
    return None


def detect_risky_operations(cfg: ServerConfig) -> list[str]:
    """Detect potentially risky operations in server configuration.

    Risky operations include:
    - Privileged ports (< 1024)
    - Non-loopback bind addresses (not 127.0.0.1 or ::1)
    - Warning bypass acknowledgement path

    Args:
        cfg: Server configuration to analyze

    Returns:
        List of risk identifiers (e.g., 'privileged_port', 'non_loopback')

    """
    risks: list[str] = []

    if cfg.port < 1024:
        risks.append("privileged_port")

    if cfg.bind_address not in ("127.0.0.1", "::1"):
        risks.append("non_loopback")

    if "warning_bypass" in cfg.risky_acknowledged:
        risks.append("warning_bypass")

    return risks


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
    # mypy false positive: both are dict[str, str | None]
    hardware_notes = _build_hardware_notes(cfg)

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


def validate_port(port: int, name: str = "port") -> ErrorDetail | None:
    """Validate port number.

    Args:
        port: Port number to validate
        name: Name of the port for error messages

    Returns:
        ErrorDetail if validation fails, None if valid
    """
    if not isinstance(port, int) or port < 1 or port > 65535:
        return ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_validation",
            why_blocked=f"{name} must be between 1 and 65535, got: {port}",
            how_to_fix=f"ensure {name} is an integer between 1 and 65535",
        )
    return None


def validate_threads(threads: int, name: str = "threads") -> ErrorDetail | None:
    """Validate thread count.

    Args:
        threads: Thread count to validate
        name: Name of the threads parameter for error messages

    Returns:
        ErrorDetail if validation fails, None if valid
    """
    if not isinstance(threads, int) or threads < 1:
        return ErrorDetail(
            error_code=ErrorCode.THREADS_INVALID,
            failed_check="thread_validation",
            why_blocked=f"{name} must be greater than 0, got: {threads}",
            how_to_fix=f"ensure {name} is a positive integer",
        )
    return None


def require_model(model_path: str) -> ErrorDetail | None:
    """Check if model file exists.

    Args:
        model_path: Path to the model file

    Returns:
        ErrorDetail if model not found, None if exists
    """
    if not os.path.isfile(model_path):
        return ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="model_path_exists",
            why_blocked=f"model not found: {model_path}",
            how_to_fix="verify model path exists and is accessible",
        )
    return None


def require_executable(bin_path: str, name: str = "binary") -> ErrorDetail | None:
    """Check if executable exists.

    Args:
        bin_path: Path to the executable
        name: Name of the executable for error messages

    Returns:
        ErrorDetail if not executable, None if exists and executable
    """
    if not os.access(bin_path, os.X_OK):
        return ErrorDetail(
            error_code=ErrorCode.PERMISSION_DENIED,
            failed_check="executable_exists",
            why_blocked=f"{name} not found or not executable: {bin_path}",
            how_to_fix="verify executable path exists and has execute permissions",
        )
    return None


def validate_ports(
    port1: int, port2: int, name1: str = "port1", name2: str = "port2"
) -> ErrorDetail | None:
    """Validate ports are different.

    Args:
        port1: First port number
        port2: Second port number
        name1: Name of first port for error messages
        name2: Name of second port for error messages

    Returns:
        ErrorDetail if ports are the same, None if different
    """
    if port1 == port2:
        return ErrorDetail(
            error_code=ErrorCode.PORT_CONFLICT,
            failed_check="port_uniqueness",
            why_blocked=f"{name1} and {name2} must be different, got: {port1}",
            how_to_fix="ensure both ports are unique values between 1 and 65535",
        )
    return None


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

    FR-003: Ensures explicit and deterministic OpenAI compatibility bundle for Qwen-class configs.
    Keys are sorted alphabetically for deterministic serialization.

    Args:
        cfg: ServerConfig to derive flags from

    Returns:
        Dict with OpenAI API compatibility flags (keys with leading dashes, mixed value types).
        Keys are deterministically ordered (sorted ascending).

    """
    # Determine if chat completion is supported based on reasoning mode
    chat_completion_supported = cfg.reasoning_mode in ("auto", "enabled")

    # Build bundle with explicit keys for Qwen-class configs
    # Keys sorted alphabetically for deterministic serialization (FR-003)
    bundle: dict[str, str | int | bool | None] = {
        "--chat-format": "chatml" if chat_completion_supported else None,
        "--host": "127.0.0.1",
        "--openai": True,
        "--port": cfg.port,
    }

    return bundle


def _build_hardware_notes(cfg: ServerConfig) -> dict[str, str | None]:
    """Build hardware notes dict describing backend and hardware.

    Args:
        cfg: ServerConfig to derive hardware info from

    Returns:
        Dict with required fields: backend, device_id, device_name;
        optional fields: driver_version, runtime_version (may be None)

    """
    backend = cfg.backend or "llama_cpp"
    device = cfg.device or "auto"
    device_id, device_name = _parse_device_details(device)

    return {
        "backend": backend,
        "device_id": device_id,
        "device_name": device_name,
        "driver_version": None,
        "runtime_version": None,
    }


def _parse_device_details(device: str) -> tuple[str | None, str]:
    if device == "auto":
        return (None, device)

    if device.startswith("cuda:"):
        parts = device.split(":", maxsplit=1)
        if len(parts) == 2 and parts[1]:
            return (parts[1], "NVIDIA GPU")
        return (None, device)

    if device.startswith("sycl:"):
        parts = device.split(":")
        if len(parts) >= 3:
            return (f"{parts[1]}:{parts[2]}", f"SYCL Device {parts[1]}")
        if len(parts) > 1:
            return (":".join(parts[1:]), device)

    return (None, device)


def compute_machine_fingerprint() -> str | None:
    """Compute a deterministic machine fingerprint from hardware identifiers.

    Gathers information from system tools (lspci, /proc/cpuinfo, /etc/os-release)
    and combines them into a SHA-256 hash. Returns None if no tools succeed.

    Returns:
        A hex-encoded SHA-256 hash string, or None if all tools fail.

    """
    import subprocess

    parts: list[str] = []

    # GPU info from lspci
    try:
        result = subprocess.run(
            ["lspci"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append("gpu:" + result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass

    # CPU info from /proc/cpuinfo
    try:
        result = subprocess.run(
            ["cat", "/proc/cpuinfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Extract model name for compact representation
            for line in result.stdout.splitlines():
                if line.startswith("model name"):
                    parts.append("cpu:" + line.split(":", 1)[1].strip())
                    break
    except (OSError, subprocess.TimeoutExpired):
        pass

    # OS info from /etc/os-release
    try:
        result = subprocess.run(
            ["cat", "/etc/os-release"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                if line.startswith("NAME="):
                    parts.append("os:" + line.split("=", 1)[1].strip().strip('"'))
                    break
    except (OSError, subprocess.TimeoutExpired):
        pass

    if not parts:
        return None

    import hashlib

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def check_hardware_allowlist(
    fingerprint: str,
    allowlist: list[str] | None = None,
) -> str:
    """Check a machine fingerprint against a hardware allowlist.

    Args:
        fingerprint: The machine fingerprint to check.
        allowlist: List of allowed fingerprints. If None, reads from
                   LLM_RUNNER_HARDWARE_ALLOWLIST env var.

    Returns:
        'match' if fingerprint is in allowlist,
        'mismatch' if allowlist has entries but fingerprint is not included,
        'invalidated' if allowlist is empty (no trusted fingerprints).

    """
    if allowlist is None:
        import os

        raw = os.environ.get("LLM_RUNNER_HARDWARE_ALLOWLIST", "")
        allowlist = [f for f in raw.split(",") if f.strip()] if raw else []

    if not allowlist:
        return "invalidated"

    if fingerprint in allowlist:
        return "match"

    return "mismatch"


def assess_vram_risk(
    vram_total_gb: float,
    vram_free_gb: float,
    model_size_gb: float,
) -> str:
    """Assess VRAM risk for loading a model.

    Heuristic per spec FR-013 / AC-016:
    warn if ``free_vram * 0.85 < model_size * 1.2``
    which simplifies to ``free_vram < model_size * (1.2 / 0.85)`` ≈ model_size * 1.411.

    Thresholds:
    - PROCEED: free VRAM >= 1.5x model size
    - WARN: free VRAM >= 1.411x model size (1.2/0.85 per spec)
    - CONFIRM_REQUIRED: free VRAM < 1.411x model size

    Args:
        vram_total_gb: Total GPU VRAM in gigabytes.
        vram_free_gb: Available (free) GPU VRAM in gigabytes.
        model_size_gb: Estimated model size in gigabytes.

    Returns:
        VRamRecommendation string value.

    """
    from .config import VRamRecommendation

    if model_size_gb <= 0:
        return VRamRecommendation.PROCEED

    # Spec formula: free_vram * 0.85 < model_size * 1.2
    # => warn when free_vram < model_size * (1.2 / 0.85)
    _WARN_THRESHOLD: Final[float] = 1.2 / 0.85  # ≈ 1.4117647
    _PROCEED_THRESHOLD: Final[float] = 1.5

    ratio = vram_free_gb / model_size_gb

    if ratio >= _PROCEED_THRESHOLD:
        return VRamRecommendation.PROCEED
    if ratio >= _WARN_THRESHOLD:
        return VRamRecommendation.WARN
    return VRamRecommendation.CONFIRM_REQUIRED
