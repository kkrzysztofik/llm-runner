"""validation package — input validation and server command building."""

from .commands.builder import (
    DoctorCheckResult,
    DoctorReport,
    DryRunSlotPayload,
    DryRunValidationSummary,
    VllmEligibility,
    assess_vram_risk,
    build_dry_run_slot_payload,
    build_server_cmd,
    check_hardware_allowlist,
    compute_machine_fingerprint,
    sort_validation_errors,
)
from .validators import (
    detect_risky_operations,
    require_executable,
    require_model,
    validate_backend_eligibility,
    validate_port,
    validate_ports,
    validate_server_config,
    validate_slots,
    validate_threads,
)

ValidationResults = DryRunValidationSummary

__all__ = [
    # Validators
    "validate_port",
    "validate_threads",
    "validate_ports",
    "validate_slots",
    "validate_backend_eligibility",
    "validate_server_config",
    "require_model",
    "require_executable",
    "detect_risky_operations",
    # Command builder
    "build_server_cmd",
    "build_dry_run_slot_payload",
    # Payload types
    "DryRunSlotPayload",
    "VllmEligibility",
    "DryRunValidationSummary",
    "ValidationResults",
    # Doctor diagnostics
    "DoctorCheckResult",
    "DoctorReport",
    # VRAM assessment
    "assess_vram_risk",
    # Hardware fingerprinting
    "compute_machine_fingerprint",
    "check_hardware_allowlist",
    # Sorting
    "sort_validation_errors",
]
