"""commands subpackage — server command building and dry-run payloads."""

from .builder import (
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

__all__ = [
    "build_server_cmd",
    "build_dry_run_slot_payload",
    "DryRunSlotPayload",
    "VllmEligibility",
    "DryRunValidationSummary",
    "DoctorCheckResult",
    "DoctorReport",
    "assess_vram_risk",
    "compute_machine_fingerprint",
    "check_hardware_allowlist",
    "sort_validation_errors",
]
