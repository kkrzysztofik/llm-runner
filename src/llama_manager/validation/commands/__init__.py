"""commands subpackage — server command building and dry-run payloads."""

from .builder import (
    DoctorCheckResult,
    DoctorReport,
    DryRunSlotPayload,
    ValidationResults,
    VllmEligibility,
    _build_environment_redacted,
    _build_hardware_notes,
    _build_openai_flag_bundle,
    _get_cpu_model,
    _get_lspci_output,
    _get_os_name,
    _parse_device_details,
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
    "ValidationResults",
    "DoctorCheckResult",
    "DoctorReport",
    "assess_vram_risk",
    "compute_machine_fingerprint",
    "check_hardware_allowlist",
    "sort_validation_errors",
    # Private (exported for backward compat)
    "_build_environment_redacted",
    "_build_hardware_notes",
    "_build_openai_flag_bundle",
    "_get_cpu_model",
    "_get_lspci_output",
    "_get_os_name",
    "_parse_device_details",
]
