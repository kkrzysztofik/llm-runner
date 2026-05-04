"""Backward compatibility shim. Import from llama_manager.validation instead."""

from .config import ServerConfig  # noqa: F401
from .validation import *  # noqa: F401, F403
from .validation.commands import (  # noqa: F401
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
from .validation.validators import (  # noqa: F401
    _convert_results_to_errors,
    _validate_duplicate_slots,
    _validate_slot,
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
