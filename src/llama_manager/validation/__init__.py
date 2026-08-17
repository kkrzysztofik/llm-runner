"""validation package — input validation and server command building."""

from .commands.builder import (
    DryRunSlotPayload,
    DryRunValidationSummary,
    VllmEligibility,
    build_dry_run_slot_payload,
    build_server_cmd,
)
from .validators import (
    detect_risky_operations,
    require_executable,
    require_model,
    validate_port,
    validate_ports,
    validate_server_config,
)

__all__ = [
    # Validators
    "validate_port",
    "validate_ports",
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
]
