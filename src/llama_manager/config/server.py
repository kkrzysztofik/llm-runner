# ServerConfig, ModelSlot, and slot utility functions

import re
from dataclasses import dataclass, field

from .errors import ErrorCode, ValidationResult

# Regex pattern for slot ID normalization: strip, lowercase, allow only a-z0-9_-
_SLOT_ID_PATTERN = re.compile(r"[^a-z0-9_-]")


@dataclass
class ServerConfig:
    """Individual server configuration"""

    model: str
    alias: str
    device: str
    port: int
    ctx_size: int
    ubatch_size: int
    threads: int
    bind_address: str = "127.0.0.1"
    tensor_split: str = ""
    reasoning_mode: str = "auto"
    reasoning_format: str = "none"
    chat_template_kwargs: str = ""
    reasoning_budget: str = ""
    use_jinja: bool = False
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    n_gpu_layers: int | str = 99
    server_bin: str = ""
    backend: str = "llama_cpp"
    risky_acknowledged: list[str] = field(default_factory=list)


@dataclass
class ModelSlot:
    """Model slot configuration for multi-slot serving"""

    slot_id: str
    model_path: str
    port: int


def normalize_slot_id(slot_id: str) -> str:
    """Normalize slot ID by stripping whitespace, lowercasing ASCII, allowing only a-z0-9_-.

    Args:
        slot_id: Raw slot identifier string

    Returns:
        Normalized slot ID with only allowed characters (lowercase a-z, digits, underscore, hyphen)

    Raises:
        ValueError: If normalized result is empty after applying allowed character filter

    """
    normalized = _SLOT_ID_PATTERN.sub("", slot_id.strip().lower())
    if not normalized:
        raise ValueError("slot_id must contain at least one valid character after normalization")
    return normalized


def detect_duplicate_slots(slots: list[ModelSlot]) -> list[str]:
    """Detect duplicate slot IDs in a list of ModelSlot entries.

    Args:
        slots: List of ModelSlot objects to check for duplicates

    Returns:
        List of normalized slot_ids that appear more than once

    """
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for slot in slots:
        normalized = normalize_slot_id(slot.slot_id)
        if normalized in seen:
            if normalized not in duplicates:
                duplicates.append(normalized)
        else:
            seen[normalized] = 1
    return duplicates


def validate_slot_id(slot_id: str) -> ValidationResult:
    """Validate and normalize a slot ID.

    Args:
        slot_id: Raw slot identifier string

    Returns:
        ValidationResult indicating success or failure with error details

    """
    try:
        normalized = normalize_slot_id(slot_id)
        return ValidationResult(
            slot_id=normalized,
            passed=True,
        )
    except ValueError as e:
        return ValidationResult(
            slot_id=slot_id,
            passed=False,
            failed_check="slot_id_validation",
            error_code=ErrorCode.INVALID_SLOT_ID,
            error_message=str(e),
        )


def validate_slot_port(port: int, slot_id: str) -> ValidationResult:
    """Validate a slot port number.

    Args:
        port: Port number to validate
        slot_id: Slot identifier for error reporting

    Returns:
        ValidationResult indicating success or failure with error details

    """
    if not isinstance(port, int) or port < 1024 or port > 65535:
        return ValidationResult(
            slot_id=slot_id,
            passed=False,
            failed_check="port_range",
            error_code=ErrorCode.PORT_INVALID,
            error_message=f"port must be between 1024 and 65535, got: {port}",
        )
    return ValidationResult(
        slot_id=slot_id,
        passed=True,
    )
