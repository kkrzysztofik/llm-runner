# Config & ServerConfig dataclasses


import re
from dataclasses import dataclass
from enum import StrEnum


@dataclass
class Config:
    """Server configuration defaults"""

    # Paths
    llama_cpp_root: str = "/home/kmk/src/llama.cpp"
    llama_server_bin_intel: str = f"{llama_cpp_root}/build/bin/llama-server"
    llama_server_bin_nvidia: str = f"{llama_cpp_root}/build_cuda/bin/llama-server"

    # Models
    model_summary_balanced: str = "/home/kmk/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf"
    model_summary_fast: str = (
        "/home/kmk/models/unsloth/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
    )
    model_qwen35: str = (
        "/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
    )
    model_qwen35_both: str = (
        "/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
    )

    # Network
    host: str = "127.0.0.1"
    summary_balanced_port: int = 8080
    summary_fast_port: int = 8082
    qwen35_port: int = 8081

    # Model-specific defaults
    summary_balanced_chat_template_kwargs: str = '{"enable_thinking":false}'

    # Server defaults
    default_n_gpu_layers: int = 99
    default_ctx_size_summary: int = 16144
    default_ctx_size_qwen35: int = 262144
    default_ctx_size_both_summary: int = 16144
    default_ctx_size_both_qwen35: int = 262144
    default_n_gpu_layers_qwen35: str = "all"
    default_n_gpu_layers_qwen35_both: str = "all"
    default_ubatch_size_summary_balanced: int = 1024
    default_ubatch_size_summary_fast: int = 512
    default_ubatch_size_qwen35: int = 1024
    default_ubatch_size_qwen35_both: int = 1024
    default_threads_summary_balanced: int = 8
    default_threads_summary_fast: int = 8
    default_threads_qwen35: int = 12
    default_threads_qwen35_both: int = 12
    default_cache_type_summary_k: str = "q8_0"
    default_cache_type_summary_v: str = "q8_0"
    default_cache_type_qwen35_k: str = "q8_0"
    default_cache_type_qwen35_v: str = "q8_0"
    default_cache_type_qwen35_both_k: str = "q8_0"
    default_cache_type_qwen35_both_v: str = "q8_0"


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


# M1 scaffolding

# Regex pattern for slot ID normalization: strip, lowercase, allow only a-z0-9_-
_SLOT_ID_PATTERN = re.compile(r"[^a-z0-9_-]")


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


def detect_duplicate_slots(slots: list["ModelSlot"]) -> list[str]:
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


@dataclass
class ModelSlot:
    """Model slot configuration for multi-slot serving"""

    slot_id: str
    model_path: str
    port: int


class ErrorCode(StrEnum):
    """Error code enum for validation with deterministic string ordering"""

    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PATH_INVALID = "PATH_INVALID"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    PORT_CONFLICT = "PORT_CONFLICT"
    PORT_INVALID = "PORT_INVALID"
    THREADS_INVALID = "THREADS_INVALID"
    CONFIG_ERROR = "CONFIG_ERROR"
    INVALID_SLOT_ID = "invalid_slot_id"
    DUPLICATE_SLOT = "duplicate_slot"
    RUNTIME_DIR_UNAVAILABLE = "runtime_dir_unavailable"
    LOCKFILE_INTEGRITY_FAILURE = "lockfile_integrity_failure"
    ARTIFACT_PERSISTENCE_FAILURE = "artifact_persistence_failure"
    BACKEND_NOT_ELIGIBLE = "backend_not_eligible"


@dataclass
class ValidationResult:
    """Result of a validation check with slot identity for T003 deterministic sorting"""

    slot_id: str
    passed: bool
    failed_check: str = ""
    error_code: ErrorCode | None = None
    error_message: str = ""

    @property
    def valid(self) -> bool:
        """Alias for passed to maintain backward compatibility"""
        return self.passed


@dataclass
class ErrorDetail:
    """FR-005 structured actionable error detail"""

    error_code: ErrorCode
    failed_check: str
    why_blocked: str
    how_to_fix: str
    docs_ref: str | None = None


@dataclass
class MultiValidationError:
    """FR-005 container for multiple validation errors with deterministic ordering"""

    errors: list[ErrorDetail]

    @property
    def error_count(self) -> int:
        """Return the number of errors in this multi-error"""
        return len(self.errors)

    def sort_errors(self) -> None:
        """Sort errors in-place by slot configuration sequence, then failed_check ascending.

        This provides stable, deterministic ordering for consistent error output.
        Slots are ordered by first appearance; within each slot, failed_check is sorted alphabetically.
        """
        if not self.errors:
            return

        # Build slot order map: slot_id -> order index (first occurrence wins)
        # We need to extract slot IDs from error details
        # For errors without slot_id, we put them at the end
        slot_order: dict[str, int] = {}
        other_errors: list[ErrorDetail] = []

        for _i, error in enumerate(self.errors):
            # Extract slot_id from failed_check if it follows pattern "slot_<id>_<check>"
            slot_id: str | None = None
            if error.failed_check.startswith("slot_"):
                parts = error.failed_check.split("_", 2)
                if len(parts) >= 2:
                    slot_id = parts[1]

            if slot_id:
                if slot_id not in slot_order:
                    slot_order[slot_id] = len(slot_order)
                else:
                    # Update slot_order to use the minimum index
                    slot_order[slot_id] = min(slot_order[slot_id], slot_order.get(slot_id, 0))
            else:
                other_errors.append(error)

        def sort_key(error: ErrorDetail) -> tuple[int, str, int]:
            """Sort key: (slot_sequence_order, failed_check, original_index)"""
            slot_idx = 0
            if error.failed_check.startswith("slot_"):
                parts = error.failed_check.split("_", 2)
                if len(parts) >= 2:
                    slot_id = parts[1]
                    slot_idx = slot_order.get(slot_id, 0)
            return (slot_idx, error.failed_check, id(error))

        # Sort main errors, then append others at the end
        sorted_errors = sorted(self.errors, key=sort_key)

        self.errors = sorted_errors
