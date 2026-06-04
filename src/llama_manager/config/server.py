"""ServerConfig, ModelSlot, and slot utility functions."""

import re
from dataclasses import dataclass, field

from ..common.validators import validate_port_range
from .errors import ErrorCode, ValidationResult

# Regex pattern for slot ID normalization: strip, lowercase, allow only a-z0-9_-
_SLOT_ID_PATTERN = re.compile(r"[^a-z0-9_-]")


@dataclass
class ServerConfig:
    """Configuration for a single llama.cpp server instance.

    Each instance targets a specific GPU device and loads a specific model.
    Fields marked with defaults are optional; missing values fall back to
    ``Config``-level defaults at launch time.

    Attributes:
        model: HuggingFace model ID or local path to the GGUF model file.
        alias: Human-readable identifier for this server instance.
        device: Backend device selector (e.g. ``"cuda:0"``, ``"sycl:0"``).
        port: HTTP API port for the server.
        ctx_size: Context window size in tokens.
        ubatch_size: Uniform batch size for prompt processing.
        threads: Number of CPU threads for inference.
        bind_address: Address to bind the HTTP server to. Defaults to ``"127.0.0.1"``.
        tensor_split: Comma-separated GPU split ratio for multi-GPU tensor parallelism.
        reasoning_mode: Reasoning mode — ``"auto"``, ``"on"``, or ``"off"``.
        reasoning_format: Output format for reasoning tokens — ``"none"``, ``"xml"``, etc.
        chat_template_kwargs: JSON string of extra chat-template keyword arguments.
        reasoning_budget: Max tokens for reasoning step (empty = auto).
        use_jinja: Enable Jinja-based chat template instead of the default.
        cache_type_k: KV-cache key type (e.g. ``"q8_0"``, ``"f16"``).
        cache_type_v: KV-cache value type (e.g. ``"q8_0"``, ``"f16"``).
        n_gpu_layers: Number of layers to offload to GPU; ``"all"`` offloads everything.
        main_gpu: Primary GPU index for this server instance (default ``0``).
        server_bin: Path to the llama.cpp server binary (empty = use Config default).
        backend: Inference backend name (e.g. ``"llama_cpp"``).
        risky_acknowledged: List of risk identifiers the user has acknowledged.
    """

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
    main_gpu: int = 0
    server_bin: str = ""
    backend: str = "llama_cpp"
    risky_acknowledged: list[str] = field(default_factory=list)
    batch_size: int = 2048
    poll_ms: int = 50
    n_predict: int = 32768
    parallel: int = 4
    threads_batch: int = 0
    mmproj: str = ""
    spec_type: str = ""
    spec_ngram_size_n: int = 0
    draft_min: int = 0
    draft_max: int = 0
    spec_draft_n_max: int = 0
    spec_draft_p_min: float = 0.0
    spec_draft_cache_type_k: str = ""
    spec_draft_cache_type_v: str = ""
    spec_draft_device: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.main_gpu, int) or self.main_gpu < 0:
            raise ValueError("main_gpu must be a non-negative integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.n_predict <= 0:
            raise ValueError("n_predict must be greater than 0")
        if self.parallel != -1 and self.parallel < 1:
            raise ValueError("parallel must be -1 or at least 1")
        if self.poll_ms < 0:
            raise ValueError("poll_ms must be non-negative")
        if self.threads_batch < 0:
            raise ValueError("threads_batch must be non-negative")
        if self.spec_ngram_size_n < 0:
            raise ValueError("spec_ngram_size_n must be non-negative")
        if self.draft_min < 0:
            raise ValueError("draft_min must be non-negative")
        if self.draft_max < 0:
            raise ValueError("draft_max must be non-negative")
        if self.draft_min > self.draft_max:
            raise ValueError("draft_min must be <= draft_max")
        if self.spec_draft_n_max < 0:
            raise ValueError("spec_draft_n_max must be non-negative")
        if self.spec_draft_p_min < 0.0 or self.spec_draft_p_min > 1.0:
            raise ValueError("spec_draft_p_min must be between 0.0 and 1.0")


@dataclass
class ModelSlot:
    """Minimal configuration for a single model serving slot.

    A slot represents one model instance bound to a specific port.
    Used for multi-GPU / multi-model deployments where each slot
    serves its own model on a dedicated HTTP port.

    Attributes:
        slot_id: Unique identifier for the slot (normalized to ``[a-z0-9_-]+``).
        model_path: Filesystem path to the GGUF model file for this slot.
        port: HTTP port this slot's server listens on.
    """

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
    seen: set[str] = set()
    duplicates: list[str] = []
    for slot in slots:
        normalized = normalize_slot_id(slot.slot_id)
        if normalized in seen:
            if normalized not in duplicates:
                duplicates.append(normalized)
        else:
            seen.add(normalized)
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
    err = validate_port_range(port)
    if err is not None:
        return ValidationResult(
            slot_id=slot_id,
            passed=False,
            failed_check="port_range",
            error_code=ErrorCode.PORT_INVALID,
            error_message=err,
        )
    return ValidationResult(
        slot_id=slot_id,
        passed=True,
    )
