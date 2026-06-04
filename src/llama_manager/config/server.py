"""ServerConfig, ModelSlot, and slot utility functions."""

import re
from dataclasses import dataclass, field

from ..common.validators import validate_port_range
from .errors import ErrorCode, ErrorDetail, ValidationResult
from .spec_decode import SpeculativeDecodingConfig

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
        spec_decode: Speculative decoding and reasoning options.
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
    chat_template_kwargs: str = ""
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
    spec_decode: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)

    def __init__(
        self,
        model: str,
        alias: str,
        device: str,
        port: int,
        ctx_size: int,
        ubatch_size: int,
        threads: int,
        bind_address: str = "127.0.0.1",
        tensor_split: str = "",
        chat_template_kwargs: str = "",
        use_jinja: bool = False,
        cache_type_k: str = "q8_0",
        cache_type_v: str = "q8_0",
        n_gpu_layers: int | str = 99,
        main_gpu: int = 0,
        server_bin: str = "",
        backend: str = "llama_cpp",
        risky_acknowledged: list[str] | None = None,
        batch_size: int = 2048,
        poll_ms: int = 50,
        n_predict: int = 32768,
        parallel: int = 4,
        threads_batch: int = 0,
        mmproj: str = "",
        spec_decode: SpeculativeDecodingConfig | None = None,
        spec_type: str | None = None,
        spec_ngram_size_n: int | None = None,
        draft_min: int | None = None,
        draft_max: int | None = None,
        spec_draft_n_max: int | None = None,
        spec_draft_p_min: float | None = None,
        spec_draft_cache_type_k: str | None = None,
        spec_draft_cache_type_v: str | None = None,
        spec_draft_device: str | None = None,
        reasoning_mode: str | None = None,
        reasoning_format: str | None = None,
        reasoning_budget: str | None = None,
    ) -> None:
        self.model = model
        self.alias = alias
        self.device = device
        self.port = port
        self.ctx_size = ctx_size
        self.ubatch_size = ubatch_size
        self.threads = threads
        self.bind_address = bind_address
        self.tensor_split = tensor_split
        self.chat_template_kwargs = chat_template_kwargs
        self.use_jinja = use_jinja
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = main_gpu
        self.server_bin = server_bin
        self.backend = backend
        self.risky_acknowledged = risky_acknowledged or []
        self.batch_size = batch_size
        self.poll_ms = poll_ms
        self.n_predict = n_predict
        self.parallel = parallel
        self.threads_batch = threads_batch
        self.mmproj = mmproj
        self.spec_decode = spec_decode or SpeculativeDecodingConfig()
        spec_overrides = {
            "spec_type": spec_type,
            "spec_ngram_size_n": spec_ngram_size_n,
            "draft_min": draft_min,
            "draft_max": draft_max,
            "spec_draft_n_max": spec_draft_n_max,
            "spec_draft_p_min": spec_draft_p_min,
            "spec_draft_cache_type_k": spec_draft_cache_type_k,
            "spec_draft_cache_type_v": spec_draft_cache_type_v,
            "spec_draft_device": spec_draft_device,
            "reasoning_mode": reasoning_mode,
            "reasoning_format": reasoning_format,
            "reasoning_budget": reasoning_budget,
        }
        active_overrides = {
            key: value for key, value in spec_overrides.items() if value is not None
        }
        if active_overrides:
            base = self.spec_decode.__dict__ | active_overrides
            self.spec_decode = SpeculativeDecodingConfig(**base)
        self.__post_init__()

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
        if not isinstance(self.spec_decode, SpeculativeDecodingConfig):
            raise ValueError("spec_decode must be a SpeculativeDecodingConfig")

    def __getattribute__(self, name: str) -> object:
        if name in SpeculativeDecodingConfig.__dataclass_fields__:
            return getattr(object.__getattribute__(self, "spec_decode"), name)
        return object.__getattribute__(self, name)

    @property
    def reasoning_mode(self) -> str:
        return self.spec_decode.reasoning_mode

    @property
    def reasoning_format(self) -> str:
        return self.spec_decode.reasoning_format

    @property
    def reasoning_budget(self) -> str:
        return self.spec_decode.reasoning_budget

    @property
    def spec_type(self) -> str:
        return self.spec_decode.spec_type

    @property
    def spec_ngram_size_n(self) -> int:
        return self.spec_decode.spec_ngram_size_n

    @property
    def draft_min(self) -> int:
        return self.spec_decode.draft_min

    @property
    def draft_max(self) -> int:
        return self.spec_decode.draft_max

    @property
    def spec_draft_n_max(self) -> int:
        return self.spec_decode.spec_draft_n_max

    @property
    def spec_draft_p_min(self) -> float:
        return self.spec_decode.spec_draft_p_min

    @property
    def spec_draft_cache_type_k(self) -> str:
        return self.spec_decode.spec_draft_cache_type_k

    @property
    def spec_draft_cache_type_v(self) -> str:
        return self.spec_decode.spec_draft_cache_type_v

    @property
    def spec_draft_device(self) -> str:
        return self.spec_decode.spec_draft_device


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


def validate_slot_id(slot_id: str) -> ErrorDetail:
    """Validate and normalize a slot ID.

    Args:
        slot_id: Raw slot identifier string

    Returns:
        None on success, ErrorDetail on failure.

    """
    try:
        normalized = normalize_slot_id(slot_id)
        return ValidationResult(
            slot_id=normalized,
            passed=True,
        )
    except ValueError as e:
        return ErrorDetail(
            error_code=ErrorCode.INVALID_SLOT_ID,
            failed_check="slot_id_validation",
            why_blocked=str(e),
            how_to_fix="use a slot_id containing letters, numbers, underscores, or hyphens",
            slot_id=slot_id,
        )


def validate_slot_port(port: int, slot_id: str) -> ErrorDetail:
    """Validate a slot port number.

    Args:
        port: Port number to validate
        slot_id: Slot identifier for error reporting

    Returns:
        None on success, ErrorDetail on failure.

    """
    err = validate_port_range(port)
    if err is not None:
        return ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="port_range",
            why_blocked=err,
            how_to_fix="use a TCP port between 1024 and 65535",
            slot_id=slot_id,
        )
    return ValidationResult(
        slot_id=slot_id,
        passed=True,
    )
