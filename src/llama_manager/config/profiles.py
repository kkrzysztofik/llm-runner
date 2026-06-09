"""Slot profile definitions."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from ..common.text import sanitize_filename_component
from ..common.validators import validate_port_range
from .spec_decode import SpeculativeDecodingConfig


class SlotProfileError(ValueError):
    """Raised when slot profile data is invalid."""


@dataclass(frozen=True, slots=True)
class SlotProfileSpec:
    """Typed data definition for one launchable llama-server slot profile."""

    profile_id: str
    model: str
    alias: str
    device: str
    port: int
    ctx_size: int
    ubatch_size: int
    threads: int
    description: str = ""
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
    risky_acknowledged: tuple[str, ...] = ()
    batch_size: int = 2048
    poll_ms: int = 50
    n_predict: int = 32768
    parallel: int = 4
    threads_batch: int = 0
    mmproj: str = ""
    spec_decode: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)
    kv_unified: bool = False
    mmproj_offload: bool = True
    mmap: bool = True
    mlock: bool = False
    no_host_buffer: bool = False

    def __init__(  # noqa: S107 - intentional explicit init with spec-decode overrides
        self,
        profile_id: str,
        model: str,
        alias: str,
        device: str,
        port: int,
        ctx_size: int,
        ubatch_size: int,
        threads: int,
        description: str = "",
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
        risky_acknowledged: tuple[str, ...] = (),
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
        spec_draft_model: str | None = None,
        spec_draft_hf: str | None = None,
        spec_draft_ngl: int | str | None = None,
        spec_dflash_cross_ctx: int | None = None,
        kv_unified: bool | None = None,
        mmproj_offload: bool | None = None,
        mmap: bool | None = None,
        mlock: bool | None = None,
        no_host_buffer: bool | None = None,
    ) -> None:
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "ctx_size", ctx_size)
        object.__setattr__(self, "ubatch_size", ubatch_size)
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "bind_address", bind_address)
        object.__setattr__(self, "tensor_split", tensor_split)
        object.__setattr__(self, "chat_template_kwargs", chat_template_kwargs)
        object.__setattr__(self, "use_jinja", use_jinja)
        object.__setattr__(self, "cache_type_k", cache_type_k)
        object.__setattr__(self, "cache_type_v", cache_type_v)
        object.__setattr__(self, "n_gpu_layers", n_gpu_layers)
        object.__setattr__(self, "main_gpu", main_gpu)
        object.__setattr__(self, "server_bin", server_bin)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "risky_acknowledged", risky_acknowledged)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "poll_ms", poll_ms)
        object.__setattr__(self, "n_predict", n_predict)
        object.__setattr__(self, "parallel", parallel)
        object.__setattr__(self, "threads_batch", threads_batch)
        object.__setattr__(self, "mmproj", mmproj)
        object.__setattr__(self, "spec_decode", spec_decode or SpeculativeDecodingConfig())
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
            "spec_draft_model": spec_draft_model,
            "spec_draft_hf": spec_draft_hf,
            "spec_draft_ngl": spec_draft_ngl,
            "spec_dflash_cross_ctx": spec_dflash_cross_ctx,
        }
        active_overrides = {
            key: value for key, value in spec_overrides.items() if value is not None
        }
        if active_overrides:
            base = self.spec_decode.__dict__ | active_overrides
            try:
                object.__setattr__(self, "spec_decode", SpeculativeDecodingConfig(**base))
            except ValueError as exc:
                raise SlotProfileError(str(exc)) from exc
        object.__setattr__(self, "kv_unified", kv_unified if kv_unified is not None else False)
        object.__setattr__(
            self, "mmproj_offload", mmproj_offload if mmproj_offload is not None else True
        )
        object.__setattr__(self, "mmap", mmap if mmap is not None else True)
        object.__setattr__(self, "mlock", mlock if mlock is not None else False)
        object.__setattr__(
            self, "no_host_buffer", no_host_buffer if no_host_buffer is not None else False
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate slot profile data at construction time."""
        _require_text(self.profile_id, "profile_id")
        _require_text(self.model, "model")
        _require_text(self.alias, "alias")
        _require_text(self.backend, "backend")
        _require_port(self.port)
        _require_positive_int(self.ctx_size, "ctx_size")
        _require_positive_int(self.ubatch_size, "ubatch_size")
        _require_positive_int(self.threads, "threads")
        _require_positive_int(self.batch_size, "batch_size")
        _require_positive_int(self.n_predict, "n_predict")
        if self.poll_ms < 0:
            raise SlotProfileError("poll_ms must be non-negative")
        if self.parallel != -1 and self.parallel < 1:
            raise SlotProfileError("parallel must be -1 or at least 1")
        if self.threads_batch < 0:
            raise SlotProfileError("threads_batch must be non-negative")
        if not isinstance(self.spec_decode, SpeculativeDecodingConfig):
            raise SlotProfileError("spec_decode must be a SpeculativeDecodingConfig")
        if not isinstance(self.main_gpu, int) or self.main_gpu < 0:
            raise SlotProfileError("main_gpu must be a non-negative integer")
        if isinstance(self.n_gpu_layers, int) and self.n_gpu_layers < 0:
            raise SlotProfileError("n_gpu_layers must be non-negative")
        if isinstance(self.n_gpu_layers, str):
            _require_text(self.n_gpu_layers, "n_gpu_layers")

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

    @property
    def spec_draft_model(self) -> str:
        return self.spec_decode.spec_draft_model

    @property
    def spec_draft_hf(self) -> str:
        return self.spec_decode.spec_draft_hf

    @property
    def spec_draft_ngl(self) -> int | str:
        return self.spec_decode.spec_draft_ngl

    @property
    def spec_dflash_cross_ctx(self) -> int:
        return self.spec_decode.spec_dflash_cross_ctx


@dataclass(frozen=True, slots=True)
class SlotProfileRegistry:
    """Collection of slot profile data used by launch surfaces."""

    profiles: tuple[SlotProfileSpec, ...]

    def __post_init__(self) -> None:
        """Validate uniqueness."""
        duplicate_profile = _first_duplicate(profile.profile_id for profile in self.profiles)
        if duplicate_profile is not None:
            raise SlotProfileError(f"duplicate profile_id: {duplicate_profile}")
        aliases = (profile.alias for profile in self.profiles if profile.alias)
        duplicate_alias = _first_duplicate(aliases)
        if duplicate_alias is not None:
            raise SlotProfileError(f"duplicate profile alias: {duplicate_alias}")

    def get_profile(self, profile_id: str) -> SlotProfileSpec:
        """Return the slot profile matching ``profile_id``.

        Args:
            profile_id: Profile identifier to look up.

        Returns:
            Matching slot profile definition.

        Raises:
            SlotProfileError: If no profile matches the identifier.
        """
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise SlotProfileError(f"unknown profile: {profile_id}")

    @property
    def profile_ids(self) -> tuple[str, ...]:
        """Return profile identifiers in registry order."""
        return tuple(profile.profile_id for profile in self.profiles)


def _require_text(value: str, field_name: str) -> None:
    """Raise if *value* is empty or whitespace-only."""
    if not value.strip():
        raise SlotProfileError(f"{field_name} must not be empty")


def _require_positive_int(value: int, field_name: str) -> None:
    """Raise if *value* is less than 1."""
    if value < 1:
        raise SlotProfileError(f"{field_name} must be greater than 0")


def _require_port(port: int) -> None:
    """Raise if *port* is outside the valid TCP range (1024–65535)."""
    err = validate_port_range(port)
    if err is not None:
        raise SlotProfileError(err)


def _first_duplicate(values: Iterable[str]) -> str | None:
    """Return the first value that appears more than once, or None."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _profile_id_from_alias(alias: str) -> str:
    """Convert an alias to comparable profile_id form.

    Profile aliases intentionally differ from slot IDs: underscores and hyphens
    are treated as equivalent because user-facing CLI aliases historically use
    both forms. Filename safety still delegates to the shared sanitizer.
    """
    return sanitize_filename_component(alias).replace("_", "-")


def _resolve_alias_to_profile_id(registry: SlotProfileRegistry, alias: str) -> str | None:
    """Resolve an alias string to a registered profile_id.

    Matches the alias against profile aliases in the registry, handling
    common normalization patterns (underscore/hyphen interchangeability).

    Args:
        registry: Profile registry containing profile definitions.
        alias: Alias string to resolve (e.g. 'summary_balanced', 'balanced').

    Returns:
        The matching profile_id, or None if no match is found.
    """
    try:
        normalized = _profile_id_from_alias(alias)
    except ValueError:
        return None

    # Direct match against profile aliases
    for profile in registry.profiles:
        if _profile_id_from_alias(profile.alias) == normalized:
            return profile.profile_id

    # Check against profile_id (for direct profile_id usage)
    for profile in registry.profiles:
        if _profile_id_from_alias(profile.profile_id) == normalized:
            return profile.profile_id

    return None


def resolve_profile_id(registry: SlotProfileRegistry, slot_id: str) -> str | None:
    """Resolve a slot_id or alias to a registered profile_id.

    Resolution order:
    1. Direct match against profile_ids
    2. Alias match via registry lookup
    3. Normalized slot_id match against profile_id
    4. None (no match found)

    Args:
        registry: Profile registry containing profile definitions.
        slot_id: Slot identifier or alias to resolve.

    Returns:
        The matching profile_id, or None if no match is found.
    """
    try:
        normalized = _profile_id_from_alias(slot_id)
    except ValueError:
        return None

    # Direct match against profile_ids
    for profile in registry.profiles:
        if profile.profile_id == slot_id:
            return profile.profile_id

    # Alias match
    if result := _resolve_alias_to_profile_id(registry, slot_id):
        return result

    # Normalized slot_id match against profile_id
    for profile in registry.profiles:
        if _profile_id_from_alias(profile.profile_id) == normalized:
            return profile.profile_id

    return None


def resolve_backend_from_profile(profile: SlotProfileSpec) -> str:
    """Derive backend string from a profile spec.

    Uses the device field to determine backend:
    - Device starts with "SYCL" (case-insensitive) → 'sycl' (Intel SYCL backend)
    - All other values (including empty, "CUDA:0", "CUDA:1", "CUDA:0,1") → 'cuda' (NVIDIA backend)

    Args:
        profile: Run profile specification.

    Returns:
        Backend string: 'cuda' or 'sycl'.
    """
    device = profile.device.strip().upper()
    return "sycl" if device.startswith("SYCL") else "cuda"


def _parse_device_indices(device: str) -> list[int]:
    """Extract GPU indices from a device string.

    Supports formats:
    - "" → [] (empty, single GPU default)
    - "SYCL0" → [0]
    - "CUDA:0" → [0]
    - "CUDA:1" → [1]
    - "CUDA:0,1" → [0, 1]

    Args:
        device: Device selector string.

    Returns:
        List of integer GPU indices parsed from the device string.
    """
    device = device.strip()
    if not device:
        return []

    # Strip backend prefix (SYCL: or CUDA:)
    upper = device.upper()
    if upper.startswith("SYCL:"):
        remainder = device[5:]
    elif upper.startswith("SYCL"):
        remainder = device[4:]
    elif upper.startswith("CUDA:"):
        remainder = device[5:]
    else:
        remainder = device

    if not remainder:
        return []

    indices: list[int] = []
    for part in remainder.split(","):
        part = part.strip()
        if part:
            try:
                indices.append(int(part))
            except ValueError:
                continue
    return indices


def _derive_tensor_split_from_device(device: str) -> str:
    """Auto-derive tensor_split from device string.

    Single GPU → "" (no --tensor-split flag)
    Dual GPU → "1,1" (equal 50/50 split, matching run_opencode_models.sh convention)

    Args:
        device: Device selector string.

    Returns:
        Tensor split string or empty string for single GPU.
    """
    indices = _parse_device_indices(device)
    count = len(indices)
    if count <= 1:
        return ""
    # Equal-weight split: "1,1" for 2 GPUs, "1,1,1" for 3, etc.
    return ",".join(["1"] * count)


def _parse_main_gpu_from_device(device: str) -> int:
    """Parse main_gpu from the first device index.

    Args:
        device: Device selector string.

    Returns:
        Primary GPU index (defaults to 0).
    """
    indices = _parse_device_indices(device)
    return indices[0] if indices else 0
