"""Slot profile definitions."""

from collections.abc import Iterable
from dataclasses import dataclass


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
    risky_acknowledged: tuple[str, ...] = ()
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
        for name in (
            "spec_ngram_size_n",
            "draft_min",
            "draft_max",
            "spec_draft_n_max",
        ):
            value = getattr(self, name)
            if value < 0:
                raise SlotProfileError(f"{name} must be non-negative")
        if self.draft_min > self.draft_max:
            raise SlotProfileError("draft_min must be <= draft_max")
        if self.spec_draft_p_min < 0:
            raise SlotProfileError("spec_draft_p_min must be non-negative")
        if self.spec_draft_p_min > 1.0:
            raise SlotProfileError("spec_draft_p_min must be <= 1.0")
        if self.spec_type not in ("", "ngram-mod", "draft-mtp"):
            raise SlotProfileError("spec_type must be '', 'ngram-mod', or 'draft-mtp'")
        if not isinstance(self.main_gpu, int) or self.main_gpu < 0:
            raise SlotProfileError("main_gpu must be a non-negative integer")
        if isinstance(self.n_gpu_layers, int) and self.n_gpu_layers < 0:
            raise SlotProfileError("n_gpu_layers must be non-negative")
        if isinstance(self.n_gpu_layers, str):
            _require_text(self.n_gpu_layers, "n_gpu_layers")


@dataclass(frozen=True, slots=True)
class SlotProfileRegistry:
    """Collection of slot profile data used by launch surfaces."""

    profiles: tuple[SlotProfileSpec, ...]

    def __post_init__(self) -> None:
        """Validate uniqueness."""
        duplicate_profile = _first_duplicate(profile.profile_id for profile in self.profiles)
        if duplicate_profile is not None:
            raise SlotProfileError(f"duplicate profile_id: {duplicate_profile}")

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
    """Raise if *port* is outside the valid TCP range (1–65535)."""
    if not (1 <= port <= 65535):
        raise SlotProfileError(f"port must be between 1 and 65535, got: {port}")


def _first_duplicate(values: Iterable[str]) -> str | None:
    """Return the first value that appears more than once, or None."""
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _normalize_alias(alias: str) -> str:
    """Normalize an alias string to profile_id form.

    Handles common variations: underscores, hyphens, and short forms.
    Converts the alias to a form that can be compared against profile_id values.

    Args:
        alias: The alias string to normalize.

    Returns:
        Normalized alias string with hyphens/underscores standardized.
    """
    return alias.strip().replace("_", "-")


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
    normalized = _normalize_alias(alias)

    # Direct match against profile aliases
    for profile in registry.profiles:
        if _normalize_alias(profile.alias) == normalized:
            return profile.profile_id

    # Check against profile_id (for direct profile_id usage)
    for profile in registry.profiles:
        if _normalize_alias(profile.profile_id) == normalized:
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
    normalized = _normalize_alias(slot_id)

    # Direct match against profile_ids
    for profile in registry.profiles:
        if profile.profile_id == slot_id:
            return profile.profile_id

    # Alias match
    if result := _resolve_alias_to_profile_id(registry, slot_id):
        return result

    # Normalized slot_id match against profile_id
    for profile in registry.profiles:
        if _normalize_alias(profile.profile_id) == normalized:
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
