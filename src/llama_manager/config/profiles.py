"""Run profile and run group definitions."""

from collections.abc import Iterable
from dataclasses import dataclass


class RunProfileError(ValueError):
    """Raised when run profile or run group data is invalid."""


@dataclass(frozen=True, slots=True)
class RunProfileSpec:
    """Typed data definition for one launchable llama-server profile."""

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
    server_bin: str = ""
    backend: str = "llama_cpp"
    risky_acknowledged: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate profile data at construction time."""
        _require_text(self.profile_id, "profile_id")
        _require_text(self.model, "model")
        _require_text(self.alias, "alias")
        _require_text(self.backend, "backend")
        _require_port(self.port)
        _require_positive_int(self.ctx_size, "ctx_size")
        _require_positive_int(self.ubatch_size, "ubatch_size")
        _require_positive_int(self.threads, "threads")
        if isinstance(self.n_gpu_layers, int) and self.n_gpu_layers < 0:
            raise RunProfileError("n_gpu_layers must be non-negative")
        if isinstance(self.n_gpu_layers, str):
            _require_text(self.n_gpu_layers, "n_gpu_layers")


@dataclass(frozen=True, slots=True)
class RunGroupSpec:
    """Typed data definition for a launch mode containing one or more profiles."""

    group_id: str
    profile_ids: tuple[str, ...]
    description: str = ""
    tui_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate run group data at construction time."""
        _require_text(self.group_id, "group_id")
        if not self.profile_ids:
            raise RunProfileError("profile_ids must contain at least one profile")
        for profile_id in self.profile_ids:
            _require_text(profile_id, "profile_id")


@dataclass(frozen=True, slots=True)
class RunProfileRegistry:
    """Collection of profile and run-group data used by launch surfaces."""

    profiles: tuple[RunProfileSpec, ...]
    run_groups: tuple[RunGroupSpec, ...]

    def __post_init__(self) -> None:
        """Validate uniqueness and group references."""
        duplicate_profile = _first_duplicate(profile.profile_id for profile in self.profiles)
        if duplicate_profile is not None:
            raise RunProfileError(f"duplicate profile_id: {duplicate_profile}")

        duplicate_group = _first_duplicate(group.group_id for group in self.run_groups)
        if duplicate_group is not None:
            raise RunProfileError(f"duplicate group_id: {duplicate_group}")

        profile_ids = {profile.profile_id for profile in self.profiles}
        for group in self.run_groups:
            for profile_id in group.profile_ids:
                if profile_id not in profile_ids:
                    raise RunProfileError(
                        f"run group {group.group_id} references unknown profile: {profile_id}"
                    )

    def get_profile(self, profile_id: str) -> RunProfileSpec:
        """Return the profile matching ``profile_id``.

        Args:
            profile_id: Profile identifier to look up.

        Returns:
            Matching run profile definition.

        Raises:
            RunProfileError: If no profile matches the identifier.
        """
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise RunProfileError(f"unknown profile: {profile_id}")

    def get_run_group(self, group_id: str) -> RunGroupSpec:
        """Return the run group matching ``group_id``.

        Args:
            group_id: Run group identifier to look up.

        Returns:
            Matching run group definition.

        Raises:
            RunProfileError: If no run group matches the identifier.
        """
        for group in self.run_groups:
            if group.group_id == group_id:
                return group
        raise RunProfileError(f"unknown run group: {group_id}")

    @property
    def profile_ids(self) -> tuple[str, ...]:
        """Return profile identifiers in registry order."""
        return tuple(profile.profile_id for profile in self.profiles)

    @property
    def run_group_ids(self) -> tuple[str, ...]:
        """Return run group identifiers in registry order."""
        return tuple(group.group_id for group in self.run_groups)


def _require_text(value: str, field_name: str) -> None:
    """Raise if *value* is empty or whitespace-only."""
    if not value.strip():
        raise RunProfileError(f"{field_name} must not be empty")


def _require_positive_int(value: int, field_name: str) -> None:
    """Raise if *value* is less than 1."""
    if value < 1:
        raise RunProfileError(f"{field_name} must be greater than 0")


def _require_port(port: int) -> None:
    """Raise if *port* is outside the valid TCP range (1–65535)."""
    if not (1 <= port <= 65535):
        raise RunProfileError(f"port must be between 1 and 65535, got: {port}")


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


def _resolve_alias_to_profile_id(registry: RunProfileRegistry, alias: str) -> str | None:
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


def resolve_profile_id(registry: RunProfileRegistry, slot_id: str) -> str | None:
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


def resolve_backend_from_profile(profile: RunProfileSpec) -> str:
    """Derive backend string from a profile spec.

    Uses the device field to determine backend:
    - Empty device field → 'cuda' (NVIDIA backend)
    - Non-empty device field → 'sycl' (Intel SYCL backend)

    Args:
        profile: Run profile specification.

    Returns:
        Backend string: 'cuda' or 'sycl'.
    """
    return "cuda" if not profile.device.strip() else "sycl"
