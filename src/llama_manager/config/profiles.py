# Run profile and run group definitions

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
    if not value.strip():
        raise RunProfileError(f"{field_name} must not be empty")


def _require_positive_int(value: int, field_name: str) -> None:
    if value < 1:
        raise RunProfileError(f"{field_name} must be greater than 0")


def _require_port(port: int) -> None:
    if not (1 <= port <= 65535):
        raise RunProfileError(f"port must be between 1 and 65535, got: {port}")


def _first_duplicate(values: Iterable[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
