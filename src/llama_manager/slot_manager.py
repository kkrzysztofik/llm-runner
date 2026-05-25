"""Pure library for slot CRUD operations.

Extracted from TUI controller so slot lifecycle management can be tested
and reused without importing Rich, Textual, or other UI libraries.
"""

import logging
from collections.abc import Callable
from typing import Any

from .config import Config, ModelSlot, ServerConfig, SlotState
from .config.builder import create_default_profile_registry, resolve_profile_config
from .config.profiles import RunProfileError, RunProfileRegistry
from .gpu_stats import GPUStats
from .log_buffer import LogBuffer
from .orchestration import ServerManager
from .slot_state import compute_slot_transition

logger = logging.getLogger(__name__)


def normalize_slot_port(port_str: str) -> tuple[int, str | None]:
    """Validate and normalise a port string, falling back to 8080.

    Args:
        port_str: Raw port value as a string.

    Returns:
        A tuple of (normalized_port, warning_message).  ``warning_message``
        is ``None`` when the input is valid.
    """
    try:
        port = int(port_str)
        if port < 1024 or port > 65535:
            return 8080, f"Invalid port {port}, using 8080"
        return port, None
    except ValueError:
        return 8080, "Invalid port, using 8080"


def device_class_for_config(cfg: ServerConfig) -> str:
    """Return normalized device class name used for replacement logic.

    Args:
        cfg: Server configuration.

    Returns:
        ``"sycl"`` when ``cfg.device`` starts with ``SYCL``,
        otherwise ``"cuda"``.
    """
    return "sycl" if cfg.device.upper().startswith("SYCL") else "cuda"


def gpu_index_for_config(
    cfg: ServerConfig,
    device_mapping: dict[str, int] | None = None,
) -> int:
    """Return dashboard GPU index for a configuration.

    Args:
        cfg: Server configuration.
        device_mapping: Optional override mapping from device class to GPU
            index.  Defaults to ``{"sycl": 1, "cuda": 0}``.

    Returns:
        GPU index for the device's dashboard panel.
    """
    mapping = device_mapping or {"sycl": 1, "cuda": 0}
    return mapping.get(device_class_for_config(cfg), 0)


def remove_slot_runtime_state(alias: str, state: dict[str, Any]) -> None:
    """Remove runtime state for one slot alias.

    Mutates *state* in place, removing entries from ``log_buffers``,
    ``server_processes``, ``slot_states``, ``unsaved_slots``, and ``slots``.

    Args:
        alias: Slot identifier to remove.
        state: Mutable runtime-state dictionary.
    """
    state["log_buffers"].pop(alias, None)
    state["server_processes"].pop(alias, None)
    state["slot_states"].pop(alias, None)
    state["unsaved_slots"].discard(alias)
    state["slots"][:] = [slot for slot in state["slots"] if slot.slot_id != alias]


def register_and_start_slot(
    cfg: ServerConfig,
    server_manager: ServerManager,
    state: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Register and start one slot.

    Args:
        cfg: Server configuration for the slot.
        server_manager: Active ``ServerManager`` instance.
        state: Mutable runtime-state dictionary.

    Returns:
        ``(updated_state, messages)`` where *messages* contains any
        status text produced by the slot-state transition.
    """
    alias = cfg.alias
    state["log_buffers"][alias] = LogBuffer(redact_sensitive=True)
    state["unsaved_slots"].add(alias)
    state["slots"].append(ModelSlot(slot_id=alias, model_path=cfg.model, port=cfg.port))

    log_buffer = state["log_buffers"][alias]
    log_handler = lambda line, buf=log_buffer: buf.add_line(line)  # noqa: E731
    procs = server_manager.start_servers([cfg], {alias: log_handler})

    old_state = state["slot_states"].get(alias)
    messages: list[str] = []
    if procs:
        state["server_processes"][alias] = procs[0]
        state["slot_states"][alias] = SlotState.RUNNING.value
        result = compute_slot_transition(alias, old_state, SlotState.RUNNING)
        if result is not None:
            message, _color = result
            messages.append(message)
            logger.info("slot %s: %s", alias, message)
    else:
        state["slot_states"][alias] = SlotState.CRASHED.value
        messages.append(f"Slot '{alias}' failed to start: no process returned")

    return state, messages


def upsert_profile_slot(
    cfg: ServerConfig,
    profile_id: str,
    configs: list[ServerConfig],
    gpu_indices: list[int],
    gpu_stats: list[GPUStats],
    server_manager: ServerManager,
    state: dict[str, Any],
    make_collector: Callable[[int], Callable[[], dict[str, Any]]],
) -> tuple[bool, list[str], dict[str, Any]]:
    """Add a profile slot or replace an existing slot on the same device.

    Args:
        cfg: Resolved server configuration.
        profile_id: Profile identifier (for status messages).
        configs: Current list of active ``ServerConfig`` objects.
        gpu_indices: Parallel list of GPU indices for *configs*.
        gpu_stats: Parallel list of ``GPUStats`` for *configs*.
        server_manager: Active ``ServerManager`` instance.
        state: Mutable runtime-state dictionary.
        make_collector: Factory that returns a GPU collector callable for a
            given device index.

    Returns:
        ``(success, messages, updated_state)``.
    """
    target_device = device_class_for_config(cfg)
    existing_index = next(
        (
            idx
            for idx, existing_cfg in enumerate(configs)
            if device_class_for_config(existing_cfg) == target_device
        ),
        None,
    )

    messages: list[str] = []

    if existing_index is None:
        configs.append(cfg)
        gpu_idx = gpu_index_for_config(cfg)
        gpu_indices.append(gpu_idx)
        gpu_stats.append(GPUStats(gpu_idx, collector=make_collector(gpu_idx)))
        state, slot_messages = register_and_start_slot(cfg, server_manager, state)
        messages.extend(slot_messages)
        messages.append(
            f"Added profile '{profile_id}' as '{cfg.alias}' on {target_device}:{cfg.port}"
        )
        return True, messages, state

    old_cfg = configs[existing_index]
    old_alias = old_cfg.alias
    if not server_manager.shutdown_slot(old_alias):
        messages.append(
            f"Unable to replace '{old_alias}' on {target_device}: shutdown verification failed"
        )
        return False, messages, state

    remove_slot_runtime_state(old_alias, state)
    configs[existing_index] = cfg
    gpu_idx = gpu_index_for_config(cfg)
    gpu_indices[existing_index] = gpu_idx
    gpu_stats[existing_index] = GPUStats(gpu_idx, collector=make_collector(gpu_idx))

    state, slot_messages = register_and_start_slot(cfg, server_manager, state)
    messages.extend(slot_messages)
    messages.append(
        f"Replaced '{old_alias}' with profile '{profile_id}' as "
        f"'{cfg.alias}' on {target_device}:{cfg.port}"
    )
    return True, messages, state


def add_slot_from_form(
    values: dict[str, str],
    config: Config,
    configs: list[ServerConfig],
    gpu_indices: list[int],
    gpu_stats: list[GPUStats],
    server_manager: ServerManager,
    state: dict[str, Any],
    make_collector: Callable[[int], Callable[[], dict[str, Any]]],
    registry: RunProfileRegistry | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Create or replace a slot from modal form values.

    Args:
        values: Form values (expected keys: ``"profile"``, ``"port"``).
        config: Base ``Config`` used to build the profile registry.
        configs: Current list of active ``ServerConfig`` objects.
        gpu_indices: Parallel list of GPU indices for *configs*.
        gpu_stats: Parallel list of ``GPUStats`` for *configs*.
        server_manager: Active ``ServerManager`` instance.
        state: Mutable runtime-state dictionary.
        make_collector: Factory that returns a GPU collector callable for a
            given device index.
        registry: Optional pre-built ``RunProfileRegistry``. When omitted,
            a fresh registry is created via ``create_default_profile_registry``.

    Returns:
        ``(success, messages, updated_state)``.
    """
    messages: list[str] = []

    profile_id = values.get("profile", "").strip()
    if not profile_id:
        messages.append("Profile is required")
        return False, messages, state

    if registry is None:
        registry = create_default_profile_registry(config)

    override_config: dict[str, int] | None = None
    port_value = values.get("port", "").strip()
    if port_value:
        port, warning = normalize_slot_port(port_value)
        if warning is not None:
            messages.append(warning)
        override_config = {"port": port}

    try:
        new_cfg = resolve_profile_config(registry, profile_id, override_config=override_config)
    except RunProfileError:
        allowed = ", ".join(registry.profile_ids)
        messages.append(f"Unknown profile '{profile_id}'. Choose one of: {allowed}")
        return False, messages, state

    success, upsert_messages, state = upsert_profile_slot(
        new_cfg,
        profile_id,
        configs,
        gpu_indices,
        gpu_stats,
        server_manager,
        state,
        make_collector,
    )
    messages.extend(upsert_messages)
    return success, messages, state
