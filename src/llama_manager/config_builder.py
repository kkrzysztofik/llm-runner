# ServerConfig creation helpers


import os
from copy import deepcopy
from typing import Any

from .config import (
    Config,
    RunGroupSpec,
    RunProfileError,
    RunProfileRegistry,
    RunProfileSpec,
    ServerConfig,
    SmokeProbeConfiguration,
)
from .profile_cache import PROFILE_OVERRIDE_FIELDS, StalenessResult


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts with override taking precedence.

    FR-006: Priority order is defaults < slot/workstation < profile < override.
    This function implements the merge logic where override values take precedence
    over base values, recursively merging nested dicts.

    Args:
        base: Base dictionary (lower precedence).
        override: Override dictionary (higher precedence).

    Returns:
        Merged dictionary with override values taking precedence.

    """
    result: dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = [*deepcopy(result[key]), *deepcopy(value)]
        else:
            result[key] = deepcopy(value)
    return result


def _validate_merged_config(
    merged: dict[str, Any],
    slot_config: dict | None,
    workstation_config: dict | None,
    profile_config: dict | StalenessResult | None,
    override_config: dict | None,
) -> None:
    """Validate merged FR-006 config values after precedence resolution."""
    port = merged.get("port")
    if not isinstance(port, int) or not (1024 <= port <= 65535):
        raise ValueError(f"port must be between 1024 and 65535, got: {port}")

    threads = merged.get("threads")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError(f"threads must be greater than 0, got: {threads}")

    model_overridden = any(
        isinstance(layer, dict) and "model" in layer
        for layer in (slot_config, workstation_config, profile_config, override_config)
    )
    if model_overridden:
        model = merged.get("model")
        if not isinstance(model, str) or not os.path.exists(model):
            raise ValueError(f"model path not found: {model}")


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    """Return a copy excluding keys with ``None`` values."""
    return {key: value for key, value in values.items() if value is not None}


def _validate_port_override_count(group: RunGroupSpec, port_overrides: tuple[int, ...]) -> None:
    if len(port_overrides) > len(group.profile_ids):
        raise RunProfileError(
            f"run group {group.group_id} accepts at most {len(group.profile_ids)} port override(s), "
            f"got {len(port_overrides)}"
        )


def _validate_resolved_profile_data(data: dict[str, Any]) -> None:
    port = data.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        raise ValueError(f"port must be between 1 and 65535, got: {port}")

    threads = data.get("threads")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError(f"threads must be greater than 0, got: {threads}")

    for key in ("model", "alias", "backend"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")


def _profile_to_config_data(profile: RunProfileSpec) -> dict[str, Any]:
    """Convert profile data to a complete ServerConfig-compatible mapping."""
    return {
        "model": profile.model,
        "alias": profile.alias,
        "device": profile.device,
        "port": profile.port,
        "bind_address": profile.bind_address,
        "ctx_size": profile.ctx_size,
        "ubatch_size": profile.ubatch_size,
        "threads": profile.threads,
        "tensor_split": profile.tensor_split,
        "reasoning_mode": profile.reasoning_mode,
        "reasoning_format": profile.reasoning_format,
        "chat_template_kwargs": profile.chat_template_kwargs,
        "reasoning_budget": profile.reasoning_budget,
        "use_jinja": profile.use_jinja,
        "cache_type_k": profile.cache_type_k,
        "cache_type_v": profile.cache_type_v,
        "n_gpu_layers": profile.n_gpu_layers,
        "server_bin": profile.server_bin,
        "backend": profile.backend,
        "risky_acknowledged": list(profile.risky_acknowledged),
    }


def _config_data_to_server_config(data: dict[str, Any]) -> ServerConfig:
    """Convert validated ServerConfig-compatible data to ServerConfig."""
    return ServerConfig(
        model=data["model"],
        alias=data["alias"],
        device=data["device"],
        port=data["port"],
        bind_address=data.get("bind_address", "127.0.0.1"),
        ctx_size=data["ctx_size"],
        ubatch_size=data["ubatch_size"],
        threads=data["threads"],
        tensor_split=data.get("tensor_split", ""),
        reasoning_mode=data["reasoning_mode"],
        reasoning_format=data["reasoning_format"],
        chat_template_kwargs=data["chat_template_kwargs"],
        reasoning_budget=data.get("reasoning_budget", ""),
        use_jinja=data["use_jinja"],
        cache_type_k=data["cache_type_k"],
        cache_type_v=data["cache_type_v"],
        n_gpu_layers=data["n_gpu_layers"],
        server_bin=data["server_bin"],
        backend=data["backend"],
        risky_acknowledged=data["risky_acknowledged"],
    )


def create_server_config_from_profile(
    profile: RunProfileSpec,
    override_config: dict[str, Any] | None = None,
) -> ServerConfig:
    """Resolve any run profile definition into a ServerConfig.

    Args:
        profile: Typed profile data to convert.
        override_config: Optional explicit overrides. These are caller-provided
            and intentionally allowed to change structural launch fields such as
            ports, models, backend, and binaries.

    Returns:
        ServerConfig resolved from profile data plus explicit overrides.

    Raises:
        ValueError: If the resolved data is not launchable.
    """
    data = _profile_to_config_data(profile)
    if override_config:
        data = _deep_merge(data, override_config)

    _validate_resolved_profile_data(data)
    return _config_data_to_server_config(data)


def resolve_profile_config(
    registry: RunProfileRegistry,
    profile_id: str,
    override_config: dict[str, Any] | None = None,
) -> ServerConfig:
    """Resolve one registered profile into ServerConfig.

    Args:
        registry: Profile registry containing profile definitions.
        profile_id: Profile identifier to resolve.
        override_config: Optional explicit profile overrides.

    Returns:
        ServerConfig for the requested profile.
    """
    return create_server_config_from_profile(registry.get_profile(profile_id), override_config)


def resolve_run_group_configs(
    registry: RunProfileRegistry,
    group_id: str,
    port_overrides: tuple[int, ...] = (),
) -> list[ServerConfig]:
    """Resolve a registered run group into ordered ServerConfig objects.

    Args:
        registry: Profile registry containing group and profile definitions.
        group_id: Run group identifier to resolve.
        port_overrides: Optional positional port overrides for group members.

    Returns:
        ServerConfig objects in run-group profile order.

    Raises:
        RunProfileError: If too many port overrides are provided.
    """
    group = registry.get_run_group(group_id)
    _validate_port_override_count(group, port_overrides)

    configs: list[ServerConfig] = []
    for index, profile_id in enumerate(group.profile_ids):
        override_config = {"port": port_overrides[index]} if index < len(port_overrides) else None
        configs.append(resolve_profile_config(registry, profile_id, override_config))
    return configs


def create_summary_balanced_cfg(
    port: int,
    ctx_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    cache_k: str | None = None,
    cache_v: str | None = None,
) -> ServerConfig:
    """Create a ServerConfig for the summary-balanced model profile.

    Args:
        port: Port to bind the server to.
        ctx_size: Context size (defaults to Config.default_ctx_size_summary).
        ubatch_size: Ubatch size (defaults to Config.default_ubatch_size_summary_balanced).
        threads: Number of threads (defaults to Config.default_threads_summary_balanced).
        cache_k: K cache type.
        cache_v: V cache type.

    Returns:
        A configured ServerConfig instance.

    """
    registry = create_default_profile_registry()
    return resolve_profile_config(
        registry,
        "summary-balanced",
        _without_none(
            {
                "port": port,
                "ctx_size": ctx_size,
                "ubatch_size": ubatch_size,
                "threads": threads,
                "cache_type_k": cache_k,
                "cache_type_v": cache_v,
            }
        ),
    )


def create_summary_fast_cfg(
    port: int,
    ctx_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    cache_k: str | None = None,
    cache_v: str | None = None,
) -> ServerConfig:
    """Create a ServerConfig for the summary-fast model profile.

    Args:
        port: Port to bind the server to.
        ctx_size: Context size (defaults to Config.default_ctx_size_summary).
        ubatch_size: Ubatch size (defaults to Config.default_ubatch_size_summary_fast).
        threads: Number of threads (defaults to Config.default_threads_summary_fast).
        cache_k: K cache type.
        cache_v: V cache type.

    Returns:
        A configured ServerConfig instance.

    """
    registry = create_default_profile_registry()
    return resolve_profile_config(
        registry,
        "summary-fast",
        _without_none(
            {
                "port": port,
                "ctx_size": ctx_size,
                "ubatch_size": ubatch_size,
                "threads": threads,
                "cache_type_k": cache_k,
                "cache_type_v": cache_v,
            }
        ),
    )


def create_qwen35_cfg(
    port: int,
    ctx_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    cache_k: str | None = None,
    cache_v: str | None = None,
    n_gpu_layers: int | str = "all",
    model: str | None = None,
    server_bin: str = "",
    backend: str = "llama_cpp",
) -> ServerConfig:
    """Create a ServerConfig for the qwen35-coding model profile.

    Args:
        port: Port to bind the server to.
        ctx_size: Context size (defaults to Config.default_ctx_size_qwen35).
        ubatch_size: Ubatch size (defaults to Config.default_ubatch_size_qwen35).
        threads: Number of threads (defaults to Config.default_threads_qwen35).
        cache_k: K cache type (defaults to Config.default_cache_type_qwen35_k).
        cache_v: V cache type (defaults to Config.default_cache_type_qwen35_v).
        n_gpu_layers: Number of GPU layers to offload (defaults to 'all').
        model: Specific model path (defaults to Config.model_qwen35).
        server_bin: Path to llama-server binary (defaults to Config.llama_server_bin_nvidia).
        backend: Inference backend (defaults to 'llama_cpp').

    Returns:
        A configured ServerConfig instance.

    """
    registry = create_default_profile_registry()
    return resolve_profile_config(
        registry,
        "qwen35",
        _without_none(
            {
                "port": port,
                "ctx_size": ctx_size,
                "ubatch_size": ubatch_size,
                "threads": threads,
                "cache_type_k": cache_k,
                "cache_type_v": cache_v,
                "n_gpu_layers": n_gpu_layers,
                "model": model,
                "server_bin": server_bin or None,
                "backend": backend,
            }
        ),
    )


def create_default_run_profiles(config: Config | None = None) -> tuple[RunProfileSpec, ...]:
    """Create built-in run profiles as typed data entries.

    Args:
        config: Optional base configuration. When omitted, environment-aware
            defaults are loaded from ``Config``.

    Returns:
        Built-in single-server profile definitions in stable CLI order.
    """
    cfg = config or Config()
    return (
        RunProfileSpec(
            profile_id="summary-balanced",
            description="Run summary-balanced model on Intel SYCL.",
            model=cfg.model_summary_balanced,
            alias="summary-balanced",
            device="SYCL0",
            port=cfg.summary_balanced_port,
            ctx_size=cfg.default_ctx_size_summary,
            ubatch_size=cfg.default_ubatch_size_summary_balanced,
            threads=cfg.default_threads_summary_balanced,
            reasoning_mode="off",
            reasoning_format="deepseek",
            chat_template_kwargs=cfg.summary_balanced_chat_template_kwargs,
            use_jinja=True,
            cache_type_k=cfg.default_cache_type_summary_k,
            cache_type_v=cfg.default_cache_type_summary_v,
            backend="llama_cpp",
        ),
        RunProfileSpec(
            profile_id="summary-fast",
            description="Run summary-fast model on Intel SYCL.",
            model=cfg.model_summary_fast,
            alias="summary-fast",
            device="SYCL0",
            port=cfg.summary_fast_port,
            ctx_size=cfg.default_ctx_size_summary,
            ubatch_size=cfg.default_ubatch_size_summary_fast,
            threads=cfg.default_threads_summary_fast,
            reasoning_mode="off",
            reasoning_format="deepseek",
            chat_template_kwargs=cfg.summary_fast_chat_template_kwargs,
            use_jinja=True,
            cache_type_k=cfg.default_cache_type_summary_k,
            cache_type_v=cfg.default_cache_type_summary_v,
            backend="llama_cpp",
        ),
        RunProfileSpec(
            profile_id="qwen35",
            description="Run qwen35-coding model on NVIDIA CUDA.",
            model=cfg.model_qwen35,
            alias="qwen35-coding",
            device="",
            port=cfg.qwen35_port,
            ctx_size=cfg.default_ctx_size_qwen35,
            ubatch_size=cfg.default_ubatch_size_qwen35,
            threads=cfg.default_threads_qwen35,
            cache_type_k=cfg.default_cache_type_qwen35_k,
            cache_type_v=cfg.default_cache_type_qwen35_v,
            n_gpu_layers=cfg.default_n_gpu_layers_qwen35,
            server_bin=cfg.llama_server_bin_nvidia,
            backend="llama_cpp",
        ),
    )


def create_default_run_groups() -> tuple[RunGroupSpec, ...]:
    """Create built-in launch modes as typed run-group data entries.

    Returns:
        Built-in run groups in stable CLI order. Single-profile modes are
        represented as one-member groups so launch surfaces can resolve every
        mode through the same data shape.
    """
    return (
        RunGroupSpec(
            group_id="summary-balanced",
            profile_ids=("summary-balanced",),
            description="Launch the summary-balanced profile.",
        ),
        RunGroupSpec(
            group_id="summary-fast",
            profile_ids=("summary-fast",),
            description="Launch the summary-fast profile.",
        ),
        RunGroupSpec(
            group_id="qwen35",
            profile_ids=("qwen35",),
            description="Launch the qwen35 profile.",
        ),
        RunGroupSpec(
            group_id="both",
            profile_ids=("summary-balanced", "qwen35"),
            description="Launch summary-balanced and qwen35 profiles together.",
        ),
    )


def create_default_profile_registry(config: Config | None = None) -> RunProfileRegistry:
    """Create the built-in dynamic profile registry.

    Args:
        config: Optional base configuration used to resolve environment-aware
            model paths, ports, binaries, and tuning defaults.

    Returns:
        Validated registry containing built-in profiles and run groups.
    """
    return RunProfileRegistry(
        profiles=create_default_run_profiles(config),
        run_groups=create_default_run_groups(),
    )


def merge_config_overrides(
    defaults: Config,
    slot_config: dict | None = None,
    workstation_config: dict | None = None,
    profile_config: dict | StalenessResult | None = None,
    override_config: dict | None = None,
    warnings: list[str] | None = None,
) -> ServerConfig:
    """FR-006: Merge configuration with 5-level precedence.

    Configuration sources are applied in the following order (lowest to
    highest precedence):

        1. **defaults** — base values from ``Config`` (``Config.model_summary_balanced``,
           ``Config.summary_balanced_port``, etc.)
        2. **slot** — per-slot overrides (e.g. ``{"port": 8080, "ctx_size": 32000}``)
        3. **workstation** — per-workstation overrides (merged alongside slot at
           the same precedence level)
        4. **profile** — cached profile overrides filtered through
           ``PROFILE_OVERRIDE_FIELDS`` whitelist
        5. **override** — explicit CLI/programmatic override (highest precedence)

    Dict fields are deep merged; scalar fields use the higher-precedence
    source.  The existing ``create_*_cfg`` functions remain compatible.

    Profile config is filtered through ``PROFILE_OVERRIDE_FIELDS`` whitelist —
    non-whitelisted keys (e.g. ``n_gpu_layers``, ``tensor_split``, ``model``,
    ``port``, ``server_bin``, ``backend``) are silently ignored.

    When ``profile_config`` is a ``StalenessResult`` (indicating stale profile
    data), a warning is appended to the ``warnings`` list and the profile
    layer is skipped for merging.

    Args:
        defaults: Base Config object with default values.
        slot_config: Optional dict for slot-level overrides (e.g.
            ``{"port": 8080, "ctx_size": 32000}``).
        workstation_config: Optional dict for workstation-level overrides.
        profile_config: Optional dict for profile-level overrides, or
            ``StalenessResult`` if stale.
        override_config: Optional dict for explicit CLI override
            (highest precedence).
        warnings: Optional list to append warnings to as a side effect.

    Returns:
        ServerConfig constructed from merged configuration.

    Example:
        cfg = Config()
        final_cfg = merge_config_overrides(
            defaults=cfg,
            slot_config={"port": 8080, "ctx_size": 32000},
            profile_config={"threads": 12, "ubatch_size": 2048},
            override_config={"port": 9000},  # Will win over slot_config
        )

    """
    # Initialize warnings list if not provided
    _warnings: list[str] = warnings if warnings is not None else []

    # Start with defaults
    base_defaults: dict = {
        "model": defaults.model_summary_balanced,
        "alias": "default",
        "device": "",
        "port": defaults.summary_balanced_port,
        "bind_address": "127.0.0.1",
        "ctx_size": defaults.default_ctx_size_summary,
        "ubatch_size": defaults.default_ubatch_size_summary_balanced,
        "threads": defaults.default_threads_summary_balanced,
        "reasoning_mode": "off",
        "reasoning_format": "deepseek",
        "chat_template_kwargs": defaults.summary_balanced_chat_template_kwargs,
        "use_jinja": True,
        "cache_type_k": defaults.default_cache_type_summary_k,
        "cache_type_v": defaults.default_cache_type_summary_v,
        "n_gpu_layers": defaults.default_n_gpu_layers,
        "server_bin": "",
        "backend": "llama_cpp",
        "risky_acknowledged": [],
    }

    # Apply precedence order: defaults < slot/workstation < profile < override
    merged = base_defaults

    # Step 1: slot_config (lower precedence)
    if slot_config:
        merged = _deep_merge(merged, slot_config)

    # Step 2: workstation_config (same level as slot, merged together)
    if workstation_config:
        merged = _deep_merge(merged, workstation_config)

    # Step 3: profile_config (higher precedence)
    if profile_config is not None:
        if isinstance(profile_config, StalenessResult):
            # Stale profile — append warning and skip merging
            _warnings.append(
                f"⚠ Profile for {profile_config.driver_version_display} is stale: "
                f"{profile_config.warning_message}"
            )
        else:
            # Filter through whitelist — only whitelisted keys are merged
            filtered_profile = {
                key: value
                for key, value in profile_config.items()
                if key in PROFILE_OVERRIDE_FIELDS
            }
            if filtered_profile:
                merged = _deep_merge(merged, filtered_profile)

    # Step 4: override_config (highest precedence)
    if override_config:
        merged = _deep_merge(merged, override_config)

    _validate_merged_config(
        merged,
        slot_config=slot_config,
        workstation_config=workstation_config,
        profile_config=profile_config,
        override_config=override_config,
    )

    # Convert dict back to ServerConfig
    return ServerConfig(
        model=merged["model"],
        alias=merged["alias"],
        device=merged["device"],
        port=merged["port"],
        bind_address=merged.get("bind_address", "127.0.0.1"),
        ctx_size=merged["ctx_size"],
        ubatch_size=merged["ubatch_size"],
        threads=merged["threads"],
        tensor_split=merged.get("tensor_split", ""),
        reasoning_mode=merged["reasoning_mode"],
        reasoning_format=merged["reasoning_format"],
        chat_template_kwargs=merged["chat_template_kwargs"],
        reasoning_budget=merged.get("reasoning_budget", ""),
        use_jinja=merged["use_jinja"],
        cache_type_k=merged["cache_type_k"],
        cache_type_v=merged["cache_type_v"],
        n_gpu_layers=merged["n_gpu_layers"],
        server_bin=merged["server_bin"],
        backend=merged["backend"],
        risky_acknowledged=merged["risky_acknowledged"],
    )


def create_smoke_config(
    config: Config,
    api_key: str = "",
    model_id_override: str | None = None,
) -> SmokeProbeConfiguration:
    """Create SmokeProbeConfiguration from a Config instance.

    Args:
        config: Base Config with smoke defaults.
        api_key: Override API key (from CLI flag).
        model_id_override: Override model ID.

    Returns:
        SmokeProbeConfiguration with resolved values.

    """
    api_key_resolved = api_key or config.smoke_api_key
    return SmokeProbeConfiguration(
        inter_slot_delay_s=config.smoke_inter_slot_delay_s,
        listen_timeout_s=config.smoke_listen_timeout_s,
        http_request_timeout_s=config.smoke_http_request_timeout_s,
        max_tokens=config.smoke_max_tokens,
        prompt=config.smoke_prompt,
        skip_models_discovery=config.smoke_skip_models_discovery,
        api_key=api_key_resolved,
        model_id_override=model_id_override,
        first_token_timeout_s=config.smoke_first_token_timeout_s,
        total_chat_timeout_s=config.smoke_total_chat_timeout_s,
    )
