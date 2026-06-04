"""ServerConfig creation helpers."""

import dataclasses
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from loguru import logger

from ..gpu_telemetry import get_gpu_identifier
from .defaults import Config, SmokeProbeConfiguration
from .profile_cache import (
    PROFILE_OVERRIDE_FIELDS,
    ProfileFlavor,
    StalenessResult,
    load_profile_with_staleness,
    profile_to_override_dict,
)
from .profiles import (
    SlotProfileRegistry,
    SlotProfileSpec,
    _derive_tensor_split_from_device,
    _parse_main_gpu_from_device,
)
from .server import ServerConfig
from .spec_decode import SpeculativeDecodingConfig

_SPEC_DECODE_FIELDS: frozenset[str] = frozenset(
    field.name for field in dataclasses.fields(SpeculativeDecodingConfig)
)


def _split_spec_decode_values(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate legacy flat spec keys from ServerConfig keys."""
    config_data = deepcopy(data)
    spec_data = dict(config_data.pop("spec_decode", {}) or {})
    for key in _SPEC_DECODE_FIELDS:
        if key in config_data:
            spec_data[key] = config_data.pop(key)
    return config_data, spec_data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dicts with override taking precedence.

    FR-006: Priority order is defaults < slot/workstation < profile < override.
    This function implements the merge logic where override values take precedence
    over base values, recursively merging nested dicts.

    List values are merged by concatenation: base list items are preserved first,
    then override list items are appended.  All values are deep-copied to prevent
    shared mutable state between caller and result.

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
        isinstance(layer, Mapping) and "model" in layer
        for layer in (slot_config, workstation_config, profile_config, override_config)
    )
    if model_overridden:
        model = merged.get("model")
        if not isinstance(model, str) or not os.path.exists(model):
            raise ValueError(f"model path not found: {model}")


def _without_none(values: dict[str, Any]) -> dict[str, Any]:
    """Return a copy excluding keys with ``None`` values."""
    return {key: value for key, value in values.items() if value is not None}


def _validate_resolved_profile_data(data: dict[str, Any]) -> None:
    """Validate required fields and value ranges in resolved profile data."""
    port = data.get("port")
    if not isinstance(port, int) or not (1024 <= port <= 65535):
        raise ValueError(f"port must be between 1024 and 65535, got: {port}")

    threads = data.get("threads")
    if not isinstance(threads, int) or threads <= 0:
        raise ValueError(f"threads must be greater than 0, got: {threads}")

    for key in ("model", "alias", "backend"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")


def _profile_to_config_data(profile: SlotProfileSpec) -> dict[str, Any]:
    """Convert slot profile data to a complete ServerConfig-compatible mapping."""
    # Auto-derive tensor_split and main_gpu from device if not explicitly set
    tensor_split = profile.tensor_split or _derive_tensor_split_from_device(profile.device)
    main_gpu = (
        profile.main_gpu if profile.main_gpu != 0 else _parse_main_gpu_from_device(profile.device)
    )
    return {
        "model": profile.model,
        "alias": profile.alias,
        "device": profile.device,
        "port": profile.port,
        "bind_address": profile.bind_address,
        "ctx_size": profile.ctx_size,
        "ubatch_size": profile.ubatch_size,
        "threads": profile.threads,
        "tensor_split": tensor_split,
        "chat_template_kwargs": profile.chat_template_kwargs,
        "spec_decode": dataclasses.asdict(profile.spec_decode),
        "use_jinja": profile.use_jinja,
        "cache_type_k": profile.cache_type_k,
        "cache_type_v": profile.cache_type_v,
        "n_gpu_layers": profile.n_gpu_layers,
        "main_gpu": main_gpu,
        "server_bin": profile.server_bin,
        "backend": profile.backend,
        "risky_acknowledged": list(profile.risky_acknowledged),
        "batch_size": profile.batch_size,
        "poll_ms": profile.poll_ms,
        "n_predict": profile.n_predict,
        "parallel": profile.parallel,
        "threads_batch": profile.threads_batch,
        "mmproj": profile.mmproj,
    }


def _config_data_to_server_config(data: dict[str, Any]) -> ServerConfig:
    """Convert validated ServerConfig-compatible data to ServerConfig."""
    config_data, spec_data = _split_spec_decode_values(data)
    return ServerConfig(
        model=config_data["model"],
        alias=config_data["alias"],
        device=config_data["device"],
        port=config_data["port"],
        bind_address=config_data.get("bind_address", "127.0.0.1"),
        ctx_size=config_data["ctx_size"],
        ubatch_size=config_data["ubatch_size"],
        threads=config_data["threads"],
        tensor_split=config_data.get("tensor_split", ""),
        chat_template_kwargs=config_data["chat_template_kwargs"],
        use_jinja=config_data["use_jinja"],
        cache_type_k=config_data["cache_type_k"],
        cache_type_v=config_data["cache_type_v"],
        n_gpu_layers=config_data["n_gpu_layers"],
        main_gpu=config_data.get("main_gpu", 0),
        server_bin=config_data["server_bin"],
        backend=config_data["backend"],
        risky_acknowledged=config_data["risky_acknowledged"],
        batch_size=int(config_data.get("batch_size", 2048)),
        poll_ms=int(config_data.get("poll_ms", 50)),
        n_predict=int(config_data.get("n_predict", 32768)),
        parallel=int(config_data.get("parallel", 4)),
        threads_batch=int(config_data.get("threads_batch", 0)),
        mmproj=str(config_data.get("mmproj", "")),
        spec_decode=SpeculativeDecodingConfig(
            spec_type=str(spec_data.get("spec_type", "")),
            spec_ngram_size_n=int(spec_data.get("spec_ngram_size_n", 0)),
            draft_min=int(spec_data.get("draft_min", 0)),
            draft_max=int(spec_data.get("draft_max", 0)),
            spec_draft_n_max=int(spec_data.get("spec_draft_n_max", 0)),
            spec_draft_p_min=float(spec_data.get("spec_draft_p_min", 0.0)),
            spec_draft_cache_type_k=str(spec_data.get("spec_draft_cache_type_k", "")),
            spec_draft_cache_type_v=str(spec_data.get("spec_draft_cache_type_v", "")),
            spec_draft_device=str(spec_data.get("spec_draft_device", "")),
            reasoning_mode=str(spec_data.get("reasoning_mode", "auto")),
            reasoning_format=str(spec_data.get("reasoning_format", "none")),
            reasoning_budget=str(spec_data.get("reasoning_budget", "")),
        ),
    )


def create_server_config_from_profile(
    profile: SlotProfileSpec,
    override_config: dict[str, Any] | None = None,
) -> ServerConfig:
    """Resolve any slot profile definition into a ServerConfig.

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
    registry: SlotProfileRegistry,
    profile_id: str,
    override_config: dict[str, Any] | None = None,
) -> ServerConfig:
    """Resolve one registered slot profile into ServerConfig.

    Args:
        registry: Slot profile registry containing profile definitions.
        profile_id: Slot profile identifier to resolve.
        override_config: Optional explicit slot profile overrides.

    Returns:
        ServerConfig for the requested profile.
    """
    return create_server_config_from_profile(registry.get_profile(profile_id), override_config)


def create_summary_balanced_cfg(
    port: int,
    ctx_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    cache_k: str | None = None,
    cache_v: str | None = None,
    registry: SlotProfileRegistry | None = None,
) -> ServerConfig:
    """Create a ServerConfig for the summary-balanced model profile.

    Args:
        port: Port to bind the server to.
        ctx_size: Context size (defaults to Config.default_ctx_size_summary).
        ubatch_size: Ubatch size (defaults to Config.default_ubatch_size_summary_balanced).
        threads: Number of threads (defaults to Config.default_threads_summary_balanced).
        cache_k: K cache type.
        cache_v: V cache type.
        registry: Optional pre-built ProfileRegistry to reuse across calls.
            When omitted, a fresh registry is created via
            ``create_default_profile_registry()``.

    Returns:
        A configured ServerConfig instance.

    """
    if registry is None:
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
    registry: SlotProfileRegistry | None = None,
) -> ServerConfig:
    """Create a ServerConfig for the summary-fast model profile.

    Args:
        port: Port to bind the server to.
        ctx_size: Context size (defaults to Config.default_ctx_size_summary).
        ubatch_size: Ubatch size (defaults to Config.default_ubatch_size_summary_fast).
        threads: Number of threads (defaults to Config.default_threads_summary_fast).
        cache_k: K cache type.
        cache_v: V cache type.
        registry: Optional pre-built ProfileRegistry to reuse across calls.
            When omitted, a fresh registry is created via
            ``create_default_profile_registry()``.

    Returns:
        A configured ServerConfig instance.

    """
    if registry is None:
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
    registry: SlotProfileRegistry | None = None,
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
        registry: Optional pre-built ProfileRegistry to reuse across calls.
            When omitted, a fresh registry is created via
            ``create_default_profile_registry()``.

    Returns:
        A configured ServerConfig instance.

    """
    if registry is None:
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


def create_default_slot_profiles(config: Config | None = None) -> tuple[SlotProfileSpec, ...]:
    """Create built-in slot profiles as typed data entries.

    Args:
        config: Optional base configuration. When omitted, environment-aware
            defaults are loaded from ``Config``.

    Returns:
        Built-in single-server slot profile definitions in stable CLI order.
    """
    cfg = config or Config()
    return (
        SlotProfileSpec(
            profile_id="summary-balanced",
            description="Run summary-balanced model on Intel SYCL.",
            model=cfg.deployment.model_summary_balanced,
            alias="summary-balanced",
            device="SYCL0",
            port=cfg.deployment.summary_balanced_port,
            ctx_size=cfg.server_defaults.ctx_size_summary,
            ubatch_size=cfg.server_defaults.ubatch_size_summary_balanced,
            threads=cfg.server_defaults.threads_summary_balanced,
            spec_decode=SpeculativeDecodingConfig(
                reasoning_mode="off",
                reasoning_format="deepseek",
            ),
            chat_template_kwargs=cfg.deployment.summary_balanced_chat_template_kwargs,
            use_jinja=True,
            cache_type_k=cfg.server_defaults.cache_type_summary_k,
            cache_type_v=cfg.server_defaults.cache_type_summary_v,
            parallel=4,
            backend="llama_cpp",
        ),
        SlotProfileSpec(
            profile_id="summary-fast",
            description="Run summary-fast model on Intel SYCL.",
            model=cfg.deployment.model_summary_fast,
            alias="summary-fast",
            device="SYCL0",
            port=cfg.deployment.summary_fast_port,
            ctx_size=cfg.server_defaults.ctx_size_summary,
            ubatch_size=cfg.server_defaults.ubatch_size_summary_fast,
            threads=cfg.server_defaults.threads_summary_fast,
            spec_decode=SpeculativeDecodingConfig(
                reasoning_mode="off",
                reasoning_format="deepseek",
            ),
            chat_template_kwargs=cfg.deployment.summary_fast_chat_template_kwargs,
            use_jinja=True,
            cache_type_k=cfg.server_defaults.cache_type_summary_k,
            cache_type_v=cfg.server_defaults.cache_type_summary_v,
            parallel=4,
            backend="llama_cpp",
        ),
        SlotProfileSpec(
            profile_id="qwen35",
            description="Run qwen35-coding model on NVIDIA CUDA.",
            model=cfg.deployment.model_qwen35,
            alias="qwen35-coding",
            device="",
            port=cfg.deployment.qwen35_port,
            ctx_size=cfg.server_defaults.ctx_size_qwen35,
            ubatch_size=256,
            threads=cfg.server_defaults.threads_qwen35,
            cache_type_k=cfg.server_defaults.cache_type_qwen35_k,
            cache_type_v=cfg.server_defaults.cache_type_qwen35_v,
            n_gpu_layers=cfg.server_defaults.n_gpu_layers_qwen35,
            server_bin=cfg.paths.llama_server_bin_nvidia,
            batch_size=1024,
            poll_ms=0,
            parallel=4,
            backend="llama_cpp",
        ),
    )


def create_default_profile_registry(config: Config | None = None) -> SlotProfileRegistry:
    """Create the built-in dynamic slot profile registry.

    Args:
        config: Optional base configuration used to resolve environment-aware
            model paths, ports, binaries, and tuning defaults.

    Returns:
        Validated registry containing built-in slot profiles.
    """
    return SlotProfileRegistry(profiles=create_default_slot_profiles(config))


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
        "model": defaults.deployment.model_summary_balanced,
        "alias": "default",
        "device": "",
        "port": defaults.deployment.summary_balanced_port,
        "bind_address": "127.0.0.1",
        "ctx_size": defaults.server_defaults.ctx_size_summary,
        "ubatch_size": defaults.server_defaults.ubatch_size_summary_balanced,
        "threads": defaults.server_defaults.threads_summary_balanced,
        "spec_decode": dataclasses.asdict(
            SpeculativeDecodingConfig(reasoning_mode="off", reasoning_format="deepseek")
        ),
        "chat_template_kwargs": defaults.deployment.summary_balanced_chat_template_kwargs,
        "use_jinja": True,
        "cache_type_k": defaults.server_defaults.cache_type_summary_k,
        "cache_type_v": defaults.server_defaults.cache_type_summary_v,
        "n_gpu_layers": defaults.server_defaults.n_gpu_layers,
        "main_gpu": 0,
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

    return _config_data_to_server_config(merged)


def apply_profile_overrides(
    configs: list[ServerConfig],
    base_config: Config,
    get_driver_version: Callable[[str], str],
) -> tuple[list[ServerConfig], list[str]]:
    """Apply cached profile overrides to configs.

    For each config, attempts to load a cached profile. If found and fresh,
    applies the whitelisted profile parameters via merge_config_overrides.
    Identity fields (model, alias, device, port, backend, etc.) are preserved
    from the original config.

    Args:
        configs: List of ServerConfig objects to apply overrides to.
        base_config: Base Config with profiles_dir, staleness settings, etc.
        get_driver_version: Callable that accepts a backend string and returns
            the driver version string.

    Returns:
        Tuple of (updated_configs, status_messages).
    """
    updated_configs: list[ServerConfig] = []
    messages: list[str] = []

    for cfg in configs:
        try:
            gpu_identifier = get_gpu_identifier(cfg.backend)
            driver_version = get_driver_version(cfg.backend)
            binary_version = base_config.server_binary_version or "unknown"

            record, staleness = load_profile_with_staleness(
                profiles_dir=base_config.paths.profiles_dir,
                gpu_identifier=gpu_identifier,
                backend=cfg.backend,
                flavor=ProfileFlavor.BALANCED,
                current_driver_version=driver_version,
                current_binary_version=binary_version,
                staleness_days=base_config.profile_staleness_days,
            )
        except OSError, ValueError, KeyError:
            logger.info("No profile found for %s; falling back to defaults", cfg.alias)
            messages.append(f"No profile found for {cfg.alias}; using defaults")
            updated_configs.append(cfg)
            continue

        if record is None:
            messages.append(f"No profile found for {cfg.alias}; using defaults")
            updated_configs.append(cfg)
            continue

        if staleness is not None and staleness.is_stale:
            reasons = "; ".join(r.value.replace("_", " ").title() for r in staleness.reasons)
            messages.append(f"Profile stale for {cfg.alias}: {reasons}; using defaults")
            updated_configs.append(cfg)
            continue

        profile_overrides = profile_to_override_dict(record)

        if not profile_overrides:
            messages.append(f"Profile empty for {cfg.alias}; using defaults")
            updated_configs.append(cfg)
            continue

        slot_config = dataclasses.asdict(cfg)
        slot_config.pop("model", None)
        merged = merge_config_overrides(
            defaults=base_config,
            slot_config=slot_config,
            workstation_config=None,
            profile_config=profile_overrides,
            override_config=None,
        )

        # Preserve identity fields that shouldn't come from profile
        merged.model = cfg.model
        merged.alias = cfg.alias
        merged.device = cfg.device
        merged.port = cfg.port
        merged.bind_address = cfg.bind_address
        merged.server_bin = cfg.server_bin
        merged.backend = cfg.backend
        merged.tensor_split = cfg.tensor_split
        merged.main_gpu = cfg.main_gpu
        merged.chat_template_kwargs = cfg.chat_template_kwargs
        merged.spec_decode = cfg.spec_decode
        merged.use_jinja = cfg.use_jinja
        merged.n_gpu_layers = cfg.n_gpu_layers
        merged.risky_acknowledged = cfg.risky_acknowledged

        messages.append(
            f"Applied profile: {cfg.alias} (balanced) "
            f"[threads={merged.threads}, ctx={merged.ctx_size}]"
        )
        updated_configs.append(merged)

    return updated_configs, messages


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
    api_key_resolved = api_key or config.smoke.api_key
    return SmokeProbeConfiguration(
        inter_slot_delay_s=config.smoke.inter_slot_delay_s,
        listen_timeout_s=config.smoke.listen_timeout_s,
        http_request_timeout_s=config.smoke.http_request_timeout_s,
        max_tokens=config.smoke.max_tokens,
        prompt=config.smoke.prompt,
        skip_models_discovery=config.smoke.skip_models_discovery,
        api_key=api_key_resolved,
        model_id_override=model_id_override,
        first_token_timeout_s=config.smoke.first_token_timeout_s,
        total_chat_timeout_s=config.smoke.total_chat_timeout_s,
    )


def create_tui_profile_registry(config: Config) -> SlotProfileRegistry:
    """Create a slot profile registry for TUI use: built-in + custom profiles from disk.

    Custom profiles are loaded from ``slot_profiles.toml``. Duplicate
    ``profile_id`` between built-in and custom is resolved by preferring
    the custom profile. Hidden built-in profiles are excluded.

    Args:
        config: Base configuration used to resolve built-in profiles.

    Returns:
        Merged ``SlotProfileRegistry`` with built-in and custom profiles.
    """
    from ..slot_profile_store import load_custom_slot_profiles, load_hidden_builtin_profile_ids

    hidden = load_hidden_builtin_profile_ids()

    builtins = create_default_profile_registry(config)
    custom = load_custom_slot_profiles()

    # Merge: skip hidden built-ins, custom profiles override built-ins with same profile_id
    all_profiles: dict[str, SlotProfileSpec] = {}
    for p in builtins.profiles:
        if p.profile_id not in hidden:
            all_profiles[p.profile_id] = p
    for p in custom:
        all_profiles[p.profile_id] = p

    return SlotProfileRegistry(profiles=tuple(all_profiles.values()))
