# ServerConfig creation helpers


import os
from copy import deepcopy
from typing import Any

from .config import Config, ServerConfig
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
    cfg = Config()
    return ServerConfig(
        model=cfg.model_summary_balanced,
        alias="summary-balanced",
        device="SYCL0",
        port=port,
        ctx_size=ctx_size or cfg.default_ctx_size_summary,
        ubatch_size=ubatch_size or cfg.default_ubatch_size_summary_balanced,
        threads=threads or cfg.default_threads_summary_balanced,
        reasoning_mode="off",
        reasoning_format="deepseek",
        chat_template_kwargs=cfg.summary_balanced_chat_template_kwargs,
        use_jinja=True,
        cache_type_k=cache_k or cfg.default_cache_type_summary_k,
        cache_type_v=cache_v or cfg.default_cache_type_summary_v,
        backend="llama_cpp",
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
    cfg = Config()
    return ServerConfig(
        model=cfg.model_summary_fast,
        alias="summary-fast",
        device="SYCL0",
        port=port,
        ctx_size=ctx_size or cfg.default_ctx_size_summary,
        ubatch_size=ubatch_size or cfg.default_ubatch_size_summary_fast,
        threads=threads or cfg.default_threads_summary_fast,
        reasoning_mode="off",
        reasoning_format="deepseek",
        chat_template_kwargs=cfg.summary_fast_chat_template_kwargs,
        use_jinja=True,
        cache_type_k=cache_k or cfg.default_cache_type_summary_k,
        cache_type_v=cache_v or cfg.default_cache_type_summary_v,
        backend="llama_cpp",
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
    cfg = Config()
    return ServerConfig(
        model=model or cfg.model_qwen35,
        alias="qwen35-coding",
        device="",
        port=port,
        ctx_size=ctx_size or cfg.default_ctx_size_qwen35,
        ubatch_size=ubatch_size or cfg.default_ubatch_size_qwen35,
        threads=threads or cfg.default_threads_qwen35,
        cache_type_k=cache_k or cfg.default_cache_type_qwen35_k,
        cache_type_v=cache_v or cfg.default_cache_type_qwen35_v,
        n_gpu_layers=n_gpu_layers,
        server_bin=server_bin or cfg.llama_server_bin_nvidia,
        backend=backend,
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
