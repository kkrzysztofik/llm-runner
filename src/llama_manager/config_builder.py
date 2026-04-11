# ServerConfig creation helpers


from copy import deepcopy

from .config import Config, ServerConfig


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts with override taking precedence.

    FR-006: Priority order is defaults < slot/workstation < profile < override.
    This function implements the merge logic where override values take precedence
    over base values, recursively merging nested dicts.

    Args:
        base: Base dictionary (lower precedence)
        override: Override dictionary (higher precedence)

    Returns:
        Merged dictionary with override values taking precedence
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_summary_balanced_cfg(
    port: int,
    ctx_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int | None = None,
    cache_k: str | None = None,
    cache_v: str | None = None,
) -> ServerConfig:
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
    cfg = Config()
    return ServerConfig(
        model=cfg.model_summary_fast,
        alias="summary-fast",
        device="SYCL0",
        port=port,
        ctx_size=ctx_size or cfg.default_ctx_size_summary,
        ubatch_size=ubatch_size or cfg.default_ubatch_size_summary_fast,
        threads=threads or cfg.default_threads_summary_fast,
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
    profile_config: dict | None = None,
    override_config: dict | None = None,
) -> ServerConfig:
    """FR-006: Merge configuration with precedence: defaults < slot/workstation < profile < override.

    This function implements the FR-006 precedence merge baseline, combining configuration
    sources in priority order. Dict fields are deep merged; scalar fields use higher-precedence
    source. The existing create_*_cfg functions remain compatible.

    Args:
        defaults: Base Config object with default values
        slot_config: Optional dict for slot-level overrides (e.g., {"port": 8080, "ctx_size": 32000})
        workstation_config: Optional dict for workstation-level overrides
        profile_config: Optional dict for profile-level overrides
        override_config: Optional dict for explicit CLI override (highest precedence)

    Returns:
        ServerConfig constructed from merged configuration

    Example:
        cfg = Config()
        final_cfg = merge_config_overrides(
            defaults=cfg,
            slot_config={"port": 8080, "ctx_size": 32000},
            profile_config={"threads": 12, "ubatch_size": 2048},
            override_config={"port": 9000},  # Will win over slot_config
        )
    """
    # Start with defaults
    base_defaults: dict = {
        "model": defaults.model_summary_balanced,
        "alias": "default",
        "device": "",
        "port": defaults.summary_balanced_port,
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
    if profile_config:
        merged = _deep_merge(merged, profile_config)

    # Step 4: override_config (highest precedence)
    if override_config:
        merged = _deep_merge(merged, override_config)

    # Convert dict back to ServerConfig
    return ServerConfig(
        model=merged["model"],
        alias=merged["alias"],
        device=merged["device"],
        port=merged["port"],
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
        n_gpu_layers=merged.get("n_gpu_layers", defaults.default_n_gpu_layers),
        server_bin=merged.get("server_bin", ""),
        backend=merged.get("backend", "llama_cpp"),
    )
