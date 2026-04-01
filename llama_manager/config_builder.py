# ServerConfig creation helpers


from .config import Config, ServerConfig


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
    )
