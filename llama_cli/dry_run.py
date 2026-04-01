# Dry run functionality


import sys

from llama_manager import (
    Config,
    build_server_cmd,
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
)


def dry_run(
    mode: str,
    primary_port: str | None = None,
    secondary_port: str | None = None,
) -> None:
    """Print command without executing"""
    cfg = Config()

    summary_balanced_port = int(primary_port) if primary_port else cfg.summary_balanced_port
    summary_fast_port = int(primary_port) if primary_port else cfg.summary_fast_port
    qwen35_port = int(primary_port) if primary_port else cfg.qwen35_port
    qwen35_port_both = int(secondary_port) if secondary_port else cfg.qwen35_port

    print("=== DRY RUN MODE ===")
    print(f"Mode: {mode}")
    print(f"llama-server (Intel): {cfg.llama_server_bin_intel}")
    print(f"llama-server (NVIDIA): {cfg.llama_server_bin_nvidia}")
    print(f"summary-balanced model: {cfg.model_summary_balanced}")
    print(f"summary-fast model: {cfg.model_summary_fast}")
    print(f"qwen35 model: {cfg.model_qwen35}")
    print(f"qwen35 both model: {cfg.model_qwen35_both}")
    print()

    if mode in ("summary-balanced", "llama32"):
        server_cfg = create_summary_balanced_cfg(summary_balanced_port)
        print("summary-balanced:")
        print(f"  Port: {summary_balanced_port}")
        print("  Device: SYCL0")
        print(f"  Context: {cfg.default_ctx_size_summary}")
        print(f"  Threads: {cfg.default_threads_summary_balanced}")
        print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
        print("  Reasoning: off")
        print("  Reasoning Format: deepseek")
        print("  Jinja: True")
        print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
        cmd = build_server_cmd(server_cfg)
        print(f"  Command: {' '.join(cmd)}")
        print()

    elif mode == "summary-fast":
        server_cfg = create_summary_fast_cfg(summary_fast_port)
        print("summary-fast:")
        print(f"  Port: {summary_fast_port}")
        print("  Device: SYCL0")
        print(f"  Context: {cfg.default_ctx_size_summary}")
        print(f"  Threads: {cfg.default_threads_summary_fast}")
        print(f"  UBatch: {cfg.default_ubatch_size_summary_fast}")
        cmd = build_server_cmd(server_cfg)
        print(f"  Command: {' '.join(cmd)}")
        print()

    elif mode == "qwen35":
        server_cfg = create_qwen35_cfg(
            qwen35_port,
            n_gpu_layers=cfg.default_n_gpu_layers_qwen35,
            model=cfg.model_qwen35,
            server_bin=cfg.llama_server_bin_nvidia,
        )
        print("qwen35:")
        print(f"  Port: {qwen35_port}")
        print("  Device: NVIDIA (CUDA)")
        print(f"  Context: {cfg.default_ctx_size_qwen35}")
        print(f"  Threads: {cfg.default_threads_qwen35}")
        print(f"  UBatch: {cfg.default_ubatch_size_qwen35}")
        print(f"  KV cache: {cfg.default_cache_type_qwen35_k}/{cfg.default_cache_type_qwen35_v}")
        print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35}")
        cmd = build_server_cmd(server_cfg)
        print(f"  Command: {' '.join(cmd)}")
        print()

    elif mode == "both":
        server_cfg1 = create_summary_balanced_cfg(summary_balanced_port)
        server_cfg2 = create_qwen35_cfg(
            qwen35_port_both,
            ctx_size=cfg.default_ctx_size_both_qwen35,
            ubatch_size=cfg.default_ubatch_size_qwen35_both,
            threads=cfg.default_threads_qwen35_both,
            cache_k=cfg.default_cache_type_qwen35_both_k,
            cache_v=cfg.default_cache_type_qwen35_both_v,
            n_gpu_layers=cfg.default_n_gpu_layers_qwen35_both,
            model=cfg.model_qwen35_both,
            server_bin=cfg.llama_server_bin_nvidia,
        )
        print("summary-balanced:")
        print(f"  Port: {summary_balanced_port}")
        print("  Device: SYCL0")
        print(f"  Context: {cfg.default_ctx_size_both_summary}")
        print(f"  Threads: {cfg.default_threads_summary_balanced}")
        print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
        print(f"  KV cache: {cfg.default_cache_type_summary_k}/{cfg.default_cache_type_summary_v}")
        print("  Reasoning: off")
        print("  Reasoning Format: deepseek")
        print("  Jinja: True")
        print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
        cmd1 = build_server_cmd(server_cfg1)
        print(f"  Command: {' '.join(cmd1)}")
        print()
        print("qwen35:")
        print(f"  Port: {qwen35_port_both}")
        print("  Device: NVIDIA (CUDA)")
        print(f"  Context: {cfg.default_ctx_size_both_qwen35}")
        print(f"  Threads: {cfg.default_threads_qwen35_both}")
        print(f"  UBatch: {cfg.default_ubatch_size_qwen35_both}")
        print(
            f"  KV cache: {cfg.default_cache_type_qwen35_both_k}/{cfg.default_cache_type_qwen35_both_v}"
        )
        print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35_both}")
        cmd2 = build_server_cmd(server_cfg2)
        print(f"  Command: {' '.join(cmd2)}")
        print()

    else:
        print(
            "error: dry-run requires a mode argument (summary-balanced|summary-fast|qwen35|both)",
            file=sys.stderr,
        )
        sys.exit(1)
