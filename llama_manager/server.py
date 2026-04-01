# Server command building and validation functions


import os
import sys

from .config import Config, ServerConfig


def build_server_cmd(cfg: ServerConfig) -> list[str]:
    """Build llama-server command arguments"""
    cmd = [
        cfg.server_bin or Config().llama_server_bin_intel,
        "--model",
        cfg.model,
        "--alias",
        cfg.alias,
        "--n-gpu-layers",
        str(cfg.n_gpu_layers),
        "--split-mode",
        "layer",
        "--ctx-size",
        str(cfg.ctx_size),
        "--flash-attn",
        "on",
        "--cache-type-k",
        cfg.cache_type_k,
        "--cache-type-v",
        cfg.cache_type_v,
        "--batch-size",
        "2048",
        "--ubatch-size",
        str(cfg.ubatch_size),
        "--threads",
        str(cfg.threads),
        "--poll",
        "50",
        "--mmap",
        "--host",
        Config().host,
        "--port",
        str(cfg.port),
        "--no-webui",
    ]

    if cfg.device:
        cmd.extend(["--device", cfg.device])
    if cfg.reasoning_mode:
        cmd.extend(["--reasoning", cfg.reasoning_mode])
    if cfg.reasoning_format:
        cmd.extend(["--reasoning-format", cfg.reasoning_format])
    if cfg.tensor_split:
        cmd.extend(["--tensor-split", cfg.tensor_split])
    if cfg.chat_template_kwargs:
        cmd.extend(["--chat-template-kwargs", cfg.chat_template_kwargs])
    if cfg.reasoning_budget:
        cmd.extend(["--reasoning-budget", cfg.reasoning_budget])
    if cfg.use_jinja:
        cmd.append("--jinja")

    return cmd


def validate_port(port: int, name: str = "port") -> None:
    """Validate port number"""
    if not isinstance(port, int) or port < 1 or port > 65535:
        print(f"error: {name} must be between 1 and 65535, got: {port}", file=sys.stderr)
        sys.exit(1)


def validate_threads(threads: int, name: str = "threads") -> None:
    """Validate thread count"""
    if not isinstance(threads, int) or threads < 1:
        print(f"error: {name} must be greater than 0, got: {threads}", file=sys.stderr)
        sys.exit(1)


def require_model(model_path: str) -> None:
    """Check if model file exists"""
    if not os.path.isfile(model_path):
        print(f"error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)


def require_executable(bin_path: str, name: str = "binary") -> None:
    """Check if executable exists"""
    if not os.access(bin_path, os.X_OK):
        print(f"error: {name} not found or not executable: {bin_path}", file=sys.stderr)
        sys.exit(1)


def validate_ports(port1: int, port2: int, name1: str = "port1", name2: str = "port2") -> None:
    """Validate ports are different"""
    if port1 == port2:
        print(
            f"error: {name1} and {name2} must be different, got: {port1}",
            file=sys.stderr,
        )
        sys.exit(1)
