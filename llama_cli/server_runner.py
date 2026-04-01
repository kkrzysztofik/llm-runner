# Server execution logic for CLI


import atexit
import os
import signal
import sys

from llama_manager import (
    Color,
    Config,
    ServerManager,
    build_server_cmd,
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
    require_executable,
    require_model,
    validate_port,
    validate_ports,
)


def usage() -> None:
    print("""Usage:
  run_opencode_models.py summary-balanced [port]
  run_opencode_models.py summary-fast [port]
  run_opencode_models.py qwen35 [port]
  run_opencode_models.py both [summary_balanced_port qwen35_port]
  run_opencode_models.py dry-run summary-balanced|summary-fast|qwen35|both [ports...]

Examples:
  run_opencode_models.py summary-balanced
  run_opencode_models.py summary-fast 8082
  run_opencode_models.py qwen35 8080
  run_opencode_models.py both 8080 8081
  run_opencode_models.py dry-run both""")


def check_prereqs() -> None:
    cfg = Config()
    require_executable(cfg.llama_server_bin_intel, "Intel llama-server")


def run_summary_balanced(port: int, manager: ServerManager) -> int:
    """Run summary-balanced server"""
    cfg = Config()
    validate_port(port, "summary-balanced port")
    require_model(cfg.model_summary_balanced)
    print(f"Starting summary-balanced at http://{cfg.host}:{port}/v1")
    server_cfg = create_summary_balanced_cfg(port)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("summary-balanced", cmd)


def run_summary_fast(port: int, manager: ServerManager) -> int:
    """Run summary-fast server"""
    cfg = Config()
    validate_port(port, "summary-fast port")
    require_model(cfg.model_summary_fast)
    print(f"Starting summary-fast at http://{cfg.host}:{port}/v1")
    server_cfg = create_summary_fast_cfg(port)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("summary-fast", cmd)


def run_qwen35(port: int, manager: ServerManager) -> int:
    """Run qwen35-coding server"""
    cfg = Config()
    validate_port(port, "qwen35 port")
    require_model(cfg.model_qwen35)
    require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")
    print(f"Starting qwen35-coding at http://{cfg.host}:{port}/v1 (NVIDIA CUDA)")
    server_cfg = create_qwen35_cfg(port)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("qwen35-coding", cmd)


def run_both(port32: int, port35: int, manager: ServerManager) -> int:
    """Run both summary-balanced and qwen35 servers"""
    cfg = Config()
    validate_port(port32, "summary-balanced port")
    validate_port(port35, "qwen35 port")
    validate_ports(port32, port35, "summary-balanced port", "qwen35 port")
    require_model(cfg.model_summary_balanced)
    require_model(cfg.model_qwen35_both)
    require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")
    server_cfg1 = create_summary_balanced_cfg(
        port32,
        ctx_size=cfg.default_ctx_size_both_summary,
        ubatch_size=cfg.default_ubatch_size_summary_balanced,
        threads=cfg.default_threads_summary_balanced,
        cache_k=cfg.default_cache_type_summary_k,
        cache_v=cfg.default_cache_type_summary_v,
    )
    server_cfg2 = create_qwen35_cfg(
        port35,
        ctx_size=cfg.default_ctx_size_both_qwen35,
        ubatch_size=cfg.default_ubatch_size_qwen35_both,
        threads=cfg.default_threads_qwen35_both,
        cache_k=cfg.default_cache_type_qwen35_both_k,
        cache_v=cfg.default_cache_type_qwen35_both_v,
        n_gpu_layers=cfg.default_n_gpu_layers_qwen35_both,
        model=cfg.model_qwen35_both,
        server_bin=cfg.llama_server_bin_nvidia,
    )
    cmd1 = build_server_cmd(server_cfg1)
    cmd2 = build_server_cmd(server_cfg2)
    print(f"summary-balanced: http://{cfg.host}:{port32}/v1")
    print(f"qwen35-coding: http://{cfg.host}:{port35}/v1")
    manager.start_server_background("summary-balanced", cmd1)
    manager.start_server_background("qwen35-coding", cmd2)
    code = manager.wait_for_any()
    manager.cleanup_servers()
    return code


def main(args: list[str]) -> int:
    """Main CLI entry point"""
    if len(args) < 2:
        usage()
        return 1

    mode = args[1]

    # Setup signal handlers
    manager = ServerManager()
    signal.signal(signal.SIGINT, manager.on_interrupt)
    signal.signal(signal.SIGTERM, manager.on_terminate)
    atexit.register(manager.cleanup_servers)

    # Initialize
    Color.is_enabled()
    check_prereqs()
    os.environ["ZES_ENABLE_SYSMAN"] = "1"

    # Handle dry-run
    if mode == "dry-run":
        if len(args) < 3:
            print("error: dry-run requires a mode argument", file=sys.stderr)
            usage()
            return 1
        mode = args[2]
        primary_port = args[3] if len(args) > 3 else ""
        secondary_port = args[4] if len(args) > 4 else ""
        from llama_cli.dry_run import dry_run

        dry_run(mode, primary_port, secondary_port)
        return 0

    # Parse and execute mode
    cfg = Config()

    try:
        if mode in ("summary-balanced", "llama32"):
            port = int(args[2]) if len(args) > 2 else cfg.summary_balanced_port
            return run_summary_balanced(port, manager)

        elif mode == "summary-fast":
            port = int(args[2]) if len(args) > 2 else cfg.summary_fast_port
            return run_summary_fast(port, manager)

        elif mode == "qwen35":
            port = int(args[2]) if len(args) > 2 else cfg.qwen35_port
            return run_qwen35(port, manager)

        elif mode == "both":
            port32 = int(args[2]) if len(args) > 2 else cfg.summary_balanced_port
            port35 = int(args[3]) if len(args) > 3 else cfg.qwen35_port
            return run_both(port32, port35, manager)

        else:
            usage()
            return 1

    except (ValueError, IndexError):
        usage()
        return 1


def cli_main() -> None:
    """Entry point for the `llm-runner` console script."""
    sys.exit(main(sys.argv[1:]))
