#!/usr/bin/env python3
"""
run_models_tui.py - TUI for managing multiple llama-server instances
2-column layout with live logs, config, and GPU stats
"""

import sys

from llama_cli import parse_tui_args
from llama_cli.tui_app import TUIApp
from llama_manager import (
    Config,
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
    require_executable,
    require_model,
    validate_port,
    validate_ports,
)


def check_prereqs() -> None:
    """Check prerequisites"""
    cfg = Config()
    require_executable(cfg.llama_server_bin_intel, "Intel llama-server")
    if cfg.llama_server_bin_nvidia:
        require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")


def main():
    args = parse_tui_args()

    # Check prerequisites
    check_prereqs()

    # Create configs based on mode
    configs = []
    gpu_indices = []

    if args.mode == "both":
        configs = [
            create_summary_balanced_cfg(args.port or 8080),
            create_qwen35_cfg(args.port2 or 8081),
        ]
        # GPU mapping: summary-balanced (SYCL) -> GPU 1, qwen35 (CUDA) -> GPU 0
        gpu_indices = [1, 0]
        print("Starting both models...")

    elif args.mode == "summary-balanced":
        configs = [create_summary_balanced_cfg(args.port or 8080)]
        gpu_indices = [1]  # SYCL -> GPU 1
        print("Starting summary-balanced...")

    elif args.mode == "summary-fast":
        configs = [create_summary_fast_cfg(args.port or 8082)]
        gpu_indices = [1]  # SYCL -> GPU 1
        print("Starting summary-fast...")

    elif args.mode == "qwen35":
        configs = [create_qwen35_cfg(args.port or 8081)]
        gpu_indices = [0]  # CUDA -> GPU 0
        print("Starting qwen35-coding...")

    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

    # Validate ports
    for cfg in configs:
        validate_port(cfg.port, cfg.alias)

    # Validate ports are different if multiple configs
    if len(configs) > 1:
        validate_ports(
            configs[0].port,
            configs[1].port,
            configs[0].alias + " port",
            configs[1].alias + " port",
        )

    # Validate models exist
    for cfg in configs:
        require_model(cfg.model)

    # Run TUI
    app = TUIApp(configs, gpu_indices)
    app.run(acknowledged=args.acknowledge_risky)


if __name__ == "__main__":
    main()
