# CLI argument parsing


import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Manage multiple llama-server instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  summary-balanced  Run summary-balanced model (Intel SYCL)
  summary-fast      Run summary-fast model (Intel SYCL)
  qwen35           Run qwen35-coding model (NVIDIA CUDA)
  both             Run summary-balanced and qwen35 side-by-side
  dry-run          Preview commands without executing

Examples:
  run_opencode_models.py summary-balanced
  run_opencode_models.py summary-fast 8082
  run_opencode_models.py qwen35 8080
  run_opencode_models.py both 8080 8081
  run_opencode_models.py dry-run both
        """,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["summary-balanced", "summary-fast", "qwen35", "both", "dry-run"],
        help="Mode to run",
    )
    parser.add_argument(
        "ports",
        nargs="*",
        type=int,
        help="Port(s) for models (1st port, 2nd port for 'both')",
    )

    return parser.parse_args()


def parse_tui_args() -> argparse.Namespace:
    """Parse TUI-specific command line arguments"""
    parser = argparse.ArgumentParser(
        description="TUI for managing multiple llama-server instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s both                     Run summary-balanced and qwen35 side-by-side
  %(prog)s both --port 8080 --port2 8081
  %(prog)s summary-balanced --port 8080
  %(prog)s qwen35 --port 8081
  %(prog)s summary-fast

GPU Mapping:
  - NVIDIA (CUDA) -> GPU 0 (RTX 3090)
  - Intel (SYCL)  -> GPU 1 (Arc B580)
        """,
    )

    modes = ["both", "summary-balanced", "summary-fast", "qwen35"]
    parser.add_argument(
        "mode",
        choices=modes,
        help=f"Mode to run: {' | '.join(modes)}",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port for primary model",
    )
    parser.add_argument(
        "--port2",
        "-p2",
        type=int,
        help="Port for secondary model",
    )

    return parser.parse_args()
