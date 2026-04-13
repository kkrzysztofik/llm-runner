# CLI argument parsing

import argparse
import sys


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Handles the special case of 'dry-run' mode which requires a second argument
    specifying the mode to preview (e.g., 'dry-run both').
    """
    if args is None:
        args = []

    # Custom parsing for dry-run mode
    if len(args) >= 1 and args[0] == "dry-run":
        # dry-run requires a second argument specifying the mode
        if len(args) < 2:
            print(
                "error: dry-run requires a mode argument (summary-balanced|summary-fast|qwen35|both)",
                file=sys.stderr,
            )
            sys.exit(1)

        dry_run_mode = args[1]
        allowed_modes = ["summary-balanced", "summary-fast", "qwen35", "both"]
        if dry_run_mode not in allowed_modes:
            print(
                f"error: invalid dry-run mode '{dry_run_mode}'. Valid modes:",
                "summary-balanced, summary-fast, qwen35, both",
                file=sys.stderr,
            )
            sys.exit(1)

        # Remaining args after mode are ports (excluding --acknowledge-risky flag)
        ports = []
        acknowledge_risky = False
        for arg in args[2:]:
            if arg == "--acknowledge-risky":
                acknowledge_risky = True
            else:
                try:
                    ports.append(int(arg))
                except ValueError:
                    print(f"error: invalid port '{arg}'", file=sys.stderr)
                    sys.exit(1)

        return argparse.Namespace(
            mode="dry-run",
            dry_run_mode=dry_run_mode,
            ports=ports,
            acknowledge_risky=acknowledge_risky,
        )

    # Normal mode parsing
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
      src/run_opencode_models.py summary-balanced
      src/run_opencode_models.py summary-fast 8082
      src/run_opencode_models.py qwen35 8080
      src/run_opencode_models.py both 8080 8081
      src/run_opencode_models.py dry-run summary-balanced
      src/run_opencode_models.py dry-run both 8080 8081
            """,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["summary-balanced", "summary-fast", "qwen35", "both"],
        help="Mode to run",
    )
    parser.add_argument(
        "ports",
        nargs="*",
        type=int,
        help="Port(s) for models (1st port, 2nd port for 'both')",
    )

    parser.add_argument(
        "--acknowledge-risky",
        action="store_true",
        help="Acknowledge risky operations (privileged ports, etc.)",
    )

    parsed = parser.parse_args(args)
    # Add dry_run_mode attribute for consistency
    parsed.dry_run_mode = None
    return parsed


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

    parser.add_argument(
        "--acknowledge-risky",
        action="store_true",
        help="Acknowledge risky operations (privileged ports, etc.)",
    )

    return parser.parse_args()
