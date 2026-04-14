"""CLI argument parsing for llm-runner.

This module handles command-line argument parsing for both standard CLI
and TUI modes, including special handling for dry-run mode which requires
a second argument specifying the mode to preview.
"""

import argparse
import sys

VALID_MODES = ("summary-balanced", "summary-fast", "qwen35", "both")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Handles the special case of 'dry-run' mode which requires a second argument
    specifying the mode to preview (e.g., 'dry-run both').

    Args:
        args: Optional list of command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    if args is None:
        # Preserve argparse default behavior: use sys.argv[1:]
        args = sys.argv[1:]

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
        if dry_run_mode not in VALID_MODES:
            print(
                f"error: invalid dry-run mode '{dry_run_mode}'. "
                f"Valid modes: {', '.join(VALID_MODES)}",
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
      %(prog)s summary-balanced
      %(prog)s summary-fast 8082
      %(prog)s qwen35 8080
      %(prog)s both 8080 8081
      %(prog)s dry-run summary-balanced
      %(prog)s dry-run both 8080 8081
            """,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=VALID_MODES,
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


def parse_tui_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse TUI-specific command line arguments.

    Args:
        args: Optional list of command-line arguments. If None, uses sys.argv[1:].

    Returns:
        Parsed arguments namespace with mode, port, port2, acknowledge_risky,
        and dry_run_mode (always None for TUI mode).
    """
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

    parser.add_argument(
        "mode",
        choices=VALID_MODES,
        help=f"Mode to run: {' | '.join(VALID_MODES)}",
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

    parsed = parser.parse_args(args)
    # Add dry_run_mode attribute for consistency with parse_args
    parsed.dry_run_mode = None
    return parsed
