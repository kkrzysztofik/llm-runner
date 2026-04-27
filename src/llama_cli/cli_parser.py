"""CLI argument parsing for llm-runner.

This module handles command-line argument parsing for both standard CLI
and TUI modes, including special handling for dry-run mode which requires
a second argument specifying the mode to preview.
"""

import argparse
import sys

from llama_manager.config_builder import create_default_profile_registry

COMMAND_MODES = (
    "build",
    "setup",
    "doctor",
)


def get_runnable_tui_modes() -> tuple[str, ...]:
    """Return registry-backed modes that can launch model servers."""
    return create_default_profile_registry().run_group_ids


# Modes that can be run via "llm-runner tui".
RUNNABLE_TUI_MODES = get_runnable_tui_modes()
VALID_MODES = (*RUNNABLE_TUI_MODES, *COMMAND_MODES)

BUILD_BACKENDS = ("sycl", "cuda", "both")
SMOKE_MODE = "smoke"

# NOTE: --strict-profiles is documented as post-MVP deferral (FR-M3-009).
# In MVP, stale profiles produce a warning only — they never block model
# launch.  When --strict-profiles is eventually implemented it would treat
# stale profiles as non-existent, requiring explicit user re-profiling before
# profile guidance is applied.


def parse_jobs_arg(arg: str) -> int:
    """Parse a jobs argument in various forms (-jN, --jobs=N).

    Args:
        arg: The current argument being parsed.

    Returns:
        Parsed jobs integer value.

    Raises:
        SystemExit: On invalid jobs value.
    """
    try:
        if "=" in arg:
            return int(arg.split("=")[1])
        elif arg.startswith("-j"):
            return int(arg[2:])
        else:
            print(f"error: invalid jobs value '{arg}'", file=sys.stderr)
            sys.exit(1)
    except (ValueError, IndexError):
        print(f"error: invalid jobs value '{arg}'", file=sys.stderr)
        sys.exit(1)


def _parse_dry_run_args(args: list[str]) -> argparse.Namespace:
    """Parse arguments for dry-run mode.

    Args:
        args: List of arguments starting from index 2 (after 'dry-run' and mode).

    Returns:
        Parsed arguments namespace.

    Raises:
        SystemExit: On invalid arguments.
    """
    dry_run_mode = args[1]
    if dry_run_mode not in VALID_MODES and dry_run_mode != SMOKE_MODE:
        print(
            f"error: invalid dry-run mode '{dry_run_mode}'. Valid modes: {', '.join(VALID_MODES)}, {SMOKE_MODE}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Remaining args after mode are ports (excluding --acknowledge-risky flag)
    ports: list[int] = []
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


def _parse_normal_mode_args(args: list[str]) -> argparse.Namespace:
    """Parse arguments for normal modes.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
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
      doctor           Run doctor diagnostics (use subcommands: check, repair)
      build            Run build pipeline
      setup            Run setup commands (use subcommands: check, venv, clean-venv)
      tui              Launch interactive TUI

    Exit codes:
      0    Success
      1-9  Doctor check failures
      10-19 Smoke test failures
      130  Interrupted (Ctrl+C)

    Examples:
      %(prog)s summary-balanced
      %(prog)s summary-fast 8082
      %(prog)s qwen35 8080
      %(prog)s both 8080 8081
      %(prog)s dry-run summary-balanced
      %(prog)s dry-run both 8080 8081
      %(prog)s doctor check
      %(prog)s doctor repair
      %(prog)s build sycl
      %(prog)s setup check
      %(prog)s tui both
      %(prog)s tui summary-balanced --port 8080
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
    parsed.dry_run_mode = None
    return parsed


def _handle_build_case(args: list[str]) -> argparse.Namespace | None:
    """Handle build command special case.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed namespace if build command, None otherwise.

    Raises:
        SystemExit: On missing or invalid build arguments.
    """
    if len(args) >= 1 and args[0] == "build":
        if len(args) < 2:
            print(
                "error: build requires a backend argument (sycl|cuda|both)",
                file=sys.stderr,
            )
            sys.exit(1)

        backend = args[1]
        if backend not in BUILD_BACKENDS:
            print(
                f"error: invalid backend '{backend}'. Valid backends: {', '.join(BUILD_BACKENDS)}",
                file=sys.stderr,
            )
            sys.exit(1)

        return argparse.Namespace(
            mode="build",
            backend=backend,
            build_args=args[1:],
            dry_run="--dry-run" in args,
        )
    return None


def _parse_setup_check_args(args: list[str]) -> argparse.Namespace:
    """Parse setup check subcommand arguments."""
    backend = "all"
    json_output = False
    jobs = None
    for arg in args:
        if arg in ("sycl", "cuda"):
            backend = arg
        elif arg == "--json":
            json_output = True
        elif arg.startswith("-j") or arg.startswith("--jobs"):
            jobs = parse_jobs_arg(arg)
    return argparse.Namespace(
        mode="setup",
        setup_command="check",
        backend=backend,
        json=json_output,
        jobs=jobs,
    )


def _parse_setup_venv_args(args: list[str]) -> argparse.Namespace:
    """Parse setup venv subcommand arguments."""
    check_integrity = False
    json_output = False
    jobs = None
    for arg in args:
        if arg == "--check-integrity":
            check_integrity = True
        elif arg == "--json":
            json_output = True
        elif arg.startswith("-j") or arg.startswith("--jobs"):
            jobs = parse_jobs_arg(arg)
    return argparse.Namespace(
        mode="setup",
        setup_command="venv",
        check_integrity=check_integrity,
        json=json_output,
        jobs=jobs,
    )


def _parse_setup_clean_venv_args(args: list[str]) -> argparse.Namespace:
    """Parse setup clean-venv subcommand arguments."""
    yes = False
    jobs = None
    for arg in args:
        if arg == "--yes":
            yes = True
        elif arg.startswith("-j") or arg.startswith("--jobs"):
            jobs = parse_jobs_arg(arg)
    return argparse.Namespace(
        mode="setup",
        setup_command="clean-venv",
        yes=yes,
        jobs=jobs,
    )


def _handle_setup_case(args: list[str]) -> argparse.Namespace | None:
    """Handle setup command special case.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed namespace if setup command, None otherwise.

    Raises:
        SystemExit: On missing or invalid setup arguments.
    """
    if not (len(args) >= 1 and args[0] == "setup"):
        return None

    if len(args) < 2:
        print(
            "error: setup requires a subcommand (check|venv|clean-venv)",
            file=sys.stderr,
        )
        sys.exit(1)

    subcommand = args[1]
    remaining_args = args[2:]

    subcommand_parsers = {
        "check": _parse_setup_check_args,
        "venv": _parse_setup_venv_args,
        "clean-venv": _parse_setup_clean_venv_args,
    }

    if subcommand in subcommand_parsers:
        return subcommand_parsers[subcommand](remaining_args)

    print(
        f"error: unknown setup subcommand '{subcommand}'. "
        f"Valid subcommands: check, venv, clean-venv",
        file=sys.stderr,
    )
    sys.exit(1)


def _parse_doctor_check_args(args: list[str]) -> argparse.Namespace:
    """Parse doctor check subcommand arguments."""
    backend = "all"
    json_output = False
    jobs = None
    for arg in args:
        if arg in ("sycl", "cuda"):
            backend = arg
        elif arg == "--json":
            json_output = True
        elif arg.startswith("-j") or arg.startswith("--jobs"):
            jobs = parse_jobs_arg(arg)
    return argparse.Namespace(
        mode="doctor",
        doctor_command="check",
        backend=backend,
        json=json_output,
        jobs=jobs,
    )


def _parse_doctor_repair_args(args: list[str]) -> argparse.Namespace:
    """Parse doctor repair subcommand arguments."""
    dry_run = False
    json_output = False
    yes = False
    jobs = None
    for arg in args:
        if arg == "--dry-run":
            dry_run = True
        elif arg == "--json":
            json_output = True
        elif arg == "--yes":
            yes = True
        elif arg.startswith("-j") or arg.startswith("--jobs"):
            jobs = parse_jobs_arg(arg)
    return argparse.Namespace(
        mode="doctor",
        doctor_command="repair",
        dry_run=dry_run,
        json=json_output,
        yes=yes,
        jobs=jobs,
    )


def _handle_doctor_case(args: list[str]) -> argparse.Namespace | None:
    """Handle doctor command special case.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed namespace if doctor command, None otherwise.

    Raises:
        SystemExit: On missing or invalid doctor arguments.
    """
    if not (len(args) >= 1 and args[0] == "doctor"):
        return None

    if len(args) < 2:
        print(
            "error: doctor requires a subcommand (check|repair)",
            file=sys.stderr,
        )
        sys.exit(1)

    subcommand = args[1]
    remaining_args = args[2:]

    subcommand_parsers = {
        "check": _parse_doctor_check_args,
        "repair": _parse_doctor_repair_args,
        "fix": _parse_doctor_repair_args,
    }

    if subcommand in subcommand_parsers:
        return subcommand_parsers[subcommand](remaining_args)

    print(
        f"error: unknown doctor subcommand '{subcommand}'. Valid subcommands: check, repair, fix",
        file=sys.stderr,
    )
    sys.exit(1)


def _handle_profile_case(args: list[str]) -> argparse.Namespace | None:
    """Handle profile subcommand — detect and forward raw args to profile_cli.main().

    Args:
        args: List of command-line arguments.

    Returns:
        Minimal namespace if profile subcommand, None otherwise.
    """
    if not (len(args) >= 1 and args[0] == "profile"):
        return None

    return argparse.Namespace(mode="profile", sub_argv=args[1:])


def _handle_dry_run_case(args: list[str]) -> argparse.Namespace | None:
    """Handle dry-run mode special case.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed namespace if dry-run mode, None otherwise.

    Raises:
        SystemExit: On missing or invalid dry-run arguments.
    """
    if len(args) >= 1 and args[0] == "dry-run":
        if len(args) < 2:
            print(
                "error: dry-run requires a mode argument (summary-balanced|summary-fast|qwen35|both)",
                file=sys.stderr,
            )
            sys.exit(1)
        return _parse_dry_run_args(args)
    return None


def _parse_tui_port(args: list[str], i: int, flag: str) -> tuple[int | None, int, bool]:
    """Parse a single port argument (--port or --port2).

    Returns:
        Tuple of (port_value_or_None, new_index, had_error).
    """
    i += 1
    if i < len(args):
        try:
            return int(args[i]), i + 1, False
        except ValueError:
            print(f"error: invalid {flag} value '{args[i]}'", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"error: {flag} requires a value", file=sys.stderr)
        sys.exit(1)


def _parse_tui_args(args: list[str], start: int) -> tuple[int | None, int | None, bool]:
    """Parse TUI optional arguments starting from index *start*.

    Returns:
        Tuple of (port, port2, acknowledge_risky).
    """
    port: int | None = None
    port2: int | None = None
    acknowledge_risky = False
    i = start

    while i < len(args):
        arg = args[i]
        if arg in ("--port", "-p"):
            port, i, _ = _parse_tui_port(args, i, arg)
        elif arg in ("--port2", "-P"):
            port2, i, _ = _parse_tui_port(args, i, arg)
        elif arg == "--acknowledge-risky":
            acknowledge_risky = True
            i += 1
        elif arg.startswith("-"):
            print(f"error: unknown tui flag '{arg}'", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"error: unexpected tui argument '{arg}'", file=sys.stderr)
            sys.exit(1)

    return port, port2, acknowledge_risky


def _handle_tui_case(args: list[str]) -> argparse.Namespace | None:
    """Handle tui subcommand special case.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed namespace if tui subcommand, None otherwise.

    Raises:
        SystemExit: On missing or invalid tui arguments.
    """
    if not (len(args) >= 1 and args[0] == "tui"):
        return None

    # Mode is now optional - if not provided, TUI starts in standalone mode
    mode = args[1] if len(args) >= 2 and args[1] not in ("--port", "--port2", "--acknowledge-risky") else None

    if mode is not None and mode not in RUNNABLE_TUI_MODES:
        print(
            f"error: invalid tui mode '{mode}'. Valid modes: {', '.join(RUNNABLE_TUI_MODES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    port, port2, acknowledge_risky = _parse_tui_args(args, 2 if mode else 1)

    return argparse.Namespace(
        mode="tui",
        tui_mode=mode,
        port=port,
        port2=port2,
        acknowledge_risky=acknowledge_risky,
    )


def _handle_smoke_case(args: list[str]) -> argparse.Namespace | None:
    """Handle smoke subcommand special case.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed namespace if smoke subcommand, None otherwise.

    Raises:
        SystemExit: On missing or invalid smoke arguments.
    """
    if not (len(args) >= 1 and args[0] == "smoke"):
        return None

    if len(args) < 2:
        print(
            "error: smoke requires a mode argument (both|slot)",
            file=sys.stderr,
        )
        sys.exit(1)

    mode = args[1]
    if mode not in ("both", "slot"):
        print(
            f"error: invalid smoke mode '{mode}'. Valid modes: both, slot",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse remaining args for slot_id (when mode is "slot") and flags.
    # Strategy: slot_id is extracted ONLY from the first token if it exists
    # and does NOT start with "--". Otherwise leave all args for parser/error
    # handling. This prevents skipping leading flags to find slot_id.
    slot_id: str | None = None
    api_key: str = ""
    model_id: str | None = None
    max_tokens: int = 0
    prompt: str = ""
    delay: int = 0
    timeout: int = 0
    json_output: bool = False

    remaining = args[2:]

    # Check if slot_id can be extracted from remaining[0] BEFORE parsing flags
    if mode == "slot" and remaining and not remaining[0].startswith("--"):
        slot_id = remaining[0]
        # Remove slot_id from remaining for flag parsing
        remaining = remaining[1:]

    i = 0

    # Flags that take a value (consume the next token)
    _FLAGS_WITH_VALUE: set[str] = {
        "--api-key",
        "--model-id",
        "--max-tokens",
        "--prompt",
        "--delay",
        "--timeout",
    }

    # Phase 1 — parse all flags, collecting known values and skipping
    # unknown ones (with their values).  slot_id is extracted from the
    # first non-flag token.
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            if "=" in arg:
                # Handle --flag=value syntax
                key, _, value = arg.partition("=")
                if key == "--api-key":
                    api_key = value
                elif key == "--model-id":
                    model_id = value
                elif key == "--max-tokens":
                    try:
                        max_tokens = int(value)
                    except ValueError:
                        print(
                            f"error: invalid --max-tokens value '{value}': must be an integer",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                elif key == "--prompt":
                    prompt = value
                elif key == "--delay":
                    try:
                        delay = int(value)
                    except ValueError:
                        print(
                            f"error: invalid --delay value '{value}': must be an integer",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                elif key == "--timeout":
                    try:
                        timeout = int(value)
                    except ValueError:
                        print(
                            f"error: invalid --timeout value '{value}': must be an integer",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                else:
                    # Unknown flag=value — hard error
                    print(f"error: unknown flag '{key}'", file=sys.stderr)
                    sys.exit(1)
                i += 1
            elif arg in _FLAGS_WITH_VALUE:
                i += 1
                if i < len(remaining):
                    val = remaining[i]
                    if arg == "--api-key":
                        api_key = val
                    elif arg == "--model-id":
                        model_id = val
                    elif arg == "--max-tokens":
                        try:
                            max_tokens = int(val)
                        except ValueError:
                            print(
                                f"error: invalid --max-tokens value '{val}': must be an integer",
                                file=sys.stderr,
                            )
                            sys.exit(1)
                    elif arg == "--prompt":
                        prompt = val
                    elif arg == "--delay":
                        try:
                            delay = int(val)
                        except ValueError:
                            print(
                                f"error: invalid --delay value '{val}': must be an integer",
                                file=sys.stderr,
                            )
                            sys.exit(1)
                    elif arg == "--timeout":
                        try:
                            timeout = int(val)
                        except ValueError:
                            print(
                                f"error: invalid --timeout value '{val}': must be an integer",
                                file=sys.stderr,
                            )
                            sys.exit(1)
                    i += 1  # skip past the value
                else:
                    print(f"error: {arg} requires a value", file=sys.stderr)
                    sys.exit(1)
            elif arg == "--json":
                json_output = True
                i += 1
            else:
                # Unknown flag — hard error
                print(f"error: unknown flag '{arg}'", file=sys.stderr)
                sys.exit(1)
        else:
            # Unexpected positional argument after flags — reject it
            print(
                f"error: unexpected positional argument '{arg}' "
                f"(expected only flags after slot ID)",
                file=sys.stderr,
            )
            sys.exit(1)

    # Phase 2 — validate max_tokens range
    if max_tokens != 0 and not (8 <= max_tokens <= 32):
        print(
            f"error: --max-tokens must be between 8 and 32, got: {max_tokens}",
            file=sys.stderr,
        )
        sys.exit(1)

    return argparse.Namespace(
        mode="smoke",
        smoke_mode=mode,
        slot_id=slot_id,
        api_key=api_key,
        model_id=model_id,
        max_tokens=max_tokens,
        prompt=prompt,
        delay=delay,
        timeout=timeout,
        json=json_output,
    )


def _try_special_case_handlers(args: list[str]) -> argparse.Namespace | None:
    """Try all special case handlers in order.

    Args:
        args: List of command-line arguments.

    Returns:
        Parsed namespace if a special case matched, None otherwise.
    """
    handlers = [
        _handle_build_case,
        _handle_setup_case,
        _handle_doctor_case,
        _handle_profile_case,
        _handle_dry_run_case,
        _handle_smoke_case,
        _handle_tui_case,
    ]
    for handler in handlers:
        result = handler(args)
        if result is not None:
            return result
    return None


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
        args = sys.argv[1:]

    result = _try_special_case_handlers(args)
    if result is not None:
        return result
    return _parse_normal_mode_args(args)


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
        choices=[*VALID_MODES, SMOKE_MODE],
        help=f"Mode to run: {' | '.join(VALID_MODES)} | {SMOKE_MODE}",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port for primary model",
    )
    parser.add_argument(
        "--port2",
        "-P",
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
