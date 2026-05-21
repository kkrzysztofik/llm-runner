"""CLI argument parsing for llm-runner.

This module handles command-line argument parsing for both standard CLI
and TUI modes, including special handling for dry-run mode which requires
a second argument specifying the mode to preview.
"""

import argparse
import sys

from llama_cli.commands.smoke import _parse_smoke_args
from llama_cli.ui_output import emit_error
from llama_manager.config import create_default_profile_registry

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
            emit_error(f"invalid jobs value '{arg}'")
            sys.exit(1)
    except ValueError, IndexError:
        emit_error(f"invalid jobs value '{arg}'")
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
        emit_error(
            f"invalid dry-run mode '{dry_run_mode}'. Valid modes: {', '.join(VALID_MODES)}, {SMOKE_MODE}",
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
                emit_error(f"invalid port '{arg}'")
                sys.exit(1)

    return argparse.Namespace(
        mode="dry-run",
        dry_run_mode=dry_run_mode,
        ports=ports,
        acknowledge_risky=acknowledge_risky,
    )


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
            emit_error("build requires a backend argument (sycl|cuda|both)")
            sys.exit(1)

        backend = args[1]
        if backend not in BUILD_BACKENDS:
            emit_error(f"invalid backend '{backend}'. Valid backends: {', '.join(BUILD_BACKENDS)}")
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
        elif arg.startswith(("-j", "--jobs")):
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
        elif arg.startswith(("-j", "--jobs")):
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
        elif arg.startswith(("-j", "--jobs")):
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
        emit_error("setup requires a subcommand (check|venv|clean-venv)")
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

    emit_error(
        f"unknown setup subcommand '{subcommand}'. Valid subcommands: check, venv, clean-venv"
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
        elif arg.startswith(("-j", "--jobs")):
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
        elif arg.startswith(("-j", "--jobs")):
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
        emit_error("doctor requires a subcommand (check|repair)")
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

    emit_error(f"unknown doctor subcommand '{subcommand}'. Valid subcommands: check, repair, fix")
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
            emit_error(
                "dry-run requires a mode argument (summary-balanced|summary-fast|qwen35|both)"
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
            emit_error(f"invalid {flag} value '{args[i]}'")
            sys.exit(1)
    else:
        emit_error(f"{flag} requires a value")
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
            emit_error(f"unknown tui flag '{arg}'")
            sys.exit(1)
        else:
            emit_error(f"unexpected tui argument '{arg}'")
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
    mode = (
        args[1]
        if len(args) >= 2
        and args[1] not in ("--port", "--port2", "--acknowledge-risky", "-p", "-P")
        else None
    )

    if mode is not None and mode not in RUNNABLE_TUI_MODES:
        emit_error(f"invalid tui mode '{mode}'. Valid modes: {', '.join(RUNNABLE_TUI_MODES)}")
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

    Delegates to the canonical smoke parser in smoke.py for argument
    parsing, ensuring a single source of truth for smoke flags.

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
        emit_error("smoke requires a mode argument (both|slot)")
        sys.exit(1)

    # Delegate to the canonical smoke parser (args[1:] = after "smoke")
    parsed = _parse_smoke_args(args[1:])

    # Map smoke parser output to cli_parser namespace format
    return argparse.Namespace(
        mode="smoke",
        smoke_mode=parsed.mode,
        slot_id=parsed.slot_id,
        api_key=parsed.api_key,
        model_id=parsed.model_id,
        max_tokens=parsed.max_tokens,
        prompt=parsed.prompt,
        delay=parsed.delay,
        timeout=parsed.timeout,
        json=parsed.json,
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

    Handles special subcommands (tui, dry-run, build, setup, doctor, profile,
    smoke) via dedicated handlers. Unknown tokens cause a usage error.

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

    if not args:
        return argparse.Namespace(mode=None)

    emit_error(f"unknown command '{args[0]}'. Use 'tui' to launch model servers.")
    sys.exit(1)


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
