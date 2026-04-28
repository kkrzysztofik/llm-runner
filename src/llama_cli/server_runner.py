"""Server execution logic for CLI.

This module provides the main entry point for running llama-server instances.
Models are launched exclusively via the TUI subcommand.
"""

import argparse
import os
import sys
from typing import NoReturn

from llama_cli.cli_parser import parse_args
from llama_cli.commands.setup import main as setup_main
from llama_manager import (
    Config,
    ErrorCode,
    ErrorDetail,
    ServerConfig,
    require_executable,
    require_model,
    validate_port,
    validate_ports,
)
from llama_manager.config import create_default_profile_registry, resolve_run_group_configs

# Server backend display names
INTEL_SERVER_NAME = "Intel llama-server"
NVIDIA_SERVER_NAME = "NVIDIA llama-server"


def usage() -> None:
    print("""Usage:
  llm-runner tui <mode> [--port PORT] [--port2 PORT2]
  llm-runner dry-run <mode> [ports...]
  llm-runner smoke <mode> [slot_id] [--json]
  llm-runner profile <slot_id> <flavor> [--json]
  llm-runner setup <subcommand>
  llm-runner build <backend>
  llm-runner doctor <subcommand>

Modes (via tui):
  summary-balanced  Run summary-balanced model (Intel SYCL)
  summary-fast      Run summary-fast model (Intel SYCL)
  qwen35           Run qwen35-coding model (NVIDIA CUDA)
  both             Run summary-balanced and qwen35 side-by-side

TUI Options:
  --port          Port for primary model
  --port2         Port for secondary model
  --acknowledge-risky  Acknowledge risky operations

Smoke Subcommands:
  both             Probe all servers
  slot <id>        Probe a specific slot

Smoke Options:
  --api-key        API key for authentication
  --model-id       Model ID override
  --max-tokens     Max tokens for chat probe (8-32)
  --prompt         Custom prompt for chat probe
  --delay          Inter-slot delay in seconds
  --timeout        Overall timeout in seconds
  --json           Output in JSON format

Setup Subcommands:
  check           Check toolchain availability
  venv            Create or reuse virtual environment
  clean-venv      Remove virtual environment

Doctor Subcommands:
  check           Check system health
  repair          Repair detected issues

Examples:
  llm-runner tui summary-balanced
  llm-runner tui both --port 8080 --port2 8081
  llm-runner dry-run summary-balanced
  llm-runner dry-run both 8080 8081
  llm-runner smoke both
  llm-runner profile slot0 balanced --json
  llm-runner setup check
  llm-runner build sycl""")


def _print_validation_error(error_detail: ErrorDetail) -> NoReturn:
    """Print validation error and exit."""
    error_code = (
        error_detail.error_code.value
        if isinstance(error_detail.error_code, ErrorCode)
        else str(error_detail.error_code)
    )
    print(f"error: {error_code}", file=sys.stderr)
    print(f"  failed_check: {error_detail.failed_check}", file=sys.stderr)
    print(f"  why_blocked: {error_detail.why_blocked}", file=sys.stderr)
    print(f"  how_to_fix: {error_detail.how_to_fix}", file=sys.stderr)
    raise SystemExit(1)


def _run_dry_run_mode(parsed: argparse.Namespace, acknowledged: bool) -> int:
    from llama_cli.commands.dry_run import dry_run

    # parsed.mode should be "dry-run"
    # parsed.dry_run_mode is the actual mode to preview
    target_mode = getattr(parsed, "dry_run_mode", None)
    if target_mode is None:
        print(
            "error: dry-run requires a mode argument (summary-balanced|summary-fast|qwen35|both)",
            file=sys.stderr,
        )
        usage()
        return 1

    primary_port = str(parsed.ports[0]) if len(parsed.ports) > 0 else ""
    secondary_port = str(parsed.ports[1]) if len(parsed.ports) > 1 else ""
    dry_run(target_mode, primary_port, secondary_port, acknowledged=acknowledged)
    return 0


def _resolve_port(ports: list[int], index: int, default: int) -> int:
    """Resolve port from ports list with default fallback.

    Args:
        ports: List of port numbers provided by user.
        index: Index of port to retrieve (0 for primary, 1 for secondary).
        default: Default port to use if index is out of range.

    Returns:
        The port number at the specified index, or the default if unavailable.
    """
    return ports[index] if len(ports) > index else default


def _build_tui_mode_configs(cfg: Config, parsed: argparse.Namespace) -> dict:
    """Build the mode configuration dict for TUI launch."""
    registry = create_default_profile_registry(cfg)
    mode_configs = {}
    for group in registry.run_groups:
        if not group.tui_enabled:
            continue
        configs = _resolve_tui_group_configs(group.group_id, cfg, parsed)
        mode_configs[group.group_id] = (
            [server_cfg.server_bin for server_cfg in configs],
            [_server_name_for_config(server_cfg) for server_cfg in configs],
            configs,
            [_gpu_index_for_config(server_cfg) for server_cfg in configs],
        )
    return mode_configs


def _resolve_tui_group_configs(
    group_id: str,
    cfg: Config,
    parsed: argparse.Namespace,
) -> list[ServerConfig]:
    """Resolve TUI configs while preserving positional port override semantics."""
    registry = create_default_profile_registry(cfg)
    default_configs = resolve_run_group_configs(registry, group_id)
    port_overrides: list[int] = []

    if parsed.port is not None:
        port_overrides.append(parsed.port)
    elif parsed.port2 is not None and default_configs:
        port_overrides.append(default_configs[0].port)

    if parsed.port2 is not None and len(default_configs) > 1:
        port_overrides.append(parsed.port2)

    if not port_overrides:
        return default_configs
    return resolve_run_group_configs(registry, group_id, tuple(port_overrides))


def _server_name_for_config(server_cfg: ServerConfig) -> str:
    """Return a human-readable executable label for a resolved server config."""
    if server_cfg.device.startswith("SYCL"):
        return INTEL_SERVER_NAME
    return NVIDIA_SERVER_NAME


def _gpu_index_for_config(server_cfg: ServerConfig) -> int:
    """Return the TUI GPU index for a resolved server config."""
    if server_cfg.device.startswith("SYCL"):
        return 1
    return 0


def _validate_tui_configs(configs: list[ServerConfig]) -> None:
    """Validate ports and models for all TUI server configs."""
    for server_cfg in configs:
        port_error = validate_port(server_cfg.port, server_cfg.alias)
        if port_error is not None:
            _print_validation_error(port_error)
        model_error = require_model(server_cfg.model)
        if model_error is not None:
            _print_validation_error(model_error)

    if len(configs) > 1:
        ports_error = validate_ports(
            configs[0].port,
            configs[1].port,
            configs[0].alias + " port",
            configs[1].alias + " port",
        )
        if ports_error is not None:
            _print_validation_error(ports_error)


def _run_tui(parsed: argparse.Namespace) -> int:
    """Run the TUI application.

    Args:
        parsed: Parsed arguments with tui_mode, port, port2, acknowledge_risky.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from llama_cli.tui import TUIApp

    cfg = Config()

    # Standalone mode (no mode specified) - start with empty slots
    if parsed.tui_mode is None:
        configs = []
        gpu_indices = []
    else:
        mode_configs = _build_tui_mode_configs(cfg, parsed)
        bins, names, configs, gpu_indices = mode_configs.get(parsed.tui_mode, ([], [], [], []))
        for bin_path, name in zip(bins, names, strict=True):
            exec_error = require_executable(bin_path, name)
            if exec_error is not None:
                _print_validation_error(exec_error)

        _validate_tui_configs(configs)

    app = TUIApp(configs, gpu_indices)
    app.run(acknowledged=parsed.acknowledge_risky)
    return 0


def _normalize_main_args(args: list[str] | None) -> list[str]:
    if args is None:
        return sys.argv[1:]
    if not args:
        return []

    modes = {
        "dry-run",
        "doctor",
        "build",
        "setup",
        "profile",
        "smoke",
        "tui",
    }
    if args[0] in modes or args[0].startswith("-"):
        return args
    return args[1:]


def _build_target_configs(parsed_mode: str, ports: list[int], cfg: Config) -> list[ServerConfig]:
    registry = create_default_profile_registry(cfg)
    if parsed_mode not in registry.run_group_ids:
        return []
    return resolve_run_group_configs(registry, parsed_mode, tuple(ports))


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point."""
    argv = _normalize_main_args(args)
    parsed = parse_args(argv)

    if not parsed.mode:
        usage()
        return 1

    # Handle build command
    if parsed.mode == "build":
        from llama_cli.commands.build import main as build_main

        return build_main(parsed.build_args)

    # Handle setup command
    if parsed.mode == "setup":
        return setup_main(argv[1:])

    # Handle doctor command (FR-004.7)
    if parsed.mode == "doctor":
        from llama_cli.commands.doctor import main as doctor_main

        return doctor_main(argv[1:])

    # Handle profile subcommand
    if parsed.mode == "profile":
        from llama_cli.commands.profile import main as profile_main

        return profile_main(argv[1:])

    # Handle smoke subcommand
    if parsed.mode == "smoke":
        from llama_cli.commands.smoke import run_smoke

        smoke_args = [
            parsed.smoke_mode,
        ]
        if parsed.slot_id:
            smoke_args.append(parsed.slot_id)
        if parsed.api_key:
            smoke_args.extend(["--api-key", parsed.api_key])
        if parsed.model_id:
            smoke_args.extend(["--model-id", parsed.model_id])
        if parsed.max_tokens:
            smoke_args.extend(["--max-tokens", str(parsed.max_tokens)])
        if parsed.prompt:
            smoke_args.extend(["--prompt", parsed.prompt])
        if parsed.delay:
            smoke_args.extend(["--delay", str(parsed.delay)])
        if parsed.timeout:
            smoke_args.extend(["--timeout", str(parsed.timeout)])
        if parsed.json:
            smoke_args.append("--json")

        return run_smoke(smoke_args)

    # Enable Intel Sysman for telemetry before any backend operations
    os.environ["ZES_ENABLE_SYSMAN"] = "1"

    # Handle tui subcommand
    if parsed.mode == "tui":
        return _run_tui(parsed)

    if parsed.mode == "dry-run":
        return _run_dry_run_mode(parsed, parsed.acknowledge_risky)

    usage()
    return 1


def cli_main() -> None:
    """Entry point for the `llm-runner` console script."""
    raise SystemExit(main(sys.argv[1:]))
