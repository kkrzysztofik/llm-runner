"""Server execution logic for CLI.

This module provides the main entry point for running llama-server instances
via command-line interface, including signal handling, process management,
and risk acknowledgment workflows.
"""

import argparse
import atexit
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

from llama_cli.cli_parser import parse_args
from llama_cli.colors import Colors
from llama_cli.setup_cli import main as setup_main
from llama_manager import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    ServerConfig,
    ServerManager,
    build_server_cmd,
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
    require_executable,
    require_model,
    validate_port,
    validate_ports,
    validate_server_config,
    validate_slots,
)
from llama_manager.server import detect_risky_operations

RISK_ACK_LABEL = "warning_bypass"
RISK_CONFIRM_PROMPT = "Confirm risky operation [y/N]: "

# Port labels for validation error messages
PORT_SUMMARY_BALANCED = "summary-balanced port"
PORT_SUMMARY_FAST = "summary-fast port"
PORT_QWEN35 = "qwen35 port"


def usage() -> None:
    print("""Usage:
  src/run_opencode_models.py summary-balanced [port]
  src/run_opencode_models.py summary-fast [port]
  src/run_opencode_models.py qwen35 [port]
  src/run_opencode_models.py both [summary_balanced_port qwen35_port]
  src/run_opencode_models.py dry-run <mode> [ports...]
  src/run_opencode_models.py setup <subcommand>

Modes:
  summary-balanced  Run summary-balanced model (Intel SYCL)
  summary-fast      Run summary-fast model (Intel SYCL)
  qwen35           Run qwen35-coding model (NVIDIA CUDA)
  both             Run summary-balanced and qwen35 side-by-side
  dry-run          Preview commands without executing
  setup            Toolchain diagnostics and venv management

Setup Subcommands:
  check           Check toolchain availability (FR-005.1)
  venv            Create or reuse virtual environment (FR-005.2)
  clean-venv      Remove virtual environment (FR-005.3)

Examples:
  src/run_opencode_models.py summary-balanced
  src/run_opencode_models.py summary-fast 8082
  src/run_opencode_models.py qwen35 8080
  src/run_opencode_models.py both 8080 8081
  src/run_opencode_models.py dry-run summary-balanced
  src/run_opencode_models.py dry-run both 8080 8081
  src/run_opencode_models.py setup --check
  src/run_opencode_models.py setup venv
  src/run_opencode_models.py setup clean-venv --yes""")


def _print_backend_error_and_exit() -> NoReturn:
    """Print backend error details and exit with code 1."""
    print("error: acknowledgement_required", file=sys.stderr)
    print("  failed_check: acknowledgement_required", file=sys.stderr)
    print("  why_blocked: risky operation detected and not acknowledged", file=sys.stderr)
    print("  how_to_fix: use --acknowledge-risky flag or confirm with 'y'", file=sys.stderr)
    raise SystemExit(1)


def check_prereqs() -> None:
    cfg = Config()
    require_executable(cfg.llama_server_bin_intel, "Intel llama-server")


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


def run_summary_balanced(port: int, manager: ServerManager) -> int:
    cfg = Config()
    port_error = validate_port(port, PORT_SUMMARY_BALANCED)
    if port_error is not None:
        _print_validation_error(port_error)
    model_error = require_model(cfg.model_summary_balanced)
    if model_error is not None:
        _print_validation_error(model_error)
    print(f"Starting summary-balanced at http://{cfg.host}:{port}/v1")
    server_cfg = create_summary_balanced_cfg(port)
    backend_error = validate_server_config(server_cfg)
    if backend_error is not None:
        _print_validation_error(backend_error)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("summary-balanced", cmd)


def run_summary_fast(port: int, manager: ServerManager) -> int:
    cfg = Config()
    port_error = validate_port(port, PORT_SUMMARY_FAST)
    if port_error is not None:
        _print_validation_error(port_error)
    model_error = require_model(cfg.model_summary_fast)
    if model_error is not None:
        _print_validation_error(model_error)
    print(f"Starting summary-fast at http://{cfg.host}:{port}/v1")
    server_cfg = create_summary_fast_cfg(port)
    backend_error = validate_server_config(server_cfg)
    if backend_error is not None:
        _print_validation_error(backend_error)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("summary-fast", cmd)


def run_qwen35(port: int, manager: ServerManager) -> int:
    cfg = Config()
    port_error = validate_port(port, PORT_QWEN35)
    if port_error is not None:
        _print_validation_error(port_error)
    model_error = require_model(cfg.model_qwen35)
    if model_error is not None:
        _print_validation_error(model_error)
    exec_error = require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")
    if exec_error is not None:
        _print_validation_error(exec_error)
    print(f"Starting qwen35-coding at http://{cfg.host}:{port}/v1 (NVIDIA CUDA)")
    server_cfg = create_qwen35_cfg(port)
    backend_error = validate_server_config(server_cfg)
    if backend_error is not None:
        _print_validation_error(backend_error)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("qwen35-coding", cmd)


def run_both(port32: int, port35: int, manager: ServerManager) -> int:
    cfg = Config()
    validate_port(port32, PORT_SUMMARY_BALANCED)
    validate_port(port35, PORT_QWEN35)
    validate_ports(port32, port35, PORT_SUMMARY_BALANCED, PORT_QWEN35)
    require_model(cfg.model_summary_balanced)
    require_model(cfg.model_qwen35_both)
    require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")

    slots: list[ModelSlot] = [
        ModelSlot(slot_id="summary-balanced", model_path=cfg.model_summary_balanced, port=port32),
        ModelSlot(slot_id="qwen35-coding", model_path=cfg.model_qwen35_both, port=port35),
    ]

    validation_error = validate_slots(slots)
    if validation_error is not None:
        for error_detail in validation_error.errors:
            print(f"error: {error_detail.error_code}", file=sys.stderr)
            print(f"  failed_check: {error_detail.failed_check}", file=sys.stderr)
            print(f"  why_blocked: {error_detail.why_blocked}", file=sys.stderr)
            print(f"  how_to_fix: {error_detail.how_to_fix}", file=sys.stderr)
        raise SystemExit(1)

    server_configs = [create_summary_balanced_cfg(port32), create_qwen35_cfg(port35)]
    print(f"Launching {len(server_configs)} server(s)...")
    for cfg_instance in server_configs:
        print(f"  {cfg_instance.alias}: http://{cfg.host}:{cfg_instance.port}/v1")

    log_handlers: dict[str, Callable[[str], None]] = {}
    for cfg_instance in server_configs:
        log_handlers[cfg_instance.alias] = lambda line, name=cfg_instance.alias: print(
            f"[{name}] {line}", flush=True
        )

    manager.start_servers(server_configs, log_handlers)
    code = manager.wait_for_any()
    manager.cleanup_servers()
    return code


def verify_risks(manager: ServerManager, configs: list[ServerConfig], acknowledged: bool) -> None:
    """Verify that risky operations are acknowledged before launch."""
    launch_attempt_id = manager.begin_launch_attempt()
    ack_token = manager.issue_ack_token(launch_attempt_id)

    for cfg in configs:
        if acknowledged and RISK_ACK_LABEL not in cfg.risky_acknowledged:
            cfg.risky_acknowledged.append(RISK_ACK_LABEL)
        for risk in detect_risky_operations(cfg):
            _acknowledge_risk_if_required(
                manager,
                cfg,
                risk,
                launch_attempt_id,
                ack_token,
                acknowledged,
            )


def _acknowledge_risk_if_required(
    manager: ServerManager,
    cfg: ServerConfig,
    risk: str,
    launch_attempt_id: str,
    ack_token: str,
    acknowledged: bool,
) -> None:
    if manager.is_risk_acknowledged(cfg.alias, risk, launch_attempt_id):
        return

    if not acknowledged:
        print(f"warning: risky operation detected in {cfg.alias}: {risk}")
        try:
            response = input(RISK_CONFIRM_PROMPT).strip().lower()
        except EOFError:
            _print_backend_error_and_exit()
        if response != "y":
            _print_backend_error_and_exit()

    manager.acknowledge_risk(
        cfg.alias,
        risk,
        launch_attempt_id=launch_attempt_id,
        ack_token=ack_token,
    )


def _run_dry_run_mode(parsed: argparse.Namespace, acknowledged: bool) -> int:
    from llama_cli.dry_run import dry_run

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


def _run_mode(parsed_mode: str, ports: list[int], manager: ServerManager, cfg: Config) -> int:
    if parsed_mode == "summary-balanced":
        port = _resolve_port(ports, 0, cfg.summary_balanced_port)
        return run_summary_balanced(port, manager)
    if parsed_mode == "summary-fast":
        port = _resolve_port(ports, 0, cfg.summary_fast_port)
        return run_summary_fast(port, manager)
    if parsed_mode == "qwen35":
        port = _resolve_port(ports, 0, cfg.qwen35_port)
        return run_qwen35(port, manager)
    if parsed_mode == "both":
        port32 = _resolve_port(ports, 0, cfg.summary_balanced_port)
        port35 = _resolve_port(ports, 1, cfg.qwen35_port)
        return run_both(port32, port35, manager)
    usage()
    return 1


def _normalize_main_args(args: list[str] | None) -> list[str]:
    if args is None:
        return sys.argv[1:]
    if not args:
        return []

    modes = {
        "summary-balanced",
        "summary-fast",
        "qwen35",
        "both",
        "dry-run",
        "doctor",
        "build",
        "setup",
    }
    if args[0] in modes or args[0].startswith("-"):
        return args
    return args[1:]


def _build_target_configs(parsed_mode: str, ports: list[int], cfg: Config) -> list[ServerConfig]:
    if parsed_mode == "summary-balanced":
        port = _resolve_port(ports, 0, cfg.summary_balanced_port)
        return [create_summary_balanced_cfg(port)]
    if parsed_mode == "summary-fast":
        port = _resolve_port(ports, 0, cfg.summary_fast_port)
        return [create_summary_fast_cfg(port)]
    if parsed_mode == "qwen35":
        port = _resolve_port(ports, 0, cfg.qwen35_port)
        return [create_qwen35_cfg(port)]
    if parsed_mode == "both":
        port32 = _resolve_port(ports, 0, cfg.summary_balanced_port)
        port35 = _resolve_port(ports, 1, cfg.qwen35_port)
        return [create_summary_balanced_cfg(port32), create_qwen35_cfg(port35)]
    return []


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point."""
    argv = _normalize_main_args(args)
    parsed = parse_args(argv)

    if not parsed.mode:
        usage()
        return 1

    # Handle build command
    if parsed.mode == "build":
        return run_build(parsed.backend, parsed.dry_run)

    # Handle setup command
    if parsed.mode == "setup":
        return setup_main()

    # Handle doctor command (FR-004.7)
    if parsed.mode == "doctor":
        from llama_cli.doctor_cli import main as doctor_main

        return doctor_main(sys.argv[2:])

    manager = ServerManager()
    signal.signal(signal.SIGINT, manager.on_interrupt)
    signal.signal(signal.SIGTERM, manager.on_terminate)
    atexit.register(manager.cleanup_servers)

    Colors.is_enabled()
    check_prereqs()
    os.environ["ZES_ENABLE_SYSMAN"] = "1"

    if parsed.mode == "dry-run":
        return _run_dry_run_mode(parsed, parsed.acknowledge_risky)

    cfg = Config()
    verify_risks(
        manager,
        _build_target_configs(parsed.mode, parsed.ports, cfg),
        parsed.acknowledge_risky,
    )

    try:
        return _run_mode(parsed.mode, parsed.ports, manager, cfg)
    except ValueError as e:
        print(f"error: invalid arguments: {e}", file=sys.stderr)
        return 1
    except IndexError as e:
        print(f"error: index error: {e}", file=sys.stderr)
        return 1


def run_build(backend: str, dry_run: bool = False) -> int:
    """Run the build command.

    Args:
        backend: Build backend ("sycl" or "cuda")
        dry_run: If True, print commands without executing

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from llama_manager.build_pipeline import BuildBackend, BuildConfig, BuildPipeline

    config = Config()

    # Determine paths based on backend
    source_dir = Path(config.llama_cpp_root)
    build_dir = source_dir / ("build_cuda" if backend == "cuda" else "build")
    output_dir = config.builds_dir

    # Create build config
    build_backend = BuildBackend.SYCL if backend == "sycl" else BuildBackend.CUDA
    build_config = BuildConfig(
        backend=build_backend,
        source_dir=source_dir,
        build_dir=build_dir,
        output_dir=output_dir,
        git_remote_url=config.build_git_remote,
        git_branch=config.build_git_branch,
        shallow_clone=True,
        retry_attempts=config.build_retry_attempts,
        retry_delay=config.build_retry_delay,
    )

    # Create and run pipeline directly
    pipeline = BuildPipeline(build_config)
    pipeline.dry_run = dry_run

    print(f"Building for {backend} backend...", file=sys.stderr)
    if dry_run:
        print("DRY RUN MODE - commands will not be executed", file=sys.stderr)

    result = pipeline.run()

    if result.success:
        print("Build completed successfully!", file=sys.stderr)
        if result.artifact:
            print(f"Artifact: {result.artifact.binary_path}", file=sys.stderr)
        return 0
    else:
        print(f"Build failed: {result.error_message}", file=sys.stderr)
        return 1


def cli_main() -> None:
    """Entry point for the `llm-runner` console script."""
    raise SystemExit(main(sys.argv[1:]))
