# Server execution logic for CLI


import atexit
import os
import signal
import sys
import time

from llama_manager import (
    Color,
    Config,
    LaunchResult,
    ModelSlot,
    ServerConfig,
    ServerManager,
    build_server_cmd,
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
    require_executable,
    require_model,
    resolve_runtime_dir,
    validate_port,
    validate_ports,
    validate_server_config,
    validate_slots,
    write_artifact,
)


def usage() -> None:
    print("""Usage:
  src/run_opencode_models.py summary-balanced [port]
  src/run_opencode_models.py summary-fast [port]
  src/run_opencode_models.py qwen35 [port]
  src/run_opencode_models.py both [summary_balanced_port qwen35_port]
  src/run_opencode_models.py dry-run summary-balanced|summary-fast|qwen35|both [ports...]

Examples:
  src/run_opencode_models.py summary-balanced
  src/run_opencode_models.py summary-fast 8082
  src/run_opencode_models.py qwen35 8080
  src/run_opencode_models.py both 8080 8081
  src/run_opencode_models.py dry-run both""")


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
    # FR-011: Validate backend eligibility
    backend_error = validate_server_config(server_cfg)
    if backend_error is not None:
        print(f"error: {backend_error.error_code}", file=sys.stderr)
        print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
        print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
        print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
        sys.exit(1)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("summary-balanced", cmd)


def run_summary_fast(port: int, manager: ServerManager) -> int:
    """Run summary-fast server"""
    cfg = Config()
    validate_port(port, "summary-fast port")
    require_model(cfg.model_summary_fast)
    print(f"Starting summary-fast at http://{cfg.host}:{port}/v1")
    server_cfg = create_summary_fast_cfg(port)
    # FR-011: Validate backend eligibility
    backend_error = validate_server_config(server_cfg)
    if backend_error is not None:
        print(f"error: {backend_error.error_code}", file=sys.stderr)
        print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
        print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
        print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
        sys.exit(1)
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
    # FR-011: Validate backend eligibility
    backend_error = validate_server_config(server_cfg)
    if backend_error is not None:
        print(f"error: {backend_error.error_code}", file=sys.stderr)
        print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
        print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
        print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
        sys.exit(1)
    cmd = build_server_cmd(server_cfg)
    return manager.run_server_foreground("qwen35-coding", cmd)


def launch_slots_with_status(
    manager: ServerManager,
    slots: list[ModelSlot],
) -> LaunchResult:
    """Launch model slots and handle LaunchResult status for T020.

    Args:
        manager: ServerManager instance
        slots: List of ModelSlot configurations to launch

    Returns:
        LaunchResult with status ('success', 'degraded', or 'blocked')

    Raises:
        SystemExit: If status is 'blocked', prints FR-005 details and exits non-zero
    """
    # Validate slots first
    validation_error = validate_slots(slots)
    if validation_error is not None:
        # Validation failed - print FR-005 details and exit
        for error_detail in validation_error.errors:
            print(f"error: {error_detail.error_code}", file=sys.stderr)
            print(f"  failed_check: {error_detail.failed_check}", file=sys.stderr)
            print(f"  why_blocked: {error_detail.why_blocked}", file=sys.stderr)
            print(f"  how_to_fix: {error_detail.how_to_fix}", file=sys.stderr)
        sys.exit(1)

    # FR-007: Write artifact for slot resolution attempt (before launch)
    runtime_dir = resolve_runtime_dir()
    slot_ids = [slot.slot_id for slot in slots]
    try:
        # Build artifact data for slot resolution
        resolution_artifact = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "slot_scope": ",".join(slot_ids),
            "resolved_command": [
                build_server_cmd(
                    ServerConfig(
                        model=slot.model_path,
                        alias=slot.slot_id,
                        port=slot.port,
                        device="auto",
                        ctx_size=8192,
                        ubatch_size=512,
                        threads=4,
                    )
                )
                for slot in slots
            ],
            "validation_results": [{"passed": True, "checks": []}],
            "warnings": [],
            "environment_redacted": {},
        }
        write_artifact(runtime_dir, f"slot-resolution-{','.join(slot_ids)}", resolution_artifact)
    except Exception as e:
        print(
            f"error: artifact persistence failed for slot resolution: {e}",
            file=sys.stderr,
        )
        print("  failed_check: artifact_persistence", file=sys.stderr)
        print(
            "  why_blocked: artifact persistence failed to enforce required permissions",
            file=sys.stderr,
        )
        print(
            "  how_to_fix: verify runtime path and permission support before retry",
            file=sys.stderr,
        )
        sys.exit(1)

    # Launch all slots and get LaunchResult
    result = manager.launch_all_slots(slots)

    # Handle status according to T020 requirements
    if result.is_blocked():
        # Blocked: print FR-005 details to stderr and return non-zero
        print("error: launch blocked - no slots could be launched", file=sys.stderr)
        if result.errors is not None:
            for error_detail in result.errors.errors:
                print(f"  {error_detail.error_code}", file=sys.stderr)
                print(f"    failed_check: {error_detail.failed_check}", file=sys.stderr)
                print(f"    why_blocked: {error_detail.why_blocked}", file=sys.stderr)
                print(f"    how_to_fix: {error_detail.how_to_fix}", file=sys.stderr)
        sys.exit(1)

    elif result.is_degraded():
        # Degraded: print warnings but continue launching available slots
        print(
            "warning: launch degraded - some slots blocked, proceeding with available slots",
            file=sys.stderr,
        )
        for warning in result.warnings or []:
            print(f"  warning: {warning}", file=sys.stderr)

    # FR-007: Write artifact for launch attempt (success/degraded/blocked)
    try:
        # Build artifact data for launch attempt
        # Note: blocked slots are inferred from launched slots vs requested slots
        requested_slots = [slot.slot_id for slot in slots]
        launched_slots = list(result.launched or [])
        blocked_slots = [s for s in requested_slots if s not in launched_slots]

        launch_artifact = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "slot_scope": ",".join(requested_slots),
            "resolved_command": [
                build_server_cmd(
                    ServerConfig(
                        model=slot.model_path,
                        alias=slot.slot_id,
                        port=slot.port,
                        device="auto",
                        ctx_size=8192,
                        ubatch_size=512,
                        threads=4,
                    )
                )
                for slot in slots
            ],
            "validation_results": [
                {
                    "passed": result.is_success() or result.is_degraded(),
                    "checks": [],
                }
            ],
            "warnings": result.warnings or [],
            "environment_redacted": {},
            "launch_status": result.status,
            "launched_slots": launched_slots,
            "blocked_slots": blocked_slots,
        }
        write_artifact(runtime_dir, f"launch-{','.join(requested_slots)}", launch_artifact)
    except Exception as e:
        print(
            f"error: artifact persistence failed for launch attempt: {e}",
            file=sys.stderr,
        )
        print("  failed_check: artifact_persistence", file=sys.stderr)
        print(
            "  why_blocked: artifact persistence failed to enforce required permissions",
            file=sys.stderr,
        )
        print(
            "  how_to_fix: verify runtime path and permission support before retry",
            file=sys.stderr,
        )
        # Don't exit on artifact failure for launch - only for dry-run
        # Log the error but continue

    # Success or degraded with available slots - proceed normally
    return result


def run_both(port32: int, port35: int, manager: ServerManager) -> int:
    """Run both summary-balanced and qwen35 servers with slot-based launch for T020"""
    cfg = Config()
    validate_port(port32, "summary-balanced port")
    validate_port(port35, "qwen35 port")
    validate_ports(port32, port35, "summary-balanced port", "qwen35 port")
    require_model(cfg.model_summary_balanced)
    require_model(cfg.model_qwen35_both)
    require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")

    # Build ModelSlot configurations for T020 slot-based launch
    slots: list[ModelSlot] = [
        ModelSlot(
            slot_id="summary-balanced",
            model_path=cfg.model_summary_balanced,
            port=port32,
        ),
        ModelSlot(
            slot_id="qwen35-coding",
            model_path=cfg.model_qwen35_both,
            port=port35,
        ),
    ]

    # Launch slots with status handling (T020)
    result = launch_slots_with_status(manager, slots)

    # If degraded, still proceed with launched slots
    # Build ServerConfig for successfully launched slots
    launched_slot_ids = set(result.launched or [])

    # Create ServerConfig for each slot based on slot_id
    server_configs: list[ServerConfig] = []
    for slot in slots:
        if slot.slot_id in launched_slot_ids:
            if slot.slot_id == "summary-balanced":
                server_cfg = create_summary_balanced_cfg(
                    slot.port,
                    ctx_size=cfg.default_ctx_size_both_summary,
                    ubatch_size=cfg.default_ubatch_size_summary_balanced,
                    threads=cfg.default_threads_summary_balanced,
                    cache_k=cfg.default_cache_type_summary_k,
                    cache_v=cfg.default_cache_type_summary_v,
                )
                # FR-011: Validate backend eligibility
                backend_error = validate_server_config(server_cfg)
                if backend_error is not None:
                    print(f"error: {backend_error.error_code}", file=sys.stderr)
                    print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
                    print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
                    print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
                    sys.exit(1)
            elif slot.slot_id == "qwen35-coding":
                server_cfg = create_qwen35_cfg(
                    slot.port,
                    ctx_size=cfg.default_ctx_size_both_qwen35,
                    ubatch_size=cfg.default_ubatch_size_qwen35_both,
                    threads=cfg.default_threads_qwen35_both,
                    cache_k=cfg.default_cache_type_qwen35_both_k,
                    cache_v=cfg.default_cache_type_qwen35_both_v,
                    n_gpu_layers=cfg.default_n_gpu_layers_qwen35_both,
                    model=cfg.model_qwen35_both,
                    server_bin=cfg.llama_server_bin_nvidia,
                )
                # FR-011: Validate backend eligibility
                backend_error = validate_server_config(server_cfg)
                if backend_error is not None:
                    print(f"error: {backend_error.error_code}", file=sys.stderr)
                    print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
                    print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
                    print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
                    sys.exit(1)
            else:
                continue
            server_configs.append(server_cfg)

    # Start servers for successfully launched slots
    print(f"Launching {len(server_configs)} server(s)...")
    for cfg_instance in server_configs:
        print(f"  {cfg_instance.alias}: http://{cfg.host}:{cfg_instance.port}/v1")

    # Use ServerManager's start_servers with log handlers
    from collections.abc import Callable

    log_handlers: dict[str, Callable[[str], None]] = {}
    for cfg_instance in server_configs:
        # For CLI mode, we'll just print logs directly (TUI handles its own buffering)
        log_handlers[cfg_instance.alias] = lambda line, name=cfg_instance.alias: print(
            f"[{name}] {line}", flush=True
        )

    manager.start_servers(server_configs, log_handlers)

    # Wait for any server to exit
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
