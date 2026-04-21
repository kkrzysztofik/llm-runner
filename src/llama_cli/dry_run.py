"""Dry run functionality for llm-runner.

This module provides command preview without execution, including
validation results, vLLM eligibility checks, and artifact persistence.
"""

import sys
import time
from typing import Any, NoReturn

from llama_manager import (
    Config,
    DryRunSlotPayload,
    ServerConfig,
    ServerManager,
    build_dry_run_slot_payload,
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
    resolve_runtime_dir,
    validate_server_config,
    write_artifact,
)
from llama_manager.server import detect_risky_operations

RISK_ACK_LABEL = "warning_bypass"
RISK_CONFIRM_PROMPT = "Confirm risky operation [y/N]: "
ELIGIBLE_LABEL = "    Eligible"
REASON_LABEL = "    Reason"
DEVICE_SYCL0_LABEL = "  Device: SYCL0"


def _print_acknowledgement_required_and_exit() -> NoReturn:
    print("error: acknowledgement_required", file=sys.stderr)
    print("  failed_check: acknowledgement_required", file=sys.stderr)
    print(
        "  why_blocked: risky operation detected and not acknowledged",
        file=sys.stderr,
    )
    print(
        "  how_to_fix: use --acknowledge-risky flag or confirm with 'y'",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _verify_dry_run_risks(
    manager: ServerManager,
    configs: list[ServerConfig],
    acknowledged: bool,
) -> None:
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
            _print_acknowledgement_required_and_exit()
        if response != "y":
            _print_acknowledgement_required_and_exit()

    manager.acknowledge_risk(
        cfg.alias,
        risk,
        launch_attempt_id=launch_attempt_id,
        ack_token=ack_token,
    )


def _print_backend_error(backend_error: Any) -> None:
    print(f"error: {backend_error.error_code}", file=sys.stderr)
    print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
    print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
    print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)


def _print_common_payload_sections(payload: Any) -> None:
    print("  OpenAI Bundle:")
    for key in sorted(payload.openai_flag_bundle.keys()):
        value = payload.openai_flag_bundle[key]
        print(f"    {key}: {value}")

    print("  vllm Eligibility:")
    print(f"{ELIGIBLE_LABEL}: {payload.vllm_eligibility.eligible}")
    print(f"{REASON_LABEL}: {payload.vllm_eligibility.reason}")

    print(f"  Command: {' '.join(payload.command_args)}")
    print()


def _build_payload(
    server_cfg: ServerConfig,
    slot_id: str,
    slot_payloads: list[Any],
) -> bool:
    backend_error = validate_server_config(server_cfg)
    if backend_error is not None:
        _print_backend_error(backend_error)
        return True

    payload = build_dry_run_slot_payload(
        server_cfg,
        slot_id=slot_id,
        validation_results=None,
        warnings=[],
    )
    slot_payloads.append(payload)
    return False


def _run_summary_balanced_mode(
    cfg: Config,
    manager: ServerManager,
    port: int,
    acknowledged: bool,
    slot_payloads: list[Any],
) -> bool:
    server_cfg = create_summary_balanced_cfg(port)
    _verify_dry_run_risks(manager, [server_cfg], acknowledged)
    if _build_payload(server_cfg, "summary-balanced", slot_payloads):
        return True

    payload = slot_payloads[-1]
    print("summary-balanced:")
    print(f"  Port: {payload.port}")
    print(DEVICE_SYCL0_LABEL)
    print(f"  Context: {cfg.default_ctx_size_summary}")
    print(f"  Threads: {cfg.default_threads_summary_balanced}")
    print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
    print("  Reasoning: off")
    print("  Reasoning Format: deepseek")
    print("  Jinja: True")
    print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
    _print_common_payload_sections(payload)
    return False


def _run_summary_fast_mode(
    cfg: Config,
    manager: ServerManager,
    port: int,
    acknowledged: bool,
    slot_payloads: list[Any],
) -> bool:
    server_cfg = create_summary_fast_cfg(port)
    _verify_dry_run_risks(manager, [server_cfg], acknowledged)
    if _build_payload(server_cfg, "summary-fast", slot_payloads):
        return True

    payload = slot_payloads[-1]
    print("summary-fast:")
    print(f"  Port: {payload.port}")
    print(DEVICE_SYCL0_LABEL)
    print(f"  Context: {cfg.default_ctx_size_summary}")
    print(f"  Threads: {cfg.default_threads_summary_fast}")
    print(f"  UBatch: {cfg.default_ubatch_size_summary_fast}")
    _print_common_payload_sections(payload)
    return False


def _run_qwen35_mode(
    cfg: Config,
    manager: ServerManager,
    port: int,
    acknowledged: bool,
    slot_payloads: list[Any],
) -> bool:
    server_cfg = create_qwen35_cfg(
        port,
        n_gpu_layers=cfg.default_n_gpu_layers_qwen35,
        model=cfg.model_qwen35,
        server_bin=cfg.llama_server_bin_nvidia,
    )
    _verify_dry_run_risks(manager, [server_cfg], acknowledged)
    if _build_payload(server_cfg, "qwen35", slot_payloads):
        return True

    payload = slot_payloads[-1]
    print("qwen35:")
    print(f"  Port: {payload.port}")
    print("  Device: NVIDIA (CUDA)")
    print(f"  Context: {cfg.default_ctx_size_qwen35}")
    print(f"  Threads: {cfg.default_threads_qwen35}")
    print(f"  UBatch: {cfg.default_ubatch_size_qwen35}")
    print(f"  KV cache: {cfg.default_cache_type_qwen35_k}/{cfg.default_cache_type_qwen35_v}")
    print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35}")
    _print_common_payload_sections(payload)
    return False


def _run_both_mode(
    cfg: Config,
    manager: ServerManager,
    summary_port: int,
    qwen35_port: int,
    acknowledged: bool,
    slot_payloads: list[Any],
) -> bool:
    server_cfg1 = create_summary_balanced_cfg(summary_port)
    _verify_dry_run_risks(manager, [server_cfg1], acknowledged)
    has_error = _build_payload(server_cfg1, "summary-balanced", slot_payloads)

    if not has_error:
        payload1 = slot_payloads[-1]
        print("summary-balanced:")
        print(f"  Port: {payload1.port}")
        print(DEVICE_SYCL0_LABEL)
        print(f"  Context: {cfg.default_ctx_size_both_summary}")
        print(f"  Threads: {cfg.default_threads_summary_balanced}")
        print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
        print(f"  KV cache: {cfg.default_cache_type_summary_k}/{cfg.default_cache_type_summary_v}")
        print("  Reasoning: off")
        print("  Reasoning Format: deepseek")
        print("  Jinja: True")
        print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
        _print_common_payload_sections(payload1)

    server_cfg = create_qwen35_cfg(
        qwen35_port,
        ctx_size=cfg.default_ctx_size_both_qwen35,
        ubatch_size=cfg.default_ubatch_size_qwen35_both,
        threads=cfg.default_threads_qwen35_both,
        cache_k=cfg.default_cache_type_qwen35_both_k,
        cache_v=cfg.default_cache_type_qwen35_both_v,
        n_gpu_layers=cfg.default_n_gpu_layers_qwen35_both,
        model=cfg.model_qwen35_both,
        server_bin=cfg.llama_server_bin_nvidia,
    )
    _verify_dry_run_risks(manager, [server_cfg], acknowledged)
    if _build_payload(server_cfg, "qwen35", slot_payloads):
        return True

    payload = slot_payloads[-1]
    print("qwen35:")
    print(f"  Port: {payload.port}")
    print("  Device: NVIDIA (CUDA)")
    print(f"  Context: {cfg.default_ctx_size_both_qwen35}")
    print(f"  Threads: {cfg.default_threads_qwen35_both}")
    print(f"  UBatch: {cfg.default_ubatch_size_qwen35_both}")
    print(
        f"  KV cache: {cfg.default_cache_type_qwen35_both_k}/{cfg.default_cache_type_qwen35_both_v}"
    )
    print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35_both}")
    _print_common_payload_sections(payload)
    return has_error


def _resolve_ports(primary_port: str | None, secondary_port: str | None) -> dict[str, int]:
    """Resolve port values with defaults.

    Args:
        primary_port: Primary port string from user input.
        secondary_port: Secondary port string from user input.

    Returns:
        Dict with resolved port values for each mode.
    """
    cfg = Config()
    return {
        "summary_balanced_port": int(primary_port) if primary_port else cfg.summary_balanced_port,
        "summary_fast_port": int(primary_port) if primary_port else cfg.summary_fast_port,
        "qwen35_port": int(primary_port) if primary_port else cfg.qwen35_port,
        "qwen35_port_both": int(secondary_port) if secondary_port else cfg.qwen35_port,
    }


def _print_dry_run_header(mode: str, cfg: Config) -> None:
    """Print dry-run mode header information.

    Args:
        mode: The dry-run mode being executed.
        cfg: Config instance with model paths and binary locations.
    """
    print("=== DRY RUN MODE ===")
    print(f"Mode: {mode}")
    print(f"llama-server (Intel): {cfg.llama_server_bin_intel}")
    print(f"llama-server (NVIDIA): {cfg.llama_server_bin_nvidia}")
    print(f"summary-balanced model: {cfg.model_summary_balanced}")
    print(f"summary-fast model: {cfg.model_summary_fast}")
    print(f"qwen35 model: {cfg.model_qwen35}")
    print(f"qwen35 both model: {cfg.model_qwen35_both}")
    print()


def _write_dry_run_artifact(mode: str, slot_payloads: list[DryRunSlotPayload]) -> None:
    """Write dry-run artifact to runtime directory.

    Args:
        mode: The dry-run mode that was executed.
        slot_payloads: List of slot payloads to include in artifact.

    Raises:
        SystemExit: On artifact persistence failure.
    """
    runtime_dir = resolve_runtime_dir()
    canonical_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "slot_scope": [p.slot_id for p in slot_payloads],
        "resolved_command": {p.slot_id: p.command_args for p in slot_payloads},
        "validation_results": {
            p.slot_id: {
                "passed": p.validation_results.passed if p.validation_results else False,
                "checks": p.validation_results.checks if p.validation_results else [],
            }
            for p in slot_payloads
        },
        "warnings": [warning for p in slot_payloads for warning in p.warnings],
        "environment_redacted": slot_payloads[0].environment_redacted if slot_payloads else {},
    }
    try:
        artifact_path = write_artifact(runtime_dir, f"dryrun-{mode}", canonical_payload)
        print(f"Artifact written: {artifact_path}")
    except Exception as e:
        print(f"error: artifact persistence failed: {e}", file=sys.stderr)
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


def dry_run(
    mode: str,
    primary_port: str | None = None,
    secondary_port: str | None = None,
    acknowledged: bool = False,
) -> None:
    """Print command without executing"""
    cfg = Config()

    ports = _resolve_ports(primary_port, secondary_port)

    _print_dry_run_header(mode, cfg)

    slot_payloads: list[DryRunSlotPayload] = []
    has_error = False
    manager = ServerManager()

    handlers = {
        "summary-balanced": lambda: _run_summary_balanced_mode(
            cfg,
            manager,
            ports["summary_balanced_port"],
            acknowledged,
            slot_payloads,
        ),
        "llama32": lambda: _run_summary_balanced_mode(
            cfg,
            manager,
            ports["summary_balanced_port"],
            acknowledged,
            slot_payloads,
        ),
        "summary-fast": lambda: _run_summary_fast_mode(
            cfg,
            manager,
            ports["summary_fast_port"],
            acknowledged,
            slot_payloads,
        ),
        "qwen35": lambda: _run_qwen35_mode(
            cfg,
            manager,
            ports["qwen35_port"],
            acknowledged,
            slot_payloads,
        ),
        "both": lambda: _run_both_mode(
            cfg,
            manager,
            ports["summary_balanced_port"],
            ports["qwen35_port_both"],
            acknowledged,
            slot_payloads,
        ),
    }

    try:
        handler = handlers.get(mode)
        if handler is None:
            allowed_modes = ", ".join(sorted(handlers.keys()))
            print(
                f"error: invalid mode '{mode}'. Valid modes: {allowed_modes}",
                file=sys.stderr,
            )
            sys.exit(1)

        has_error = handler()

        # FR-007: Write artifact for dry-run attempt
        if not has_error and slot_payloads:
            _write_dry_run_artifact(mode, slot_payloads)

        if has_error:
            sys.exit(1)

    except Exception as e:
        print(f"error: dry-run failed unexpectedly: {e}", file=sys.stderr)
        sys.exit(1)
