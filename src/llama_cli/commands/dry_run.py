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
    resolve_runtime_dir,
    validate_server_config,
    write_artifact,
)
from llama_manager.config_builder import create_default_profile_registry, resolve_run_group_configs
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


def _print_smoke_probe_info(cfg: Config) -> None:
    """Print smoke probe configuration for dry-run output.

    Shows smoke-relevant flags: model ID, prompt, /v1/models probe, API key source.

    Args:
        cfg: Config instance with smoke defaults.
    """
    print("  Smoke Probe:")
    print(f"    /v1/models: {'skip' if cfg.smoke_skip_models_discovery else 'enabled'}")
    print(f"    Prompt: {cfg.smoke_prompt}")
    print(f"    Max tokens: {cfg.smoke_max_tokens}")
    if cfg.smoke_api_key:
        print("    API key: [configured]")
    else:
        print("    API key: [not set]")
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


def _print_dry_run_header(mode: str, cfg: Config) -> None:
    """Print dry-run mode header information.

    Args:
        mode: The dry-run mode being executed.
        cfg: Config instance with model paths and binary locations.
    """
    registry = create_default_profile_registry(cfg)
    print("=== DRY RUN MODE ===")
    print(f"Mode: {mode}")
    print(f"llama-server (Intel): {cfg.llama_server_bin_intel}")
    print(f"llama-server (NVIDIA): {cfg.llama_server_bin_nvidia}")
    for profile_id in registry.profile_ids:
        profile = registry.get_profile(profile_id)
        print(f"{profile_id} model: {profile.model}")
    print()


def _parse_port_overrides(primary_port: str | None, secondary_port: str | None) -> tuple[int, ...]:
    """Return positional port overrides for a dry-run group."""
    overrides: list[int] = []
    if primary_port:
        overrides.append(int(primary_port))
    if secondary_port:
        overrides.append(int(secondary_port))
    return tuple(overrides)


def _slot_ids_for_group(mode: str, cfg: Config) -> tuple[str, ...]:
    """Return profile identifiers for the dry-run group in launch order."""
    registry = create_default_profile_registry(cfg)
    return registry.get_run_group(mode).profile_ids


def _resolve_dry_run_configs(
    mode: str,
    cfg: Config,
    primary_port: str | None,
    secondary_port: str | None,
) -> list[ServerConfig]:
    """Resolve dry-run configs through the profile registry."""
    registry = create_default_profile_registry(cfg)
    if mode not in registry.run_group_ids:
        allowed_modes = ", ".join(registry.run_group_ids)
        print(
            f"error: invalid mode '{mode}'. Valid modes: {allowed_modes}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return resolve_run_group_configs(
        registry,
        mode,
        _parse_port_overrides(primary_port, secondary_port),
    )


def _print_resolved_slot(
    slot_id: str, server_cfg: ServerConfig, payload: DryRunSlotPayload
) -> None:
    """Print a dry-run slot from canonical ServerConfig data."""
    print(f"{slot_id}:")
    print(f"  Port: {payload.port}")
    if server_cfg.device.startswith("SYCL"):
        print(f"  Device: {server_cfg.device}")
    else:
        print("  Device: NVIDIA (CUDA)")
    print(f"  Context: {server_cfg.ctx_size}")
    print(f"  Threads: {server_cfg.threads}")
    print(f"  UBatch: {server_cfg.ubatch_size}")
    print(f"  KV cache: {server_cfg.cache_type_k}/{server_cfg.cache_type_v}")
    print(f"  n-gpu-layers: {server_cfg.n_gpu_layers}")
    if server_cfg.reasoning_mode != "auto":
        print(f"  Reasoning: {server_cfg.reasoning_mode}")
    if server_cfg.reasoning_format != "none":
        print(f"  Reasoning Format: {server_cfg.reasoning_format}")
    if server_cfg.use_jinja:
        print(f"  Jinja: {server_cfg.use_jinja}")
    if server_cfg.chat_template_kwargs:
        print(f"  Chat Template Kwargs: {server_cfg.chat_template_kwargs}")
    _print_common_payload_sections(payload)


def _run_registry_mode(
    mode: str,
    cfg: Config,
    manager: ServerManager,
    primary_port: str | None,
    secondary_port: str | None,
    acknowledged: bool,
    slot_payloads: list[DryRunSlotPayload],
) -> bool:
    """Resolve and print any run group through the profile registry."""
    configs = _resolve_dry_run_configs(mode, cfg, primary_port, secondary_port)
    slot_ids = _slot_ids_for_group(mode, cfg)
    _verify_dry_run_risks(manager, configs, acknowledged)

    has_error = False
    for slot_id, server_cfg in zip(slot_ids, configs, strict=True):
        if _build_payload(server_cfg, slot_id, slot_payloads):
            has_error = True
            continue
        _print_resolved_slot(slot_id, server_cfg, slot_payloads[-1])
    return has_error


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
    """Print command without executing."""
    cfg = Config()

    _print_dry_run_header(mode, cfg)

    slot_payloads: list[DryRunSlotPayload] = []
    manager = ServerManager()

    try:
        has_error = _run_registry_mode(
            mode,
            cfg,
            manager,
            primary_port,
            secondary_port,
            acknowledged,
            slot_payloads,
        )

        if not has_error:
            _print_smoke_probe_info(cfg)

        if not has_error and slot_payloads:
            _write_dry_run_artifact(mode, slot_payloads)

        if has_error:
            sys.exit(1)

    except SystemExit:
        raise
    except (TypeError, ValueError) as e:
        print(f"error: dry-run failed unexpectedly: {e}", file=sys.stderr)
        sys.exit(1)
