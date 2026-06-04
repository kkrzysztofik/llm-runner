"""Dry run functionality for llm-runner.

This module provides command preview without execution, including
validation results, vLLM eligibility checks, and artifact persistence.

Delegates to llama_manager.dry_run for domain logic; owns all I/O.
"""

import sys
from typing import Any, NoReturn

from llama_cli.ui_output import emit_error, emit_heading, emit_info, emit_success, emit_warn
from llama_manager import (
    RISK_ACK_LABEL,
    Config,
    DryRunResult,
    DryRunSlotPayload,
    ServerConfig,
    ServerManager,
    create_default_profile_registry,
    run_dry_run,
    write_dry_run_artifact,
)

RISK_CONFIRM_PROMPT = "Confirm risky operation [y/N]: "
ELIGIBLE_LABEL = "    Eligible"
REASON_LABEL = "    Reason"
DEVICE_SYCL0_LABEL = "  Device: SYCL0"


def _print_acknowledgement_required_and_exit() -> NoReturn:
    emit_error("acknowledgement_required")
    emit_error("  failed_check: acknowledgement_required")
    emit_error(
        "  why_blocked: risky operation detected and not acknowledged",
    )
    emit_error(
        "  how_to_fix: use --acknowledge-risky flag or confirm with 'y'",
    )
    raise SystemExit(1)


def _print_backend_error(backend_error: Any) -> None:
    emit_error(backend_error.error_code)
    emit_error(f"  failed_check: {backend_error.failed_check}")
    emit_error(f"  why_blocked: {backend_error.why_blocked}")
    emit_error(f"  how_to_fix: {backend_error.how_to_fix}")


def _print_common_payload_sections(payload: Any) -> None:
    emit_info("OpenAI Bundle:")
    for key in sorted(payload.openai_flag_bundle.keys()):
        value = payload.openai_flag_bundle[key]
        emit_info(f"    {key}: {value}")

    emit_info("vllm Eligibility:")
    emit_info(f"{ELIGIBLE_LABEL}: {payload.vllm_eligibility.eligible}")
    emit_info(f"{REASON_LABEL}: {payload.vllm_eligibility.reason}")

    emit_info(f"  Command: {' '.join(payload.command_args)}")


def _print_smoke_probe_info(cfg: Config) -> None:
    """Print smoke probe configuration for dry-run output.

    Shows smoke-relevant flags: model ID, prompt, /v1/models probe, API key source.

    Args:
        cfg: Config instance with smoke defaults.
    """
    emit_info("Smoke Probe:")
    emit_info(f"    /v1/models: {'skip' if cfg.smoke_skip_models_discovery else 'enabled'}")
    emit_info(f"    Prompt: {cfg.smoke_prompt}")
    emit_info(f"    Max tokens: {cfg.smoke_max_tokens}")
    if cfg.smoke_api_key:
        emit_info("    API key: [configured]")
    else:
        emit_info("    API key: [not set]")


def _print_resolved_slot(
    slot_id: str, server_cfg: ServerConfig, payload: DryRunSlotPayload
) -> None:
    """Print a dry-run slot from canonical ServerConfig data."""
    emit_info(f"{slot_id}:")
    emit_info(f"  Port: {payload.port}")
    emit_info(f"  Device: {server_cfg.device}")
    emit_info(f"  Context: {server_cfg.ctx_size}")
    emit_info(f"  Threads: {server_cfg.threads}")
    emit_info(f"  UBatch: {server_cfg.ubatch_size}")
    emit_info(f"  KV cache: {server_cfg.cache_type_k}/{server_cfg.cache_type_v}")
    emit_info(f"  n-gpu-layers: {server_cfg.n_gpu_layers}")
    if server_cfg.reasoning_mode != "auto":
        emit_info(f"  Reasoning: {server_cfg.reasoning_mode}")
    if server_cfg.reasoning_format != "none":
        emit_info(f"  Reasoning Format: {server_cfg.reasoning_format}")
    if server_cfg.use_jinja:
        emit_info(f"  Jinja: {server_cfg.use_jinja}")
    if server_cfg.chat_template_kwargs:
        emit_info(f"  Chat Template Kwargs: {server_cfg.chat_template_kwargs}")
    _print_common_payload_sections(payload)


def _payload_server_config(payload: DryRunSlotPayload) -> ServerConfig:
    """Return the resolved ServerConfig carried by a dry-run payload."""
    if payload.server_config is not None:
        return payload.server_config

    return ServerConfig(
        model=payload.model_path,
        alias=payload.slot_id,
        device="",
        port=payload.port,
        ctx_size=4096,
        ubatch_size=512,
        threads=4,
        server_bin=payload.binary_path,
    )


def _print_dry_run_header(mode: str, cfg: Config, registry: Any) -> None:
    """Print dry-run mode header information.

    Args:
        mode: The dry-run mode being executed.
        cfg: Config instance with model paths and binary locations.
        registry: Pre-built profile registry.
    """
    emit_heading("DRY RUN MODE", level=2)
    emit_info(f"Mode: {mode}")
    emit_info(f"llama-server (Intel): {cfg.llama_server_bin_intel}")
    emit_info(f"llama-server (NVIDIA): {cfg.llama_server_bin_nvidia}")
    for profile_id in registry.profile_ids:
        profile = registry.get_profile(profile_id)
        emit_info(f"{profile_id} model: {profile.model}")


def _parse_port_overrides(primary_port: str | None, secondary_port: str | None) -> dict[str, int]:
    """Return positional port overrides for a dry-run mode."""
    overrides: dict[str, int] = {}
    for name, raw in (("primary", primary_port), ("secondary", secondary_port)):
        if raw:
            if not raw.isdigit():
                raise ValueError(f"{name} port {raw!r} is not a valid integer")
            port = int(raw)
            if not (1 <= port <= 65535):
                raise ValueError(f"{name} port {port} is out of range (1\u201365535)")
            overrides[name] = port
    return overrides


def _run_with_risk_acknowledgement(
    result: DryRunResult,
    manager: ServerManager,
    configs: list[Any],
    acknowledged: bool,
) -> None:
    """Handle risk acknowledgement for dry-run results.

    Exits via SystemExit when the user declines a risky operation.
    """
    if not result.warnings:
        return

    for warning in result.warnings:
        emit_warn(warning)

    if not acknowledged:
        try:
            response = input(RISK_CONFIRM_PROMPT).strip().lower()
        except EOFError:
            _print_acknowledgement_required_and_exit()
        if response != "y":
            _print_acknowledgement_required_and_exit()

    # Update configs with risk acknowledgement
    for cfg in configs:
        if acknowledged and RISK_ACK_LABEL not in cfg.risky_acknowledged:
            cfg.risky_acknowledged.append(RISK_ACK_LABEL)
        manager.acknowledge_risk(cfg.alias, RISK_ACK_LABEL)


def dry_run(
    mode: str,
    primary_port: str | None = None,
    secondary_port: str | None = None,
    acknowledged: bool = False,
) -> None:
    """Print command without executing."""
    app_cfg = Config()
    registry = create_default_profile_registry(app_cfg)

    _print_dry_run_header(mode, app_cfg, registry)

    try:
        port_overrides = _parse_port_overrides(primary_port, secondary_port)
        result = run_dry_run(
            mode=mode,
            config=app_cfg,
            registry=registry,
            port_overrides=port_overrides,
            acknowledged=acknowledged,
        )

        # Handle errors from manager
        if result.errors:
            for error in result.errors:
                emit_error(error)
            sys.exit(1)

        # Handle risk acknowledgement
        configs = [_payload_server_config(payload) for payload in result.slot_payloads]

        manager = ServerManager()
        _run_with_risk_acknowledgement(result, manager, configs, acknowledged)

        # Print slot details — use the resolved ServerConfig from the manager
        for payload in result.slot_payloads:
            cfg = _payload_server_config(payload)
            _print_resolved_slot(payload.slot_id, cfg, payload)

        if not result.has_error:
            _print_smoke_probe_info(app_cfg)

        if not result.has_error and result.slot_payloads:
            try:
                artifact_path = write_dry_run_artifact(mode, result.slot_payloads)
                emit_success(f"Artifact written: {artifact_path}")
            except Exception as e:
                emit_error(f"artifact persistence failed: {e}")
                emit_error("  failed_check: artifact_persistence")
                emit_error(
                    "  why_blocked: artifact persistence failed to enforce required permissions",
                )
                emit_error(
                    "  how_to_fix: verify runtime path and permission support before retry",
                )
                sys.exit(1)

    except SystemExit:
        raise
    except (TypeError, ValueError) as e:
        emit_error(f"dry-run failed unexpectedly: {e}")
        sys.exit(1)
