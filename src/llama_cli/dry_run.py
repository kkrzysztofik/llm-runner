# Dry run functionality


import sys
import time

from llama_manager import (
    Config,
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


def _print_acknowledgement_required_and_exit() -> None:
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
        if acknowledged and "warning_bypass" not in cfg.risky_acknowledged:
            cfg.risky_acknowledged.append("warning_bypass")
        for risk in detect_risky_operations(cfg):
            if manager.is_risk_acknowledged(cfg.alias, risk, launch_attempt_id):
                continue
            if not acknowledged:
                print(f"warning: risky operation detected in {cfg.alias}: {risk}")
                response = input("Confirm risky operation [y/N]: ").strip().lower()
                if response != "y":
                    _print_acknowledgement_required_and_exit()
            manager.acknowledge_risk(
                cfg.alias,
                risk,
                launch_attempt_id=launch_attempt_id,
                ack_token=ack_token,
            )


def dry_run(
    mode: str,
    primary_port: str | None = None,
    secondary_port: str | None = None,
    acknowledged: bool = False,
) -> None:
    """Print command without executing"""
    cfg = Config()

    summary_balanced_port = int(primary_port) if primary_port else cfg.summary_balanced_port
    summary_fast_port = int(primary_port) if primary_port else cfg.summary_fast_port
    qwen35_port = int(primary_port) if primary_port else cfg.qwen35_port
    qwen35_port_both = int(secondary_port) if secondary_port else cfg.qwen35_port

    print("=== DRY RUN MODE ===")
    print(f"Mode: {mode}")
    print(f"llama-server (Intel): {cfg.llama_server_bin_intel}")
    print(f"llama-server (NVIDIA): {cfg.llama_server_bin_nvidia}")
    print(f"summary-balanced model: {cfg.model_summary_balanced}")
    print(f"summary-fast model: {cfg.model_summary_fast}")
    print(f"qwen35 model: {cfg.model_qwen35}")
    print(f"qwen35 both model: {cfg.model_qwen35_both}")
    print()

    # Build canonical slot payloads for FR-003
    from llama_manager import DryRunSlotPayload

    slot_payloads: list[DryRunSlotPayload] = []
    has_error = False
    manager = ServerManager()

    try:
        if mode in ("summary-balanced", "llama32"):
            server_cfg = create_summary_balanced_cfg(summary_balanced_port)
            _verify_dry_run_risks(manager, [server_cfg], acknowledged)
            # FR-011: Validate backend eligibility
            backend_error = validate_server_config(server_cfg)
            if backend_error is not None:
                print(f"error: {backend_error.error_code}", file=sys.stderr)
                print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
                print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
                print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
                has_error = True
            else:
                # FR-003: Build canonical dry-run slot payload
                payload = build_dry_run_slot_payload(
                    server_cfg,
                    slot_id="summary-balanced",
                    validation_results=None,  # Will default to passed=True
                    warnings=[],
                )
                slot_payloads.append(payload)

                # Human-readable output derived from canonical payload
                print("summary-balanced:")
                print(f"  Port: {payload.port}")
                print("  Device: SYCL0")
                print(f"  Context: {cfg.default_ctx_size_summary}")
                print(f"  Threads: {cfg.default_threads_summary_balanced}")
                print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
                print("  Reasoning: off")
                print("  Reasoning Format: deepseek")
                print("  Jinja: True")
                print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
                print(f"  Command: {' '.join(payload.command_args)}")
                print()

        elif mode == "summary-fast":
            server_cfg = create_summary_fast_cfg(summary_fast_port)
            _verify_dry_run_risks(manager, [server_cfg], acknowledged)
            # FR-011: Validate backend eligibility
            backend_error = validate_server_config(server_cfg)
            if backend_error is not None:
                print(f"error: {backend_error.error_code}", file=sys.stderr)
                print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
                print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
                print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
                has_error = True
            else:
                # FR-003: Build canonical dry-run slot payload
                payload = build_dry_run_slot_payload(
                    server_cfg,
                    slot_id="summary-fast",
                    validation_results=None,
                    warnings=[],
                )
                slot_payloads.append(payload)

                # Human-readable output derived from canonical payload
                print("summary-fast:")
                print(f"  Port: {payload.port}")
                print("  Device: SYCL0")
                print(f"  Context: {cfg.default_ctx_size_summary}")
                print(f"  Threads: {cfg.default_threads_summary_fast}")
                print(f"  UBatch: {cfg.default_ubatch_size_summary_fast}")
                print(f"  Command: {' '.join(payload.command_args)}")
                print()

        elif mode == "qwen35":
            server_cfg = create_qwen35_cfg(
                qwen35_port,
                n_gpu_layers=cfg.default_n_gpu_layers_qwen35,
                model=cfg.model_qwen35,
                server_bin=cfg.llama_server_bin_nvidia,
            )
            _verify_dry_run_risks(manager, [server_cfg], acknowledged)
            # FR-011: Validate backend eligibility
            backend_error = validate_server_config(server_cfg)
            if backend_error is not None:
                print(f"error: {backend_error.error_code}", file=sys.stderr)
                print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
                print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
                print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
                has_error = True
            else:
                # FR-003: Build canonical dry-run slot payload
                payload = build_dry_run_slot_payload(
                    server_cfg,
                    slot_id="qwen35",
                    validation_results=None,
                    warnings=[],
                )
                slot_payloads.append(payload)

                # Human-readable output derived from canonical payload
                print("qwen35:")
                print(f"  Port: {payload.port}")
                print("  Device: NVIDIA (CUDA)")
                print(f"  Context: {cfg.default_ctx_size_qwen35}")
                print(f"  Threads: {cfg.default_threads_qwen35}")
                print(f"  UBatch: {cfg.default_ubatch_size_qwen35}")
                print(
                    f"  KV cache: {cfg.default_cache_type_qwen35_k}/{cfg.default_cache_type_qwen35_v}"
                )
                print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35}")
                print(f"  Command: {' '.join(payload.command_args)}")
                print()

        elif mode == "both":
            server_cfg1 = create_summary_balanced_cfg(summary_balanced_port)
            _verify_dry_run_risks(manager, [server_cfg1], acknowledged)
            # FR-011: Validate backend eligibility for summary-balanced
            backend_error = validate_server_config(server_cfg1)
            if backend_error is not None:
                print(f"error: {backend_error.error_code}", file=sys.stderr)
                print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
                print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
                print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
                has_error = True
            else:
                payload1 = build_dry_run_slot_payload(
                    server_cfg1,
                    slot_id="summary-balanced",
                    validation_results=None,
                    warnings=[],
                )
                slot_payloads.append(payload1)

                # Human-readable output derived from canonical payload
                print("summary-balanced:")
                print(f"  Port: {payload1.port}")
                print("  Device: SYCL0")
                print(f"  Context: {cfg.default_ctx_size_both_summary}")
                print(f"  Threads: {cfg.default_threads_summary_balanced}")
                print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
                print(
                    f"  KV cache: {cfg.default_cache_type_summary_k}/{cfg.default_cache_type_summary_v}"
                )
                print("  Reasoning: off")
                print("  Reasoning Format: deepseek")
                print("  Jinja: True")
                print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
                print(f"  Command: {' '.join(payload1.command_args)}")
                print()

            server_cfg2 = create_qwen35_cfg(
                qwen35_port_both,
                ctx_size=cfg.default_ctx_size_both_qwen35,
                ubatch_size=cfg.default_ubatch_size_qwen35_both,
                threads=cfg.default_threads_qwen35_both,
                cache_k=cfg.default_cache_type_qwen35_both_k,
                cache_v=cfg.default_cache_type_qwen35_both_v,
                n_gpu_layers=cfg.default_n_gpu_layers_qwen35_both,
                model=cfg.model_qwen35_both,
                server_bin=cfg.llama_server_bin_nvidia,
            )
            _verify_dry_run_risks(manager, [server_cfg2], acknowledged)
            # FR-011: Validate backend eligibility for qwen35
            backend_error = validate_server_config(server_cfg2)
            if backend_error is not None:
                print(f"error: {backend_error.error_code}", file=sys.stderr)
                print(f"  failed_check: {backend_error.failed_check}", file=sys.stderr)
                print(f"  why_blocked: {backend_error.why_blocked}", file=sys.stderr)
                print(f"  how_to_fix: {backend_error.how_to_fix}", file=sys.stderr)
                has_error = True
            else:
                payload2 = build_dry_run_slot_payload(
                    server_cfg2,
                    slot_id="qwen35",
                    validation_results=None,
                    warnings=[],
                )
                slot_payloads.append(payload2)

                # Human-readable output derived from canonical payload
                print("qwen35:")
                print(f"  Port: {payload2.port}")
                print("  Device: NVIDIA (CUDA)")
                print(f"  Context: {cfg.default_ctx_size_both_qwen35}")
                print(f"  Threads: {cfg.default_threads_qwen35_both}")
                print(f"  UBatch: {cfg.default_ubatch_size_qwen35_both}")
                print(
                    f"  KV cache: {cfg.default_cache_type_qwen35_both_k}/{cfg.default_cache_type_qwen35_both_v}"
                )
                print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35_both}")
                print(f"  Command: {' '.join(payload2.command_args)}")
                print()

        else:
            print(
                "error: dry-run requires a mode argument (summary-balanced|summary-fast|qwen35|both)",
                file=sys.stderr,
            )
            sys.exit(1)

        # FR-007: Write artifact for dry-run attempt
        if not has_error and slot_payloads:
            runtime_dir = resolve_runtime_dir()
            # Build canonical top-level payload containing ordered slot payloads
            canonical_payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "slot_scope": [p.slot_id for p in slot_payloads],
                "resolved_command": {p.slot_id: p.command_args for p in slot_payloads},
                "validation_results": {
                    p.slot_id: {
                        "passed": p.validation_results.passed,
                        "checks": p.validation_results.checks,
                    }
                    for p in slot_payloads
                },
                "warnings": [warning for p in slot_payloads for warning in p.warnings],
                "environment_redacted": slot_payloads[0].environment_redacted
                if slot_payloads
                else {},
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

        if has_error:
            sys.exit(1)

    except Exception as e:
        print(f"error: dry-run failed unexpectedly: {e}", file=sys.stderr)
        sys.exit(1)
