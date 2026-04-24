"""Smoke test CLI entry point.

Runs smoke probes against one or more llama-server instances and reports
results in human-readable or JSON format.

Usage::

    llm-runner smoke both
    llm-runner smoke slot <slot-id>
    llm-runner smoke both --json --api-key KEY
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from llama_manager import (
    Config,
    SmokeCompositeReport,
    SmokeProbeConfiguration,
)
from llama_manager.smoke import (
    SmokeProbeResult,
    probe_slot,
)


def _build_slot_configs(
    mode: str,
) -> list[tuple[str, str, str, int]]:
    """Build list of (slot_id, model, host, port) for the given mode.

    Args:
        mode: Either "both" or "slot".

    Returns:
        List of (slot_id, model_path, host, port) tuples.

    Raises:
        SystemExit: If mode is invalid.
    """
    cfg = Config()

    if mode == "both":
        return [
            ("summary-balanced", cfg.model_summary_balanced, cfg.host, cfg.summary_balanced_port),
            ("qwen35-coding", cfg.model_qwen35_both, cfg.host, cfg.qwen35_port),
        ]
    if mode == "slot":
        return [("slot", cfg.model_summary_balanced, cfg.host, cfg.summary_balanced_port)]

    print(f"error: unknown smoke mode '{mode}'. Valid modes: both, slot", file=sys.stderr)
    sys.exit(1)


def _probe_server(
    slot_id: str,
    model: str,
    host: str,
    port: int,
    smoke_cfg: SmokeProbeConfiguration,
) -> SmokeProbeResult:
    """Probe a single server instance.

    Args:
        slot_id: Slot identifier.
        model: Model path.
        host: Server hostname.
        port: Server port.
        smoke_cfg: Smoke probe configuration.

    Returns:
        SmokeProbeResult with probe outcome.
    """
    return probe_slot(
        host=host,
        port=port,
        smoke_cfg=smoke_cfg,
        model_id=model,
        expected_model_id=model,
    )


def run_smoke(args: list[str]) -> int:
    """Run smoke tests against server instances.

    Parses CLI arguments, probes servers, and outputs results.

    Args:
        args: List of command-line arguments (after "smoke").

    Returns:
        Exit code (0 for all pass, otherwise worst failure).
    """
    parser = _build_smoke_parser()
    parsed = parser.parse_args(args)

    mode: str = parsed.mode
    json_output: bool = parsed.json
    api_key: str = parsed.api_key or ""
    model_id: str | None = parsed.model_id
    max_tokens: int = parsed.max_tokens
    prompt: str = parsed.prompt
    delay: int = parsed.delay
    timeout: int = parsed.timeout

    # Validate mode
    if mode not in ("both", "slot"):
        print(
            f"error: invalid smoke mode '{mode}'. Valid modes: both, slot",
            file=sys.stderr,
        )
        return 1

    # Resolve slot_id for "slot" mode
    slot_id: str | None = parsed.slot_id
    if mode == "slot" and not slot_id:
        print("error: 'slot' mode requires a slot ID argument", file=sys.stderr)
        return 1

    # Build server targets
    targets = _build_slot_configs(mode)

    # If slot mode with explicit slot_id, filter to that slot
    if mode == "slot" and slot_id:
        targets = [
            (slot_id, model, host, port) for sid, model, host, port in targets if sid == slot_id
        ]
        if not targets:
            print(f"error: no server found for slot '{slot_id}'", file=sys.stderr)
            return 1

    # Build smoke probe configuration
    cfg = Config()
    smoke_cfg = SmokeProbeConfiguration(
        inter_slot_delay_s=delay or cfg.smoke_inter_slot_delay_s,
        listen_timeout_s=timeout or cfg.smoke_listen_timeout_s,
        http_request_timeout_s=cfg.smoke_http_request_timeout_s,
        max_tokens=max_tokens or cfg.smoke_max_tokens,
        prompt=prompt or cfg.smoke_prompt,
        skip_models_discovery=cfg.smoke_skip_models_discovery,
        api_key=api_key or cfg.smoke_api_key,
        model_id_override=model_id,
        first_token_timeout_s=cfg.smoke_first_token_timeout_s,
        total_chat_timeout_s=cfg.smoke_total_chat_timeout_s,
    )

    # Run probes sequentially with delay between slots
    results: list[SmokeProbeResult] = []
    for idx, (sid, model, host, port) in enumerate(targets):
        if idx > 0 and smoke_cfg.inter_slot_delay_s > 0:
            time.sleep(smoke_cfg.inter_slot_delay_s)

        result = _probe_server(sid, model, host, port, smoke_cfg)
        results.append(result)

    # Build composite report
    report = SmokeCompositeReport(results=results)

    # Output results
    if json_output:
        _print_report_json(report)
    else:
        _print_report_human(report, mode)

    return report.overall_exit_code


def _build_smoke_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the smoke subcommand.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description="Run smoke tests against llama-server instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s both                    Probe all servers
  %(prog)s slot summary-balanced   Probe a specific slot
  %(prog)s both --json             Output in JSON format
        """,
    )

    parser.add_argument(
        "mode",
        choices=["both", "slot"],
        help="Mode: 'both' probes all servers, 'slot <id>' probes a specific slot",
    )
    parser.add_argument(
        "slot_id",
        nargs="?",
        default=None,
        help="Slot ID (required when mode is 'slot')",
    )

    # Smoke probe options
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for authentication",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model ID override for smoke probe",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Max tokens for chat probe (8-32, defaults to config)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Custom prompt for chat probe",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Inter-slot delay in seconds (defaults to config)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Overall timeout in seconds for listen phase (defaults to config)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results in JSON format",
    )

    return parser


def _print_report_human(report: SmokeCompositeReport, mode: str) -> None:
    """Print human-readable smoke test report.

    Args:
        report: Composite report with probe results.
        mode: The smoke mode ("both" or "slot").
    """
    status_label = report.overall_status.value.upper()
    print("=== SMOKE TEST REPORT ===")
    print(f"Mode: {mode}")
    print(f"Overall: {status_label}")
    print(f"Passed: {report.pass_count} / {len(report.results)}")
    print()

    for result in report.results:
        status = result.status.value.upper()
        phase = result.phase_reached.value
        slot = result.slot_id
        latency = f"{result.latency_ms}ms" if result.latency_ms is not None else "N/A"

        print(f"  [{slot}] {status}")
        print(f"    Phase reached: {phase}")
        print(f"    Latency: {latency}")

        if result.failure_phase:
            print(f"    Failed at: {result.failure_phase.value}")
        if result.model_id:
            print(f"    Model: {result.model_id}")
        print()


def _print_report_json(report: SmokeCompositeReport) -> None:
    """Print JSON smoke test report.

    Args:
        report: Composite report with probe results.
    """
    output: dict[str, Any] = {
        "overall_status": report.overall_status.value,
        "overall_exit_code": report.overall_exit_code,
        "pass_count": report.pass_count,
        "fail_count": report.fail_count,
        "results": [
            {
                "slot_id": r.slot_id,
                "status": r.status.value,
                "phase_reached": r.phase_reached.value,
                "failure_phase": r.failure_phase.value if r.failure_phase else None,
                "model_id": r.model_id,
                "latency_ms": r.latency_ms,
                "exit_code": r.exit_code,
                "provenance": {
                    "sha": r.provenance.sha,
                    "version": r.provenance.version,
                },
            }
            for r in report.results
        ],
    }
    print(json.dumps(output, indent=2))
