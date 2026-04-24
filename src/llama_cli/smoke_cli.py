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
import dataclasses
import json
import sys
import time
from typing import Any

from llama_manager import (
    Config,
    SmokeCompositeReport,
    SmokeProbeConfiguration,
    resolve_runtime_dir,
)
from llama_manager.config_builder import create_smoke_config
from llama_manager.smoke import (
    SmokeProbeResult,
    _ensure_report_dir,
    probe_slot,
)


def _build_slot_configs(
    mode: str,
    slot_id: str | None = None,
) -> list[tuple[str, str, str, int]]:
    """Build list of (slot_id, model, host, port) for the given mode.

    Args:
        mode: Either "both" or "slot".
        slot_id: User-supplied slot identifier (required when mode is "slot").

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
        if not slot_id:
            print("error: 'slot' mode requires a slot ID argument", file=sys.stderr)
            sys.exit(1)
        slot_port = _resolve_slot_port(cfg, slot_id)
        slot_model = _resolve_slot_model(cfg, slot_id)
        return [(slot_id, slot_model, cfg.host, slot_port)]

    print(f"error: unknown smoke mode '{mode}'. Valid modes: both, slot", file=sys.stderr)
    sys.exit(1)


def _resolve_slot_port(cfg: Config, slot_id: str) -> int:
    """Resolve port for a given slot_id.

    Args:
        cfg: Config instance.
        slot_id: Slot identifier (e.g., "summary-balanced", "qwen35-coding").

    Returns:
        Port number for the slot.

    Raises:
        SystemExit: If slot_id is unknown.
    """
    slot_port_map = {
        "summary-balanced": cfg.summary_balanced_port,
        "summary-fast": cfg.summary_fast_port,
        "qwen35-coding": cfg.qwen35_port,
    }
    if slot_id not in slot_port_map:
        print(
            f"error: unknown slot '{slot_id}'. Valid slots: {', '.join(slot_port_map.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    return slot_port_map[slot_id]


def _resolve_slot_model(cfg: Config, slot_id: str) -> str:
    """Resolve model path for a given slot_id.

    Args:
        cfg: Config instance.
        slot_id: Slot identifier (e.g., "summary-balanced", "qwen35-coding").

    Returns:
        Model path for the slot.

    Raises:
        SystemExit: If slot_id is unknown.
    """
    slot_model_map = {
        "summary-balanced": cfg.model_summary_balanced,
        "summary-fast": cfg.model_summary_fast,
        "qwen35-coding": cfg.model_qwen35,
    }
    if slot_id not in slot_model_map:
        print(
            f"error: unknown slot '{slot_id}'. Valid slots: {', '.join(slot_model_map.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)
    return slot_model_map[slot_id]


def _probe_server(
    model_path: str,
    host: str,
    port: int,
    smoke_cfg: SmokeProbeConfiguration,
) -> SmokeProbeResult:
    """Probe a single server instance.

    Args:
        model_path: Path to the GGUF model file.
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
        model_path=model_path,
        model_id=None,
        expected_model_id=None,
    )


def _validate_smoke_args(parsed: argparse.Namespace) -> int | None:
    """Validate smoke CLI arguments.

    Args:
        parsed: Parsed argparse namespace.

    Returns:
        Exit code on validation failure, or None if valid.
    """
    mode: str = parsed.mode
    if mode not in ("both", "slot"):
        print(
            f"error: invalid smoke mode '{mode}'. Valid modes: both, slot",
            file=sys.stderr,
        )
        return 1

    slot_id: str | None = parsed.slot_id
    if mode == "slot" and not slot_id:
        print("error: 'slot' mode requires a slot ID argument", file=sys.stderr)
        return 1
    return None


def _build_smoke_config(parsed: argparse.Namespace) -> SmokeProbeConfiguration:
    """Build SmokeProbeConfiguration from parsed CLI arguments.

    Uses create_smoke_config() as the base factory, then applies CLI overrides
    via dataclasses.replace(). Validation failures in __post_init__ are caught
    and reported as clean error messages (no traceback).

    Args:
        parsed: Parsed argparse namespace.

    Returns:
        Configured SmokeProbeConfiguration.

    Raises:
        SystemExit: If validation fails.
    """
    cfg = Config()

    # Build base config via factory
    smoke_cfg = create_smoke_config(
        config=cfg,
        api_key=parsed.api_key or cfg.smoke_api_key,
        model_id_override=parsed.model_id,
    )

    # Apply CLI overrides where provided (0 means "use default")
    overrides: dict[str, int | str] = {}
    if parsed.delay > 0:
        overrides["inter_slot_delay_s"] = parsed.delay
    if parsed.timeout > 0:
        overrides["listen_timeout_s"] = parsed.timeout
    if parsed.max_tokens > 0:
        overrides["max_tokens"] = parsed.max_tokens
    if parsed.prompt:
        overrides["prompt"] = parsed.prompt

    if overrides:
        try:
            smoke_cfg = dataclasses.replace(smoke_cfg, **overrides)  # type: ignore[arg-type]
        except ValueError as exc:
            print(f"error: invalid smoke configuration: {exc}", file=sys.stderr)
            sys.exit(1)

    return smoke_cfg


def _run_probes(
    targets: list[tuple[str, str, str, int]],
    smoke_cfg: SmokeProbeConfiguration,
) -> list[SmokeProbeResult]:
    """Run smoke probes sequentially against server targets.

    Args:
        targets: List of (slot_id, model, host, port) tuples.
        smoke_cfg: Smoke probe configuration.

    Returns:
        List of SmokeProbeResult objects.
    """
    results: list[SmokeProbeResult] = []
    for idx, (_sid, model, host, port) in enumerate(targets):
        if idx > 0 and smoke_cfg.inter_slot_delay_s > 0:
            time.sleep(smoke_cfg.inter_slot_delay_s)

        result = _probe_server(model, host, port, smoke_cfg)
        results.append(result)
    return results


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

    # Validate arguments
    validation_result = _validate_smoke_args(parsed)
    if validation_result is not None:
        return validation_result

    mode: str = parsed.mode
    json_output: bool = parsed.json

    # Build server targets
    slot_id: str | None = parsed.slot_id
    targets = _build_slot_configs(mode, slot_id)

    # Build smoke probe configuration
    smoke_cfg = _build_smoke_config(parsed)

    # Run probes
    results = _run_probes(targets, smoke_cfg)

    # Build composite report
    report = SmokeCompositeReport(results=results)

    # Ensure report directory exists when there are failures (T072)
    if report.fail_count > 0:
        runtime_dir = resolve_runtime_dir()
        _ensure_report_dir(runtime_dir / "smoke_reports")

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
