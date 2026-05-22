"""Smoke target resolution and probe execution for llama.cpp servers.

Manager-level smoke service that handles:
- Target resolution (both mode -> all slots, slot_id -> specific slot)
- Probe execution with inter-slot delay
- Report assembly

Pure library — no argparse, no Rich, no user-facing I/O.
All functions return structured results, never print.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from llama_manager.config import (
    Config,
    SmokeProbeConfiguration,
    create_default_profile_registry,
    resolve_run_group_configs,
)
from llama_manager.probe import SmokeCompositeReport, SmokeProbeResult, probe_slot

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SmokeTarget:
    """Target for a smoke probe.

    Attributes:
        slot_id: Normalized slot identifier (e.g. "summary-balanced").
        model: Path to the GGUF model file.
        host: Server hostname or IP address.
        port: Server HTTP port.
        backend: Inference backend name (e.g. "llama_cpp").
    """

    slot_id: str
    model: str
    host: str
    port: int
    backend: str


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


def resolve_smoke_targets(
    config: Config,
    mode: str,
    slot_id: str | None = None,
) -> list[SmokeTarget]:
    """Resolve which slots to smoke test based on mode and optional slot filter.

    Args:
        config: Base configuration with profile registry.
        mode: Either "both" (all slots) or "slot" (specific slot).
        slot_id: Slot identifier when mode is "slot".

    Returns:
        List of SmokeTarget objects to probe. Empty list for unknown slots.
    """
    registry = create_default_profile_registry(config)

    if mode == "both":
        configs = resolve_run_group_configs(registry, "both")
        return [
            SmokeTarget(
                slot_id=cfg.alias,
                model=cfg.model,
                host=cfg.bind_address,
                port=cfg.port,
                backend=cfg.backend,
            )
            for cfg in configs
        ]

    if mode == "slot" and slot_id:
        try:
            configs = resolve_run_group_configs(registry, slot_id)
            return [
                SmokeTarget(
                    slot_id=cfg.alias,
                    model=cfg.model,
                    host=cfg.bind_address,
                    port=cfg.port,
                    backend=cfg.backend,
                )
                for cfg in configs
            ]
        except Exception:
            return []

    return []


# ---------------------------------------------------------------------------
# Probe execution
# ---------------------------------------------------------------------------


def run_smoke_probes(
    targets: list[SmokeTarget],
    smoke_cfg: SmokeProbeConfiguration,
    sleep: Callable[[float], None] | None = None,
) -> SmokeCompositeReport:
    """Run smoke probes sequentially against each target with inter-slot delay.

    Probes are executed in order. Between each target (except the last),
    a configurable inter-slot delay is applied.

    Args:
        targets: List of SmokeTarget objects to probe.
        smoke_cfg: Smoke probe configuration.
        sleep: Sleep callable for test injection (defaults to time.sleep).

    Returns:
        SmokeCompositeReport with per-slot results.
    """
    if sleep is None:
        sleep = time.sleep  # type: ignore[assignment]

    results: list[SmokeProbeResult] = []
    for idx, target in enumerate(targets):
        if idx > 0 and smoke_cfg.inter_slot_delay_s > 0:
            sleep(smoke_cfg.inter_slot_delay_s)

        result = probe_slot(
            host=target.host,
            port=target.port,
            smoke_cfg=smoke_cfg,
            model_path=target.model,
            model_id=smoke_cfg.model_id_override,
            expected_model_id=None,
        )
        results.append(result)

    return SmokeCompositeReport(results=results)
