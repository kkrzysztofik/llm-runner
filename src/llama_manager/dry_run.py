"""Dry-run domain service for llama_manager.

Provides the pure-library layer for dry-run operations:
- Resolving configs from the profile registry
- Validating server configurations
- Building slot payloads
- Applying/recording risk acknowledgement
- Assembling canonical artifact payloads

The CLI layer (llama_cli.commands.dry_run) owns all user-facing I/O:
input() prompts, emit_* calls, formatting, and sys.exit.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from llama_manager.config import Config, create_default_profile_registry, resolve_run_group_configs
from llama_manager.orchestration import DryRunArtifactPayload, ServerManager, write_artifact
from llama_manager.orchestration.lockfile import resolve_runtime_dir
from llama_manager.risk_ack import evaluate_risks
from llama_manager.validation import (
    DryRunSlotPayload,
    ValidationResults,
    build_dry_run_slot_payload,
    validate_server_config,
)

if TYPE_CHECKING:
    from llama_manager.config import RunProfileRegistry


@dataclass
class DryRunResult:
    """Result of a dry-run operation."""

    mode: str
    slot_payloads: list[DryRunSlotPayload]
    has_error: bool
    errors: list[str]
    warnings: list[str] = field(default_factory=list)
    artifact_payload: DryRunArtifactPayload | None = None


def run_dry_run(
    mode: str,
    config: Config,
    registry: RunProfileRegistry | None = None,
    port_overrides: dict[str, int] | None = None,
    acknowledged: bool = False,
) -> DryRunResult:
    """Execute a dry-run operation and return structured results.

    This is the pure-library function that handles all dry-run logic
    without any user-facing I/O. The CLI layer wraps this function
    to handle input prompts, emit_* calls, formatting, and sys.exit.

    Args:
        mode: The dry-run mode (e.g. "summary-balanced", "both").
        config: Base configuration with paths and defaults.
        registry: Optional pre-built profile registry.
            When omitted, a fresh registry is created.
        port_overrides: Optional dict mapping profile IDs to port numbers.
        acknowledged: Whether risky operations are pre-acknowledged.

    Returns:
        DryRunResult with slot payloads, errors, and optional artifact.

    Raises:
        SystemExit: On invalid mode (handled by caller).
    """
    if registry is None:
        registry = create_default_profile_registry(config)

    run_group_ids = registry.run_group_ids
    if mode not in run_group_ids:
        allowed_modes = ", ".join(run_group_ids)
        return DryRunResult(
            mode=mode,
            slot_payloads=[],
            has_error=True,
            errors=[f"invalid mode '{mode}'. Valid modes: {allowed_modes}"],
        )

    # Parse port overrides from positional format
    position_overrides: tuple[int, ...] = ()
    if port_overrides:
        group = registry.get_run_group(mode)
        position_overrides = tuple(port_overrides.get(pid, 0) for pid in group.profile_ids)

    try:
        configs = resolve_run_group_configs(
            registry,
            mode,
            port_overrides=position_overrides,
        )
    except (ValueError, TypeError) as exc:
        return DryRunResult(
            mode=mode,
            slot_payloads=[],
            has_error=True,
            errors=[f"port override error: {exc}"],
        )

    slot_ids = registry.get_run_group(mode).profile_ids
    slot_payloads: list[DryRunSlotPayload] = []
    errors: list[str] = []
    warnings: list[str] = []

    # Risk acknowledgement via ServerManager
    manager = ServerManager()
    launch_attempt_id = manager.begin_launch_attempt()
    ack_token = manager.issue_ack_token(launch_attempt_id)

    risk_result = evaluate_risks(
        configs,
        manager,
        launch_attempt_id,
        ack_token,
        acknowledged,
    )

    if risk_result.has_risks and not risk_result.risks_acknowledged and risk_result.risk_details:
        for detail in risk_result.risk_details:
            warnings.append(f"risky operation detected in {detail['alias']}: {detail['risk']}")

    # Build payloads for each slot
    has_error = False
    for slot_id, server_cfg in zip(slot_ids, configs, strict=True):
        backend_error = validate_server_config(server_cfg)
        if backend_error is not None:
            errors.append(f"{backend_error.error_code}: {backend_error.why_blocked}")
            has_error = True
            continue

        payload = build_dry_run_slot_payload(
            server_cfg,
            slot_id=slot_id,
            validation_results=ValidationResults(passed=True, checks=[]),
            warnings=[],
        )
        slot_payloads.append(payload)

    # Build artifact payload
    artifact_payload = _build_artifact_payload(mode, slot_payloads, warnings)

    return DryRunResult(
        mode=mode,
        slot_payloads=slot_payloads,
        has_error=has_error,
        errors=errors,
        warnings=warnings,
        artifact_payload=artifact_payload,
    )


def _build_artifact_payload(
    mode: str,
    slot_payloads: list[DryRunSlotPayload],
    run_warnings: list[str] | None = None,
) -> DryRunArtifactPayload:
    """Build the canonical artifact payload from slot payloads.

    Args:
        mode: The dry-run mode that was executed.
        slot_payloads: List of slot payloads to include.
        run_warnings: Optional run-level warnings to merge into the artifact.

    Returns:
        DryRunArtifactPayload with the canonical schema.
    """
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    slot_scope = [p.slot_id for p in slot_payloads]
    resolved_commands = {p.slot_id: p.command_args for p in slot_payloads}
    validation_results: dict[str, dict[str, Any]] = {}
    all_warnings: list[str] = []

    if run_warnings:
        all_warnings.extend(run_warnings)

    for p in slot_payloads:
        validation_results[p.slot_id] = {
            "passed": p.validation_results.passed if p.validation_results else False,
            "checks": p.validation_results.checks if p.validation_results else [],
        }
        all_warnings.extend(p.warnings)

    environment_redacted: dict[str, Any] = {}
    if slot_payloads:
        environment_redacted = slot_payloads[0].environment_redacted

    return DryRunArtifactPayload(
        timestamp=timestamp,
        slot_scope=slot_scope,
        resolved_command=resolved_commands,
        validation_results=validation_results,
        warnings=all_warnings,
        environment_redacted=environment_redacted,
    )


def write_dry_run_artifact(
    mode: str,
    slot_payloads: list[DryRunSlotPayload],
    runtime_dir: Any | None = None,
) -> Any:
    """Write dry-run artifact to runtime directory.

    Args:
        mode: The dry-run mode that was executed.
        slot_payloads: List of slot payloads to include.
        runtime_dir: Optional runtime directory path.
            When omitted, uses the default runtime directory.

    Returns:
        Path to the written artifact file.

    Raises:
        Exception: On artifact persistence failure.
    """
    if runtime_dir is None:
        runtime_dir = resolve_runtime_dir()

    artifact_payload = _build_artifact_payload(mode, slot_payloads)
    return write_artifact(
        runtime_dir,
        f"dryrun-{mode}",
        {
            "timestamp": artifact_payload["timestamp"],
            "slot_scope": artifact_payload["slot_scope"],
            "resolved_command": artifact_payload["resolved_command"],
            "validation_results": artifact_payload["validation_results"],
            "warnings": artifact_payload["warnings"],
            "environment_redacted": artifact_payload["environment_redacted"],
        },
    )
