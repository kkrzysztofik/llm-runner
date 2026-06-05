"""Launch orchestration — error builders, risk evaluation, and launch_orchestrate."""

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ..config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    SlotState,
    ValidationException,
    apply_profile_overrides,
)
from ..log_buffer import LogBuffer
from .lockfile import LOCKFILE_CHECK_NAME
from .types import ARTIFACT_CHECK_NAME, LaunchOrchestrationResult, LaunchResult

if TYPE_CHECKING:
    from ..risk_ack import RiskAckResult
    from .manager import ServerManager

logger = logging.getLogger(__name__)


def _make_validation_error(
    error_code: ErrorCode,
    failed_check: str,
    why_blocked: str,
    how_to_fix: str,
) -> ValidationException:
    """Build a single-error ValidationException from raw fields."""
    detail = ErrorDetail(
        error_code=error_code,
        failed_check=failed_check,
        why_blocked=why_blocked,
        how_to_fix=how_to_fix,
    )
    return ValidationException(MultiValidationError(errors=[detail]))


def _lockfile_error(why_blocked: str, how_to_fix: str) -> ValidationException:
    """Build a lockfile-integrity ValidationException."""
    return _make_validation_error(
        ErrorCode.LOCKFILE_INTEGRITY_FAILURE, LOCKFILE_CHECK_NAME, why_blocked, how_to_fix
    )


def _artifact_error(
    why_blocked: str, how_to_fix: str, check: str = ARTIFACT_CHECK_NAME
) -> ValidationException:
    """Build an artifact-persistence ValidationException."""
    return _make_validation_error(
        ErrorCode.ARTIFACT_PERSISTENCE_FAILURE, check, why_blocked, how_to_fix
    )


def _evaluate_and_handle_risks(
    updated_configs: list[ServerConfig],
    server_manager: ServerManager,
    launch_attempt_id: str,
    ack_token: str,
    acknowledged: bool,
    profile_messages: list[str],
    risk_result: RiskAckResult | None = None,
) -> tuple[list[str], list[ServerConfig] | None, RiskAckResult | None]:
    """Evaluate risks and return (status_messages, configs_to_launch, risk_result).

    If risks are unacknowledged, returns early with status messages and no configs.
    Otherwise returns the full configs list and the risk_result.
    """
    from ..risk_ack import evaluate_risks

    if risk_result is None:
        risk_result = evaluate_risks(
            updated_configs,
            server_manager,
            launch_attempt_id,
            ack_token,
            acknowledged,
        )

    if risk_result.has_risks and not risk_result.risks_acknowledged and risk_result.risk_details:
        status_messages: list[str] = list(profile_messages)
        status_messages.append(
            "Launch blocked: unacknowledged risks detected. "
            f"Details: {len(risk_result.risk_details)} risk(s) require"
            " acknowledgement.",
        )
        return status_messages, None, risk_result

    return profile_messages, updated_configs, risk_result


def _build_launch_status_messages(
    launch_result: LaunchResult,
    profile_messages: list[str],
) -> list[str]:
    """Build status messages based on launch result (blocked/degraded).

    Returns a new list starting with profile_messages plus any launch-specific messages.
    """
    status_messages: list[str] = list(profile_messages)

    if launch_result.is_blocked():
        status_messages.append("Launch blocked: no slots could be launched")
        if launch_result.errors is not None:
            for error_detail in launch_result.errors.errors:
                status_messages.append(f"  {error_detail.error_code} - {error_detail.why_blocked}")
        return status_messages

    if launch_result.is_degraded():
        status_messages.append("Launch degraded: some slots blocked")
        for warning in launch_result.warnings or []:
            status_messages.append(f"  warning: {warning}")

    return status_messages


def _build_launch_only_messages(launch_result: LaunchResult) -> list[str]:
    """Build launch-specific status messages (excludes profile_messages)."""
    messages: list[str] = []

    if launch_result.is_blocked():
        messages.append("Launch blocked: no slots could be launched")
        if launch_result.errors is not None:
            for error_detail in launch_result.errors.errors:
                messages.append(f"  {error_detail.error_code} - {error_detail.why_blocked}")
        return messages

    if launch_result.is_degraded():
        messages.append("Launch degraded: some slots blocked")
        for warning in launch_result.warnings or []:
            messages.append(f"  warning: {warning}")

    return messages


def _build_log_handlers(
    launched_configs: list[ServerConfig],
    log_buffers: Mapping[str, LogBuffer],
    launched_set: list[str],
) -> dict[str, Callable[[str], None]]:
    """Build log handlers for launched configs."""
    launched_set_s = set(launched_set)
    launched_log_buffers = {
        alias: buf for alias, buf in log_buffers.items() if alias in launched_set_s
    }

    log_handlers: dict[str, Callable[[str], None]] = {}
    for cfg in launched_configs:
        buf = launched_log_buffers.get(cfg.alias)
        if buf is not None:
            log_handlers[cfg.alias] = lambda line, b=buf: b.add_line(line)

    return log_handlers


def _start_and_map_servers(
    launched_configs: list[ServerConfig],
    log_handlers: dict[str, Callable[[str], None]],
    server_manager: ServerManager,
) -> dict[str, Any]:
    """Start servers and map processes by alias."""
    processes: dict[str, Any] = {}
    try:
        processes_list = server_manager.start_servers(launched_configs, log_handlers)
    except Exception:
        server_manager.cleanup_servers()
        raise

    for cfg, proc in zip(launched_configs, processes_list, strict=True):
        processes[cfg.alias] = proc

    return processes


def _empty_result() -> LaunchOrchestrationResult:
    """Return an empty orchestration result for no-configs."""
    from ..risk_ack import RiskAckResult

    return LaunchOrchestrationResult(
        updated_configs=[],
        launch_result=LaunchResult(status="success", launched=[]),
        processes={},
        slot_states={},
        status_messages=["No slots configured. Press 'a' to add a slot."],
        risk_result=RiskAckResult(),
        empty=True,
    )


def _blocked_result(
    updated_configs: list[ServerConfig],
    launch_result: LaunchResult | None,
    status_messages: list[str],
    risk_result: RiskAckResult | None,
) -> LaunchOrchestrationResult:
    """Return a blocked/degraded orchestration result."""
    return LaunchOrchestrationResult(
        updated_configs=updated_configs,
        launch_result=launch_result,
        processes={},
        slot_states={},
        status_messages=status_messages,
        risk_result=risk_result,
        empty=False,
    )


def _build_slot_states_and_messages(
    launched_configs: list[ServerConfig],
    status_messages: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Build slot states and append transition messages."""
    from ..slot_state import compute_slot_transition

    slot_states: dict[str, str] = {}
    for cfg in launched_configs:
        transition = compute_slot_transition(cfg.alias, None, SlotState.RUNNING)
        slot_states[cfg.alias] = SlotState.RUNNING.value
        if transition is not None:
            message, _color = transition
            status_messages.append(message)
    return slot_states, status_messages


def launch_orchestrate(
    configs: list[ServerConfig],
    base_config: Config,
    server_manager: ServerManager,
    log_buffers: Mapping[str, LogBuffer],
    get_driver_version: Callable[[str], str],
    acknowledged: bool = False,
) -> LaunchOrchestrationResult:
    """Orchestrate the full launch sequence for model slots."""
    logger.info("launch_orchestrate: %d config(s)", len(configs))
    updated_configs, profile_messages = apply_profile_overrides(
        configs, base_config, get_driver_version
    )

    if not updated_configs:
        return _empty_result()

    slots = [
        ModelSlot(slot_id=cfg.alias, model_path=cfg.model, port=cfg.port) for cfg in updated_configs
    ]

    launch_attempt_id = server_manager.begin_launch_attempt()
    ack_token = server_manager.issue_ack_token(launch_attempt_id)

    status_messages, configs_to_launch, risk_result = _evaluate_and_handle_risks(
        updated_configs,
        server_manager,
        launch_attempt_id,
        ack_token,
        acknowledged,
        profile_messages,
    )
    if not configs_to_launch:
        return _blocked_result(updated_configs, None, status_messages, risk_result)

    launch_result = server_manager.launch_all_slots(slots, configs=updated_configs)

    status_messages.extend(_build_launch_only_messages(launch_result))

    if launch_result.is_blocked():
        return _blocked_result(updated_configs, launch_result, status_messages, risk_result)

    launched_slots = launch_result.launched or []
    launched_set = set(launched_slots)

    launched_configs = [cfg for cfg in updated_configs if cfg.alias in launched_set]

    log_handlers = _build_log_handlers(launched_configs, log_buffers, launched_slots)
    processes = _start_and_map_servers(launched_configs, log_handlers, server_manager)

    slot_states, status_messages = _build_slot_states_and_messages(
        launched_configs, status_messages
    )

    return LaunchOrchestrationResult(
        updated_configs=updated_configs,
        launch_result=launch_result,
        processes=processes,
        slot_states=slot_states,
        status_messages=status_messages,
        risk_result=risk_result,
        empty=False,
    )
