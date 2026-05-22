"""Risk acknowledgement logic — UI-agnostic.

This module provides pure business logic for evaluating risky server
configurations and resolving user key-press actions.  It does not import
Rich, argparse, or subprocess and must not depend on ``llama_cli``.
"""

from copy import copy
from dataclasses import dataclass, field
from typing import Any

from .config import ServerConfig
from .orchestration import ServerManager

RISK_ACK_LABEL: str = "warning_bypass"


@dataclass
class RiskAckResult:
    """Structured result from risk evaluation.

    Attributes:
        has_risks: Whether any risky operations were detected.
        risks_acknowledged: Whether risks have been pre-acknowledged.
        risk_details: List of unacknowledged risk details.  Each dict
            contains ``alias``, ``risk``, and ``risk_kind`` keys.
    """

    has_risks: bool = False
    risks_acknowledged: bool = False
    risk_details: list[dict[str, Any]] = field(default_factory=list)


def _collect_risky_details(
    cfg: ServerConfig,
    server_manager: ServerManager,
    launch_attempt_id: str,
    acknowledged: bool,
    ack_token: str,
) -> tuple[bool, list[dict[str, Any]]]:
    """Collect unacknowledged risk details for a single config.

    Args:
        cfg: Server configuration to evaluate.
        server_manager: ServerManager for risk acknowledgement tracking.
        launch_attempt_id: Current launch attempt identifier.
        acknowledged: Whether risks have been pre-acknowledged.
        ack_token: Acknowledgement token for this attempt.

    Returns:
        Tuple of ``(has_unacknowledged_risks, risk_detail_entries)``.
    """
    from .validation import detect_risky_operations

    has_unacknowledged = False
    entries: list[dict[str, Any]] = []

    for risk in detect_risky_operations(cfg):
        if server_manager.is_risk_acknowledged(cfg.alias, risk, launch_attempt_id):
            continue

        has_unacknowledged = True
        risk_kind = "vram" if "vram" in risk.lower() else "hardware"
        entries.append(
            {
                "alias": cfg.alias,
                "risk": risk,
                "risk_kind": risk_kind,
            }
        )
        if acknowledged:
            server_manager.acknowledge_risk(
                cfg.alias,
                risk,
                launch_attempt_id=launch_attempt_id,
                ack_token=ack_token,
            )

    return has_unacknowledged, entries


def evaluate_risks(
    configs: list[ServerConfig],
    server_manager: ServerManager,
    launch_attempt_id: str,
    ack_token: str,
    acknowledged: bool,
) -> RiskAckResult:
    """Evaluate risky operations across all configs and manage acknowledgement state.

    Args:
        configs: List of server configurations to evaluate.
        server_manager: ServerManager for risk acknowledgement tracking.
        launch_attempt_id: Current launch attempt identifier.
        ack_token: Acknowledgement token for this attempt.
        acknowledged: Whether risks have been pre-acknowledged
            (e.g. ``--acknowledge-risky``).

    Returns:
        RiskAckResult with ``has_risks``, ``risks_acknowledged``, and
        ``risk_details``.

    Note:
        When ``acknowledged`` is True, a shallow copy of each config is
        returned with ``risky_acknowledged`` updated; the original
        ``ServerConfig`` objects are not mutated.
    """
    has_risks = False
    risk_details: list[dict[str, Any]] = []

    for idx, cfg in enumerate(configs):
        if acknowledged and RISK_ACK_LABEL not in cfg.risky_acknowledged:
            # Make a shallow copy to avoid mutating the original config
            new_ack_list = list(cfg.risky_acknowledged)
            new_ack_list.append(RISK_ACK_LABEL)
            configs[idx] = copy(cfg)
            configs[idx].risky_acknowledged = new_ack_list

        has_risk, details = _collect_risky_details(
            cfg, server_manager, launch_attempt_id, acknowledged, ack_token
        )
        has_risks = has_risks or has_risk
        risk_details.extend(details)

    risks_acknowledged = has_risks and acknowledged

    return RiskAckResult(
        has_risks=has_risks,
        risks_acknowledged=risks_acknowledged,
        risk_details=risk_details,
    )


def resolve_risk_action(key: str, risk_kind: str | None) -> str:
    """Resolve a key press into a risk action for the given risk kind.

    Args:
        key: The key pressed by the user.
        risk_kind: The kind of active risk (``"vram"`` or ``"hardware"``).

    Returns:
        Action string: ``"acknowledge"``, ``"proceed"``, ``"abort"``,
        ``"quit"``, or ``"ignore"``.
    """
    key = key.lower()
    if key == "y":
        return "proceed" if risk_kind == "vram" else "acknowledge"
    if key == "n":
        return "abort"
    if key == "q":
        return "quit"
    return "ignore"
