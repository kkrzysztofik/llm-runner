from unittest.mock import MagicMock

import pytest

from llama_manager.risk_ack import (
    RISK_ACK_LABEL,
    evaluate_risks,
    resolve_risk_action,
)

_ACK_TOKEN = "ack:attempt-1"  # noqa: S105

# Detect risky operations is optional for M1 - skip tests if not implemented
try:
    server_module = pytest.importorskip(
        "llama_manager.server", reason="detect_risky_operations not implemented in M1"
    )
    detect_risky_operations = server_module.detect_risky_operations
except AttributeError:
    pytest.skip(
        "detect_risky_operations attribute not found in llama_manager.server",
        allow_module_level=True,
    )


def test_privileged_port_requires_acknowledgement() -> None:
    """Privileged ports (< 1024) must be flagged as risky."""
    cfg = MagicMock()
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "privileged_port" in risks


def test_non_loopback_bind_requires_acknowledgement() -> None:
    """Binding to non-loopback address must be flagged as risky."""
    cfg = MagicMock()
    cfg.port = 8080
    cfg.bind_address = "192.168.1.1"
    cfg.risky_acknowledged = []  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "non_loopback" in risks


def test_combined_risks() -> None:
    """Multiple risky operations should all be detected."""
    cfg = MagicMock()
    cfg.port = 80
    cfg.bind_address = "192.168.1.1"
    cfg.risky_acknowledged = []  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "privileged_port" in risks
    assert "non_loopback" in risks


def test_warning_bypass_risk_class_detected() -> None:
    """warning_bypass marker should be reported as a risk class."""
    cfg = MagicMock()
    cfg.port = 8080
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = ["warning_bypass"]  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "warning_bypass" in risks


# =============================================================================
# evaluate_risks
# =============================================================================


def test_evaluate_risks_no_risks() -> None:
    """No risks detected → has_risks=False, empty details."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 8080
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    result = evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is False
    assert result.risks_acknowledged is False
    assert result.risk_details == []
    sm.acknowledge_risk.assert_not_called()


def test_evaluate_risks_detects_privileged_port() -> None:
    """Privileged port is detected and reported in risk_details."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    result = evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is True
    assert result.risks_acknowledged is False
    assert len(result.risk_details) == 1
    assert result.risk_details[0]["alias"] == "test"
    assert result.risk_details[0]["risk"] == "privileged_port"
    assert result.risk_details[0]["risk_kind"] == "hardware"
    sm.acknowledge_risk.assert_called_once_with(
        "test",
        "privileged_port",
        launch_attempt_id="attempt-1",
        ack_token=_ACK_TOKEN,
    )


def test_evaluate_risks_skips_already_acknowledged() -> None:
    """Risks already acknowledged in server_manager are skipped."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = True

    result = evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is True
    assert result.risk_details == []
    sm.acknowledge_risk.assert_not_called()


def test_evaluate_risks_pre_acknowledged_flag() -> None:
    """acknowledged=True sets risks_acknowledged and mutates config."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    result = evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=True)

    assert result.has_risks is True
    assert result.risks_acknowledged is True
    assert RISK_ACK_LABEL in cfg.risky_acknowledged


def test_evaluate_risks_does_not_double_append_label() -> None:
    """acknowledged=True does not double-append RISK_ACK_LABEL."""
    cfg = MagicMock()
    cfg.alias = "test"
    cfg.port = 8080
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = [RISK_ACK_LABEL]

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    evaluate_risks([cfg], sm, "attempt-1", _ACK_TOKEN, acknowledged=True)

    assert cfg.risky_acknowledged.count(RISK_ACK_LABEL) == 1


def test_evaluate_risks_multiple_configs() -> None:
    """Risks across multiple configs are aggregated."""
    cfg1 = MagicMock()
    cfg1.alias = "a"
    cfg1.port = 80
    cfg1.bind_address = "127.0.0.1"
    cfg1.risky_acknowledged = []

    cfg2 = MagicMock()
    cfg2.alias = "b"
    cfg2.port = 8080
    cfg2.bind_address = "0.0.0.0"
    cfg2.risky_acknowledged = []

    sm = MagicMock()
    sm.is_risk_acknowledged.return_value = False

    result = evaluate_risks([cfg1, cfg2], sm, "attempt-1", _ACK_TOKEN, acknowledged=False)

    assert result.has_risks is True
    assert len(result.risk_details) == 2
    aliases = {d["alias"] for d in result.risk_details}
    assert aliases == {"a", "b"}
    assert sm.acknowledge_risk.call_count == 2


# =============================================================================
# resolve_risk_action
# =============================================================================


def test_resolve_risk_action_y_hardware() -> None:
    assert resolve_risk_action("y", "hardware") == "acknowledge"


def test_resolve_risk_action_y_vram() -> None:
    assert resolve_risk_action("y", "vram") == "proceed"


def test_resolve_risk_action_n() -> None:
    assert resolve_risk_action("n", "hardware") == "abort"
    assert resolve_risk_action("n", "vram") == "abort"


def test_resolve_risk_action_q() -> None:
    assert resolve_risk_action("q", "hardware") == "quit"
    assert resolve_risk_action("q", "vram") == "quit"


def test_resolve_risk_action_unknown() -> None:
    assert resolve_risk_action("x", "hardware") == "ignore"
    assert resolve_risk_action("x", "vram") == "ignore"
    assert resolve_risk_action("", None) == "ignore"
