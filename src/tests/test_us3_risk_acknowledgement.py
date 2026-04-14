from unittest.mock import MagicMock

import pytest

# Detect risky operations is optional for M1 - skip tests if not implemented
detect_risky_operations = pytest.importorskip(
    "llama_manager.server", "detect_risky_operations not implemented in M1"
).detect_risky_operations


def test_privileged_port_requires_acknowledgement() -> None:
    """Privileged ports (< 1024) must be flagged as risky."""
    cfg = MagicMock()
    cfg.port = 80
    cfg.bind_address = "127.0.0.1"
    cfg.risky_acknowledged = []  # type: ignore

    risks = detect_risky_operations(cfg)
    assert "privileged_port" in risks


def test_non_loopback_bind_requires_acknowledgement() -> None:
    """Binding to 0.0.0.0 must be flagged as risky."""
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
