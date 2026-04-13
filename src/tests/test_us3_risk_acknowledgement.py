from unittest.mock import MagicMock

import pytest


def test_privileged_port_requires_acknowledgement():
    """Privileged ports (< 1024) must be flagged as risky."""
    try:
        from llama_manager.server import detect_risky_operations
    except ImportError:
        pytest.fail("detect_risky_operations not implemented in llama_manager.server")

    cfg = MagicMock()

    cfg.port = 80
    cfg.bind_address = "127.0.0.1"

    risks = detect_risky_operations(cfg)
    assert "privileged_port" in risks


def test_non_loopback_bind_requires_acknowledgement():
    """Binding to 0.0.0.0 must be flagged as risky."""
    try:
        from llama_manager.server import detect_risky_operations
    except ImportError:
        pytest.fail("detect_risky_operations not implemented in llama_manager.server")

    cfg = MagicMock()
    cfg.port = 8080
    cfg.bind_address = "0.0.0.0"

    risks = detect_risky_operations(cfg)
    assert "non_loopback" in risks


def test_combined_risks():
    """Multiple risky operations should all be detected."""
    try:
        from llama_manager.server import detect_risky_operations
    except ImportError:
        pytest.fail("detect_risky_operations not implemented in llama_manager.server")

    cfg = MagicMock()
    cfg.port = 80
    cfg.bind_address = "0.0.0.0"

    risks = detect_risky_operations(cfg)
    assert "privileged_port" in risks
    assert "non_loopback" in risks
