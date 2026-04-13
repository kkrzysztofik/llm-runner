import sys
from unittest.mock import patch

import pytest

from llama_cli.cli_parser import parse_args, parse_tui_args
from llama_cli.server_runner import verify_risks
from llama_cli.tui_app import TUIApp
from llama_manager import ServerConfig, ServerManager


def _risky_cfg() -> ServerConfig:
    return ServerConfig(
        model="/home/kmk/models/test-model.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=80,
        ctx_size=2048,
        ubatch_size=512,
        threads=4,
    )


def test_parse_args_supports_acknowledge_risky_flag() -> None:
    parsed = parse_args(["summary-balanced", "8080", "--acknowledge-risky"])
    assert parsed.mode == "summary-balanced"
    assert parsed.acknowledge_risky is True


def test_parse_tui_args_supports_acknowledge_risky_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_models_tui.py", "summary-balanced", "--acknowledge-risky"],
    )
    parsed = parse_tui_args()
    assert parsed.mode == "summary-balanced"
    assert parsed.acknowledge_risky is True


def test_verify_risks_prompts_and_exits_with_actionable_acknowledgement_required(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = ServerManager()
    cfg = _risky_cfg()

    with patch("builtins.input", return_value="n") as mock_input, pytest.raises(SystemExit) as exc:
        verify_risks(manager, [cfg], acknowledged=False)

    assert exc.value.code == 1
    mock_input.assert_called_once_with("Confirm risky operation [y/N]: ")
    captured = capsys.readouterr()
    assert "error: acknowledgement_required" in captured.err
    assert "failed_check: acknowledgement_required" in captured.err
    assert "why_blocked: risky operation detected and not acknowledged" in captured.err
    assert "how_to_fix: use --acknowledge-risky flag or confirm with 'y'" in captured.err


def test_verify_risks_acknowledges_without_prompt_when_flag_is_set() -> None:
    manager = ServerManager()
    cfg = _risky_cfg()

    with patch("builtins.input") as mock_input:
        verify_risks(manager, [cfg], acknowledged=True)

    mock_input.assert_not_called()
    assert manager.is_risk_acknowledged(cfg.alias, "privileged_port")


def test_tui_risk_panels_render_required_and_acknowledged_states() -> None:
    app = TUIApp([_risky_cfg()], [0])

    app._build_risk_panel_required()
    required_layout = app.render()
    assert required_layout is not None
    assert app.risk_panel is not None
    assert "ACKNOWLEDGEMENT REQUIRED" in str(app.risk_panel.renderable)
    assert app.risks_acknowledged is False

    app._build_risk_panel_acknowledged()
    acknowledged_layout = app.render()
    assert acknowledged_layout is not None
    assert app.risk_panel is not None
    assert "ACKNOWLEDGED" in str(app.risk_panel.renderable)
    assert app.risks_acknowledged is True
