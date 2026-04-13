import sys
from unittest.mock import patch

import pytest

from llama_cli.cli_parser import parse_args, parse_tui_args
from llama_cli.dry_run import dry_run
from llama_cli.server_runner import verify_risks
from llama_cli.tui_app import TUIApp
from llama_manager import LaunchResult, ServerConfig, ServerManager


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
    assert manager.is_risk_acknowledged(
        cfg.alias,
        "privileged_port",
        manager._current_launch_attempt_id,
    )


def test_verify_risks_uses_attempt_scoped_ack_token_for_prompted_confirmation() -> None:
    manager = ServerManager()
    cfg = _risky_cfg()

    with (
        patch("builtins.input", return_value="y") as mock_input,
        patch.object(manager, "acknowledge_risk", wraps=manager.acknowledge_risk) as mock_ack,
    ):
        verify_risks(manager, [cfg], acknowledged=False)

    mock_input.assert_called_once_with("Confirm risky operation [y/N]: ")
    assert mock_ack.call_count == 1
    call_kwargs = mock_ack.call_args.kwargs
    assert call_kwargs["launch_attempt_id"] == manager._current_launch_attempt_id
    assert call_kwargs["ack_token"] == f"ack:{manager._current_launch_attempt_id}"


def test_ack_token_validation_is_attempt_scoped() -> None:
    manager = ServerManager()
    attempt_id = manager.begin_launch_attempt("attempt-1")
    valid_token = manager.issue_ack_token(attempt_id)

    assert manager.validate_ack_token(attempt_id, valid_token) is True
    assert manager.validate_ack_token(attempt_id, "ack:other") is False


def test_cleanup_clears_attempt_ack_cache() -> None:
    manager = ServerManager()
    attempt_id = manager.begin_launch_attempt("attempt-1")
    manager.acknowledge_risk("summary-balanced", "privileged_port", attempt_id)
    assert manager.is_risk_acknowledged("summary-balanced", "privileged_port", attempt_id)

    manager.cleanup_servers()

    assert manager.is_risk_acknowledged("summary-balanced", "privileged_port", attempt_id) is False


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


def test_tui_run_keeps_acknowledged_risk_panel_visible() -> None:
    app = TUIApp([_risky_cfg()], [0])
    app.running = False

    class _FakeLive:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def update(self, *_args, **_kwargs) -> None:
            return None

    with (
        patch("llama_cli.tui_app.Live", _FakeLive),
        patch.object(
            app.server_manager,
            "launch_all_slots",
            return_value=LaunchResult(status="success", launched=["summary-balanced"]),
        ),
        patch.object(app.server_manager, "start_servers"),
        patch.object(app.server_manager, "cleanup_servers"),
    ):
        app.run(acknowledged=True)

    assert app.risk_panel is not None
    assert "ACKNOWLEDGED" in str(app.risk_panel.renderable)
    assert app.risks_acknowledged is True


def test_dry_run_prompts_for_risky_operation_with_exact_prompt() -> None:
    with patch("builtins.input", return_value="n") as mock_input, pytest.raises(SystemExit) as exc:
        dry_run("summary-balanced", primary_port="80")

    assert exc.value.code == 1
    mock_input.assert_called_once_with("Confirm risky operation [y/N]: ")
