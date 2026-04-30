import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from llama_cli import server_runner
from llama_cli.cli_parser import parse_args, parse_tui_args
from llama_cli.commands.dry_run import dry_run
from llama_cli.tui import TUIApp
from llama_manager import LaunchResult, ServerConfig, ServerManager
from llama_manager.config import ErrorCode, ErrorDetail, MultiValidationError


class _FakeTextualDashboardApp:
    """Fake Textual app for TUI tests without terminal rendering."""

    def __init__(self, controller: TUIApp) -> None:
        self.controller = controller

    def run(self) -> None:
        self.controller.render()


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


def test_parse_args_tui_supports_acknowledge_risky_flag() -> None:
    parsed = parse_args(["tui", "summary-balanced", "--acknowledge-risky"])
    assert parsed.mode == "tui"
    assert parsed.tui_mode == "summary-balanced"
    assert parsed.acknowledge_risky is True


def test_parse_tui_args_supports_acknowledge_risky_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-runner", "summary-balanced", "--acknowledge-risky"],
    )
    parsed = parse_tui_args()
    assert parsed.mode == "summary-balanced"
    assert parsed.acknowledge_risky is True


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

    with (
        patch("llama_cli.tui.controller.TextualDashboardApp", _FakeTextualDashboardApp),
        patch("llama_cli.tui.components.panels.psutil.pid_exists", return_value=True),
        patch.object(
            app.server_manager,
            "launch_all_slots",
            return_value=LaunchResult(status="success", launched=["summary-balanced"]),
        ),
        patch.object(
            app.server_manager,
            "start_servers",
            side_effect=lambda configs, log_handlers=None: [MagicMock() for _ in configs],
        ),
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


def test_dry_run_invalid_mode_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        dry_run("invalid-mode")

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "invalid mode" in captured.err
    assert "Valid modes:" in captured.err


def test_dry_run_exits_when_backend_validation_fails() -> None:
    backend_error = ErrorDetail(
        error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
        failed_check="vllm_launch_eligibility",
        why_blocked="backend blocked",
        how_to_fix="switch backend",
    )

    with (
        patch("llama_cli.commands.dry_run.validate_server_config", return_value=backend_error),
        pytest.raises(SystemExit) as exc,
    ):
        dry_run("summary-fast")

    assert exc.value.code == 1


def test_tui_run_exits_when_launch_is_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    app = TUIApp([_risky_cfg()], [0])
    app.running = False

    blocked_error = ErrorDetail(
        error_code=ErrorCode.PORT_CONFLICT,
        failed_check="lockfile_creation",
        why_blocked="blocked",
        how_to_fix="fix",
    )

    with (
        patch("llama_cli.tui.controller.TextualDashboardApp", _FakeTextualDashboardApp),
        patch.object(
            app.server_manager,
            "launch_all_slots",
            return_value=LaunchResult(
                status="blocked",
                launched=[],
                errors=MultiValidationError(errors=[blocked_error]),
            ),
        ),
        patch.object(app.server_manager, "start_servers"),
        patch("builtins.input", return_value="y"),
        pytest.raises(SystemExit) as exc,
    ):
        app.run(acknowledged=False)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "error: launch blocked - no slots could be launched" in captured.err


def test_tui_run_prints_degraded_warnings(capsys: pytest.CaptureFixture[str]) -> None:
    app = TUIApp([_risky_cfg()], [0])
    app.running = False

    with (
        patch("llama_cli.tui.controller.TextualDashboardApp", _FakeTextualDashboardApp),
        patch("llama_cli.tui.components.panels.psutil.pid_exists", return_value=True),
        patch.object(
            app.server_manager,
            "launch_all_slots",
            return_value=LaunchResult(
                status="degraded",
                launched=["summary-balanced"],
                warnings=["slot blocked"],
            ),
        ),
        patch.object(
            app.server_manager,
            "start_servers",
            side_effect=lambda configs, log_handlers=None: [MagicMock() for _ in configs],
        ),
        patch.object(app.server_manager, "cleanup_servers"),
    ):
        app.run(acknowledged=True)

    captured = capsys.readouterr()
    assert "warning: launch degraded - some slots blocked" in captured.err
    assert "warning: slot blocked" in captured.err


def test_server_runner_main_dispatches_dry_run_mode() -> None:
    parsed = Namespace(
        mode="dry-run",
        dry_run_mode="both",
        ports=[8080, 8081],
        acknowledge_risky=True,
    )
    with (
        patch("llama_cli.server_runner.parse_args", return_value=parsed),
        patch("llama_cli.colors.Colors.is_enabled"),
        patch("llama_cli.server_runner._run_dry_run_mode", return_value=0) as mock_run,
    ):
        code = server_runner.main(["dry-run", "both", "8080", "8081"])

    assert code == 0
    mock_run.assert_called_once_with(parsed, True)


def test_server_runner_main_dry_run_without_target_mode_returns_one() -> None:
    parsed = Namespace(mode="dry-run", dry_run_mode=None, ports=[], acknowledge_risky=False)
    code = server_runner._run_dry_run_mode(parsed, acknowledged=False)
    assert code == 1
