import sys
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from llama_cli import server_runner
from llama_cli.cli_parser import parse_args, parse_tui_args
from llama_cli.dry_run import dry_run
from llama_cli.server_runner import (
    run_both,
    run_qwen35,
    run_summary_balanced,
    run_summary_fast,
    verify_risks,
)
from llama_cli.tui_app import TUIApp
from llama_manager import LaunchResult, ServerConfig, ServerManager
from llama_manager.config import ErrorCode, ErrorDetail, MultiValidationError


class _FakeLive:
    """Fake Rich Live context for TUI tests.

    This class mocks Rich's Live context manager to allow testing
    TUI components without actually rendering to the terminal.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize fake Live context."""
        self.args = args
        self.kwargs = kwargs

    def __enter__(self) -> "_FakeLive":
        """Enter context manager."""
        return self

    def __exit__(self, *args) -> None:
        """Exit context manager."""
        pass

    def update(self, *args, **kwargs) -> None:
        """Update the live display (no-op for tests)."""
        pass


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

    with (
        patch("llama_cli.tui_app.Live", _FakeLive),
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
        patch("llama_cli.dry_run.validate_server_config", return_value=backend_error),
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
        patch("llama_cli.tui_app.Live", _FakeLive),
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
        patch("llama_cli.tui_app.Live", _FakeLive),
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
        patch("llama_cli.server_runner.check_prereqs"),
        patch("llama_cli.server_runner.ServerManager"),
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


def test_server_runner_main_returns_one_on_value_error() -> None:
    parsed = Namespace(mode="summary-fast", ports=[], acknowledge_risky=False)
    with (
        patch("llama_cli.server_runner.parse_args", return_value=parsed),
        patch("llama_cli.server_runner.check_prereqs"),
        patch("llama_cli.server_runner.ServerManager") as mock_manager_cls,
        patch("llama_cli.colors.Colors.is_enabled"),
        patch("llama_cli.server_runner.verify_risks"),
        patch("llama_cli.server_runner.run_summary_fast", side_effect=ValueError),
    ):
        code = server_runner.main(["summary-fast"])

    assert code == 1
    mock_manager_cls.assert_called_once()


def test_run_summary_balanced_success_calls_foreground_manager() -> None:
    manager = ServerManager()
    with (
        patch("llama_cli.server_runner.require_model", return_value=None),
        patch("llama_cli.server_runner.validate_server_config", return_value=None),
        patch("llama_cli.server_runner.build_server_cmd", return_value=["bin", "--x"]),
        patch.object(manager, "run_server_foreground", return_value=0) as run_fg,
    ):
        code = run_summary_balanced(8080, manager)

    assert code == 0
    run_fg.assert_called_once_with("summary-balanced", ["bin", "--x"])


def test_run_summary_fast_exits_on_backend_validation_error() -> None:
    manager = ServerManager()
    backend_error = ErrorDetail(
        error_code=ErrorCode.BACKEND_NOT_ELIGIBLE,
        failed_check="vllm_launch_eligibility",
        why_blocked="backend blocked",
        how_to_fix="switch backend",
    )
    with (
        patch("llama_cli.server_runner.require_model"),
        patch("llama_cli.server_runner.validate_server_config", return_value=backend_error),
        pytest.raises(SystemExit) as exc,
    ):
        run_summary_fast(8082, manager)

    assert exc.value.code == 1


def test_run_qwen35_success_calls_foreground_manager() -> None:
    manager = ServerManager()
    with (
        patch("llama_cli.server_runner.require_model", return_value=None),
        patch("llama_cli.server_runner.require_executable", return_value=None),
        patch("llama_cli.server_runner.validate_server_config", return_value=None),
        patch("llama_cli.server_runner.build_server_cmd", return_value=["bin", "--y"]),
        patch.object(manager, "run_server_foreground", return_value=0) as run_fg,
    ):
        code = run_qwen35(8081, manager)

    assert code == 0
    run_fg.assert_called_once_with("qwen35-coding", ["bin", "--y"])


def test_run_both_success_starts_waits_and_cleans() -> None:
    manager = ServerManager()
    with (
        patch("llama_cli.server_runner.require_model"),
        patch("llama_cli.server_runner.require_executable"),
        patch("llama_cli.server_runner.validate_slots", return_value=None),
        patch.object(manager, "start_servers") as start_servers,
        patch.object(manager, "wait_for_any", return_value=0),
        patch.object(manager, "cleanup_servers") as cleanup,
    ):
        code = run_both(8080, 8081, manager)

    assert code == 0
    start_servers.assert_called_once()
    cleanup.assert_called_once()
