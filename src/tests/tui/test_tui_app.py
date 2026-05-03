"""Tests for TUI application module (tui_app.py).

Covers:
- TUIApp initialization
- Layout building and rendering
- Build pipeline integration
- Risk acknowledgment
- Status panel building
"""

import signal
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.tui import DashboardController
from llama_cli.tui.textual_app import DashboardApp
from llama_manager.build_pipeline import BuildProgress, BuildResult
from llama_manager.config import ServerConfig
from tests.support.factories import make_server_config

TUIApp = DashboardController
TextualDashboardApp = DashboardApp


def _make_config(
    alias: str = "test",
    port: int = 8080,
    device: str = "CUDA",
) -> ServerConfig:
    """Helper to create a ServerConfig for tests."""
    return make_server_config(
        model="/path/to/model.gguf",
        alias=alias,
        device=device,
        port=port,
    )


# =============================================================================
# TUIApp initialization
# =============================================================================


class TestTUIAppInit:
    """Tests for TUIApp.__init__."""

    def test_init_basic(self) -> None:
        """TUIApp should initialize with basic config."""
        configs = [_make_config()]
        app = TUIApp(configs=configs, gpu_indices=[0])

        assert len(app.configs) == 1
        assert app.gpu_indices == [0]
        assert app.running is True
        assert app.launch_result is None
        assert app.risk_panel is None
        assert app.risks_acknowledged is False

    def test_init_multiple_configs(self) -> None:
        """TUIApp should create log buffers for all configs."""
        configs = [_make_config("model1", 8080, "CUDA"), _make_config("model2", 8081, "SYCL")]
        app = TUIApp(configs=configs, gpu_indices=[0, 1])

        assert len(app.log_buffers) == 2
        assert "model1" in app.log_buffers
        assert "model2" in app.log_buffers

    def test_init_no_slots(self) -> None:
        """TUIApp should initialize with empty slots when not provided."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        assert app.slots == []

    def test_init_with_slots(self) -> None:
        """TUIApp should accept slots parameter."""
        from llama_manager import ModelSlot

        slots = [ModelSlot(slot_id="test", model_path="/path/to/model.gguf", port=8080)]
        app = TUIApp(configs=[_make_config()], gpu_indices=[0], slots=slots)
        assert len(app.slots) == 1


class TestTextualDashboardAppActions:
    """Tests for TextualDashboardApp action delegation."""

    def test_actions_delegate_to_controller_methods(self) -> None:
        controller = MagicMock()
        controller.running = True
        controller.config = MagicMock()
        controller.configs = [_make_config()]
        app = TextualDashboardApp(controller)

        with patch.object(app, "refresh_dashboard") as mock_refresh:
            app.action_profile()
            app.action_build()
            app.action_smoke()
            app.action_confirm()
            app.action_reject()
            app.action_cancel_pending_prompt()
            app.action_refresh_dashboard()
            app.action_interrupt_dashboard()

        controller.request_profile.assert_called_once()
        controller.request_build.assert_called_once()
        controller.request_smoke.assert_called_once()
        controller.acknowledge_risk.assert_called_once()
        controller.reject_risk.assert_called_once()
        controller.cancel_pending_prompt.assert_called_once()
        controller.refresh_display.assert_called_once()
        controller.interrupt.assert_called_once()
        assert mock_refresh.call_count == 7


# =============================================================================
# stop
# =============================================================================


class TestTUIAppStop:
    """Tests for TUIApp.stop."""

    def test_stop_sets_running_false(self) -> None:
        """stop should set running to False."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.running = True
        app.stop()
        assert app.running is False


# =============================================================================
# render
# =============================================================================


class TestTUIAppRender:
    """Tests for TUIApp.render."""

    def test_render_no_configs(self) -> None:
        """render should handle no configs gracefully."""
        app = TUIApp(configs=[], gpu_indices=[])
        layout = app.render_panels()
        assert layout is not None

    def test_render_single_config(self) -> None:
        """render should render single config with placeholder."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        layout = app.render_panels()
        assert layout is not None

    def test_render_two_configs(self) -> None:
        """render should render both configs side by side."""
        configs = [_make_config("model1", 8080, "CUDA"), _make_config("model2", 8081, "SYCL")]
        app = TUIApp(configs=configs, gpu_indices=[0, 1])
        layout = app.render_panels()
        assert layout is not None

    def test_render_with_status_panel(self) -> None:
        """render should include status panel in alerts."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.status_panel = MagicMock()
        layout = app.render_panels()
        assert layout is not None

    def test_render_populates_menu(self) -> None:
        """render should populate the dashboard snapshot menu."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        snapshot = app.render_panels()
        assert snapshot[3] is not None


# =============================================================================
# _build_status_panel
# =============================================================================


class TestBuildStatusPanel:
    """Tests for TUIApp._build_status_panel."""

    def test_status_panel_success_clears_panel(self) -> None:
        """_build_status_panel should clear status panel on success."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(status="success")
        app._build_status_panel(launch_result)
        assert app.status_panel is None

    def test_status_panel_blocked(self) -> None:
        """_build_status_panel should create blocked panel."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(
            status="blocked",
            errors=MagicMock(errors=[]),
        )
        app._build_status_panel(launch_result)
        assert app.status_panel is not None

    def test_status_panel_degraded(self) -> None:
        """_build_status_panel should create degraded panel."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(
            status="degraded",
            launched=["slot1"],
            warnings=["partial success"],
        )
        app._build_status_panel(launch_result)
        assert app.status_panel is not None


# =============================================================================
# Risk panels
# =============================================================================


class TestRiskPanels:
    """Tests for risk panel building."""

    def test_build_risk_panel_required(self) -> None:
        """_build_risk_panel_required should set risk_panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_risk_panel_required()
        assert app.risk_panel is not None
        assert app.risks_acknowledged is False

    def test_build_risk_panel_acknowledged(self) -> None:
        """_build_risk_panel_acknowledged should set risk_panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_risk_panel_acknowledged()
        assert app.risk_panel is not None
        assert app.risks_acknowledged is True

    def test_update_risk_panel_state_with_risks(self) -> None:
        """_update_risk_panel_state should set required panel when risks not acknowledged."""
        from llama_manager import RiskAckResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        result = RiskAckResult(has_risks=True, risks_acknowledged=False)
        app._update_risk_panel_state(result)
        assert app.risks_acknowledged is False

    def test_update_risk_panel_state_with_acknowledged_risks(self) -> None:
        """_update_risk_panel_state should set acknowledged panel when risks acknowledged."""
        from llama_manager import RiskAckResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        result = RiskAckResult(has_risks=True, risks_acknowledged=True)
        app._update_risk_panel_state(result)
        assert app.risks_acknowledged is True

    def test_update_risk_panel_state_without_risks(self) -> None:
        """_update_risk_panel_state should clear risk_panel when no risks."""
        from llama_manager import RiskAckResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.risk_panel = MagicMock()
        result = RiskAckResult(has_risks=False, risks_acknowledged=False)
        app._update_risk_panel_state(result)
        assert app.risk_panel is None
        assert app.risks_acknowledged is False


# =============================================================================
# _handle_build_progress
# =============================================================================


class TestHandleBuildProgress:
    """Tests for TUIApp._handle_build_progress."""

    def test_handle_progress_retry(self) -> None:
        """_handle_build_progress should create retrying status panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="retrying",
            message="Retrying...",
            progress_percent=50,
            retries_remaining=2,
        )
        app._handle_build_progress(progress)

        assert app.build_progress is progress
        assert app.status_panel is not None

    def test_handle_progress_failure(self) -> None:
        """_handle_build_progress should create failed status panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="failed",
            message="Build failed",
            progress_percent=0,
        )
        app._handle_build_progress(progress)

        assert app.status_panel is not None

    def test_handle_progress_success_clears(self) -> None:
        """_handle_build_progress should clear status panel on success."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.build_in_progress = True
        app.status_panel = MagicMock()

        progress = BuildProgress(
            stage="build",
            status="success",
            message="Build successful",
            progress_percent=100,
        )
        app._handle_build_progress(progress)

        assert app.status_panel is None


# =============================================================================
# _build_placeholder_panel
# =============================================================================


class TestBuildPlaceholderPanel:
    """Tests for TUIApp._build_placeholder_panel."""

    def test_placeholder_panel(self) -> None:
        """_build_placeholder_panel should create a dim placeholder."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        panel = app._build_placeholder_panel()
        assert panel is not None


# =============================================================================
# _handle_launch_result
# =============================================================================


class TestHandleLaunchResult:
    """Tests for TUIApp._handle_launch_result."""

    def test_handle_blocked_result(self, capsys) -> None:
        """_handle_launch_result should raise SystemExit for blocked result."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(status="blocked", errors=MagicMock(errors=[]))

        with pytest.raises(SystemExit):
            app._handle_launch_result(launch_result)

    def test_handle_degraded_result(self, capsys) -> None:
        """_handle_launch_result should print warning for degraded result."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(status="degraded", warnings=["slot1 blocked"])

        # Should not raise, just print warning
        app._handle_launch_result(launch_result)

        captured = capsys.readouterr()
        assert "degraded" in captured.err.lower()


# =============================================================================
# build_llama_cpp
# =============================================================================


class TestBuildLlamaCpp:
    """Tests for TUIApp.build_llama_cpp."""

    def _make_mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock Config with build attributes."""
        mock_config = MagicMock()
        mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
        mock_config.builds_dir = tmp_path / "output"
        mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
        mock_config.build_git_branch = "main"
        mock_config.build_retry_attempts = 2
        mock_config.build_retry_delay = 5
        return mock_config

    def test_build_llama_cpp_success(self, tmp_path: Path) -> None:
        """build_llama_cpp should return True on success."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config_cls.return_value = self._make_mock_config(tmp_path)

            with (
                patch("llama_cli.tui.controller.signal.signal") as mock_signal,
                patch("llama_cli.tui.controller.run_build_for_backend") as mock_run_build,
            ):
                mock_run_build.return_value = BuildResult(success=True)

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                result = app.build_llama_cpp(backend="sycl", dry_run=False)

                assert result is True
                assert app.build_in_progress is False
                mock_signal.assert_called()

    def test_build_llama_cpp_failure(self, tmp_path: Path) -> None:
        """build_llama_cpp should return False on failure."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config_cls.return_value = self._make_mock_config(tmp_path)

            with (
                patch("llama_cli.tui.controller.signal.signal"),
                patch("llama_cli.tui.controller.run_build_for_backend") as mock_run_build,
            ):
                mock_run_build.return_value = BuildResult(success=False, error_message="failed")

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                result = app.build_llama_cpp(backend="sycl", dry_run=False)

                assert result is False
                assert app.build_in_progress is False

    def test_build_llama_cpp_dry_run(self, tmp_path: Path) -> None:
        """build_llama_cpp should pass dry_run to orchestration."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config_cls.return_value = self._make_mock_config(tmp_path)

            with (
                patch("llama_cli.tui.controller.signal.signal"),
                patch("llama_cli.tui.controller.run_build_for_backend") as mock_run_build,
            ):
                mock_run_build.return_value = BuildResult(success=True)

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                app.build_llama_cpp(backend="sycl", dry_run=True)

                assert mock_run_build.call_args.kwargs["dry_run"] is True

    def test_build_llama_cpp_cuda_backend(self, tmp_path: Path) -> None:
        """build_llama_cpp should pass cuda backend to orchestration."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config_cls.return_value = self._make_mock_config(tmp_path)

            with (
                patch("llama_cli.tui.controller.signal.signal"),
                patch("llama_cli.tui.controller.run_build_for_backend") as mock_run_build,
            ):
                mock_run_build.return_value = BuildResult(success=True)

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                app.build_llama_cpp(backend="cuda", dry_run=False)

                assert mock_run_build.call_args.kwargs["backend"] == "cuda"

    def test_build_llama_cpp_restores_sigint(self, tmp_path: Path) -> None:
        """build_llama_cpp should restore the original SIGINT handler."""
        original_handler = object()

        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config_cls.return_value = self._make_mock_config(tmp_path)

            with (
                patch(
                    "llama_cli.tui.controller.signal.signal",
                    return_value=original_handler,
                ) as mock_signal,
                patch("llama_cli.tui.controller.run_build_for_backend") as mock_run_build,
            ):
                mock_run_build.return_value = BuildResult(success=True)

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                app.build_llama_cpp(backend="sycl", dry_run=False)

                # signal.signal is also called in TUIApp.__init__
                sigint_calls = [c for c in mock_signal.call_args_list if c[0][0] == signal.SIGINT]
                # Last SIGINT call should restore the original handler
                assert sigint_calls[-1][0] == (signal.SIGINT, original_handler)


# =============================================================================
# _signal_handler
# =============================================================================


class TestSignalHandler:
    """Tests for signal handlers."""

    def test_signal_handler_stops(self) -> None:
        """_signal_handler should call stop()."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.running = True

        app._signal_handler(2, None)

        assert app.running is False

    def test_signal_handler_build_releases_lock(self) -> None:
        """_signal_handler_build should release build lock."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.build_in_progress = True

        with patch.object(app, "_build_pipeline") as mock_pipeline:
            mock_pipeline.release_lock.return_value = None

            with pytest.raises(SystemExit) as exc_info:
                app._signal_handler_build(2, None)

            assert exc_info.value.code == 130
            mock_pipeline.release_lock.assert_called_once()


# =============================================================================
# Profiling input/cancellation and staleness wiring
# =============================================================================


class TestProfilingFlow:
    """Tests for non-blocking profiling input and cancellation behavior."""

    def test_execute_profile_returns_1_when_cancel_event_missing(self) -> None:
        app = TUIApp(configs=[_make_config(alias="slot0")], gpu_indices=[0])

        exit_code = app._execute_profile("slot0", "balanced")

        assert exit_code == 1

    def test_execute_profile_uses_silent_callback_mode(self) -> None:
        app = TUIApp(configs=[_make_config(alias="slot0")], gpu_indices=[0])
        cancel_event = threading.Event()
        app._profile_cancel_events["slot0"] = cancel_event

        with patch("llama_cli.commands.profile.cmd_profile", return_value=0) as mock_cmd_profile:
            exit_code = app._execute_profile("slot0", "balanced")

        assert exit_code == 0
        mock_cmd_profile.assert_called_once_with(
            slot_id="slot0",
            flavor="balanced",
            quiet=True,
            progress_callback=app._push_status_message,
            cancel_event=cancel_event,
        )

    def test_abort_profile_sets_cancel_event_and_failed_status(self) -> None:
        app = TUIApp(configs=[_make_config(alias="slot0")], gpu_indices=[0])
        event = threading.Event()
        app._profile_status["slot0"] = "running"
        app._profile_cancel_events["slot0"] = event

        app._abort_profile()

        assert event.is_set()
        assert app._profile_status["slot0"] == "failed"

    def test_get_stale_warning_uses_gpu_identifier_and_driver_binary_versions(self) -> None:
        app = TUIApp(configs=[_make_config(alias="slot0", device="SYCL0")], gpu_indices=[0])
        cfg = app.configs[0]
        app.config.server_binary_version = "v1.2.3"
        app.config.profile_staleness_days = 30

        stale_result = MagicMock()
        stale_result.is_stale = True
        stale_reason = MagicMock()
        stale_reason.value = "driver_changed"
        stale_result.reasons = [stale_reason]

        with (
            patch(
                "llama_cli.tui.viewmodel.get_gpu_identifier", return_value="intel-arc_b580-00"
            ) as mock_gpu,
            patch(
                "llama_cli.commands.profile.get_driver_version", return_value="driver-1"
            ) as mock_driver,
            patch(
                "llama_cli.tui.viewmodel.load_profile_with_staleness",
                return_value=(MagicMock(), stale_result),
            ) as mock_load,
        ):
            warning = app.get_stale_warning(cfg)

        assert warning is not None
        assert "profile stale" in warning.lower()
        mock_gpu.assert_called_once_with(cfg.backend)
        mock_driver.assert_called_once_with(cfg.backend)
        assert mock_load.call_args.kwargs["gpu_identifier"] == "intel-arc_b580-00"
        assert mock_load.call_args.kwargs["current_driver_version"] == "driver-1"
        assert mock_load.call_args.kwargs["current_binary_version"] == "v1.2.3"


class TestBuildCommandMenu:
    """Tests for TUIApp._build_command_menu."""

    def test_normal_mode_shows_expected_commands(self) -> None:
        menu = TUIApp(configs=[_make_config()], gpu_indices=[0])._build_command_menu()
        text = menu.plain
        assert "Quit" in text
        assert "Refresh" in text
        assert "Add slot" in text
        assert "Profile" in text
        assert "Stop" in text

    def test_profile_pending_shows_flavor_commands(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.profile_request = "slot0"
        menu = app._build_command_menu()
        text = menu.plain
        assert "Balanced" in text
        assert "Fast" in text
        assert "Quality" in text
        assert "Cancel" in text

    def test_risk_panel_shows_confirm_commands(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.risk_panel = MagicMock()
        menu = app._build_command_menu()
        text = menu.plain
        assert "Confirm" in text
        assert "Abort" in text

    def test_risk_panel_vram_hides_quit(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.risk_panel = MagicMock()
        app.active_risk_kind = "vram"
        menu = app._build_command_menu()
        text = menu.plain
        assert "Quit" not in text


# =============================================================================
# Slot creation via delegation to llama_manager.slot_manager
# =============================================================================


class TestAddSlotFromForm:
    """Tests for modal-backed slot creation."""

    def test_add_slot_from_form_creates_slot_from_profile(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        form_values = {
            "profile": "summary-fast",
            "port": "8090",
        }

        with patch.object(app.server_manager, "start_servers", return_value=[]):
            ok = app.add_slot_from_form(form_values)

        assert ok is True
        assert any(cfg.alias == "summary-fast" for cfg in app.configs)
        assert "summary-fast" in app.log_buffers
        assert "summary-fast" in app.unsaved_slots

    def test_add_slot_from_form_rejects_empty_profile(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        initial_count = len(app.configs)

        ok = app.add_slot_from_form(
            {
                "profile": "   ",
                "port": "8091",
            }
        )

        assert ok is False
        assert len(app.configs) == initial_count

    def test_add_slot_from_form_replaces_existing_device_slot(self) -> None:
        existing = _make_config(alias="summary-balanced", port=8080, device="SYCL0")
        app = TUIApp(configs=[existing], gpu_indices=[1])

        with (
            patch.object(app.server_manager, "shutdown_slot", return_value=True),
            patch.object(app.server_manager, "start_servers", return_value=[]),
        ):
            ok = app.add_slot_from_form({"profile": "summary-fast", "port": "8092"})

        assert ok is True
        assert len(app.configs) == 1
        assert app.configs[0].alias == "summary-fast"
        assert app.configs[0].port == 8092

    def test_add_slot_from_form_replacement_aborts_when_shutdown_fails(self) -> None:
        existing = _make_config(alias="summary-balanced", port=8080, device="SYCL0")
        app = TUIApp(configs=[existing], gpu_indices=[1])

        with patch.object(app.server_manager, "shutdown_slot", return_value=False):
            ok = app.add_slot_from_form({"profile": "summary-fast", "port": "8092"})

        assert ok is False
        assert len(app.configs) == 1
        assert app.configs[0].alias == "summary-balanced"
