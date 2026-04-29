"""Tests for TUI application module (tui_app.py).

Covers:
- TUIApp initialization
- Layout building and rendering
- Build pipeline integration
- Risk acknowledgment
- Status panel building
"""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.tui import TUIApp
from llama_manager.build_pipeline import BuildBackend, BuildProgress, BuildResult
from llama_manager.config import ServerConfig
from tests.support.factories import make_server_config


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
        assert app.width == 80
        assert app.height == 24
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

    def test_init_default_width_height(self) -> None:
        """TUIApp should have default 80x24 dimensions."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        assert app.width == 80
        assert app.height == 24


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
# on_resize
# =============================================================================


class TestTUIAppOnResize:
    """Tests for TUIApp.on_resize."""

    def test_on_resize_updates_dimensions(self) -> None:
        """on_resize should update width and height."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        mock_event = MagicMock()
        mock_event.width = 120
        mock_event.height = 40
        app.on_resize(mock_event)  # type: ignore[arg-type]
        assert app.width == 120
        assert app.height == 40


# =============================================================================
# build_layout
# =============================================================================


class TestTUIAppBuildLayout:
    """Tests for TUIApp.build_layout."""

    def test_build_layout_structure(self) -> None:
        """build_layout should describe the responsive Textual content layout."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.width = 120
        layout = app.build_layout()

        assert layout.content_orientation == "horizontal"

    def test_build_layout_row_split(self) -> None:
        """build_layout should use horizontal content when width >= 80."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.width = 120
        layout = app.build_layout()

        assert layout.content_orientation == "horizontal"

    def test_build_layout_column_split(self) -> None:
        """build_layout should use vertical content when width < 80."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.width = 60
        layout = app.build_layout()

        assert layout.content_orientation == "vertical"

    def test_build_layout_boundary_width_is_horizontal(self) -> None:
        """build_layout should switch to horizontal at the 80-column boundary."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.width = 80
        layout = app.build_layout()
        assert layout.content_orientation == "horizontal"


# =============================================================================
# render
# =============================================================================


class TestTUIAppRender:
    """Tests for TUIApp.render."""

    def test_render_no_configs(self) -> None:
        """render should handle no configs gracefully."""
        app = TUIApp(configs=[], gpu_indices=[])
        layout = app.render()
        assert layout is not None

    def test_render_single_config(self) -> None:
        """render should render single config with placeholder."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        layout = app.render()
        assert layout is not None

    def test_render_two_configs(self) -> None:
        """render should render both configs side by side."""
        configs = [_make_config("model1", 8080, "CUDA"), _make_config("model2", 8081, "SYCL")]
        app = TUIApp(configs=configs, gpu_indices=[0, 1])
        layout = app.render()
        assert layout is not None

    def test_render_with_status_panel(self) -> None:
        """render should include status panel in alerts."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.status_panel = MagicMock()
        layout = app.render()
        assert layout is not None

    def test_render_populates_menu(self) -> None:
        """render should populate the dashboard snapshot menu."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        snapshot = app.render()
        assert snapshot.menu is not None


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
        """_update_risk_panel_state should set acknowledged panel when has_risks."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._update_risk_panel_state(has_risks=True)
        assert app.risks_acknowledged is False

    def test_update_risk_panel_state_without_risks(self) -> None:
        """_update_risk_panel_state should clear risk_panel when no risks."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.risk_panel = MagicMock()
        app._update_risk_panel_state(has_risks=False)
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
        app._build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="retrying",
            message="Retrying...",
            progress_percent=50,
            retries_remaining=2,
        )
        app._handle_build_progress(progress)

        assert app._build_progress is progress
        assert app.status_panel is not None

    def test_handle_progress_failure(self) -> None:
        """_handle_build_progress should create failed status panel."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_in_progress = True

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
        app._build_in_progress = True
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

    def test_build_llama_cpp_success(self, tmp_path: Path) -> None:
        """build_llama_cpp should return True on success."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.tui.controller.BuildPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value.run.return_value = BuildResult(success=True)
                mock_pipeline_cls.return_value.dry_run = False

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                result = app.build_llama_cpp(backend="sycl", dry_run=False)

                assert result is True
                assert app._build_in_progress is False

    def test_build_llama_cpp_failure(self, tmp_path: Path) -> None:
        """build_llama_cpp should return False on failure."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.tui.controller.BuildPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value.run.return_value = BuildResult(
                    success=False, error_message="failed"
                )
                mock_pipeline_cls.return_value.dry_run = False

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                result = app.build_llama_cpp(backend="sycl", dry_run=False)

                assert result is False
                assert app._build_in_progress is False

    def test_build_llama_cpp_dry_run(self, tmp_path: Path) -> None:
        """build_llama_cpp should set dry_run on pipeline."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.tui.controller.BuildPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value.run.return_value = BuildResult(success=True)

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                app.build_llama_cpp(backend="sycl", dry_run=True)

                assert mock_pipeline_cls.return_value.dry_run is True

    def test_build_llama_cpp_cuda_backend(self, tmp_path: Path) -> None:
        """build_llama_cpp should use correct backend for CUDA."""
        with patch("llama_cli.tui.controller.Config") as mock_config_cls:
            mock_config = MagicMock()
            mock_config.llama_cpp_root = str(tmp_path / "llama.cpp")
            mock_config.builds_dir = tmp_path / "output"
            mock_config.build_git_remote = "https://github.com/ggerganov/llama.cpp"
            mock_config.build_git_branch = "main"
            mock_config.build_retry_attempts = 2
            mock_config.build_retry_delay = 5
            mock_config_cls.return_value = mock_config

            with patch("llama_cli.tui.controller.BuildPipeline") as mock_pipeline_cls:
                mock_pipeline_cls.return_value.run.return_value = BuildResult(success=True)

                app = TUIApp(configs=[_make_config()], gpu_indices=[0])
                app.build_llama_cpp(backend="cuda", dry_run=False)

                # Verify pipeline was created with CUDA backend
                call_args = mock_pipeline_cls.call_args
                build_config = call_args[0][0]
                assert build_config.backend == BuildBackend.CUDA


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
        app._build_in_progress = True

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

    def test_process_keypresses_prioritizes_flavor_key_for_pending_request(self) -> None:
        app = TUIApp(configs=[_make_config(alias="slot0")], gpu_indices=[0])
        app._profile_request = "slot0"
        app._keypress_queue.put("1")

        with patch.object(app, "_wait_for_flavor_selection") as mock_wait:
            app._process_keypresses()

        mock_wait.assert_called_once_with("slot0", preselected_key="1")
        assert app._profile_request is None

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
                "llama_cli.tui.controller.get_gpu_identifier", return_value="intel-arc_b580-00"
            ) as mock_gpu,
            patch(
                "llama_cli.commands.profile._get_driver_version", return_value="driver-1"
            ) as mock_driver,
            patch(
                "llama_cli.tui.controller.load_profile_with_staleness",
                return_value=(MagicMock(), stale_result),
            ) as mock_load,
        ):
            warning = app._get_stale_warning(cfg)

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
        app._profile_request = "slot0"
        menu = app._build_command_menu()
        text = menu.plain
        assert "Balanced" in text
        assert "Fast" in text
        assert "Quality" in text
        assert "Cancel" in text

    def test_slot_config_shows_input_commands(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._slot_config_state["slot1"] = "model"
        menu = app._build_command_menu()
        text = menu.plain
        assert "Next" in text
        assert "Edit" in text
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
# Slot config normalizer helpers
# =============================================================================


class TestNormalizeSlotPort:
    """Tests for TUIApp._normalize_slot_port."""

    def _make_app(self) -> TUIApp:
        return TUIApp(configs=[_make_config()], gpu_indices=[0])

    def test_valid_port_unchanged(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"port": "8080"}
        app._normalize_slot_port(values)
        assert values["port"] == "8080"

    def test_port_below_range_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"port": "80"}
        app._normalize_slot_port(values)
        assert values["port"] == "8080"

    def test_port_above_range_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"port": "99999"}
        app._normalize_slot_port(values)
        assert values["port"] == "8080"

    def test_non_numeric_port_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"port": "abc"}
        app._normalize_slot_port(values)
        assert values["port"] == "8080"

    def test_empty_port_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"port": ""}
        app._normalize_slot_port(values)
        assert values["port"] == "8080"


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
        assert "summary-fast" in app._unsaved_slots

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


class TestNormalizeSlotThreads:
    """Tests for TUIApp._normalize_slot_threads."""

    def _make_app(self) -> TUIApp:
        return TUIApp(configs=[_make_config()], gpu_indices=[0])

    def test_valid_threads_unchanged(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"threads": "4"}
        app._normalize_slot_threads(values)
        assert values["threads"] == "4"

    def test_zero_threads_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"threads": "0"}
        app._normalize_slot_threads(values)
        assert values["threads"] == "4"

    def test_non_numeric_threads_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"threads": "xyz"}
        app._normalize_slot_threads(values)
        assert values["threads"] == "4"


class TestNormalizeSlotCtxSize:
    """Tests for TUIApp._normalize_slot_ctx_size."""

    def _make_app(self) -> TUIApp:
        return TUIApp(configs=[_make_config()], gpu_indices=[0])

    def test_valid_ctx_size_unchanged(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"ctx_size": "2048"}
        app._normalize_slot_ctx_size(values)
        assert values["ctx_size"] == "2048"

    def test_small_ctx_size_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"ctx_size": "256"}
        app._normalize_slot_ctx_size(values)
        assert values["ctx_size"] == "2048"

    def test_non_numeric_ctx_size_resets_to_default(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"ctx_size": "big"}
        app._normalize_slot_ctx_size(values)
        assert values["ctx_size"] == "2048"


class TestNormalizeSlotBackend:
    """Tests for TUIApp._normalize_slot_backend."""

    def _make_app(self) -> TUIApp:
        return TUIApp(configs=[_make_config()], gpu_indices=[0])

    def test_cuda_backend_accepted(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"backend": "CUDA"}
        app._normalize_slot_backend(values)
        assert values["backend"] == "cuda"

    def test_sycl_backend_accepted(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"backend": "SYCL"}
        app._normalize_slot_backend(values)
        assert values["backend"] == "sycl"

    def test_invalid_backend_resets_to_sycl(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"backend": "metal"}
        app._normalize_slot_backend(values)
        assert values["backend"] == "sycl"

    def test_empty_backend_resets_to_sycl(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"backend": ""}
        app._normalize_slot_backend(values)
        assert values["backend"] == "sycl"


class TestValidateSlotField:
    """Tests for TUIApp._validate_slot_field."""

    def _make_app(self) -> TUIApp:
        return TUIApp(configs=[_make_config()], gpu_indices=[0])

    def test_returns_true_for_valid_port(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"port": "8080"}
        assert app._validate_slot_field("port", values) is True

    def test_returns_true_for_valid_threads(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"threads": "4"}
        assert app._validate_slot_field("threads", values) is True

    def test_returns_true_for_valid_ctx_size(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"ctx_size": "2048"}
        assert app._validate_slot_field("ctx_size", values) is True

    def test_returns_true_for_valid_backend(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"backend": "cuda"}
        assert app._validate_slot_field("backend", values) is True

    def test_returns_false_for_empty_model(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"model": "  "}
        assert app._validate_slot_field("model", values) is False

    def test_returns_true_for_non_empty_model(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {"model": "/path/to/model.gguf"}
        assert app._validate_slot_field("model", values) is True

    def test_returns_true_for_unknown_field(self) -> None:
        app = self._make_app()
        values: dict[str, str] = {}
        assert app._validate_slot_field("other_field", values) is True


class TestAdvanceSlotConfigField:
    """Tests for TUIApp._advance_slot_config_field."""

    def _make_app(self) -> TUIApp:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._slot_config_state["slot1"] = "model"
        app._slot_config_values["slot1"] = {"model": "/path/to/model.gguf"}
        return app

    def test_advances_from_model_to_port(self) -> None:
        app = self._make_app()
        app._advance_slot_config_field("slot1")
        assert app._slot_config_state.get("slot1") == "port"

    def test_aborts_when_model_is_empty(self) -> None:
        app = self._make_app()
        app._slot_config_values["slot1"]["model"] = ""
        app._advance_slot_config_field("slot1")
        # State should not advance (still "model" since we can't continue)
        assert app._slot_config_state.get("slot1") == "model"

    def test_returns_early_when_field_is_none(self) -> None:
        app = self._make_app()
        app._slot_config_state["slot1"] = None  # type: ignore[assignment]
        app._advance_slot_config_field("slot1")
        # Should not crash

    def test_finalizes_when_all_fields_complete(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._slot_config_state["slot1"] = "ctx_size"
        app._slot_config_values["slot1"] = {
            "model": "/path/to/model.gguf",
            "port": "8081",
            "backend": "cuda",
            "threads": "4",
            "ctx_size": "2048",
        }
        app._advance_slot_config_field("slot1")
        # slot1 should be removed from state (finalized)
        assert "slot1" not in app._slot_config_state

    def test_cancels_on_unknown_field(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._slot_config_state["slot1"] = "nonexistent_field"
        app._slot_config_values["slot1"] = {}
        app._advance_slot_config_field("slot1")
        # Should cancel (slot cleaned up)
        assert "slot1" not in app._slot_config_state


class TestProcessSlotConfigInput:
    """Tests for TUIApp._process_slot_config_input."""

    def _make_app_in_slot_config(self) -> TUIApp:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._slot_config_state["slot1"] = "model"
        app._slot_config_values["slot1"] = {}
        return app

    def test_enter_key_advances_field(self) -> None:
        app = self._make_app_in_slot_config()
        app._slot_config_values["slot1"]["model"] = "/path/to/model.gguf"
        app._process_slot_config_input("\n")
        assert app._slot_config_state.get("slot1") == "port"

    def test_escape_key_cancels(self) -> None:
        app = self._make_app_in_slot_config()
        app._process_slot_config_input("\x1b")
        assert "slot1" not in app._slot_config_state

    def test_backspace_removes_last_char(self) -> None:
        app = self._make_app_in_slot_config()
        app._slot_config_values["slot1"]["model"] = "/path/to/model"
        app._process_slot_config_input("\x7f")
        assert app._slot_config_values["slot1"]["model"] == "/path/to/mode"

    def test_backspace_on_empty_value_is_noop(self) -> None:
        app = self._make_app_in_slot_config()
        app._slot_config_values["slot1"]["model"] = ""
        app._process_slot_config_input("\x7f")
        assert app._slot_config_values["slot1"]["model"] == ""

    def test_regular_char_appends_to_field(self) -> None:
        app = self._make_app_in_slot_config()
        app._process_slot_config_input("/")
        app._process_slot_config_input("p")
        assert app._slot_config_values["slot1"].get("model", "") == "/p"

    def test_skips_slot_with_none_field(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._slot_config_state["slot1"] = None  # type: ignore[assignment]
        app._slot_config_values["slot1"] = {}
        # Should not crash
        app._process_slot_config_input("x")
