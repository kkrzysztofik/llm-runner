from __future__ import annotations

"""Tests for TUI application (llama_cli.tui_app).

Tests for T016c-T016f:
- T016c: Per-slot status display
- T016d: GPU telemetry panel update
- T016e: Slot state transition handling
- T016f: Graceful shutdown key handler (Ctrl+C)
"""


import signal
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import SlotState
from tests.support.helpers import make_server_config

_make_minimal_config = make_server_config


class TestPerSlotStatusDisplay:
    """T016c: Tests for per-slot status display in TUI."""

    def test_tui_app_instantiation_with_configs(self) -> None:
        """DashboardController should be instantiable with configs and GPU indices."""
        from llama_cli.tui import DashboardController

        configs = [_make_minimal_config(alias="slot1")]
        app = DashboardController(configs=configs, gpu_indices=[0])

        assert app.configs == configs
        assert app.gpu_indices == [0]
        assert app.running is True
        assert app.server_manager is not None

    def test_tui_app_instantiation_with_multiple_configs(self) -> None:
        """DashboardController should handle multiple configs."""
        from llama_cli.tui import DashboardController

        configs = [
            _make_minimal_config(alias="slot1", port=8080),
            _make_minimal_config(alias="slot2", port=8081),
        ]
        app = DashboardController(configs=configs, gpu_indices=[0, 1])

        assert len(app.configs) == 2
        assert len(app.gpu_stats) == 2
        assert len(app.log_buffers) == 2

    def test_log_buffers_created_per_config(self) -> None:
        """DashboardModel should create a LogBuffer for each config."""
        from llama_cli.tui import DashboardController

        configs = [
            _make_minimal_config(alias="log-test-1"),
            _make_minimal_config(alias="log-test-2"),
        ]
        app = DashboardController(configs=configs, gpu_indices=[0])

        for cfg in configs:
            assert cfg.alias in app.log_buffers

    def test_status_panel_initialized(self) -> None:
        """DashboardController should initialize status_panel as None."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        assert app.status_panel is None

    def test_risk_panel_initialized(self) -> None:
        """DashboardController should initialize risk_panel as None."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        assert app.risk_panel is None

    def test_build_column_panel_creates_panel(self) -> None:
        """_build_column_panel should return a Panel with config info."""
        from llama_cli.tui import DashboardController
        from llama_manager.log_buffer import LogBuffer

        cfg = _make_minimal_config(alias="panel-test")
        app = DashboardController(configs=[cfg], gpu_indices=[])

        buffer = LogBuffer()
        panel = app._build_column_panel(cfg, buffer, None)

        assert panel is not None
        # Panel should contain the config alias in its content
        # The panel is a Rich Panel with Group containing header, gpu, logs

    def test_build_placeholder_panel(self) -> None:
        """_build_placeholder_panel should return a dim panel."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[])
        panel = app._build_placeholder_panel()

        assert panel is not None


class TestGPUTelemetryPanel:
    """T016d: Tests for GPU telemetry panel update."""

    def test_gpu_stats_initialized(self) -> None:
        """GPUStats should be initialized for each GPU index."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0, 1])
        assert len(app.gpu_stats) == 2

    def test_gpu_stats_collects_data(self) -> None:
        """GPUStats should collect data when updated."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        stats = app.gpu_stats[0]

        # Force an update with a custom collector
        fake_stats = {
            "device": "Intel Arc B580",
            "gpu_util": "45%",
            "mem_util": "60%",
            "temp": "65C",
        }
        stats._collector = lambda: fake_stats  # type: ignore[assignment]
        stats.update()

        assert stats.stats["device"] == "Intel Arc B580"
        assert stats.stats["gpu_util"] == "45%"

    def test_gpu_stats_format_text(self) -> None:
        """GPUStats.format_stats_text should produce readable output."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        stats = app.gpu_stats[0]

        # Use psutil-only collector (no GPU)
        text = stats.format_stats_text()
        assert "Device:" in text

    def test_gpu_stats_with_mock_collector(self) -> None:
        """GPUStats should use injected collector for testing."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        stats = app.gpu_stats[0]

        # Inject a mock collector
        mock_data: dict[str, Any] = {
            "device": "Mock GPU",
            "gpu_util": "100%",
            "mem_util": "50%",
            "temp": "80C",
        }

        stats._collector = lambda: mock_data  # type: ignore[assignment]
        stats.update()

        snapshot = stats.get_stats_snapshot()
        assert snapshot["device"] == "Mock GPU"
        assert snapshot["gpu_util"] == "100%"

    def test_gpu_stats_update_interval(self) -> None:
        """GPUStats should respect update_interval to avoid excessive updates."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        stats = app.gpu_stats[0]

        # Set a long update interval
        stats.update_interval = 3600  # 1 hour

        call_count = 0

        def counting_collector() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"device": "test"}

        stats._collector = counting_collector  # type: ignore[assignment]

        # First update
        stats.update()
        first_count = call_count

        # Second update should be skipped due to interval
        stats.update()
        assert call_count == first_count

    def test_gpu_stats_with_none_gpu_in_column_panel(self) -> None:
        """_build_column_panel should handle None GPU gracefully."""
        from llama_cli.tui import DashboardController
        from llama_manager.log_buffer import LogBuffer

        cfg = _make_minimal_config(alias="no-gpu-test")
        app = DashboardController(configs=[cfg], gpu_indices=[])

        buffer = LogBuffer()
        # Pass None for GPU
        panel = app._build_column_panel(cfg, buffer, None)

        assert panel is not None


class TestSlotStateTransitionHandling:
    """T016e: Tests for slot state transition handling in TUI."""

    def test_tui_app_has_server_manager(self) -> None:
        """DashboardController should have a ServerManager for lifecycle management."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        assert app.server_manager is not None

    def test_server_manager_lifecycle_audit(self) -> None:
        """ServerManager should maintain lifecycle audit trail."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Record a lifecycle event
        app.server_manager._record_lifecycle_event("test_event", pid=12345)

        audit = app.server_manager._lifecycle_audit
        assert len(audit) >= 1
        assert any(e["event"] == "test_event" for e in audit)

    def test_slot_state_enum_values(self) -> None:
        """SlotState should have all expected values for TUI display."""
        assert SlotState.IDLE.value == "idle"
        assert SlotState.LAUNCHING.value == "launching"
        assert SlotState.RUNNING.value == "running"
        assert SlotState.DEGRADED.value == "degraded"
        assert SlotState.CRASHED.value == "crashed"
        assert SlotState.OFFLINE.value == "offline"

    def test_tui_status_panel_on_blocked(self) -> None:
        """_build_status_panel should create error panel for blocked launch."""
        from llama_cli.tui import DashboardController
        from llama_manager.config import ErrorCode, ErrorDetail, MultiValidationError
        from llama_manager.orchestration import LaunchResult

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        blocked_result = LaunchResult(
            status="blocked",
            launched=[],
            errors=MultiValidationError(
                errors=[
                    ErrorDetail(
                        error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE,
                        failed_check="lockfile_integrity",
                        why_blocked="lock already held",
                        how_to_fix="remove stale lock",
                    ),
                ]
            ),
        )

        app._build_status_panel(blocked_result)
        assert app.status_panel is not None

    def test_tui_status_panel_on_degraded(self) -> None:
        """_build_status_panel should create warning panel for degraded launch."""
        from llama_cli.tui import DashboardController
        from llama_manager.orchestration import LaunchResult

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        degraded_result = LaunchResult(
            status="degraded",
            launched=["slot1"],
            warnings=["slot2: lock already held"],
        )

        app._build_status_panel(degraded_result)
        assert app.status_panel is not None

    def test_tui_status_panel_clears_on_success(self) -> None:
        """_build_status_panel should clear panel on successful launch."""
        from llama_cli.tui import DashboardController
        from llama_manager.orchestration import LaunchResult

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set a non-None status panel first
        app.status_panel = MagicMock()

        success_result = LaunchResult(status="success", launched=["slot1"])
        app._build_status_panel(success_result)

        # Panel should be cleared (set to None)
        assert app.status_panel is None

    def test_risk_panel_required(self) -> None:
        """_build_risk_panel_required should set risk_panel with required style."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        app._build_risk_panel_required()

        assert app.risk_panel is not None
        assert app.risks_acknowledged is False

    def test_risk_panel_acknowledged(self) -> None:
        """_build_risk_panel_acknowledged should set risk_panel with acknowledged style."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        app._build_risk_panel_acknowledged()

        assert app.risk_panel is not None
        assert app.risks_acknowledged is True


class TestGracefulShutdownKeyHandler:
    """T016f: Tests for graceful shutdown key handler (Ctrl+C)."""

    def test_stop_sets_running_false(self) -> None:
        """DashboardController.stop() should set running=False."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        assert app.running is True

        app.stop()
        assert app.running is False

    def test_signal_handler_calls_stop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_signal_handler should call stop() to stop the TUI loop."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        assert app.running is True

        app._signal_handler(signal.SIGINT, None)

        assert app.running is False

    def test_cleanup_calls_server_manager_cleanup(self) -> None:
        """DashboardController._cleanup() should call server_manager.cleanup_servers()."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Mock cleanup_servers to track calls
        cleanup_called = False

        def track_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        app.server_manager.cleanup_servers = track_cleanup  # type: ignore[assignment]

        # Should not raise
        app._cleanup()
        assert cleanup_called is True

    def test_cleanup_does_not_require_input_polling_thread(self) -> None:
        """DashboardController._cleanup() should not depend on legacy input polling."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        app._cleanup()
        assert app.running is True

    def test_on_interrupt_calls_cleanup_and_exits(self) -> None:
        """ServerManager.on_interrupt should call cleanup_servers and exit with code 130."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        cleanup_called = False

        def track_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        app.server_manager.cleanup_servers = track_cleanup  # type: ignore[assignment]

        with pytest.raises(SystemExit) as exc_info:
            app.server_manager.on_interrupt(signal.SIGINT, None)

        assert exc_info.value.code == 130
        assert cleanup_called is True

    def test_on_terminate_calls_cleanup_and_exits(self) -> None:
        """ServerManager.on_terminate should call cleanup_servers and exit with code 143."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        cleanup_called = False

        def track_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        app.server_manager.cleanup_servers = track_cleanup  # type: ignore[assignment]

        with pytest.raises(SystemExit) as exc_info:
            app.server_manager.on_terminate(signal.SIGTERM, None)

        assert exc_info.value.code == 143
        assert cleanup_called is True

    def test_signal_handler_releases_build_lock(self) -> None:
        """_signal_handler should release build lock if build in progress."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Mock build pipeline
        mock_pipeline = MagicMock()
        app._build_pipeline = mock_pipeline
        app.build_in_progress = True

        app._signal_handler(signal.SIGINT, None)

        mock_pipeline.release_lock.assert_called_once()
        assert app.build_in_progress is False

    def test_request_quit_calls_graceful_shutdown(self) -> None:
        """request_quit should initiate graceful shutdown when idle."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        with patch.object(app, "_graceful_shutdown") as mock_shutdown:
            app.request_quit()

        mock_shutdown.assert_called_once()

    def test_interrupt_aborts_running_profile(self) -> None:
        """interrupt should abort a running profile before shutdown."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config(alias="slot0")], gpu_indices=[0])
        cancel_event = threading.Event()
        with app._profile_lock:
            app._profile_status["slot0"] = "running"
            app._profile_cancel_events["slot0"] = cancel_event

        app.interrupt()

        assert cancel_event.is_set()
        assert app._profile_status["slot0"] == "failed"

    def test_refresh_display_appends_message(self) -> None:
        """refresh_display should add a visible status message."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.refresh_display()

        assert any("refreshed" in msg.lower() for _, msg in app._status_messages)

    def test_request_profile_sets_pending_request(self) -> None:
        """request_profile should queue the first profile for selection."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config(alias="slot0")], gpu_indices=[0])

        app.request_profile()

        assert app.profile_request == "slot0"
        assert app._profile_status["slot0"] == "idle"

    def test_request_build_and_cancel_pending_prompt(self) -> None:
        """request_build should set build state and cancel_pending_prompt should clear it."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.request_build()
        assert app._build_request is True

        cancelled = app.cancel_pending_prompt()
        assert cancelled is True
        assert app._build_request is False

    def test_request_smoke_and_cancel_pending_prompt(self) -> None:
        """request_smoke should set smoke state and cancel_pending_prompt should clear it."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.request_smoke()
        assert app._smoke_request is True

        cancelled = app.cancel_pending_prompt()
        assert cancelled is True
        assert app._smoke_request is False

    def test_push_status_message(self) -> None:
        """_push_status_message should add messages to the status buffer."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app._push_status_message("test message 1")
        app._push_status_message("test message 2")

        # Messages should be in the buffer
        assert len(app._status_messages) == 2
        assert any(msg == "test message 1" for _, msg in app._status_messages)
        assert any(msg == "test message 2" for _, msg in app._status_messages)

    def test_push_status_message_limited_to_five(self) -> None:
        """_push_status_message should keep at most 5 messages."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        for i in range(10):
            app._push_status_message(f"message {i}")

        assert len(app._status_messages) <= 5

    def test_build_status_messages_panel(self) -> None:
        """_build_status_messages_panel should create panel from status messages."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        app._push_status_message("status update 1")

        panel = app._build_status_messages_panel()
        assert panel is not None

    def test_build_status_messages_panel_empty(self) -> None:
        """_build_status_messages_panel should return None when no messages."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        panel = app._build_status_messages_panel()
        assert panel is None

    def test_abort_profile(self) -> None:
        """_abort_profile should cancel any running profile."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set up a fake running profile
        with app._profile_lock:
            app._profile_status["test-slot"] = "running"
            app._profile_cancel_events["test-slot"] = threading.Event()

        # Abort should not raise
        app._abort_profile()

        # Profile should be marked as failed
        assert app._profile_status["test-slot"] == "failed"


class TestHandleHardwareWarning:
    """T061: Tests for hardware warning TUI key handler."""

    def test_handle_hardware_warning_y_acknowledges(self) -> None:
        """handle_hardware_warning should acknowledge and clear panel on 'y'."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set up a risk panel to be cleared
        app.risk_panel = MagicMock()

        result = app.handle_hardware_warning("y")

        assert result == "acknowledge"
        assert app.risk_panel is None

    def test_handle_hardware_warning_n_aborts(self) -> None:
        """handle_hardware_warning should abort on 'n' key."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set up a risk panel so handler is invoked
        app.risk_panel = MagicMock()

        result = app.handle_hardware_warning("n")

        assert result == "abort"
        assert app.running is False

    def test_handle_hardware_warning_q_quits(self) -> None:
        """handle_hardware_warning should quit on 'q' key."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set up a risk panel so handler is invoked
        app.risk_panel = MagicMock()

        result = app.handle_hardware_warning("q")

        assert result == "quit"
        assert app.running is False

    def test_handle_hardware_warning_other_ignored(self) -> None:
        """handle_hardware_warning should ignore non-action keys."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set up a risk panel
        app.risk_panel = MagicMock()

        result = app.handle_hardware_warning("x")

        assert result == "ignore"
        assert app.risk_panel is not None


class TestHandleVramRisk:
    """T061b: Tests for VRAM risk confirmation TUI key handler."""

    def test_handle_vram_risk_y_proceeds(self) -> None:
        """handle_vram_risk should proceed on 'y' key and clear panel."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set up a risk panel and VRAM risk kind for routing
        app.risk_panel = MagicMock()
        app.active_risk_kind = "vram"

        result = app.handle_vram_risk("y")

        assert result == "proceed"
        assert app.risk_panel is None

    def test_handle_vram_risk_n_aborts(self) -> None:
        """handle_vram_risk should abort on 'n' key."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Call handle_vram_risk directly
        result = app.handle_vram_risk("n")

        assert result == "abort"
        assert app.running is False

    def test_handle_vram_risk_other_ignored(self) -> None:
        """handle_vram_risk should ignore non-action keys."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        # Set up a risk panel and VRAM risk kind for routing
        app.risk_panel = MagicMock()
        app.active_risk_kind = "vram"

        result = app.handle_vram_risk("x")

        assert result == "ignore"
        assert app.risk_panel is not None


class TestMVVMArchitecture:
    """Tests for the class-based MVVM TUI split."""

    def test_public_tui_api_exports_new_class_names_only(self) -> None:
        import llama_cli.tui as tui

        assert hasattr(tui, "DashboardApp")
        assert hasattr(tui, "DashboardController")
        assert hasattr(tui, "DashboardModel")
        assert hasattr(tui, "DashboardViewModel")
        assert not hasattr(tui, "TUIApp")
        assert not hasattr(tui, "TextualDashboardApp")
        assert not hasattr(tui, "DashboardSnapshot")

    def test_controller_owns_model_and_view_model(self) -> None:
        from llama_cli.tui import DashboardController, DashboardModel, DashboardViewModel

        controller = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        assert isinstance(controller.model, DashboardModel)
        assert isinstance(controller.view_model, DashboardViewModel)

    def test_view_model_exposes_plain_command_state(self) -> None:
        from llama_cli.tui import DashboardController

        controller = DashboardController(
            configs=[_make_minimal_config(alias="slot0")], gpu_indices=[]
        )
        controller.request_profile()

        state = controller.view_model.command_menu()

        assert state.profile_request == "slot0"
        assert state.risk_prompt is None

    def test_risk_prompt_lives_in_model_state(self) -> None:
        from llama_cli.tui import DashboardController

        controller = DashboardController(configs=[_make_minimal_config()], gpu_indices=[])
        controller._build_risk_panel_required("vram")

        state = controller.view_model.command_menu()

        assert state.risk_prompt is not None
        assert state.risk_prompt.kind == "vram"
        assert state.risk_prompt.acknowledged is False


"""Tests for TUI application module (tui_app.py).

Covers:
- TUIApp initialization
- Layout building and rendering
- Build pipeline integration
- Risk acknowledgment
- Status panel building
"""


from pathlib import Path

from llama_cli.tui import DashboardController
from llama_cli.tui.textual_app import DashboardApp
from llama_manager.build_pipeline import BuildProgress, BuildResult
from llama_manager.config import ServerConfig
from tests.support.helpers import make_server_config

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

        mock_proc = MagicMock()
        with patch.object(app.server_manager, "start_servers", return_value=[mock_proc]):
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

        mock_proc = MagicMock()
        with (
            patch.object(app.server_manager, "shutdown_slot", return_value=True),
            patch.object(app.server_manager, "start_servers", return_value=[mock_proc]),
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


"""Unit tests for smaller TUI component modules."""


from types import SimpleNamespace
from typing import cast

from rich.text import Text
from textual.widgets import Select

from llama_cli.tui.components import alerts, panels
from llama_cli.tui.components.modal import AddSlotModal
from llama_cli.tui.components.profile_status import ProfileStatusPanelRenderer


def test_alerts_module_exports_expected_symbols() -> None:
    assert "ProfileStatusPanelRenderer" in alerts.__all__
    assert "SystemHealthRenderer" in alerts.__all__
    assert alerts.ProfileStatusPanelRenderer is ProfileStatusPanelRenderer


def test_panels_module_exports_expected_symbols() -> None:
    assert "SlotStatusResolver" in panels.__all__
    assert "GPUStatsPanel" in panels.__all__
    assert hasattr(panels, "ServerColumnPanel")


def test_profile_status_renderer_returns_none_when_empty() -> None:
    renderer = ProfileStatusPanelRenderer()
    assert renderer.render({}, {}) is None


def test_profile_status_renderer_renders_all_status_variants() -> None:
    renderer = ProfileStatusPanelRenderer()
    panel = renderer.render(
        profile_status={
            "slot-a": "running",
            "slot-b": "done",
            "slot-c": "failed",
            "slot-d": "custom-status",
        },
        profile_flavor={"slot-a": "fast", "slot-b": "balanced", "slot-c": "deep"},
    )

    assert panel is not None
    assert panel.title == "Profile Status"
    body = cast(Text, panel.renderable).plain
    assert "Profiling slot-a: fast [running...]" in body
    assert "Profile slot-b: balanced [done]" in body
    assert "Profile slot-c: deep [failed]" in body
    assert "Profile slot-d: unknown [custom-status]" in body


def test_add_slot_modal_rejects_empty_profile_options() -> None:
    with pytest.raises(ValueError, match="profile_options must not be empty"):
        AddSlotModal([])


def test_add_slot_modal_collect_values_valid_and_strips_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value="qwen")
    port = SimpleNamespace(value=" 8081 ")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    values = modal._collect_values()

    assert values == {"profile": "qwen", "port": "8081"}


def test_add_slot_modal_collect_values_blank_profile_maps_to_empty() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value=Select.BLANK)
    port = SimpleNamespace(value="")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    values = modal._collect_values()

    assert values == {"profile": "", "port": ""}


def test_add_slot_modal_collect_values_rejects_non_numeric_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value="qwen")
    port = SimpleNamespace(value="abc")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    modal.notify = MagicMock()  # type: ignore[method-assign]

    assert modal._collect_values() is None
    modal.notify.assert_called_once_with("Port must be a number", severity="error")


def test_add_slot_modal_collect_values_rejects_out_of_range_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    profile = SimpleNamespace(value="qwen")
    port = SimpleNamespace(value="70000")

    modal.query_one = MagicMock(side_effect=[profile, port])  # type: ignore[method-assign]
    modal.notify = MagicMock()  # type: ignore[method-assign]

    assert modal._collect_values() is None
    modal.notify.assert_called_once_with("Port must be 1-65535", severity="error")


def test_add_slot_modal_action_cancel_dismisses_none() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    modal.dismiss = MagicMock()  # type: ignore[method-assign]

    modal.action_cancel()

    modal.dismiss.assert_called_once_with(None)


def test_add_slot_modal_on_mount_focuses_profile_select() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    select_widget = MagicMock()
    modal.query_one = MagicMock(return_value=select_widget)  # type: ignore[method-assign]

    modal.on_mount()

    select_widget.focus.assert_called_once_with()
    modal.query_one.assert_called_once_with("#slot-profile", Select)


def test_add_slot_modal_on_button_pressed_paths() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    modal.dismiss = MagicMock()  # type: ignore[method-assign]

    modal.on_button_pressed(cast(Any, SimpleNamespace(button=SimpleNamespace(id="cancel-slot"))))
    modal.dismiss.assert_called_once_with(None)

    modal.dismiss.reset_mock()
    modal._collect_values = MagicMock(return_value={"profile": "qwen", "port": "8080"})  # type: ignore[method-assign]
    modal.on_button_pressed(cast(Any, SimpleNamespace(button=SimpleNamespace(id="submit-slot"))))
    modal.dismiss.assert_called_once_with({"profile": "qwen", "port": "8080"})

    modal.dismiss.reset_mock()
    modal._collect_values = MagicMock(return_value=None)  # type: ignore[method-assign]
    modal.on_button_pressed(cast(Any, SimpleNamespace(button=SimpleNamespace(id="submit-slot"))))
    modal.dismiss.assert_not_called()


def test_add_slot_modal_on_input_submitted_only_for_slot_port() -> None:
    modal = AddSlotModal([("Qwen", "qwen")])
    modal.dismiss = MagicMock()  # type: ignore[method-assign]
    modal._collect_values = MagicMock(return_value={"profile": "qwen", "port": "8080"})  # type: ignore[method-assign]

    modal.on_input_submitted(cast(Any, SimpleNamespace(input=SimpleNamespace(id="other-input"))))
    modal.dismiss.assert_not_called()

    modal.on_input_submitted(cast(Any, SimpleNamespace(input=SimpleNamespace(id="slot-port"))))
    modal.dismiss.assert_called_once_with({"profile": "qwen", "port": "8080"})
