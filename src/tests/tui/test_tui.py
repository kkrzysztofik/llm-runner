"""Tests for TUI application (llama_cli.tui_app).

Tests for T016c-T016f:
- T016c: Per-slot status display
- T016d: GPU telemetry panel update
- T016e: Slot state transition handling
- T016f: Graceful shutdown key handler (Ctrl+C)
"""

from __future__ import annotations

import signal
import threading
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import SlotState
from tests.support.factories import make_server_config

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
        from llama_manager.process_manager import LaunchResult

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
        from llama_manager.process_manager import LaunchResult

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
        from llama_manager.process_manager import LaunchResult

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
