"""Tests for TUI application (llama_cli.tui_app).

Tests for T016c-T016f:
- T016c: Per-slot status display
- T016d: GPU telemetry panel update
- T016e: Slot state transition handling
- T016f: Graceful shutdown key handler (Ctrl+C)
"""

import signal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import SlotState
from tests.support.helpers import make_server_config

_make_minimal_config = make_server_config


class TestSystemHealthAlignment:
    """Tests for terminal-width aware system health rendering."""

    def test_dashboard_app_uses_split_tcss_files(self) -> None:
        from llama_cli.tui.textual_app import DashboardApp

        assert DashboardApp.CSS_PATH == [
            "textual_app.tcss",
            "system_status.tcss",
            "dashboard_panels.tcss",
            "modals.tcss",
        ]

    def test_dashboard_ctrl_c_binding_does_not_override_input_copy(self) -> None:
        from textual.binding import Binding

        from llama_cli.tui.textual_app import DashboardApp

        ctrl_c_bindings = [
            binding
            for binding in DashboardApp.BINDINGS
            if isinstance(binding, Binding) and binding.key == "ctrl+c"
        ]

        assert len(ctrl_c_bindings) == 1
        assert ctrl_c_bindings[0].priority is False

    def test_system_health_widget_composes_stylable_sections(self) -> None:
        from textual.containers import Horizontal

        from llama_cli.tui.components.system_health import (
            CPUUsageWidget,
            DateTimeWidget,
            MemorySwapWidget,
            SystemHealthWidget,
            SystemInfoWidget,
        )

        sections = list(SystemHealthWidget().compose())

        assert [type(section) for section in sections] == [
            DateTimeWidget,
            CPUUsageWidget,
            Horizontal,
        ]
        assert sections[0].has_class("system-health-datetime")
        assert sections[1].has_class("system-health-cpu")
        assert sections[2].has_class("system-health-resource-row")
        resource_sections = sections[2]._pending_children
        assert [type(section) for section in resource_sections] == [
            MemorySwapWidget,
            SystemInfoWidget,
        ]
        assert resource_sections[0].has_class("system-health-memory-swap")
        assert resource_sections[1].has_class("system-health-system-info")

    def test_datetime_widget_composes_stylable_row(self) -> None:
        from textual.containers import Horizontal
        from textual.widgets import Digits, Static

        from llama_cli.tui.components.digital_clock import DigitalClockWidget
        from llama_cli.tui.components.system_health import DateTimeWidget, SystemHealthRenderer
        from llama_cli.tui.types import DateTimeSnapshot

        renderer = SystemHealthRenderer()
        renderer.current_datetime_snapshot = lambda: DateTimeSnapshot(  # type: ignore[method-assign]
            date_text="Wed 2026-05-20"
        )
        sections = list(DateTimeWidget(renderer).compose())

        assert len(sections) == 1
        row = sections[0]
        assert isinstance(row, Horizontal)
        assert row.has_class("system-health-datetime-row")
        children = row._pending_children
        assert isinstance(children[0], Static)
        assert children[0].has_class("llm-runner-logo")
        assert isinstance(children[1], Static)
        assert children[1].has_class("datetime-header-spacer")
        assert isinstance(children[2], Horizontal)
        assert children[2].has_class("datetime-far-right")
        far_right = children[2]._pending_children
        assert isinstance(far_right[0], Static)
        assert far_right[0].has_class("datetime-date")
        assert isinstance(far_right[1], DigitalClockWidget)
        clock_parts = list(far_right[1].compose())
        assert len(clock_parts) == 1
        assert isinstance(clock_parts[0], Digits)
        assert clock_parts[0].has_class("datetime-digits")

    def test_digital_clock_widget_composes_digits_only(self) -> None:
        from textual.widgets import Digits

        from llama_cli.tui.components.digital_clock import DigitalClockWidget

        sections = list(DigitalClockWidget().compose())
        assert len(sections) == 1
        assert isinstance(sections[0], Digits)
        assert sections[0].has_class("datetime-digits")

    def test_llm_runner_logo_reads_llm_block_letters(self) -> None:
        from llama_cli.tui.components.digital_clock import LLM_RUNNER_LOGO, _clean_markup

        clean_logo = _clean_markup(LLM_RUNNER_LOGO)
        assert "██╗       ██╗" in clean_logo
        assert "███╗   ███╗" in clean_logo
        assert "╚█████╝" in clean_logo
        assert LLM_RUNNER_LOGO.count("\n") == 6

    def test_system_info_widget_composes_stylable_rows(self) -> None:
        from textual.containers import Horizontal
        from textual.widgets import Static

        from llama_cli.tui.components.system_health import (
            SystemHealthRenderer,
            SystemInfoSnapshot,
            SystemInfoWidget,
        )

        renderer = SystemHealthRenderer()
        renderer.system_info_snapshot = MagicMock(  # type: ignore[method-assign]
            return_value=SystemInfoSnapshot(
                tasks=12,
                threads=345,
                running=2,
                load_values=(0.10, 0.20, 0.30),
                uptime="01:02:03",
            )
        )

        sections = list(SystemInfoWidget(renderer).compose())

        assert len(sections) == 3
        assert all(isinstance(section, Horizontal) for section in sections)
        assert all(section.has_class("system-info-row") for section in sections)
        first_row_children = sections[0]._pending_children
        assert isinstance(first_row_children[0], Static)
        assert first_row_children[0].has_class("system-info-label")
        assert isinstance(first_row_children[1], Static)
        assert first_row_children[1].has_class("system-info-primary-value")

    def test_system_info_widget_composes_load_na_row_when_missing(self) -> None:
        from llama_cli.tui.components.system_health import (
            SystemHealthRenderer,
            SystemInfoSnapshot,
            SystemInfoWidget,
        )

        renderer = SystemHealthRenderer()
        renderer.system_info_snapshot = MagicMock(  # type: ignore[method-assign]
            return_value=SystemInfoSnapshot(
                tasks=12,
                threads=345,
                running=2,
                load_values=None,
                uptime="01:02:03",
            )
        )

        sections = list(SystemInfoWidget(renderer).compose())

        assert len(sections) == 3
        assert sections[1]._pending_children[1].has_class("system-info-muted-value")

    def test_cpu_usage_widget_composes_stylable_rows(self) -> None:
        from textual.containers import Container, Horizontal

        from llama_cli.tui.components.system_health import CPUUsageWidget, SystemHealthRenderer

        renderer = SystemHealthRenderer()
        renderer.cpu_usage_rows = MagicMock(  # type: ignore[method-assign]
            return_value=[
                [SimpleNamespace(index=0, percent=12.5), SimpleNamespace(index=1, percent=87.0)]
            ]
        )

        rows = list(CPUUsageWidget(renderer).compose())

        assert len(rows) == 1
        assert isinstance(rows[0], Horizontal)
        assert rows[0].has_class("system-health-cpu-row")
        assert isinstance(rows[0]._pending_children[0], Container)
        assert rows[0]._pending_children[0].has_class("cpu-core-cell")
        assert rows[0]._pending_children[0]._pending_children[2].has_class("cpu-core-percent")

    def test_memory_swap_widget_composes_stylable_rows(self) -> None:
        from textual.containers import Horizontal

        from llama_cli.tui.components.system_health import MemorySwapWidget, SystemHealthRenderer

        renderer = SystemHealthRenderer()
        renderer.memory_usage_rows = MagicMock(  # type: ignore[method-assign]
            return_value=[
                SimpleNamespace(label="Mem", percent=50.0, value_text="8.00G/16.0G"),
                SimpleNamespace(label="Swp", percent=0.0, value_text="0.00G/2.00G"),
            ]
        )

        rows = list(MemorySwapWidget(renderer).compose())

        assert len(rows) == 2
        assert all(isinstance(row, Horizontal) for row in rows)
        assert all(row.has_class("memory-swap-row") for row in rows)

    def test_core_grid_respects_narrow_terminal_width(self) -> None:
        from llama_cli.tui.components.system_health import SystemHealthRenderer

        renderer = SystemHealthRenderer()

        lines = renderer._build_core_grid_lines([0.0] * 24, content_width=80)

        assert len(lines) > 3
        assert all(len(line) <= 80 for line in lines)

    def test_core_grid_auto_fits_more_columns_on_wide_terminals(self) -> None:
        from llama_cli.tui.components.system_health import SystemHealthRenderer

        renderer = SystemHealthRenderer()
        cpu_samples = [0.0] * 24

        standard_rows = renderer._build_core_grid_rows(cpu_samples, content_width=120)
        wide_rows = renderer._build_core_grid_rows(cpu_samples, content_width=240)

        assert len(standard_rows) == 4
        assert len(wide_rows) == 2

    def test_system_health_sections_use_available_width(self) -> None:
        import llama_cli.tui.components.system_health as system_health
        from llama_cli.tui.types import DateTimeSnapshot

        base_renderer = system_health.SystemHealthRenderer()
        provider = cast(
            system_health.SystemHealthProvider,
            SimpleNamespace(
                cpu_usage_rows=lambda width=None: base_renderer._build_core_grid_rows(
                    [0.0] * 24, base_renderer._content_width(width)
                ),
                memory_usage_rows=lambda: [
                    system_health.MemoryUsageSnapshot("Mem", 50.0, "8.00G/16.0G"),
                    system_health.MemoryUsageSnapshot("Swp", 0.0, "0.00G/2.00G"),
                ],
                system_info_snapshot=lambda: system_health.SystemInfoSnapshot(
                    tasks=0,
                    threads=0,
                    running=0,
                    load_values=(0.1, 0.2, 0.3),
                    uptime="00:00:00",
                ),
                current_datetime_snapshot=lambda: DateTimeSnapshot(date_text="Wed 2026-05-13"),
            ),
        )
        renderer = system_health.SystemHealthRenderer(provider)

        narrow_lines = (
            renderer.render_cpu_usage(width=80).splitlines()
            + renderer.render_memory_swap_usage(width=80).splitlines()
            + renderer.render_system_info().splitlines()
        )
        wide_lines = (
            renderer.render_cpu_usage(width=240).splitlines()
            + renderer.render_memory_swap_usage(width=240).splitlines()
            + renderer.render_system_info().splitlines()
        )

        assert all(len(line) <= 80 for line in narrow_lines)
        assert any(len(line) > 116 for line in wide_lines)
        assert all(len(line) <= renderer.MAX_CONTENT_WIDTH for line in wide_lines)

    def test_core_widgets_expose_semantic_css_classes(self) -> None:
        from llama_cli.tui.components.gpu_telemetry import GPUTelemetryWidget
        from llama_cli.tui.components.server_log import ServerLogPanel
        from llama_cli.tui.components.system_status import SystemStatusWidget

        view_model = MagicMock()

        status = SystemStatusWidget()
        gpu = GPUTelemetryWidget(view_model)
        panel = ServerLogPanel(0, view_model)

        assert status.id == "alerts"
        assert status.has_class("system-status")
        assert gpu.has_class("gpu-telemetry")
        assert panel.has_class("column")
        assert panel.has_class("server-log-panel")

    def test_gpu_telemetry_widget_composes_stylable_row(self) -> None:
        from textual.containers import Horizontal
        from textual.widgets import Static

        from llama_cli.tui.components.gpu_telemetry import GPUTelemetryWidget

        view_model = MagicMock()
        view_model.gpu_telemetry_lines.return_value = ["GPU0 42%", "GPU1 77%"]

        sections = list(GPUTelemetryWidget(view_model).compose())

        assert len(sections) == 1
        assert isinstance(sections[0], Horizontal)
        assert sections[0].has_class("gpu-telemetry-row")
        children = sections[0]._pending_children
        assert isinstance(children[0], Static)
        assert children[0].has_class("gpu-telemetry-label")
        assert isinstance(children[1], Static)
        assert children[1].has_class("gpu-telemetry-value")

    def test_gpu_telemetry_widget_hides_when_empty(self) -> None:
        from llama_cli.tui.components.gpu_telemetry import GPUTelemetryWidget

        view_model = MagicMock()
        view_model.gpu_telemetry_lines.return_value = []

        widget = GPUTelemetryWidget(view_model)

        assert list(widget.compose()) == []
        assert widget.has_class("hidden")

    def test_gpu_stats_panel_composes_stylable_sections(self) -> None:
        from textual.containers import Container, Horizontal
        from textual.widgets import Static

        from llama_cli.tui.components.gpu_stats import GPUStatsPanel

        stats = {
            "device": "Mock GPU",
            "gpu_util": "45%",
            "mem_util": "61%",
            "temp": "67C",
            "power": "120W",
        }

        sections = list(GPUStatsPanel(stats).compose())

        assert isinstance(sections[0], Static)
        assert sections[0].has_class("gpu-stats-title")
        assert isinstance(sections[1], Horizontal)
        assert sections[1].has_class("gpu-stats-row")
        assert isinstance(sections[2], Horizontal)
        assert sections[2].has_class("gpu-stats-usage-row")
        usage_items = sections[2]._pending_children
        assert isinstance(usage_items[0], Container)
        assert usage_items[0].has_class("gpu-stats-usage-item")

    def test_server_log_panel_composes_placeholder_when_empty(self) -> None:
        from textual.containers import Container

        from llama_cli.tui.components.server_log import ServerLogPanel

        view_model = MagicMock()
        view_model.column.return_value = None

        sections = list(ServerLogPanel(0, view_model).compose())

        assert len(sections) == 1
        assert isinstance(sections[0], Container)
        assert sections[0].has_class("server-placeholder")

    def test_server_log_panel_composes_server_column_widget(self) -> None:
        from llama_cli.tui.components.server_column import ServerColumnPanel
        from llama_cli.tui.components.server_log import ServerLogPanel
        from llama_cli.tui.types import ServerColumnState, SlotRuntimeStats

        view_model = MagicMock()
        view_model.column.return_value = ServerColumnState(
            alias="slot-a",
            profile_name="slot-a",
            status="offline",
            status_label="Offline",
            status_class="server-column-status-offline",
            backend_label="SYCL",
            url="http://127.0.0.1:8080",
            config_summary="Device: SYCL0 | Ctx: 2048 | Threads: 4",
            log_lines=("Waiting for output...",),
            runtime_stats=SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0"),
            gpu_stats=None,
            stale_warning=None,
        )

        sections = list(ServerLogPanel(0, view_model).compose())

        assert len(sections) == 1
        assert isinstance(sections[0], ServerColumnPanel)


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

    def test_launch_result_initialized(self) -> None:
        """DashboardController should initialize launch_result as None."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        assert app.launch_result is None

    def test_risk_prompt_initialized(self) -> None:
        """DashboardController should initialize without an active risk prompt."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        assert app.active_risk_kind is None


class TestGPUTelemetryPanel:
    """T016d: Tests for GPU telemetry panel update."""

    def test_gpu_stats_initialized(self) -> None:
        """GPUStats should be initialized for each configured slot."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0, 1])
        assert len(app.gpu_stats) == 1

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

    def test_gpu_stats_panel_handles_none_gpu(self) -> None:
        """GPUStatsPanel should render an unavailable state without GPU data."""
        from textual.widgets import Static

        from llama_cli.tui.components.gpu_stats import GPUStatsPanel

        sections = list(GPUStatsPanel(None).compose())

        assert len(sections) == 2
        assert isinstance(sections[1], Static)
        assert sections[1].has_class("gpu-stats-unavailable")


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

        audit = app.server_manager._audit.lifecycle_audit
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

    def test_blocked_launch_result_surfaces_notice(self) -> None:
        """Blocked launch results should surface a Textual notice."""
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

        app.launch_result = blocked_result

        assert "Launch blocked: no slots could be launched" in app.view_model.system_notices()

    def test_degraded_launch_result_surfaces_notice(self) -> None:
        """Degraded launch results should surface a Textual notice."""
        from llama_cli.tui import DashboardController
        from llama_manager.orchestration import LaunchResult

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        degraded_result = LaunchResult(
            status="degraded",
            launched=["slot1"],
            warnings=["slot2: lock already held"],
        )

        app.launch_result = degraded_result

        assert "Launch degraded: some slots blocked" in app.view_model.system_notices()

    def test_successful_launch_result_surfaces_no_notice(self) -> None:
        """Successful launches should not surface degraded or blocked notices."""
        from llama_cli.tui import DashboardController
        from llama_manager.orchestration import LaunchResult

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        success_result = LaunchResult(status="success", launched=["slot1"])
        app.launch_result = success_result

        assert "Launch blocked: no slots could be launched" not in app.view_model.system_notices()
        assert "Launch degraded: some slots blocked" not in app.view_model.system_notices()

    def test_risk_prompt_required(self) -> None:
        """_build_risk_panel_required should set the risk prompt state."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        app._build_risk_panel_required()

        assert app.active_risk_kind == "hardware"
        assert app.risks_acknowledged is False

    def test_risk_prompt_acknowledged(self) -> None:
        """_build_risk_panel_acknowledged should set acknowledged prompt state."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])
        app._build_risk_panel_acknowledged()

        assert app.active_risk_kind == "hardware"
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

    def test_signal_handler_calls_stop(self) -> None:
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
        """ServerManager.on_interrupt should call cleanup_servers and return code 130."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        cleanup_called = False

        def track_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        app.server_manager.cleanup_servers = track_cleanup  # type: ignore[assignment]

        exit_code = app.server_manager.on_interrupt(signal.SIGINT, None)

        assert exit_code == 130
        assert cleanup_called is True

    def test_on_terminate_calls_cleanup_and_exits(self) -> None:
        """ServerManager.on_terminate should call cleanup_servers and return code 143."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        cleanup_called = False

        def track_cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        app.server_manager.cleanup_servers = track_cleanup  # type: ignore[assignment]

        exit_code = app.server_manager.on_terminate(signal.SIGTERM, None)

        assert exit_code == 143
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

    def test_interrupt_triggers_graceful_shutdown(self) -> None:
        """interrupt should shut down when no risk prompt is active."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config(alias="slot0")], gpu_indices=[0])

        with patch.object(app, "_graceful_shutdown") as mock_shutdown:
            app.interrupt()

        mock_shutdown.assert_called_once()

    def test_refresh_display_appends_message(self) -> None:
        """refresh_display should add a visible status message."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.refresh_display()

        assert any("refreshed" in msg.lower() for _, msg in app._status_messages)

    def test_request_build_and_cancel_pending_prompt(self) -> None:
        """request_build should set build state and cancel_pending_prompt should clear it."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.request_build()
        assert app._build_request is True

        cancelled = app.cancel_pending_prompt()
        assert cancelled is True
        assert app._build_request is False

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


class TestHandleHardwareWarning:
    """T061: Tests for hardware warning TUI key handler."""

    def test_handle_hardware_warning_y_acknowledges(self) -> None:
        """handle_hardware_warning should acknowledge and clear panel on 'y'."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.active_risk_kind = "hardware"

        result = app.handle_hardware_warning("y")

        assert result == "acknowledge"
        assert app.active_risk_kind is None

    def test_handle_hardware_warning_n_aborts(self) -> None:
        """handle_hardware_warning should abort on 'n' key."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.active_risk_kind = "hardware"

        result = app.handle_hardware_warning("n")

        assert result == "abort"
        assert app.running is False

    def test_handle_hardware_warning_q_quits(self) -> None:
        """handle_hardware_warning should quit on 'q' key."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.active_risk_kind = "hardware"

        result = app.handle_hardware_warning("q")

        assert result == "quit"
        assert app.running is False

    def test_handle_hardware_warning_other_ignored(self) -> None:
        """handle_hardware_warning should ignore non-action keys."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.active_risk_kind = "hardware"

        result = app.handle_hardware_warning("x")

        assert result == "ignore"
        assert app.active_risk_kind == "hardware"


class TestHandleVramRisk:
    """T061b: Tests for VRAM risk confirmation TUI key handler."""

    def test_handle_vram_risk_y_proceeds(self) -> None:
        """handle_vram_risk should proceed on 'y' key and clear panel."""
        from llama_cli.tui import DashboardController

        app = DashboardController(configs=[_make_minimal_config()], gpu_indices=[0])

        app.active_risk_kind = "vram"

        result = app.handle_vram_risk("y")

        assert result == "proceed"
        assert app.active_risk_kind is None

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

        app.active_risk_kind = "vram"

        result = app.handle_vram_risk("x")

        assert result == "ignore"
        assert app.active_risk_kind == "vram"


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
        controller.request_build()

        state = controller.view_model.command_menu()

        assert state.build_request is True
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
        assert app.active_risk_kind is None
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

        with (
            patch.object(app, "refresh_dashboard") as mock_refresh,
            patch.object(app, "push_screen") as mock_push,
        ):
            app.action_build()
            app.action_confirm()
            app.action_reject()
            app.action_cancel_pending_prompt()
            app.action_refresh_dashboard()
            app.action_interrupt_dashboard()

        mock_push.assert_called_once()
        controller.acknowledge_risk.assert_called_once()
        controller.reject_risk.assert_called_once()
        controller.cancel_pending_prompt.assert_called_once()
        controller.refresh_display.assert_called_once()
        controller.interrupt.assert_called_once()
        assert mock_refresh.call_count == 4

    def test_emit_status_toasts_uses_popups_for_notices_and_status(self) -> None:
        controller = MagicMock()
        controller.view_model.system_notices.return_value = ["Launch degraded: some slots blocked"]
        controller.get_status_messages_since.return_value = [(1.0, "Slot launched")]
        app = TextualDashboardApp(controller)

        with patch.object(app, "notify") as notify:
            app._emit_status_toasts()
            app._emit_status_toasts()

        notify.assert_any_call(
            "Launch degraded: some slots blocked",
            title="Alert",
            severity="warning",
        )
        notify.assert_any_call("Slot launched", title="Status", severity="information")
        alert_calls = [
            call
            for call in notify.call_args_list
            if call.args == ("Launch degraded: some slots blocked",)
        ]
        assert len(alert_calls) == 1


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

    def test_textual_app_instantiates_for_empty_configs(self) -> None:
        """DashboardApp should be instantiable even with no slots."""
        controller = TUIApp(configs=[], gpu_indices=[])
        app = TextualDashboardApp(controller)
        assert app is not None

    def test_textual_app_instantiates_for_single_config(self) -> None:
        """DashboardApp should be instantiable with a single slot."""
        controller = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app = TextualDashboardApp(controller)
        assert app is not None

    def test_textual_app_instantiates_for_two_configs(self) -> None:
        """DashboardApp should be instantiable with two slots."""
        configs = [_make_config("model1", 8080, "CUDA"), _make_config("model2", 8081, "SYCL")]
        controller = TUIApp(configs=configs, gpu_indices=[0, 1])
        app = TextualDashboardApp(controller)
        assert app is not None


# =============================================================================
# Risk prompt state
# =============================================================================


class TestRiskPanels:
    """Tests for risk prompt state transitions."""

    def test_build_risk_panel_required(self) -> None:
        """_build_risk_panel_required should set risk prompt state."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_risk_panel_required()
        assert app.active_risk_kind == "hardware"
        assert app.risks_acknowledged is False

    def test_build_risk_panel_acknowledged(self) -> None:
        """_build_risk_panel_acknowledged should set acknowledged state."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app._build_risk_panel_acknowledged()
        assert app.active_risk_kind == "hardware"
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
        """_update_risk_panel_state should clear prompt state when no risks."""
        from llama_manager import RiskAckResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.active_risk_kind = "hardware"
        result = RiskAckResult(has_risks=False, risks_acknowledged=False)
        app._update_risk_panel_state(result)
        assert app.active_risk_kind is None
        assert app.risks_acknowledged is False


# =============================================================================
# Build wizard async status
# =============================================================================


def _minimal_build_status(backend: Any) -> Any:
    """Minimal BuildStatus for build wizard unit tests."""
    from llama_manager.build_pipeline import BuildStatus

    return BuildStatus(
        backend=backend,
        artifact_exists=False,
        artifact=None,
        binary_version_output=None,
        binary_exists_untracked=False,
        untracked_binary_path=None,
        source_exists=True,
        source_is_repo=True,
        source_branch="master",
        source_head_sha="a" * 40,
        source_remote_url="https://github.com/ggerganov/llama.cpp.git",
        configured_branch="master",
        remote_branch_sha="b" * 40,
    )


class TestBackendReadiness:
    """Tests for derive_backend_readiness badge and detail lines."""

    def test_loading_when_status_none(self) -> None:
        from llama_cli.tui.components.build import derive_backend_readiness

        readiness = derive_backend_readiness(None)
        assert readiness.level == "loading"
        assert readiness.badge == ""

    def test_missing_when_source_absent(self) -> None:
        from llama_cli.tui.components.build import derive_backend_readiness
        from llama_manager.build_pipeline import BuildBackend, BuildStatus

        status = BuildStatus(
            backend=BuildBackend.SYCL,
            artifact_exists=False,
            artifact=None,
            binary_version_output=None,
            binary_exists_untracked=False,
            untracked_binary_path=None,
            source_exists=False,
            source_is_repo=False,
            source_branch=None,
            source_head_sha=None,
            source_remote_url=None,
            configured_branch="master",
            remote_branch_sha=None,
        )
        readiness = derive_backend_readiness(status)
        assert readiness.level == "missing"
        assert readiness.badge == "Missing"

    def test_needs_update_when_no_binary(self) -> None:
        from llama_cli.tui.components.build import derive_backend_readiness
        from llama_manager.build_pipeline import BuildBackend

        status = _minimal_build_status(BuildBackend.SYCL)
        readiness = derive_backend_readiness(status)
        assert readiness.level == "needs_update"
        assert readiness.badge == "Needs update"

    def test_needs_update_when_source_behind_remote(self) -> None:
        from llama_cli.tui.components.build import derive_backend_readiness
        from llama_manager.build_pipeline import BuildBackend, BuildStatus

        sha = "a" * 40
        status = BuildStatus(
            backend=BuildBackend.CUDA,
            artifact_exists=True,
            artifact=None,
            binary_version_output="version: 1 (bbbbbbbb)",
            binary_exists_untracked=False,
            untracked_binary_path=None,
            source_exists=True,
            source_is_repo=True,
            source_branch="master",
            source_head_sha=sha,
            source_remote_url="https://github.com/ggerganov/llama.cpp.git",
            configured_branch="master",
            remote_branch_sha="c" * 40,
        )
        readiness = derive_backend_readiness(status)
        assert readiness.level == "needs_update"

    def test_needs_update_when_binary_commit_mismatch(self) -> None:
        from llama_cli.tui.components.build import derive_backend_readiness
        from llama_manager.build_pipeline import BuildBackend, BuildStatus

        sha = "a" * 40
        status = BuildStatus(
            backend=BuildBackend.SYCL,
            artifact_exists=True,
            artifact=None,
            binary_version_output="version: 1 (bbbbbbbb)",
            binary_exists_untracked=False,
            untracked_binary_path=None,
            source_exists=True,
            source_is_repo=True,
            source_branch="master",
            source_head_sha=sha,
            source_remote_url="https://github.com/ggerganov/llama.cpp.git",
            configured_branch="master",
            remote_branch_sha=sha,
        )
        readiness = derive_backend_readiness(status)
        assert readiness.level == "needs_update"

    def test_needs_update_untracked_binary_commit_behind_source(self) -> None:
        """CUDA default binary without provenance must not show Current when SHA differs."""
        from llama_cli.tui.components.build import derive_backend_readiness
        from llama_manager.build_pipeline import BuildBackend, BuildStatus

        source_sha = "29f14822" + "0" * 32
        status = BuildStatus(
            backend=BuildBackend.CUDA,
            artifact_exists=False,
            artifact=None,
            binary_version_output="version: 1 (f535774)",
            binary_exists_untracked=True,
            untracked_binary_path=None,
            source_exists=True,
            source_is_repo=True,
            source_branch="master",
            source_head_sha=source_sha,
            source_remote_url="https://github.com/ggerganov/llama.cpp.git",
            configured_branch="master",
            remote_branch_sha=source_sha,
        )
        readiness = derive_backend_readiness(status)
        assert readiness.level == "needs_update"

    def test_current_when_version_line_has_build_number_before_commit(self) -> None:
        """version: 312 (29f14822) must use the commit hash, not the build number."""
        from llama_cli.tui.components.build import derive_backend_readiness
        from llama_manager.build_pipeline import BuildArtifact, BuildBackend, BuildStatus

        sha = "29f14822" + "a" * 32
        status = BuildStatus(
            backend=BuildBackend.SYCL,
            artifact_exists=True,
            artifact=BuildArtifact(
                artifact_type="llama-server",
                backend=BuildBackend.SYCL,
                created_at=0.0,
                git_remote_url="",
                git_commit_sha=sha,
                git_branch="master",
                build_command=[],
                build_duration_seconds=1.0,
                exit_code=0,
                binary_path=None,
                binary_size_bytes=None,
                build_log_path=None,
                failure_report_path=None,
            ),
            binary_version_output=f"version: 312 ({sha[:8]})",
            binary_exists_untracked=False,
            untracked_binary_path=None,
            source_exists=True,
            source_is_repo=True,
            source_branch="master",
            source_head_sha=sha,
            source_remote_url="https://github.com/ggerganov/llama.cpp.git",
            configured_branch="master",
            remote_branch_sha=sha,
        )
        readiness = derive_backend_readiness(status)
        assert readiness.level == "current"

    def test_current_when_aligned(self) -> None:
        from llama_cli.tui.components.build import derive_backend_readiness
        from llama_manager.build_pipeline import BuildBackend, BuildStatus

        sha = "a" * 40
        short = sha[:8]
        status = BuildStatus(
            backend=BuildBackend.CUDA,
            artifact_exists=True,
            artifact=None,
            binary_version_output=f"version: 1 ({short})",
            binary_exists_untracked=False,
            untracked_binary_path=None,
            source_exists=True,
            source_is_repo=True,
            source_branch="master",
            source_head_sha=sha,
            source_remote_url="https://github.com/ggerganov/llama.cpp.git",
            configured_branch="master",
            remote_branch_sha=sha,
        )
        readiness = derive_backend_readiness(status)
        assert readiness.level == "current"
        assert readiness.badge == "Current"
        assert "[bold]Binary:[/]" in readiness.binary_line


class TestBuildModalAsyncStatus:
    """Tests for async build status fetch in BuildModalScreen."""

    def test_apply_single_backend_status_updates_sycl_only(self) -> None:
        """_apply_single_backend_status should update one backend without touching the other."""
        from llama_cli.tui.components.build import BuildModalScreen
        from llama_manager.build_pipeline import BuildBackend

        screen = BuildModalScreen()
        sycl = _minimal_build_status(BuildBackend.SYCL)
        mock_sycl_card = MagicMock()

        with (
            patch.object(screen, "_screen_can_apply_status", return_value=True),
            patch.object(screen, "_sycl_card", mock_sycl_card),
            patch.object(screen, "_cuda_card", None),
        ):
            screen._apply_single_backend_status("sycl", sycl)

        assert screen._wizard_state["sycl_status"] is sycl
        assert screen._wizard_state["cuda_status"] is None
        mock_sycl_card.set_status.assert_called_once_with(sycl)

    def test_apply_single_backend_status_skipped_when_detached(self) -> None:
        """_apply_single_backend_status should no-op when the screen is not attached."""
        from llama_cli.tui.components.build import BuildModalScreen
        from llama_manager.build_pipeline import BuildBackend

        screen = BuildModalScreen()
        sycl = _minimal_build_status(BuildBackend.SYCL)

        with patch.object(screen, "_screen_can_apply_status", return_value=False):
            screen._apply_single_backend_status("sycl", sycl)

        assert screen._wizard_state["sycl_status"] is None

    def test_apply_single_backend_status_stores_state_without_stale_card(self) -> None:
        """Status fetch after leaving step 1 must not touch removed cards."""
        from llama_cli.tui.components.build import BuildModalScreen
        from llama_manager.build_pipeline import BuildBackend

        screen = BuildModalScreen()
        cuda = _minimal_build_status(BuildBackend.CUDA)
        mock_cuda_card = MagicMock()
        mock_cuda_card.is_mounted = False

        screen._wizard_state["step"] = screen.STEP_BUILDING
        with (
            patch.object(screen, "_screen_can_apply_status", return_value=True),
            patch.object(screen, "_cuda_card", mock_cuda_card),
            patch.object(screen, "_status_card_for_backend", return_value=None),
        ):
            screen._apply_single_backend_status("cuda", cuda)

        assert screen._wizard_state["cuda_status"] is cuda
        mock_cuda_card.set_status.assert_not_called()

    def test_on_mount_starts_worker_without_sync_get_build_status(self) -> None:
        """on_mount should render immediately and defer get_build_status to a worker."""
        from llama_cli.tui.components.build import BuildModalScreen

        screen = BuildModalScreen()

        with (
            patch("llama_cli.tui.components.build.get_build_status") as mock_get_status,
            patch.object(screen, "call_later") as mock_call_later,
            patch.object(screen, "_fetch_build_status_worker") as mock_worker,
        ):
            screen.on_mount()

        mock_get_status.assert_not_called()
        mock_call_later.assert_called_once()
        mock_worker.assert_called_once()


# =============================================================================
# _handle_build_progress
# =============================================================================


class TestHandleBuildProgress:
    """Tests for TUIApp._handle_build_progress."""

    def test_handle_progress_running_does_not_push_status_message(self) -> None:
        """Routine running progress should update state without creating toast spam."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="running",
            message="Building object...",
            progress_percent=7,
        )
        app._handle_build_progress(progress)

        assert app.build_progress is progress
        assert app.get_status_messages_since(0.0) == []

    def test_handle_progress_retry(self) -> None:
        """_handle_build_progress should push a retry status message."""
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
        assert any(
            "Build retrying:" in message for _, message in app.get_status_messages_since(0.0)
        )

    def test_handle_progress_failure(self) -> None:
        """_handle_build_progress should push a failed status message."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="failed",
            message="Build failed",
            progress_percent=0,
        )
        app._handle_build_progress(progress)

        assert any("Build failed:" in message for _, message in app.get_status_messages_since(0.0))

    def test_handle_progress_success_reports_completion(self) -> None:
        """_handle_build_progress should report build completion on success."""
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.build_in_progress = True

        progress = BuildProgress(
            stage="build",
            status="success",
            message="Build successful",
            progress_percent=100,
        )
        app._handle_build_progress(progress)

        assert any(
            message == "Build completed successfully."
            for _, message in app.get_status_messages_since(0.0)
        )


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

    def test_handle_degraded_result(self) -> None:
        """_handle_launch_result should buffer warning for degraded result."""
        from llama_manager import LaunchResult

        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        launch_result = LaunchResult(status="degraded", warnings=["slot1 blocked"])

        app._handle_launch_result(launch_result)

        messages = [message for _ts, message in app._status_messages]
        assert any("degraded" in message.lower() for message in messages)


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
            app.model.build_cancel_event = MagicMock()

            app._signal_handler_build(2, None)

            mock_pipeline.release_lock.assert_called_once()
            assert app.build_in_progress is False
            app.model.build_cancel_event.set.assert_called_once()


# =============================================================================
# Stale profile cache warnings (read-only in TUI)
# =============================================================================


class TestStaleProfileWarnings:
    """Tests for cached benchmark profile staleness display in the TUI."""

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
                "llama_cli.tui.controller.load_profile_with_staleness",
                return_value=(MagicMock(), stale_result),
            ) as mock_load,
        ):
            mock_driver = MagicMock(return_value="driver-1")
            app.refresh_stale_warnings(mock_driver)
            warning = app.get_stale_warning(cfg)

        assert warning is not None
        assert "profile stale" in warning.lower()
        mock_gpu.assert_called_once_with(cfg.backend)
        mock_driver.assert_called_once_with(cfg.backend)
        assert mock_load.call_args.kwargs["gpu_identifier"] == "intel-arc_b580-00"
        assert mock_load.call_args.kwargs["current_driver_version"] == "driver-1"
        assert mock_load.call_args.kwargs["current_binary_version"] == "v1.2.3"


class TestBuildWizardResultMarkup:
    """Build wizard result text must escape user errors that contain Rich markup."""

    def test_result_content_preserves_brackets_in_error(self) -> None:
        from llama_cli.tui.components.build import BuildModalScreen

        screen = BuildModalScreen()
        err = "Configure failed: ['cmake', '-DGGML_SYCL=ON'] timed out after 0.25 seconds"
        screen._wizard_state["build_result_error"] = err
        content = screen._result_content(False)
        assert err in str(content)
        assert "Build failed" in str(content)


class TestBuildCommandMenu:
    """Command menu state lives in the view model for the Textual widget."""

    def test_normal_mode_shows_expected_commands(self) -> None:
        state = TUIApp(configs=[_make_config()], gpu_indices=[0]).view_model.command_menu()
        assert state.risk_prompt is None
        assert state.build_request is False

    def test_risk_prompt_shows_confirm_commands(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.active_risk_kind = "hardware"
        state = app.view_model.command_menu()
        assert state.risk_prompt is not None
        assert state.risk_prompt.kind == "hardware"

    def test_vram_prompt_hides_quit_in_state(self) -> None:
        app = TUIApp(configs=[_make_config()], gpu_indices=[0])
        app.active_risk_kind = "vram"
        state = app.view_model.command_menu()
        assert state.risk_prompt is not None
        assert state.risk_prompt.kind == "vram"


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


from textual.widgets import Select

import llama_cli.tui.components as components
from llama_cli.tui.components.modal import AddSlotModal


def test_panels_module_exports_expected_symbols() -> None:
    assert "GPUStatsPanel" in components.__all__
    assert "ServerColumnPanel" in components.__all__
    assert hasattr(components, "SystemStatusWidget")


def test_add_slot_modal_rejects_empty_profile_options() -> None:
    with pytest.raises(ValueError, match="profile_options must not be empty"):
        AddSlotModal([])


def test_add_slot_modal_composes_shared_modal_classes() -> None:
    from textual.containers import Container, Horizontal
    from textual.widgets import Button, Label

    modal = AddSlotModal([("Qwen", "qwen")])

    children = list(modal.compose())

    assert len(children) == 1
    dialog = children[0]
    assert isinstance(dialog, Container)
    assert dialog.id == "add-slot-dialog"
    assert dialog.has_class("modal-dialog")
    assert dialog.has_class("add-slot-dialog")

    dialog_children = dialog._pending_children
    assert isinstance(dialog_children[0], Label)
    assert dialog_children[0].has_class("modal-title")
    assert isinstance(dialog_children[1], Horizontal)
    assert dialog_children[1].has_class("form-row")
    assert isinstance(dialog_children[3], Horizontal)
    assert dialog_children[3].has_class("modal-actions")
    action_buttons = dialog_children[3]._pending_children
    assert isinstance(action_buttons[0], Button)
    assert action_buttons[0].has_class("modal-button-cancel")
    assert isinstance(action_buttons[1], Button)
    assert action_buttons[1].has_class("modal-button-success")


class TestDashboardAppCheckAction:
    """Footer visibility is driven by DashboardApp.check_action()."""

    @staticmethod
    def _app_for_state(
        *,
        risk_prompt: Any = None,
        build_request: bool = False,
    ) -> Any:
        from llama_cli.tui.textual_app import DashboardApp
        from llama_cli.tui.types import CommandMenuState

        controller = MagicMock()
        controller.view_model.command_menu.return_value = CommandMenuState(
            risk_prompt=risk_prompt,
            build_request=build_request,
        )
        return DashboardApp(controller)

    def test_normal_mode_shows_dashboard_actions(self) -> None:
        app = self._app_for_state()
        assert app.check_action("quit_dashboard", ()) is True
        assert app.check_action("refresh_dashboard", ()) is True
        assert app.check_action("confirm", ()) is False
        assert app.check_action("reject", ()) is False

    def test_risk_prompt_hides_dashboard_actions(self) -> None:
        from llama_cli.tui.types import RiskPromptState

        app = self._app_for_state(
            risk_prompt=RiskPromptState(kind="hardware", acknowledged=False),
        )
        assert app.check_action("confirm", ()) is True
        assert app.check_action("build", ()) is False
        assert app.check_action("quit_dashboard", ()) is True

    def test_vram_risk_hides_quit(self) -> None:
        from llama_cli.tui.types import RiskPromptState

        app = self._app_for_state(
            risk_prompt=RiskPromptState(kind="vram", acknowledged=False),
        )
        assert app.check_action("quit_dashboard", ()) is False
        assert app.check_action("confirm", ()) is True

    def test_build_request_only_shows_cancel(self) -> None:
        app = self._app_for_state(build_request=True)
        assert app.check_action("cancel_pending_prompt", ()) is True
        assert app.check_action("quit_dashboard", ()) is False
        assert app.check_action("build", ()) is False


@pytest.mark.anyio
async def test_config_modal_composes_shared_modal_classes() -> None:
    from textual.app import App
    from textual.containers import Container, Horizontal, VerticalScroll
    from textual.widgets import Button, Label

    from llama_cli.tui.components.config_modal import ConfigModal
    from llama_manager.config import Config

    class ModalHostApp(App[None]):
        pass

    modal = ConfigModal(Config())

    app = ModalHostApp()
    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        dialog = modal.query_one("#config-dialog", Container)
        assert dialog.has_class("modal-dialog")
        assert dialog.has_class("config-dialog")

        dialog_children = list(dialog.children)
        assert isinstance(dialog_children[0], Label)
        assert dialog_children[0].has_class("modal-title")
        assert dialog_children[0].has_class("config-title")
        assert isinstance(dialog_children[1], VerticalScroll)
        assert dialog_children[1].has_class("modal-scroll-body")
        assert isinstance(dialog_children[2], Horizontal)
        assert dialog_children[2].has_class("modal-actions")
        assert dialog_children[2].has_class("config-actions")
        action_buttons = list(dialog_children[2].children)
        assert isinstance(action_buttons[0], Button)
        assert action_buttons[0].has_class("modal-button-cancel")
        assert isinstance(action_buttons[1], Button)
        assert action_buttons[1].has_class("modal-button-success")
        assert isinstance(action_buttons[2], Button)
        assert action_buttons[2].has_class("modal-button-warning")


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


# =============================================================================
# Layout geometry regression
# =============================================================================


@pytest.mark.anyio
async def test_dashboard_app_layout_geometry_regression() -> None:
    """CPUUsageWidget stays compact; system widgets above #content; non-zero sizes."""
    from llama_cli.tui.components.gpu_stats import GPUStatsPanel
    from llama_cli.tui.components.gpu_telemetry import GPUTelemetryWidget
    from llama_cli.tui.components.server_column import ServerColumnPanel
    from llama_cli.tui.components.system_health import (
        CPUUsageWidget,
        DateTimeWidget,
        MemorySwapWidget,
        SystemInfoWidget,
    )
    from llama_cli.tui.textual_app import DashboardApp
    from llama_cli.tui.types import (
        CPUCoreSnapshot,
        DateTimeSnapshot,
        MemoryUsageSnapshot,
        ServerColumnState,
        SlotRuntimeStats,
        SystemInfoSnapshot,
    )

    controller = DashboardController(
        configs=[_make_config()],
        gpu_indices=[],
        register_signals=False,
    )
    controller.view_model.cpu_usage_rows = MagicMock(  # type: ignore[method-assign]
        return_value=[
            [CPUCoreSnapshot(index=col * 3 + row, percent=0.0) for col in range(8)]
            for row in range(3)
        ]
    )
    controller.view_model.memory_usage_rows = MagicMock(  # type: ignore[method-assign]
        return_value=[
            MemoryUsageSnapshot("Mem", 50.0, "8.00G/16.0G"),
            MemoryUsageSnapshot("Swp", 0.0, "0.00G/2.00G"),
        ]
    )
    controller.view_model.system_info_snapshot = MagicMock(  # type: ignore[method-assign]
        return_value=SystemInfoSnapshot(
            tasks=0,
            threads=0,
            running=0,
            load_values=(0.1, 0.2, 0.3),
            uptime="00:00:00",
        )
    )
    controller.view_model.current_datetime_snapshot = MagicMock(  # type: ignore[method-assign]
        return_value=DateTimeSnapshot(date_text="Wed 2026-05-13")
    )

    controller.view_model.gpu_telemetry_lines = MagicMock(return_value=[])  # type: ignore[method-assign]
    controller.view_model.system_notices = MagicMock(return_value=[])  # type: ignore[method-assign]
    controller.view_model.column = MagicMock(  # type: ignore[method-assign]
        return_value=ServerColumnState(
            alias="slot0",
            profile_name="slot0",
            status="offline",
            status_label="Offline",
            status_class="server-column-status-offline",
            backend_label="SYCL",
            url="http://127.0.0.1:8080",
            config_summary="Device: SYCL0 | Ctx: 2048 | Threads: 4",
            log_lines=("Waiting for output...",),
            runtime_stats=SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0"),
            gpu_stats=None,
            stale_warning=None,
        ),
    )

    app = DashboardApp(controller)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        datetime_row = app.query_one(DateTimeWidget)
        cpu = app.query_one(CPUUsageWidget)
        mem = app.query_one(MemorySwapWidget)
        info = app.query_one(SystemInfoWidget)
        content = app.query_one("#content")

        assert datetime_row.region.height >= 7, (
            f"DateTimeWidget height {datetime_row.region.height} < 7 (logo row)"
        )
        assert cpu.region.height < 10, f"CPUUsageWidget height {cpu.region.height} >= 10"

        datetime_bottom = datetime_row.region.y + datetime_row.region.height
        assert content.region.y >= datetime_bottom

        assert mem.region.y + mem.region.height <= content.region.y
        assert info.region.y + info.region.height <= content.region.y
        assert not list(app.query(GPUTelemetryWidget))

        assert list(app.query(ServerColumnPanel))
        for _ in range(5):
            if list(app.query(GPUStatsPanel)):
                break
            await pilot.pause(0.05)
        assert list(app.query(GPUStatsPanel))

        from textual.widgets import Footer

        footer = app.query_one(Footer)
        assert footer.region.height == 1, f"Footer height {footer.region.height} != 1"
        assert footer.show_command_palette is False
        descriptions = {
            binding.description
            for (_, binding, _enabled, _tooltip) in app.screen.active_bindings.values()
            if binding.show and app.check_action(binding.action, ()) is not False
        }
        assert "Quit" in descriptions
        assert "Refresh" in descriptions


# =============================================================================
# ViewModel profile_options (Task 7: MVVM registry leakage cleanup)
# =============================================================================


class TestViewModelProfileOptions:
    """Tests for DashboardViewModel.profile_options()."""

    def test_profile_options_returns_label_value_tuples(self) -> None:
        """profile_options should return (label, value) pairs for each profile."""
        controller = DashboardController(configs=[_make_config()], gpu_indices=[])
        options = controller.view_model.profile_options()

        assert isinstance(options, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in options)

    def test_profile_options_contains_expected_profiles(self) -> None:
        """profile_options should include the built-in profiles."""
        controller = DashboardController(configs=[_make_config()], gpu_indices=[])
        options = controller.view_model.profile_options()

        profile_ids = [value for _, value in options]
        assert "summary-balanced" in profile_ids
        assert "summary-fast" in profile_ids
        assert "qwen35" in profile_ids

    def test_profile_options_labels_include_profile_id_and_description(self) -> None:
        """Profile labels should combine profile_id and description."""
        controller = DashboardController(configs=[_make_config()], gpu_indices=[])
        options = controller.view_model.profile_options()

        labels = [label for label, _ in options]
        assert any("summary-balanced" in label for label in labels)
        assert any("summary-fast" in label for label in labels)
        assert any("qwen35" in label for label in labels)

    def test_profile_options_with_config_passes_config_to_registry(self) -> None:
        """profile_options should accept and pass a Config to create_default_profile_registry."""
        from llama_manager.config import Config

        controller = DashboardController(configs=[_make_config()], gpu_indices=[])
        custom_config = Config()
        options = controller.view_model.profile_options(custom_config)

        assert isinstance(options, list)
        assert len(options) > 0

    def test_app_delegates_profile_options_to_viewmodel(self) -> None:
        """DashboardApp._build_profile_options should call viewmodel.profile_options()."""
        controller = DashboardController(configs=[_make_config()], gpu_indices=[0])
        app = DashboardApp(controller)

        # The app caches results, so first call populates the cache
        options = app._build_profile_options()

        profile_ids = [value for _, value in options]
        assert "summary-balanced" in profile_ids
        assert "summary-fast" in profile_ids
        assert "qwen35" in profile_ids

    def test_app_profile_options_uses_controller_config(self) -> None:
        """_build_profile_options should pass controller.config to viewmodel."""
        controller = DashboardController(configs=[_make_config()], gpu_indices=[0])
        app = DashboardApp(controller)

        # Clear cache to force a fresh call
        app._profile_options_cache = None
        app._profile_cache_config_id = None

        # Should not raise and should use controller.config
        options = app._build_profile_options()
        assert len(options) > 0
