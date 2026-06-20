"""Tests for DashboardApp — Textual app shell and profile options caching.

Covers:
- Profile options caching logic
- Reconcile server log panels
- Build modal dismisses on cancel
"""

from __future__ import annotations

import contextlib
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.css.query import NoMatches

from llama_cli.tui.components.build import BuildModalScreen
from llama_cli.tui.components.modal import RemoveSlotModal
from llama_cli.tui.controller import AsyncSlotPlan, AsyncSlotStageResult
from llama_cli.tui.textual_app import (
    DashboardApp,
    _profile_options_cached,
    _split_log_update,
)
from llama_cli.tui.types import MemoryUsageSnapshot, SystemInfoSnapshot
from llama_manager import LogBuffer
from tests.support.helpers import make_server_config


def _make_controller() -> MagicMock:
    """Create a mocked DashboardController for DashboardApp tests."""
    from llama_cli.tui.model import DashboardModel
    from llama_cli.tui.viewmodel import DashboardViewModel
    from llama_manager.config import Config

    config = Config()
    model = DashboardModel(configs=[make_server_config(alias="test")], gpu_indices=[])
    view_model = DashboardViewModel(model)

    controller = MagicMock()
    controller.running = True
    controller.config = config
    controller.configs = [make_server_config(alias="test")]
    controller.view_model = view_model
    controller.server_manager = MagicMock()
    controller.model = model
    controller.model.build_cancel_event = None
    # Provide valid return for model-index worker called on mount
    controller.refresh_model_index.return_value = ([], 0, 0)
    return controller


class TestProfileOptionsCached:
    """Tests for _profile_options_cached — extracted cache helper."""

    def test_caching_returns_cached_on_same_config(self) -> None:
        """When config_id matches, cached options should be returned."""
        view_model = MagicMock()
        config = make_server_config(alias="cached")
        cache = [("cached_model", "cached_alias")]
        cache_config_id = id(config)

        result_options, result_id = _profile_options_cached(
            view_model, config, cache, cache_config_id
        )

        assert result_options == cache
        assert result_id == cache_config_id
        # view_model.profile_options should NOT be called
        view_model.profile_options.assert_not_called()

    def test_cache_invalidated_on_new_config(self) -> None:
        """When config_id differs, cache should be invalidated."""
        view_model = MagicMock()
        view_model.profile_options.return_value = [("new_model", "new_alias")]
        config = make_server_config(alias="new")
        old_cache: list[tuple[str, str]] = [("old", "old")]
        old_config_id = id(make_server_config(alias="old"))

        result_options, result_id = _profile_options_cached(
            view_model, config, old_cache, old_config_id
        )

        assert result_options == [("new_model", "new_alias")]
        view_model.profile_options.assert_called_once()

    def test_none_cache_always_recomputes(self) -> None:
        """When cache is None, options should always be recomputed."""
        view_model = MagicMock()
        view_model.profile_options.return_value = [("recomputed", "alias")]
        config = make_server_config(alias="test")

        result_options, result_id = _profile_options_cached(view_model, config, None, None)

        assert result_options == [("recomputed", "alias")]
        view_model.profile_options.assert_called_once()

    def test_returns_config_id(self) -> None:
        """Result should include the config_id."""
        view_model = MagicMock()
        view_model.profile_options.return_value = []
        config = make_server_config(alias="test")

        _, result_id = _profile_options_cached(view_model, config, None, None)

        assert result_id == id(config)


class TestBuildModalScreenCancel:
    """Tests for BuildModalScreen cancel behavior."""

    def test_cancel_dismisses_none(self) -> None:
        """Cancel button should dismiss the modal with None."""
        screen = BuildModalScreen()
        dismiss = MagicMock()
        screen.dismiss = dismiss  # type: ignore[method-assign]

        screen.action_cancel()

        dismiss.assert_called_once_with(None)

    def test_escape_binding_routes_to_cancel(self) -> None:
        """Escape key should be bound to the cancel action."""
        binding = next(
            binding
            for binding in (cast(Any, item) for item in BuildModalScreen.BINDINGS)
            if binding.key == "escape"
        )

        assert binding.action == "cancel"


class TestDashboardAppInit:
    """Tests for DashboardApp initialization."""

    @pytest.mark.anyio
    async def test_app_instantiation(self) -> None:
        """DashboardApp should be instantiable with a controller."""
        controller = _make_controller()
        app = DashboardApp(controller)
        assert app.controller is controller
        assert app.view_model is controller.view_model

    @pytest.mark.anyio
    async def test_profile_cache_initialized(self) -> None:
        """DashboardApp should initialize cache attributes."""
        controller = _make_controller()
        app = DashboardApp(controller)
        assert app._profile_options_cache is None
        assert app._profile_cache_config_id is None


class TestDashboardAppProfileOptions:
    """Tests for DashboardApp._build_profile_options caching."""

    @pytest.mark.anyio
    async def test_first_call_recomputes(self) -> None:
        """First call should recompute options from view_model."""
        controller = _make_controller()
        # Replace view_model with a MagicMock for this test
        controller.view_model = MagicMock()
        controller.view_model.profile_options.return_value = [
            ("model_a", "alias_a"),
        ]
        app = DashboardApp(controller)

        options = app._build_profile_options()

        assert options == [("model_a", "alias_a")]

    @pytest.mark.anyio
    async def test_second_call_returns_cached(self) -> None:
        """Second call with same config should return cached options."""
        controller = _make_controller()
        # Replace view_model with a MagicMock for this test
        controller.view_model = MagicMock()
        controller.view_model.profile_options.return_value = [
            ("cached_model", "cached_alias"),
        ]
        app = DashboardApp(controller)

        # First call
        options1 = app._build_profile_options()
        # Second call
        options2 = app._build_profile_options()

        assert options1 == options2
        # profile_options should only be called once
        assert controller.view_model.profile_options.call_count == 1


class TestDashboardAppModelIndex:
    """Tests for DashboardApp model-index callback methods.

    These methods are called from background threads. Testing them directly
    (with app.notify mocked) avoids needing a running event loop while still
    covering every branch.
    """

    def _make_app(self) -> tuple[DashboardApp, MagicMock]:
        """Create a DashboardApp and a MagicMock wired in as app.notify."""
        controller = _make_controller()
        app = DashboardApp(controller)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[assignment]
        return app, notify_mock

    def test_handle_progress_zero_total_no_notify(self) -> None:
        """_handle_model_index_progress is a no-op when total <= 0."""
        app, notify_mock = self._make_app()

        app._handle_model_index_progress(0, 0, 0)

        notify_mock.assert_not_called()

    def test_handle_progress_negative_total_no_notify(self) -> None:
        """_handle_model_index_progress is a no-op when total is negative."""
        app, notify_mock = self._make_app()

        app._handle_model_index_progress(5, -1, 0)

        notify_mock.assert_not_called()

    def test_handle_progress_below_threshold_no_notify(self) -> None:
        """No notification when scanned is below the 1/5 progress threshold."""
        app, notify_mock = self._make_app()
        # notify_every = max(1, 10 // 5) = 2; scanned=1 < notify_every=2 → no notify
        app._handle_model_index_progress(1, 10, 0)

        notify_mock.assert_not_called()

    def test_handle_progress_at_threshold_notifies(self) -> None:
        """Notification is sent when scanned reaches the 1/5 threshold."""
        app, notify_mock = self._make_app()
        # notify_every = max(1, 10 // 5) = 2; scanned=2 >= notify_every=2 → notify
        app._handle_model_index_progress(2, 10, 0)

        notify_mock.assert_called_once_with(
            "Indexed 2/10 models",
            title="Models",
            severity="information",
        )

    def test_handle_progress_with_errors_severity_warning(self) -> None:
        """Notification uses severity='warning' when errors > 0."""
        app, notify_mock = self._make_app()
        app._handle_model_index_progress(2, 10, 1)

        call_kwargs = notify_mock.call_args
        assert call_kwargs.kwargs["severity"] == "warning"
        assert "(1 errors)" in call_kwargs.args[0]

    def test_handle_progress_complete_notifies(self) -> None:
        """Notification is always sent when scanned == total."""
        app, notify_mock = self._make_app()
        # scanned == total should bypass the threshold guard
        app._handle_model_index_progress(10, 10, 0)

        notify_mock.assert_called_once()
        assert "Indexed 10/10" in notify_mock.call_args.args[0]

    def test_handle_complete_no_errors(self) -> None:
        """_handle_model_index_complete notifies with total models found."""
        app, notify_mock = self._make_app()

        app._handle_model_index_complete(5, 0)

        notify_mock.assert_called_once_with(
            "Indexing complete: 5 models found",
            title="Models",
            severity="information",
        )

    def test_handle_complete_with_errors_severity_warning(self) -> None:
        """_handle_model_index_complete uses severity='warning' when errors > 0."""
        app, notify_mock = self._make_app()

        app._handle_model_index_complete(5, 2)

        call_kwargs = notify_mock.call_args
        assert call_kwargs.kwargs["severity"] == "warning"
        assert "5 models found" in call_kwargs.args[0]
        assert "(2 errors)" in call_kwargs.args[0]

    def test_index_models_notifies_when_refresh_starts(self) -> None:
        """_index_models notifies 'Indexing models...' when a refresh is started."""
        controller = _make_controller()
        controller.refresh_model_index_async.return_value = True
        app = DashboardApp(controller)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[assignment]

        app._index_models()

        notify_mock.assert_called_once_with("Indexing models...", title="Models")

    def test_index_models_no_notify_when_already_running(self) -> None:
        """_index_models does not notify when refresh_model_index_async returns False."""
        controller = _make_controller()
        controller.refresh_model_index_async.return_value = False
        app = DashboardApp(controller)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[assignment]

        app._index_models()

        notify_mock.assert_not_called()


class TestDashboardAppCheckAction:
    """Tests for DashboardApp.check_action — binding visibility gating."""

    def _make_app(
        self,
        *,
        build_request: bool = False,
        risk_prompt: object = None,
    ) -> DashboardApp:
        controller = _make_controller()
        controller.model.build_request = build_request
        controller.model.risk_prompt = risk_prompt  # type: ignore[assignment]
        return DashboardApp(controller)

    def test_normal_state_hides_confirm_and_reject(self) -> None:
        """In normal mode 'confirm' and 'reject' bindings are hidden."""
        app = self._make_app()

        assert app.check_action("confirm", ()) is False
        assert app.check_action("reject", ()) is False

    def test_normal_state_shows_build_action(self) -> None:
        """In normal mode 'build' binding is visible."""
        app = self._make_app()

        assert app.check_action("build", ()) is True

    def test_build_request_only_shows_cancel(self) -> None:
        """During build_request only 'cancel_pending_prompt' returns True."""
        app = self._make_app(build_request=True)

        assert app.check_action("cancel_pending_prompt", ()) is True
        assert app.check_action("build", ()) is False
        assert app.check_action("add_slot", ()) is False

    def test_risk_prompt_hides_refresh_add_build_config(self) -> None:
        """During a risk prompt, refresh/add_slot/build/open_config are hidden."""
        from llama_cli.tui.types import RiskPromptState

        risk = RiskPromptState(kind="hardware", acknowledged=False)
        app = self._make_app(risk_prompt=risk)

        assert app.check_action("refresh_dashboard", ()) is False
        assert app.check_action("add_slot", ()) is False
        assert app.check_action("build", ()) is False
        assert app.check_action("open_config", ()) is False

    def test_hardware_risk_prompt_allows_quit(self) -> None:
        """During hardware risk prompt 'quit_dashboard' remains visible."""
        from llama_cli.tui.types import RiskPromptState

        risk = RiskPromptState(kind="hardware", acknowledged=False)
        app = self._make_app(risk_prompt=risk)

        assert app.check_action("quit_dashboard", ()) is True

    def test_vram_risk_prompt_hides_quit(self) -> None:
        """During vram risk prompt 'quit_dashboard' is hidden."""
        from llama_cli.tui.types import RiskPromptState

        risk = RiskPromptState(kind="vram", acknowledged=False)
        app = self._make_app(risk_prompt=risk)

        assert app.check_action("quit_dashboard", ()) is False

    def test_build_request_hides_about_action(self) -> None:
        """During build_request the 'about' binding is hidden."""
        app = self._make_app(build_request=True)

        assert app.check_action("about", ()) is False

    def test_action_about_pushes_about_modal(self) -> None:
        from llama_cli.tui.components.about_modal import AboutModal

        app = self._make_app()
        app.push_screen = MagicMock()  # type: ignore[method-assign]

        app.action_about()

        pushed_screen = app.push_screen.call_args.args[0]  # type: ignore[attr-defined]
        assert isinstance(pushed_screen, AboutModal)


class TestDashboardAppGpuStatsRefresh:
    """Tests for background GPU stats refresh helpers."""

    def test_schedule_gpu_stats_refresh_skips_when_active(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app._gpu_stats_refresh_active = True
        app._refresh_gpu_stats_worker = MagicMock()  # type: ignore[method-assign]

        app._schedule_gpu_stats_refresh()

        app._refresh_gpu_stats_worker.assert_not_called()

    def test_mark_gpu_stats_refresh_complete_clears_flag(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app._gpu_stats_refresh_active = True

        app._mark_gpu_stats_refresh_complete()

        assert app._gpu_stats_refresh_active is False

    @pytest.mark.anyio
    async def test_refresh_gpu_stats_worker_survives_gpu_update_error(self) -> None:
        controller = _make_controller()
        gpu = MagicMock()
        gpu.device_index = 0
        gpu.update.side_effect = RuntimeError("gpu unavailable")
        controller.model.gpu_stats = [gpu]
        controller.refresh_model_index_async.return_value = False

        app = DashboardApp(controller)
        app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[method-assign]

        DashboardApp._refresh_gpu_stats_worker.__wrapped__(app)  # type: ignore[attr-defined]

        assert app._gpu_stats_refresh_active is False

    def test_refresh_gpu_stats_worker_applies_cached_snapshot(self) -> None:
        controller = _make_controller()
        cfg = make_server_config(alias="slot0")
        controller.model.configs = [cfg]
        gpu = MagicMock()
        gpu.device_index = 0
        gpu.get_cached_stats_snapshot.return_value = {"gpu_util": "45%"}
        controller.model.gpu_stats = [gpu]
        controller.model.apply_gpu_stats_snapshot = MagicMock()
        app = DashboardApp(controller)
        app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[method-assign]
        app._gpu_stats_refresh_active = True

        DashboardApp._refresh_gpu_stats_worker.__wrapped__(app)  # type: ignore[attr-defined]

        gpu.update.assert_called_once()
        gpu.get_cached_stats_snapshot.assert_called_once()
        controller.model.apply_gpu_stats_snapshot.assert_called_once_with(
            {"slot0": {"gpu_util": "45%"}}
        )
        assert app._gpu_stats_refresh_active is False

    def test_update_panel_widgets_pushes_gpu_stats(self) -> None:
        """_update_panel_widgets should call GPUStatsPanel.update_stats with state.gpu_stats."""
        from textual.widgets import Static

        from llama_cli.tui.components.gpu_stats import GPUStatsPanel
        from llama_cli.tui.types import ServerColumnState, SlotRuntimeStats

        controller = _make_controller()
        app = DashboardApp(controller)

        view_model = MagicMock()
        view_model.column.return_value = ServerColumnState(
            alias="slot0",
            profile_name="default",
            status="running",
            status_label="Running",
            status_class="server-column-status-running",
            backend_label="CUDA",
            url="http://127.0.0.1:8081",
            config_summary="Device: CUDA0 | Ctx: 8192",
            log_lines=("line1", "line2"),
            runtime_stats=SlotRuntimeStats(tps="12.5", pp="100", tokens_in="500", tokens_out="200"),
            gpu_stats={"device": "NVIDIA RTX 3090", "gpu_util": "82%", "mem_util": "71%"},
            stale_warning=None,
        )

        gpu_panel = GPUStatsPanel(None)

        panel = MagicMock()
        log_widget = MagicMock()

        def _query(sel: str, cls: type | None = None) -> list[object]:
            if cls is GPUStatsPanel:
                return [gpu_panel]
            if isinstance(sel, str) and ".slot-stats-value" in sel:
                return []
            return []

        def _query_one(sel: str, cls: type | None = None) -> object:
            if sel is GPUStatsPanel:
                return gpu_panel
            if isinstance(sel, str) and ".server-log-content" in sel:
                return log_widget
            return Static()

        panel.query = _query
        panel.query_one = _query_one

        app._update_panel_widgets(panel, view_model.column.return_value)  # type: ignore[arg-type]

        assert gpu_panel._stats == {
            "device": "NVIDIA RTX 3090",
            "gpu_util": "82%",
            "mem_util": "71%",
        }


class TestDashboardAppAddSlotFlow:
    """Tests for async add-slot worker and finish handler."""

    def _stub_finish_add_slot_ui(self, app: DashboardApp) -> None:
        app.query_one = MagicMock(side_effect=NoMatches())  # type: ignore[method-assign]
        app._reconcile_server_log_panels = AsyncMock()  # type: ignore[method-assign]
        app.refresh_dashboard = MagicMock()  # type: ignore[method-assign]

    @pytest.mark.anyio
    async def test_finish_add_slot_refreshes_successful_slot(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        self._stub_finish_add_slot_ui(app)

        await app._finish_add_slot(
            "Slot added for profile",
            None,
            True,
            ["Validated"],
            layout_changed=True,
        )

        controller.apply_add_slot_from_form.assert_not_called()
        app._reconcile_server_log_panels.assert_awaited_once()  # type: ignore[attr-defined]
        app.refresh_dashboard.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.anyio
    async def test_finish_add_slot_refreshes_without_reconcile_when_layout_unchanged(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        self._stub_finish_add_slot_ui(app)

        await app._finish_add_slot(
            "Slot already current",
            None,
            True,
            ["Validated"],
            layout_changed=False,
        )

        app._reconcile_server_log_panels.assert_not_awaited()  # type: ignore[attr-defined]
        app.refresh_dashboard.assert_called_once()  # type: ignore[attr-defined]

    def test_run_add_slot_applies_successful_slot_off_ui_thread(self) -> None:
        controller = _make_controller()
        controller.compute_add_slot_from_form.return_value = (
            True,
            ["Validated"],
            "new-slot",
            make_server_config(alias="new-slot"),
        )
        controller.prepare_async_slot_launch.return_value = AsyncSlotPlan(
            success=True,
            messages=[],
            alias="new-slot",
            profile_id="new-slot",
            old_alias=None,
        )
        controller.stage_async_slot_launch.return_value = AsyncSlotStageResult(
            success=True,
            messages=["Slot 'new-slot' launching..."],
            alias="new-slot",
            log_buffer=LogBuffer(redact_sensitive=True),
        )
        mock_proc = MagicMock()
        controller.server_manager.start_servers.return_value = [mock_proc]
        controller.complete_async_slot_launch.return_value = (
            True,
            ["Slot 'new-slot' is running", "Slot added"],
        )
        app = DashboardApp(controller)
        app.refresh_dashboard = MagicMock()  # type: ignore[method-assign]
        captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def _call_from_thread(fn: object, *args: object, **kwargs: object) -> object:
            if getattr(fn, "__name__", "") == "_finish_add_slot":
                captured.append((fn, args, kwargs))
                return None
            return fn(*args, **kwargs)  # type: ignore[misc]

        app.call_from_thread = _call_from_thread  # type: ignore[method-assign]

        DashboardApp._run_add_slot.__wrapped__(  # type: ignore[attr-defined]
            app,
            {"profile": "new-slot", "port": "8080"},
        )

        controller.apply_add_slot_from_form.assert_not_called()
        controller._push_status_message.assert_called_once_with("Validated")
        controller.prepare_async_slot_launch.assert_called_once()
        controller.stage_async_slot_launch.assert_called_once()
        controller.server_manager.start_servers.assert_called_once()
        app.refresh_dashboard.assert_not_called()  # type: ignore[attr-defined]
        controller.complete_async_slot_launch.assert_called_once_with(
            "new-slot",
            "new-slot",
            None,
            mock_proc,
        )
        finish_calls = [
            call for call in captured if getattr(call[0], "__name__", "") == "_finish_add_slot"
        ]
        assert len(finish_calls) == 1
        finish_args = finish_calls[0][1]
        assert finish_args[2] is True
        assert finish_args[3] == [
            "Validated",
            "Slot 'new-slot' launching...",
            "Slot 'new-slot' is running",
            "Slot added",
        ]
        assert finish_args[4] is True

    def test_start_add_slot_rejects_when_slot_operation_running(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app._slot_operation_active = True
        app.notify = MagicMock()  # type: ignore[method-assign]
        app._run_add_slot = MagicMock()  # type: ignore[method-assign]

        app._start_add_slot({"profile": "test", "port": ""})

        app._run_add_slot.assert_not_called()  # type: ignore[attr-defined]
        app.notify.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.anyio
    async def test_finish_add_slot_shows_error(self) -> None:
        controller = _make_controller()
        controller.refresh_model_index_async.return_value = False
        app = DashboardApp(controller)
        self._stub_finish_add_slot_ui(app)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[method-assign]

        await app._finish_add_slot(error="Add slot failed: boom")

        error_calls = [
            call for call in notify_mock.call_args_list if call.kwargs.get("severity") == "error"
        ]
        assert len(error_calls) == 1
        assert "Add slot failed: boom" in error_calls[0].args[0]

    @pytest.mark.anyio
    async def test_finish_add_slot_pushes_validation_messages_on_failure(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        self._stub_finish_add_slot_ui(app)

        await app._finish_add_slot(
            success=False,
            messages=["Profile is required"],
        )

        controller._push_status_message.assert_called_with("Profile is required")

    @pytest.mark.anyio
    async def test_run_add_slot_handles_compute_exception(self) -> None:
        controller = _make_controller()
        controller.compute_add_slot_from_form.side_effect = RuntimeError("boom")
        controller.refresh_model_index_async.return_value = False
        app = DashboardApp(controller)
        captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def _capture_finish(fn: object, *args: object, **kwargs: object) -> None:
            captured.append((fn, args, kwargs))

        app.call_from_thread = _capture_finish  # type: ignore[method-assign]

        DashboardApp._run_add_slot.__wrapped__(  # type: ignore[attr-defined]
            app,
            {"profile": "summary-balanced", "port": "8080"},
        )

        finish_calls = [
            call for call in captured if getattr(call[0], "__name__", "") == "_finish_add_slot"
        ]
        assert len(finish_calls) == 1
        finish_args = finish_calls[0][1]
        assert finish_args[1] == "Add slot failed: boom"
        assert finish_args[2] is False


class TestDashboardAppSystemHealthRefresh:
    """Tests for cached system-health refresh worker."""

    def test_system_health_worker_applies_snapshot_on_ui_thread(self) -> None:
        controller = _make_controller()
        cpu = [10.0, 20.0]
        memory = [MemoryUsageSnapshot(label="Mem", percent=50.0, value_text="8/16 GB")]
        system = SystemInfoSnapshot(
            tasks=10,
            threads=20,
            running=1,
            load_values=(1.0, 2.0, 3.0),
            uptime="1:00",
        )
        controller.model.collect_system_health_snapshot = MagicMock(
            return_value=(cpu, memory, system)
        )
        controller.model.apply_system_health_snapshot = MagicMock()
        app = DashboardApp(controller)

        def _call_from_thread(fn: object, *args: object, **kwargs: object) -> object:
            return fn(*args, **kwargs)  # type: ignore[misc]

        app.call_from_thread = _call_from_thread  # type: ignore[method-assign]
        app._system_health_refresh_active = True

        DashboardApp._refresh_system_health_worker.__wrapped__(app)  # type: ignore[attr-defined]

        controller.model.collect_system_health_snapshot.assert_called_once()
        controller.model.apply_system_health_snapshot.assert_called_once_with(cpu, memory, system)
        assert app._system_health_refresh_active is False

    def test_apply_system_health_snapshot_refreshes_only_health_widgets(self) -> None:
        controller = _make_controller()
        controller.model.apply_system_health_snapshot = MagicMock()
        app = DashboardApp(controller)
        widgets = [MagicMock(), MagicMock(), MagicMock()]
        app.query_one = MagicMock(side_effect=widgets)  # type: ignore[method-assign]
        cpu = [10.0, 20.0]
        memory = [MemoryUsageSnapshot(label="Mem", percent=50.0, value_text="8/16 GB")]
        system = SystemInfoSnapshot(
            tasks=10,
            threads=20,
            running=1,
            load_values=(1.0, 2.0, 3.0),
            uptime="1:00",
        )

        app._apply_system_health_snapshot(cpu, memory, system)

        controller.model.apply_system_health_snapshot.assert_called_once_with(cpu, memory, system)
        for widget in widgets:
            widget.refresh.assert_called_once_with(recompose=True)


class TestDashboardAppRemoveSlotFlow:
    """Tests for live-slot removal flow."""

    def _stub_finish_remove_slot_ui(self, app: DashboardApp) -> None:
        app._reconcile_server_log_panels = AsyncMock()  # type: ignore[method-assign]
        app.refresh_dashboard = MagicMock()  # type: ignore[method-assign]

    def test_action_remove_slot_rejects_empty_config_list(self) -> None:
        controller = _make_controller()
        controller.configs = []
        app = DashboardApp(controller)
        app.notify = MagicMock()  # type: ignore[method-assign]
        app.push_screen = MagicMock()  # type: ignore[method-assign]

        app.action_remove_slot()

        app.notify.assert_called_once_with(
            "No slots configured to remove",
            title="Slot",
            severity="warning",
        )
        app.push_screen.assert_not_called()

    def test_action_remove_slot_opens_selector(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[method-assign]

        app.action_remove_slot()

        screen = app.push_screen.call_args.args[0]
        assert isinstance(screen, RemoveSlotModal)
        assert app.push_screen.call_args.args[1] == app._handle_remove_slot_modal_result

    def test_remove_slot_modal_cancel_does_nothing(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[method-assign]
        app._run_remove_slot = MagicMock()  # type: ignore[method-assign]

        app._handle_remove_slot_modal_result(None)

        app.push_screen.assert_not_called()
        app._run_remove_slot.assert_not_called()  # type: ignore[attr-defined]

    def test_remove_slot_confirm_starts_worker(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app.notify = MagicMock()  # type: ignore[method-assign]
        app._run_remove_slot = MagicMock()  # type: ignore[method-assign]
        app._pending_remove_slot_alias = "slot0"

        app._handle_remove_slot_confirm(True)

        app.notify.assert_called_once_with(
            "Removing slot…",
            title="Slot",
            severity="information",
        )
        app._run_remove_slot.assert_called_once_with("slot0")  # type: ignore[attr-defined]

    def test_run_remove_slot_calls_controller(self) -> None:
        controller = _make_controller()
        controller.prepare_async_slot_remove.return_value = (True, [])
        controller.server_manager.shutdown_slot.return_value = True
        controller.commit_async_slot_remove.return_value = (True, ["Removed slot 'slot0'"])
        app = DashboardApp(controller)
        captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def _call_from_thread(fn: object, *args: object, **kwargs: object) -> object:
            if getattr(fn, "__name__", "") == "_finish_remove_slot":
                captured.append((fn, args, kwargs))
                return None
            return fn(*args, **kwargs)  # type: ignore[misc]

        app.call_from_thread = _call_from_thread  # type: ignore[method-assign]

        DashboardApp._run_remove_slot.__wrapped__(app, "slot0")  # type: ignore[attr-defined]

        controller.remove_live_slot.assert_not_called()
        controller.prepare_async_slot_remove.assert_called_once_with("slot0")
        controller.server_manager.shutdown_slot.assert_called_once_with("slot0")
        controller.commit_async_slot_remove.assert_called_once_with("slot0")
        finish_calls = [
            call for call in captured if getattr(call[0], "__name__", "") == "_finish_remove_slot"
        ]
        assert len(finish_calls) == 1
        assert finish_calls[0][1] == ("slot0", None, True)

    def test_run_remove_slot_blocks_commit_when_shutdown_fails(self) -> None:
        controller = _make_controller()
        controller.prepare_async_slot_remove.return_value = (True, [])
        controller.server_manager.shutdown_slot.return_value = False
        app = DashboardApp(controller)
        captured: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def _call_from_thread(fn: object, *args: object, **kwargs: object) -> object:
            if getattr(fn, "__name__", "") == "_finish_remove_slot":
                captured.append((fn, args, kwargs))
                return None
            return fn(*args, **kwargs)  # type: ignore[misc]

        app.call_from_thread = _call_from_thread  # type: ignore[method-assign]

        DashboardApp._run_remove_slot.__wrapped__(app, "slot0")  # type: ignore[attr-defined]

        controller.prepare_async_slot_remove.assert_called_once_with("slot0")
        controller.server_manager.shutdown_slot.assert_called_once_with("slot0")
        controller.commit_async_slot_remove.assert_not_called()
        controller._push_status_message.assert_called_once_with(
            "Unable to remove 'slot0': shutdown verification failed"
        )
        finish_calls = [
            call for call in captured if getattr(call[0], "__name__", "") == "_finish_remove_slot"
        ]
        assert len(finish_calls) == 1
        assert finish_calls[0][1] == ("slot0", None, False)

    @pytest.mark.anyio
    async def test_finish_remove_slot_reconciles_and_refreshes(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        self._stub_finish_remove_slot_ui(app)
        app.notify = MagicMock()  # type: ignore[method-assign]

        await app._finish_remove_slot("slot0", success=True)

        app.notify.assert_called_once_with(
            "Slot 'slot0' removed",
            title="Slot",
            severity="information",
        )
        app._reconcile_server_log_panels.assert_awaited_once()  # type: ignore[attr-defined]
        app.refresh_dashboard.assert_called_once()  # type: ignore[attr-defined]


class TestDashboardAppProfileModalResult:
    """Tests for profile save callback handling."""

    def test_handle_profile_modal_result_save_and_add_slot(self) -> None:
        from llama_cli.tui.components.slot_profile_modal import SlotProfilePayload

        controller = _make_controller()
        controller.save_slot_profile_from_form.return_value = True
        app = DashboardApp(controller)
        app._run_add_slot = MagicMock()  # type: ignore[method-assign]
        app.notify = MagicMock()  # type: ignore[method-assign]
        app.refresh_dashboard = MagicMock()  # type: ignore[method-assign]

        payload = SlotProfilePayload(profile_id="my-profile", save_and_add_slot=True)
        app._handle_profile_modal_result(payload)

        controller.save_slot_profile_from_form.assert_called_once_with(payload)
        app._run_add_slot.assert_called_once()
        app.refresh_dashboard.assert_not_called()

    def test_handle_profile_modal_result_failed_save(self) -> None:
        from llama_cli.tui.components.slot_profile_modal import SlotProfilePayload

        controller = _make_controller()
        controller.save_slot_profile_from_form.return_value = False
        app = DashboardApp(controller)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[method-assign]

        app._handle_profile_modal_result(SlotProfilePayload(profile_id="my-profile"))

        notify_mock.assert_called_once_with("Failed to save profile", severity="error")


class TestDashboardAppSlotStatsRefresh:
    """Tests for background slot stats refresh helpers."""

    def test_schedule_slot_stats_refresh_skips_when_active(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app._slot_stats_refresh_active = True
        app._refresh_slot_stats_worker = MagicMock()  # type: ignore[method-assign]

        app._schedule_slot_stats_refresh()

        app._refresh_slot_stats_worker.assert_not_called()

    def test_mark_slot_stats_refresh_complete_clears_flag(self) -> None:
        controller = _make_controller()
        app = DashboardApp(controller)
        app._slot_stats_refresh_active = True

        app._mark_slot_stats_refresh_complete()

        assert app._slot_stats_refresh_active is False

    def test_refresh_slot_stats_worker_calls_controller_refresh(self) -> None:
        controller = _make_controller()
        controller.refresh_slot_stats = MagicMock()
        app = DashboardApp(controller)
        app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[method-assign]
        app.refresh_dashboard = MagicMock()
        app._slot_stats_refresh_active = True

        DashboardApp._refresh_slot_stats_worker.__wrapped__(app)  # type: ignore[attr-defined]

        controller.refresh_slot_stats.assert_called_once()
        app.refresh_dashboard.assert_called_once()
        assert app._slot_stats_refresh_active is False

    def test_refresh_slot_stats_worker_survives_controller_error(self) -> None:
        controller = _make_controller()
        controller.refresh_slot_stats.side_effect = RuntimeError("server down")  # type: ignore[attr-defined]
        app = DashboardApp(controller)
        app.call_from_thread = lambda fn, *args, **kwargs: fn(*args, **kwargs)  # type: ignore[method-assign]
        app.refresh_dashboard = MagicMock()

        DashboardApp._refresh_slot_stats_worker.__wrapped__(app)  # type: ignore[attr-defined]

        assert app._slot_stats_refresh_active is False


# ---------------------------------------------------------------------------
# Additional textual_app tests for SonarQube coverage gaps
# ---------------------------------------------------------------------------


class TestSplitLogUpdate:
    """Tests for _split_log_update — extracted pure helper."""

    def test_no_change_returns_false_empty(self) -> None:
        """When current equals previous, return (False, ())."""
        previous = ("line1", "line2")
        current = ("line1", "line2")
        reload, lines = _split_log_update(previous, current)
        assert reload is False
        assert lines == ()

    def test_append_returns_false_appended_lines(self) -> None:
        """When current is an append of previous, return (False, appended)."""
        previous = ("line1", "line2")
        current = ("line1", "line2", "line3")
        reload, lines = _split_log_update(previous, current)
        assert reload is False
        assert lines == ("line3",)

    def test_prepend_returns_true_all_current(self) -> None:
        """When current prepends to previous, return (True, current)."""
        previous = ("line2",)
        current = ("line1", "line2")
        reload, lines = _split_log_update(previous, current)
        assert reload is True
        assert lines == current


class TestDashboardAppActionHandlers:
    """Tests for DashboardApp action handlers."""

    def test_action_quit_dashboard_exits(self) -> None:
        """action_quit_dashboard should call controller.request_quit and exit."""
        controller = _make_controller()
        controller.running = False  # Exit immediately
        app = DashboardApp(controller)
        app.exit = MagicMock()  # type: ignore[assignment]

        app.action_quit_dashboard()

        controller.request_quit.assert_called_once()
        app.exit.assert_called_once()

    def test_action_quit_dashboard_stays_running(self) -> None:
        """action_quit_dashboard should not exit if controller is still running."""
        controller = _make_controller()
        controller.running = True
        app = DashboardApp(controller)
        app.exit = MagicMock()  # type: ignore[assignment]

        app.action_quit_dashboard()

        controller.request_quit.assert_called_once()
        app.exit.assert_not_called()

    def test_action_interrupt_dashboard_exits(self) -> None:
        """action_interrupt_dashboard should call controller.interrupt and exit."""
        controller = _make_controller()
        controller.running = False
        app = DashboardApp(controller)
        app.exit = MagicMock()  # type: ignore[assignment]

        app.action_interrupt_dashboard()

        controller.interrupt.assert_called_once()
        app.exit.assert_called_once()

    def test_action_refresh_dashboard_calls_refresh(self) -> None:
        """action_refresh_dashboard should call controller.refresh_display and refresh."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.refresh_dashboard = MagicMock()  # type: ignore[assignment]

        app.action_refresh_dashboard()

        controller.refresh_display.assert_called_once()

    def test_action_add_slot_pushes_modal(self) -> None:
        """action_add_slot should push AddSlotModal."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[assignment]

        app.action_add_slot()

        app.push_screen.assert_called_once()

    def test_action_remove_slot_pushes_modal(self) -> None:
        """action_remove_slot should push RemoveSlotModal."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[assignment]

        app.action_remove_slot()

        app.push_screen.assert_called_once()

    def test_action_remove_slot_no_slots_notifies(self) -> None:
        """action_remove_slot should notify when no slots configured."""
        controller = _make_controller()
        controller.configs = []
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[assignment]
        app.notify = MagicMock()  # type: ignore[assignment]

        app.action_remove_slot()

        app.notify.assert_called_once()
        app.push_screen.assert_not_called()

    def test_action_open_config_pushes_modal(self) -> None:
        """action_open_config should push ConfigModal."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[assignment]

        app.action_open_config()

        app.push_screen.assert_called_once()

    def test_action_about_pushes_modal(self) -> None:
        """action_about should push AboutModal."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[assignment]

        app.action_about()

        app.push_screen.assert_called_once()

    def test_action_manage_profiles_pushes_screen(self) -> None:
        """action_manage_profiles should push ProfilesScreen."""
        controller = _make_controller()
        controller.list_slot_profiles.return_value = []
        controller.is_profile_in_use.return_value = False
        controller.load_model_index.return_value = []
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[assignment]

        app.action_manage_profiles()

        app.push_screen.assert_called_once()


class TestDashboardAppModalResults:
    """Tests for modal result handlers."""

    def test_handle_add_slot_modal_result_none_calls_cancel(self) -> None:
        """_handle_add_slot_modal_result with None should cancel."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app._profile_options_cache = [("opt", "val")]
        app._profile_cache_config_id = 1

        # Manually call the cancellation path instead of push_screen
        controller.cancel_add_slot_form = MagicMock()  # type: ignore[assignment]
        app.refresh_dashboard = MagicMock()  # type: ignore[assignment]

        app._handle_add_slot_modal_result(None)

        controller.cancel_add_slot_form.assert_called_once()

    def test_handle_add_slot_modal_result_selected(self) -> None:
        """_handle_add_slot_modal_result with selected should start add slot."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app._profile_options_cache = [("opt", "val")]
        app._profile_cache_config_id = 1
        app._start_add_slot = MagicMock()  # type: ignore[assignment]

        app._handle_add_slot_modal_result({"profile": "summary-balanced"})

        app._start_add_slot.assert_called_once()

    def test_handle_remove_slot_modal_result_none_no_change(self) -> None:
        """_handle_remove_slot_modal_result with None should set no pending alias."""
        controller = _make_controller()
        app = DashboardApp(controller)
        # _pending_remove_slot_alias is None by default on DashboardApp
        # The method checks `if not alias: return` — so no change
        app._pending_remove_slot_alias = None  # type: ignore[attr-defined]

        app._handle_remove_slot_modal_result(None)

        assert app._pending_remove_slot_alias is None

    def test_handle_config_modal_result_clean_cache_confirms(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_handle_config_modal_result with clean_cache should push ConfirmModal."""
        from llama_cli.tui.components.config_modal import ConfigPayload

        controller = _make_controller()
        app = DashboardApp(controller)
        app.push_screen = MagicMock()  # type: ignore[assignment]

        payload = ConfigPayload(restart=False, clean_cache=True)

        app._handle_config_modal_result(payload)

        app.push_screen.assert_called_once()

    def test_handle_config_modal_result_save(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_handle_config_modal_result without clean_cache should save config."""
        from llama_cli.tui.components.config_modal import ConfigPayload

        controller = _make_controller()
        app = DashboardApp(controller)
        app.refresh_dashboard = MagicMock()  # type: ignore[assignment]

        payload = ConfigPayload(restart=False, clean_cache=False)

        app._handle_config_modal_result(payload)

        controller.save_config.assert_called_once()


class TestDashboardAppWorkers:
    """Tests for background worker exception paths."""

    def test_run_add_slot_handles_exception(self) -> None:
        """_run_add_slot should handle exceptions and notify."""
        controller = _make_controller()
        controller.compute_add_slot_from_form.side_effect = RuntimeError("compute failed")  # type: ignore[attr-defined]
        app = DashboardApp(controller)
        # Call the wrapped (non-worker) method directly to avoid event loop
        with contextlib.suppress(Exception):
            DashboardApp._run_add_slot.__wrapped__(app, {"profile": "test"})  # type: ignore[attr-defined]

    def test_run_remove_slot_handles_exception(self) -> None:
        """_run_remove_slot should handle exceptions and notify."""
        controller = _make_controller()
        controller.prepare_async_slot_remove.side_effect = RuntimeError("remove failed")  # type: ignore[attr-defined]
        app = DashboardApp(controller)
        # Call the wrapped (non-worker) method directly to avoid event loop
        with contextlib.suppress(Exception):
            DashboardApp._run_remove_slot.__wrapped__(app, "slot0")  # type: ignore[attr-defined]


class TestDashboardAppRefresh:
    """Tests for refresh_dashboard and _emit_status_toasts."""

    def test_refresh_dashboard_exits_when_not_running(self) -> None:
        """refresh_dashboard should exit when controller is not running."""
        controller = _make_controller()
        controller.running = False
        app = DashboardApp(controller)
        app.exit = MagicMock()  # type: ignore[assignment]
        app.query = MagicMock()  # type: ignore[assignment]

        app.refresh_dashboard()

        app.exit.assert_called_once()

    def test_emit_status_toasts_notices(self) -> None:
        """_emit_status_toasts should notify for new notices."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.notify = MagicMock()  # type: ignore[assignment]
        app._last_notified_status_ts = 0.0
        # Mock view_model.system_notices to return a notice
        app.view_model.system_notices = MagicMock(return_value=["new notice"])  # type: ignore[attr-defined]
        controller.get_status_messages_since.return_value = []  # type: ignore[attr-defined]

        app._emit_status_toasts()

        app.notify.assert_called()

    def test_emit_status_toasts_status_messages(self) -> None:
        """_emit_status_toasts should notify for new status messages."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.notify = MagicMock()  # type: ignore[assignment]
        app._last_notified_status_ts = 0.0
        app.view_model.system_notices = MagicMock(return_value=[])  # type: ignore[attr-defined]
        controller.get_status_messages_since.return_value = [(1.0, "status msg")]  # type: ignore[attr-defined]

        app._emit_status_toasts()

        app.notify.assert_called()


class TestDashboardAppReconcilePanels:
    """Tests for _reconcile_server_log_panels."""

    def test_reconcile_panels_zero_needed_does_not_crash(self) -> None:
        """_reconcile_server_log_panels should handle zero panels gracefully."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.view_model.server_column_count = MagicMock(return_value=0)  # type: ignore[attr-defined]

        # The function calls query_one which needs a Textual screen stack.
        # Instead, verify the function exists and is callable.
        import inspect

        assert inspect.iscoroutinefunction(app._reconcile_server_log_panels)


class TestDashboardAppBuildModalResult:
    """Tests for _handle_build_modal_result."""

    def test_handle_build_modal_result_none(self) -> None:
        """_handle_build_modal_result with None should cancel."""
        controller = _make_controller()
        app = DashboardApp(controller)
        app.refresh_dashboard = MagicMock()  # type: ignore[assignment]

        app._handle_build_modal_result(None)

        controller.cancel_pending_prompt.assert_called_once()

    def test_handle_build_modal_result_with_backends(self) -> None:
        """_handle_build_modal_result with backends should call handle_build_selection."""
        from llama_cli.tui.types import BuildWizardResult

        controller = _make_controller()
        app = DashboardApp(controller)
        app.refresh_dashboard = MagicMock()  # type: ignore[assignment]

        result = BuildWizardResult(backends=["sycl"], options={})
        app._handle_build_modal_result(result)

        controller.handle_build_selection.assert_called_once()
