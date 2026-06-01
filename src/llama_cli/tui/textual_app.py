"""Textual App shell for the llm-runner TUI dashboard."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.css.query import NoMatches
from textual.widgets import Footer

if TYPE_CHECKING:
    from .controller import DashboardController

from .components.about_modal import AboutModal
from .components.build import BuildModalScreen
from .components.config_modal import ConfigModal, ConfigPayload
from .components.modal import AddSlotModal
from .components.run_profile_modal import RunProfileModal, RunProfilePayload
from .components.server_column import ServerColumnPanel
from .components.server_log import ServerLogPanel
from .components.system_health import (
    CPUUsageWidget,
    MemorySwapWidget,
    SystemInfoWidget,
)
from .components.system_status import SystemStatusWidget
from .types import BuildWizardResult

# ---------------------------------------------------------------------------
# Extracted pure helper: profile options caching
# ---------------------------------------------------------------------------


def _profile_options_cached(
    view_model: object,
    config: object,
    cache: list[tuple[str, str]] | None,
    cache_config_id: int | None,
) -> tuple[list[tuple[str, str]], int | None]:
    """Return (options, config_id) with caching logic.

    Extracted from ``DashboardApp._build_profile_options`` for testability.
    """
    config_id = id(config)
    if cache is not None and cache_config_id == config_id:
        return cache, config_id
    options = view_model.profile_options(config)  # type: ignore[union-attr]
    return options, config_id


_RISK_HIDDEN_ACTIONS = frozenset(
    {"refresh_dashboard", "add_slot", "build", "open_config", "about"},
)
_NORMAL_HIDDEN_ACTIONS = frozenset({"confirm", "reject"})


class DashboardApp(App[None]):
    """Textual shell for the llm-runner dashboard."""

    TITLE = "llm-runner"
    CSS_PATH = [
        "textual_app.tcss",
        "system_status.tcss",
        "dashboard_panels.tcss",
        "modals.tcss",
    ]
    BINDINGS = [
        Binding("q", "quit_dashboard", "Quit", priority=True),
        Binding(
            "ctrl+c",
            "cancel_pending_prompt",
            "Cancel",
            key_display="^C",
        ),
        Binding("escape", "cancel_pending_prompt", "Cancel", show=False),
        Binding("r", "refresh_dashboard", "Refresh"),
        Binding("b", "build", "Build"),
        Binding("a", "add_slot", "Add Slot"),
        Binding("c", "open_config", "Config"),
        Binding("p", "manage_profiles", "Profiles"),
        Binding("h", "about", "About"),
        Binding("y", "confirm", "Confirm"),
        Binding("n", "reject", "Abort"),
    ]

    def __init__(self, controller: DashboardController) -> None:
        super().__init__()
        self.controller = controller
        self.view_model = controller.view_model
        self._last_notified_status_ts: float = 0.0
        self._active_notice_toasts: set[str] = set()
        self._profile_options_cache: list[tuple[str, str]] | None = None
        self._profile_cache_config_id: int | None = None
        self._last_model_index_notice_scanned = 0
        self._gpu_stats_refresh_active = False
        self.last_build_backend: str = "sycl"

    def compose(self) -> ComposeResult:
        with Container(id="dashboard"):
            yield SystemStatusWidget(self.view_model)
            with Container(id="content"):
                for i in range(self.view_model.server_column_count()):
                    yield ServerLogPanel(i, self.view_model)
        yield Footer(show_command_palette=False)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Control which bindings appear in the footer for the current mode."""
        state = self.view_model.command_menu()

        if state.build_request:
            return action == "cancel_pending_prompt"

        if state.risk_prompt is not None:
            if action in _RISK_HIDDEN_ACTIONS:
                return False
            return not (action == "quit_dashboard" and state.risk_prompt.kind == "vram")

        return action not in _NORMAL_HIDDEN_ACTIONS

    def on_mount(self) -> None:
        interval_s = max(0.1, self.controller.config.tui_refresh_interval_ms / 1000)
        logger.info(
            "DashboardApp.on_mount: initial refresh start interval_s=%.3f configs=%r",
            interval_s,
            [c.alias for c in self.controller.configs],
        )
        self.refresh_dashboard()
        logger.info("DashboardApp.on_mount: initial refresh complete")
        self.set_interval(interval_s, self.refresh_dashboard)
        self.set_interval(max(1.0, interval_s), self._schedule_gpu_stats_refresh)
        self._schedule_gpu_stats_refresh()
        self._index_models()

    def _schedule_gpu_stats_refresh(self) -> None:
        """Start one background GPU stats refresh if no refresh is already running."""
        if self._gpu_stats_refresh_active:
            logger.debug("_schedule_gpu_stats_refresh: skipped, refresh already active")
            return
        self._gpu_stats_refresh_active = True
        self._refresh_gpu_stats_worker()

    @work(thread=True)
    def _refresh_gpu_stats_worker(self) -> None:
        """Refresh GPU stats off the render thread."""
        start = time.perf_counter()
        stats = list(self.controller.model.gpu_stats)
        logger.debug("_refresh_gpu_stats_worker: start count=%d", len(stats))
        try:
            for index, gpu in enumerate(stats):
                gpu_start = time.perf_counter()
                gpu.update()
                logger.debug(
                    "_refresh_gpu_stats_worker: updated index=%d device_index=%s duration_ms=%.1f",
                    index,
                    getattr(gpu, "device_index", "unknown"),
                    (time.perf_counter() - gpu_start) * 1000,
                )
        except BaseException:
            logger.exception("_refresh_gpu_stats_worker: unhandled exception")
        finally:
            logger.debug(
                "_refresh_gpu_stats_worker: complete count=%d duration_ms=%.1f",
                len(stats),
                (time.perf_counter() - start) * 1000,
            )
            self.call_from_thread(self._mark_gpu_stats_refresh_complete)

    def _mark_gpu_stats_refresh_complete(self) -> None:
        self._gpu_stats_refresh_active = False

    def _index_models(self) -> None:
        """Refresh model index in a controller-owned background thread."""
        started = self.controller.refresh_model_index_async(
            progress_callback=self._index_models_progress_from_thread,
            complete_callback=self._index_models_complete_from_thread,
        )
        logger.info("_index_models: background refresh started=%s", started)
        if started:
            self._last_model_index_notice_scanned = 0
            self.notify("Indexing models...", title="Models")

    def _index_models_progress_from_thread(
        self,
        _entries: object,
        scanned: int,
        total: int,
        errors: int,
    ) -> None:
        """Forward model index progress from the controller worker to the UI thread."""
        self.call_from_thread(self._handle_model_index_progress, scanned, total, errors)

    def _index_models_complete_from_thread(
        self,
        _entries: object,
        total: int,
        errors: int,
    ) -> None:
        """Forward model index completion from the controller worker to the UI thread."""
        self.call_from_thread(self._handle_model_index_complete, total, errors)

    def _handle_model_index_progress(self, scanned: int, total: int, errors: int) -> None:
        """Show sparse progress notifications while model indexing continues."""
        if total <= 0:
            return
        notify_every = max(1, total // 5)
        if scanned < total and scanned - self._last_model_index_notice_scanned < notify_every:
            return
        self._last_model_index_notice_scanned = scanned
        message = f"Indexed {scanned}/{total} models"
        if errors:
            message += f" ({errors} errors)"
        self.notify(message, title="Models", severity="warning" if errors else "information")

    def _handle_model_index_complete(self, total: int, errors: int) -> None:
        """Show final model index status."""
        message = f"Indexing complete: {total} models found"
        if errors:
            message += f" ({errors} errors)"
        self.notify(message, title="Models", severity="warning" if errors else "information")

    def action_quit_dashboard(self) -> None:
        self.controller.request_quit()
        if not self.controller.running:
            self.exit()

    def action_interrupt_dashboard(self) -> None:
        self.controller.interrupt()
        if not self.controller.running:
            self.exit()

    def action_refresh_dashboard(self) -> None:
        self.controller.refresh_display()
        self.refresh_dashboard()

    def action_add_slot(self) -> None:
        self.push_screen(
            AddSlotModal(profile_options=self._build_profile_options()),
            self._handle_add_slot_modal_result,
        )

    def action_open_config(self) -> None:
        self.push_screen(
            ConfigModal(config=self.controller.config),
            self._handle_config_modal_result,
        )

    def action_about(self) -> None:
        self.push_screen(AboutModal())

    def action_manage_profiles(self) -> None:
        """Open the profiles management screen."""
        from .components.profiles_screen import ProfilesScreen

        profiles = self.controller.list_run_profiles()
        in_use_ids = {
            spec.profile_id
            for spec, _ in profiles
            if self.controller.is_profile_in_use(spec.profile_id)
        }
        model_index = self.controller.load_model_index()

        self.push_screen(
            ProfilesScreen(profiles=profiles, in_use_ids=in_use_ids, model_index=model_index),
            self._handle_profiles_screen_result,
        )

    def _handle_profiles_screen_result(self, result: dict | None) -> None:
        """Handle result from the ProfilesScreen."""
        if result is None:
            return

        action = result.get("action")

        if action == "add":
            model_index = self.controller.load_model_index()
            self.push_screen(
                RunProfileModal(model_index=model_index, config=self.controller.config),
                self._handle_profile_modal_result,
            )

        elif action == "edit":
            profile_id = result.get("profile_id", "")
            if not profile_id:
                return

            from llama_manager.config.builder import create_tui_profile_registry
            from llama_manager.run_profile_store import custom_profile_exists

            registry = create_tui_profile_registry(self.controller.config)
            try:
                spec = registry.get_profile(profile_id)
            except KeyError, ValueError:
                self.notify(f"Profile '{profile_id}' not found", severity="error")
                return

            source = "custom" if custom_profile_exists(profile_id) else "builtin"

            model_index = self.controller.load_model_index()
            self.push_screen(
                RunProfileModal(
                    profile=spec,
                    edit_source=source,
                    model_index=model_index,
                    config=self.controller.config,
                ),
                self._handle_edit_modal_result,
            )

        elif action == "delete":
            profile_id = result.get("profile_id", "")
            if not profile_id:
                return

            success = self.controller.delete_run_profile(profile_id)
            if success:
                self.notify(f"Profile '{profile_id}' deleted", severity="information")
                self._profile_options_cache = None
                self._profile_cache_config_id = None
            self.refresh_dashboard()

    def _handle_edit_modal_result(self, result: RunProfilePayload | None) -> None:
        """Handle result from the edit profile modal."""
        if result is None:
            return

        original_id = result.original_profile_id or result.profile_id
        saved = self.controller.update_run_profile(original_id, result)
        if not saved:
            self.notify("Failed to update profile", severity="error")
            return

        self.notify(f"Profile '{result.profile_id}' updated", severity="information")

        # Invalidate profile options cache
        self._profile_options_cache = None
        self._profile_cache_config_id = None

        if result.save_and_add_slot:
            slot_form: dict[str, str] = {
                "profile": result.profile_id,
                "port": "",
            }
            self._run_add_slot(slot_form, "Slot added for profile")
            return

        self.refresh_dashboard()

    def action_create_profile(self) -> None:
        """Open the run profile creation modal (legacy alias)."""
        self.push_screen(
            RunProfileModal(config=self.controller.config),
            self._handle_profile_modal_result,
        )

    def _handle_config_modal_result(self, result: ConfigPayload | None) -> None:
        if result is not None:
            if result.clean_cache:
                from .components.confirm_modal import ConfirmModal

                self.push_screen(
                    ConfirmModal(
                        title="Clean Model Cache",
                        message="Delete the cached model index? Models will be re-scanned.",
                    ),
                    self._handle_cache_clean_confirm,
                )
                return

            self.controller.save_config(result)
        self.refresh_dashboard()

    def _handle_cache_clean_confirm(self, confirmed: bool | None) -> None:
        """Handle confirmation result from the cache clean dialog."""
        if confirmed:
            success, message = self.controller.clean_model_cache()
            if success:
                self.notify(message, title="Models", severity="information")
                self._index_models()
            else:
                self.notify(message, title="Models", severity="error")
        else:
            self.notify("Cancelled", title="Models", severity="information")

    def _handle_profile_modal_result(self, result: RunProfilePayload | None) -> None:
        if result is None:
            return

        saved = self.controller.save_run_profile_from_form(result)
        if not saved:
            self.notify("Failed to save profile", severity="error")
            return

        self.notify(f"Profile '{result.profile_id}' saved", severity="information")

        # Invalidate profile options cache
        self._profile_options_cache = None
        self._profile_cache_config_id = None

        if result.save_and_add_slot:
            slot_form: dict[str, str] = {
                "profile": result.profile_id,
                "port": "",
            }
            self._run_add_slot(slot_form, "Slot added for profile")
            return

        self.refresh_dashboard()

    @work(thread=True)
    def _run_add_slot(self, slot_form: dict[str, str], notify_message: str = "") -> None:
        """Run add_slot_from_form on a background thread to keep the UI responsive."""
        logger.debug("_run_add_slot: starting, slot_form=%r", slot_form)
        error: str | None = None
        try:
            success = self.controller.add_slot_from_form(slot_form)
            logger.debug(
                "_run_add_slot: add_slot_from_form returned success=%s, configs=%r",
                success,
                [c.alias for c in self.controller.configs],
            )
        except BaseException as exc:
            logger.exception("_run_add_slot: unhandled exception")
            error = f"Add slot failed: {exc}"
        self.call_from_thread(self._finish_add_slot, notify_message, error)

    async def _finish_add_slot(self, notify_message: str = "", error: str | None = None) -> None:
        """Called on the UI thread after the background add-slot work completes."""
        configs_aliases = [c.alias for c in self.controller.configs]
        try:
            container = self.query_one("#content")
            panel_count_before = len(list(container.query(ServerLogPanel)))
        except NoMatches:
            panel_count_before = -1
        logger.debug(
            "_finish_add_slot: configs=%r panels_before=%d column_count=%d",
            configs_aliases,
            panel_count_before,
            self.view_model.server_column_count(),
        )
        if error:
            self.notify(error, title="Add Slot", severity="error")
        elif notify_message:
            self.notify(notify_message, severity="information")
        reconcile_start = time.perf_counter()
        logger.debug("_finish_add_slot: reconcile panels start")
        await self._reconcile_server_log_panels()
        logger.debug(
            "_finish_add_slot: reconcile panels complete duration_ms=%.1f",
            (time.perf_counter() - reconcile_start) * 1000,
        )
        try:
            container = self.query_one("#content")
            panel_count_after = len(list(container.query(ServerLogPanel)))
        except NoMatches:
            panel_count_after = -1
        logger.debug(
            "_finish_add_slot: panels_after=%d — calling refresh_dashboard",
            panel_count_after,
        )
        refresh_start = time.perf_counter()
        self.refresh_dashboard()
        logger.debug(
            "_finish_add_slot: refresh_dashboard returned duration_ms=%.1f",
            (time.perf_counter() - refresh_start) * 1000,
        )

    def _handle_add_slot_modal_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self.controller.cancel_add_slot_form()
            self.refresh_dashboard()
        else:
            self.notify("Adding slot…", title="Slot", severity="information")
            self._run_add_slot(result)

    def _handle_build_modal_result(self, result: BuildWizardResult | None) -> None:
        if result is None:
            self.controller.cancel_pending_prompt()
        else:
            self.last_build_backend = result.backends[0] if result.backends else "sycl"
            self.controller.handle_build_selection(result.backends, result.options)
        self.refresh_dashboard()

    async def _reconcile_server_log_panels(self) -> None:
        """Ensure ServerLogPanel widgets match the current slot count."""
        start = time.perf_counter()
        container = self.query_one("#content", Container)
        current_panels = list(container.query(ServerLogPanel))
        needed = self.view_model.server_column_count()
        logger.debug(
            "_reconcile_server_log_panels: start current=%d needed=%d",
            len(current_panels),
            needed,
        )
        # Remove excess panels.
        if len(current_panels) > needed:
            for panel in current_panels[needed:]:
                logger.debug("_reconcile_server_log_panels: removing excess panel=%r", panel)
                await panel.remove()
        # Replace placeholder panels that now have real data (e.g. 0→1 slot).
        # A placeholder has no ServerColumnPanel child; replacing it with a
        # fresh ServerLogPanel ensures compose() runs with live view-model state.
        for i, panel in enumerate(current_panels[:needed]):
            if self.view_model.column(i) is not None and not panel.query(ServerColumnPanel):
                logger.debug("_reconcile_server_log_panels: replacing placeholder slot_index=%d", i)
                await panel.remove()
                await container.mount(ServerLogPanel(i, self.view_model))
                logger.debug(
                    "_reconcile_server_log_panels: replaced placeholder duration_ms=%.1f",
                    (time.perf_counter() - start) * 1000,
                )
                return  # one replacement per call
        # Mount any panels still missing.
        for i in range(len(current_panels), needed):
            logger.debug("_reconcile_server_log_panels: mounting missing slot_index=%d", i)
            await container.mount(ServerLogPanel(i, self.view_model))
        logger.debug(
            "_reconcile_server_log_panels: complete current=%d needed=%d duration_ms=%.1f",
            len(current_panels),
            needed,
            (time.perf_counter() - start) * 1000,
        )

    def _build_profile_options(self) -> list[tuple[str, str]]:
        options, config_id = _profile_options_cached(
            self.view_model,
            self.controller.config,
            self._profile_options_cache,
            self._profile_cache_config_id,
        )
        self._profile_options_cache = options
        self._profile_cache_config_id = config_id
        return options

    def action_build(self) -> None:
        screen = BuildModalScreen(last_backend=self.last_build_backend)
        self.push_screen(screen, self._handle_build_modal_result)

    def action_confirm(self) -> None:
        self.controller.acknowledge_risk()
        self.refresh_dashboard()

    def action_reject(self) -> None:
        self.controller.reject_risk()
        if not self.controller.running:
            self.exit()
        self.refresh_dashboard()

    def action_cancel_pending_prompt(self) -> None:
        cancelled = self.controller.cancel_pending_prompt()
        if not cancelled:
            self.controller.interrupt()
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        start = time.perf_counter()
        if not self.controller.running:
            logger.info("refresh_dashboard: controller stopped, exiting app")
            self.exit()
            return

        panel_count = len(list(self.query(ServerLogPanel)))
        logger.debug(
            "refresh_dashboard: start panels=%d configs=%d",
            panel_count,
            len(self.controller.configs),
        )
        self._emit_status_toasts()

        # Refresh each leaf widget.  SystemStatusWidget and SystemHealthWidget
        # use compose(), so their children own their own repaints.
        for widget_type in (CPUUsageWidget, MemorySwapWidget, SystemInfoWidget):
            with contextlib.suppress(NoMatches):
                widget_start = time.perf_counter()
                self.query_one(widget_type).refresh(recompose=True)
                logger.debug(
                    "refresh_dashboard: refreshed widget=%s duration_ms=%.1f",
                    widget_type.__name__,
                    (time.perf_counter() - widget_start) * 1000,
                )
        for panel in self.query(ServerLogPanel):
            panel_start = time.perf_counter()
            panel.refresh(recompose=True)
            logger.debug(
                "refresh_dashboard: refreshed ServerLogPanel slot_index=%s duration_ms=%.1f",
                getattr(panel, "_slot_index", "unknown"),
                (time.perf_counter() - panel_start) * 1000,
            )
        self.refresh_bindings()
        logger.debug(
            "refresh_dashboard: complete panels=%d duration_ms=%.1f",
            panel_count,
            (time.perf_counter() - start) * 1000,
        )

    def _emit_status_toasts(self) -> None:
        notices = self.view_model.system_notices()
        current_notices = set(notices)
        for notice in notices:
            if notice not in self._active_notice_toasts:
                self.notify(notice, title="Alert", severity="warning")
        self._active_notice_toasts = current_notices

        updates = self.controller.get_status_messages_since(self._last_notified_status_ts)
        if not updates:
            return
        for ts, message in updates:
            self.notify(message, title="Status", severity="information")
            self._last_notified_status_ts = max(self._last_notified_status_ts, ts)
