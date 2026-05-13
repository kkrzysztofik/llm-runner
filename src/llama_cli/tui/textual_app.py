"""Textual App shell for the llm-runner TUI dashboard."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.css.query import NoMatches

from llama_manager.config import create_default_profile_registry

from .components.config_modal import ConfigModal, ConfigPayload
from .components.menu import CommandMenu
from .components.modal import AddSlotModal
from .components.server_log import ServerLogPanel
from .components.system_health import (
    CPUUsageWidget,
    DateTimeWidget,
    MemorySwapWidget,
    SystemInfoWidget,
)
from .components.system_status import SystemStatusWidget

if TYPE_CHECKING:
    from .controller import DashboardController


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
        Binding("ctrl+c", "cancel_pending_prompt", "Cancel"),
        Binding("escape", "cancel_pending_prompt", "Cancel"),
        Binding("r", "refresh_dashboard", "Refresh"),
        Binding("p", "profile", "Profile"),
        Binding("P", "profile", "Profile"),
        Binding("b", "build", "Build"),
        Binding("s", "smoke", "Smoke"),
        Binding("a", "add_slot", "Add Slot"),
        Binding("c", "open_config", "Config"),
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

    def compose(self) -> ComposeResult:
        with Container(id="dashboard"):
            yield SystemStatusWidget(self.view_model)
            with Container(id="content"):
                for i in range(self.view_model.server_column_count()):
                    yield ServerLogPanel(i, self.view_model)
            yield CommandMenu(self.view_model)

    def on_mount(self) -> None:
        self.refresh_dashboard()
        self.set_interval(0.25, self.refresh_dashboard)

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

    def _handle_config_modal_result(self, result: ConfigPayload | None) -> None:
        if result is not None:
            self.controller.save_config(result)
        self.refresh_dashboard()

    def _handle_add_slot_modal_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self.controller.cancel_add_slot_form()
        else:
            self.controller.add_slot_from_form(result)
            self._reconcile_server_log_panels()
        self.refresh_dashboard()

    def _reconcile_server_log_panels(self) -> None:
        """Ensure ServerLogPanel widgets match the current slot count."""
        container = self.query_one("#content", Container)
        current_panels = list(container.query(ServerLogPanel))
        needed = self.view_model.server_column_count()
        for i in range(len(current_panels), needed):
            container.mount(ServerLogPanel(i, self.view_model))

    def _build_profile_options(self) -> list[tuple[str, str]]:
        config_id = id(self.controller.config)
        if self._profile_options_cache is not None and self._profile_cache_config_id == config_id:
            return self._profile_options_cache

        registry = create_default_profile_registry(self.controller.config)
        self._profile_options_cache = [
            (
                f"{profile.profile_id} - {profile.description}",
                profile.profile_id,
            )
            for profile in registry.profiles
        ]
        self._profile_cache_config_id = config_id
        return self._profile_options_cache

    def action_build(self) -> None:
        self.controller.request_build()
        self.refresh_dashboard()

    def action_smoke(self) -> None:
        self.controller.request_smoke()
        self.refresh_dashboard()

    def action_profile(self) -> None:
        self.controller.request_profile()
        self.refresh_dashboard()

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
        if not self.controller.running:
            self.exit()
            return

        self._emit_status_toasts()

        # Refresh each leaf widget.  SystemStatusWidget and SystemHealthWidget
        # use compose(), so their children own their own repaints.
        for widget_type in (DateTimeWidget, CPUUsageWidget, MemorySwapWidget, SystemInfoWidget):
            with contextlib.suppress(NoMatches):
                self.query_one(widget_type).refresh(recompose=True)
        for panel in self.query(ServerLogPanel):
            panel.refresh(recompose=True)
        with contextlib.suppress(NoMatches):
            self.query_one(CommandMenu).refresh(recompose=True)

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
