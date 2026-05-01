"""Textual App shell for the llm-runner TUI dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container

from llama_manager.config import create_default_profile_registry

from .components.alerts import (
    GPUTelemetryWidget,
    NoticesWidget,
    SystemHealthWidget,
    SystemStatusWidget,
)
from .components.menu import CommandMenu
from .components.modal import AddSlotModal
from .components.panels import ServerLogPanel

if TYPE_CHECKING:
    from .controller import TUIApp


class TextualDashboardApp(App[None]):
    """Textual shell for the llm-runner dashboard."""

    TITLE = "llm-runner"
    CSS_PATH = "textual_app.tcss"
    BINDINGS = [
        Binding("q", "quit_dashboard", "Quit", priority=True),
        Binding("ctrl+c", "interrupt_dashboard", "Stop", priority=True),
        Binding("escape", "cancel_pending_prompt", "Cancel"),
        Binding("r", "refresh_dashboard", "Refresh"),
        Binding("p", "profile", "Profile"),
        Binding("b", "build", "Build"),
        Binding("s", "smoke", "Smoke"),
        Binding("a", "add_slot", "Add Slot"),
        Binding("y", "confirm", "Confirm"),
        Binding("n", "reject", "Abort"),
        Binding("1", "select_flavor('1')", "Balanced"),
        Binding("2", "select_flavor('2')", "Fast"),
        Binding("3", "select_flavor('3')", "Quality"),
    ]

    def __init__(self, controller: TUIApp) -> None:
        super().__init__()
        self.controller = controller
        self._last_notified_status_ts: float = 0.0

    def compose(self) -> ComposeResult:
        with Container(id="dashboard"):
            yield SystemStatusWidget(self.controller)
            with Container(id="content"):
                yield ServerLogPanel(0, self.controller)
                yield ServerLogPanel(1, self.controller)
            yield CommandMenu(self.controller)

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

    def _handle_add_slot_modal_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self.controller.cancel_add_slot_form()
        else:
            self.controller.add_slot_from_form(result)
        self.refresh_dashboard()

    def _build_profile_options(self) -> list[tuple[str, str]]:
        registry = create_default_profile_registry(self.controller.config)
        return [
            (
                f"{profile.profile_id} - {profile.description}",
                profile.profile_id,
            )
            for profile in registry.profiles
        ]

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

    def action_select_flavor(self, key: str) -> None:
        self.controller.select_pending_option(key)
        self.refresh_dashboard()

    def action_cancel_pending_prompt(self) -> None:
        self.controller.cancel_pending_prompt()
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        if not self.controller.running:
            self.exit()
            return

        self._emit_status_toasts()

        # Refresh each leaf widget.  SystemStatusWidget uses compose() so it
        # is just a layout container — its children own their own repaints.
        self.query_one(SystemHealthWidget).refresh()
        self.query_one(GPUTelemetryWidget).refresh()
        self.query_one(NoticesWidget).refresh()
        for panel in self.query(ServerLogPanel):
            panel.refresh()
        self.query_one(CommandMenu).refresh()

    def _emit_status_toasts(self) -> None:
        updates = self.controller.get_status_messages_since(self._last_notified_status_ts)
        if not updates:
            return
        for ts, message in updates:
            self.notify(message, title="Status", severity="information")
            self._last_notified_status_ts = max(self._last_notified_status_ts, ts)
