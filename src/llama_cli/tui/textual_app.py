"""Textual App shell for the llm-runner TUI dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from llama_manager.config import create_default_profile_registry

if TYPE_CHECKING:
    from .controller import TUIApp


class TextualDashboardApp(App[None]):
    """Textual shell for the llm-runner dashboard."""

    TITLE = "llm-runner"
    CSS_PATH = "textual_app.tcss"
    _LEFT_PANEL_ID = "#left"
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
            yield Static(id="alerts")
            with Container(id="content"):
                yield Static(id="left", classes="column")
                yield Static(id="right", classes="column")
            yield Static(id="menu")

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

        snapshot = self.controller.render()
        self.query_one("#alerts", Static).update(snapshot.alerts)
        left = self.query_one(self._LEFT_PANEL_ID, Static)
        left.set_class(snapshot.left is None, "empty")
        if snapshot.left is not None:
            left.update(snapshot.left)
        self.query_one("#right", Static).update(snapshot.right)
        self.query_one("#menu", Static).update(snapshot.menu)

    def _emit_status_toasts(self) -> None:
        updates = self.controller.get_status_messages_since(self._last_notified_status_ts)
        if not updates:
            return

        for ts, message in updates:
            self.notify(message, title="Status", severity="information")
            self._last_notified_status_ts = max(self._last_notified_status_ts, ts)


class AddSlotModal(ModalScreen[dict[str, str] | None]):
    """Modal form for adding a new slot."""

    CSS_PATH = "textual_app.tcss"

    def __init__(self, profile_options: list[tuple[str, str]]) -> None:
        super().__init__()
        self._profile_options = profile_options

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    _FIELD_ORDER = (
        "slot-port",
    )

    def compose(self) -> ComposeResult:
        with Container(id="add-slot-dialog"):
            yield Label("Add Slot", id="add-slot-title")

            with Horizontal(classes="add-slot-row"):
                yield Label("Profile", classes="add-slot-label")
                yield Select(
                    options=self._profile_options,
                    allow_blank=False,
                    value=self._profile_options[0][1],
                    prompt="Choose a profile",
                    id="slot-profile",
                    classes="add-slot-input",
                )

            with Horizontal(classes="add-slot-row"):
                yield Label("Port override", classes="add-slot-label")
                yield Input(
                    value="",
                    placeholder="optional; leave blank for profile default",
                    id="slot-port",
                    classes="add-slot-input",
                )

            with Horizontal(id="add-slot-actions"):
                yield Button("Cancel", id="cancel-slot")
                yield Button("Add Slot", id="submit-slot")

    def on_mount(self) -> None:
        self.query_one("#slot-profile", Select).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-slot":
            self.dismiss(None)
            return
        if event.button.id == "submit-slot":
            self.dismiss(self._collect_values())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        input_id = event.input.id
        if input_id != "slot-port":
            return
        self.dismiss(self._collect_values())

    def _collect_values(self) -> dict[str, str]:
        selected_profile = self.query_one("#slot-profile", Select).value
        return {
            "profile": "" if selected_profile == Select.BLANK else str(selected_profile),
            "port": self.query_one("#slot-port", Input).value,
        }
