"""Textual App shell for the llm-runner TUI dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.events import Key, Resize
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

if TYPE_CHECKING:
    from .controller import TUIApp


class TextualDashboardApp(App[None]):
    """Textual shell for the llm-runner dashboard."""

    TITLE = "llm-runner"
    _LEFT_PANEL_ID = "#left"
    CSS = """
    Screen {
        layout: vertical;
    }

    #dashboard {
        height: 1fr;
        layout: vertical;
    }

    #alerts {
        height: auto;
        max-height: 35%;
    }

    #content {
        height: 1fr;
    }

    #content.horizontal {
        layout: horizontal;
    }

    #content.vertical {
        layout: vertical;
    }

    .column {
        width: 1fr;
        height: 1fr;
    }

    #menu {
        height: 1;
    }
    """
    BINDINGS = [
        Binding("q", "quit_dashboard", "Quit", priority=True),
        Binding("ctrl+c", "interrupt_dashboard", "Stop", priority=True),
        Binding("r", "refresh_dashboard", "Refresh"),
        Binding("p", "profile", "Profile"),
        Binding("b", "build", "Build"),
        Binding("s", "smoke", "Smoke"),
        Binding("y", "confirm", "Confirm"),
        Binding("n", "reject", "Abort"),
        Binding("1", "select_flavor('1')", "Balanced"),
        Binding("2", "select_flavor('2')", "Fast"),
        Binding("3", "select_flavor('3')", "Quality"),
    ]

    def __init__(self, controller: TUIApp) -> None:
        super().__init__()
        self.controller = controller

    def compose(self) -> ComposeResult:
        with Container(id="dashboard"):
            yield Static(id="alerts")
            with Horizontal(id="content", classes="horizontal"):
                yield Static(id="left", classes="column")
                yield Static(id="right", classes="column")
            yield Static(id="menu")

    def on_mount(self) -> None:
        self.refresh_dashboard()
        self.set_interval(0.25, self.refresh_dashboard)

    def on_resize(self, event: Resize) -> None:
        self.controller.width = event.size.width
        self.controller.height = event.size.height
        self.refresh_dashboard()

    def on_key(self, event: Key) -> None:
        if event.key.lower() == "a":
            self.action_add_slot()
            event.stop()
            return

        key = self._textual_key_to_controller_key(event)
        if key is None:
            return
        self.controller.handle_keypress(key)
        self.refresh_dashboard()
        event.stop()

    def action_quit_dashboard(self) -> None:
        self.controller.handle_keypress("q")
        if not self.controller.running:
            self.exit()

    def action_interrupt_dashboard(self) -> None:
        self.controller.handle_keypress("^C")
        if not self.controller.running:
            self.exit()

    def action_refresh_dashboard(self) -> None:
        self.controller.handle_keypress("r")
        self.refresh_dashboard()

    def action_add_slot(self) -> None:
        self.push_screen(AddSlotModal(), self._handle_add_slot_modal_result)

    def _handle_add_slot_modal_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self.controller.cancel_add_slot_form()
        else:
            self.controller.add_slot_from_form(result)
        self.refresh_dashboard()

    def action_build(self) -> None:
        self.controller.handle_keypress("b")
        self.refresh_dashboard()

    def action_smoke(self) -> None:
        self.controller.handle_keypress("s")
        self.refresh_dashboard()

    def action_profile(self) -> None:
        self.controller.handle_keypress("P")
        self.refresh_dashboard()

    def action_confirm(self) -> None:
        self.controller.handle_keypress("y")
        self.refresh_dashboard()

    def action_reject(self) -> None:
        self.controller.handle_keypress("n")
        if not self.controller.running:
            self.exit()
        self.refresh_dashboard()

    def action_select_flavor(self, key: str) -> None:
        self.controller.handle_keypress(key)
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        if not self.controller.running:
            self.exit()
            return

        snapshot = self.controller.render()
        content = self.query_one("#content", Horizontal)
        content_orientation = self.controller.build_layout().content_orientation
        content.set_classes(content_orientation)

        self.query_one("#alerts", Static).update(snapshot.alerts)
        if snapshot.left is not None:
            self.query_one(self._LEFT_PANEL_ID, Static).display = True
            self.query_one(self._LEFT_PANEL_ID, Static).update(snapshot.left)
        else:
            self.query_one(self._LEFT_PANEL_ID, Static).display = False
        self.query_one("#right", Static).update(snapshot.right)
        self.query_one("#menu", Static).update(snapshot.menu)

    def _textual_key_to_controller_key(self, event: Key) -> str | None:
        if event.key in {"enter", "return"}:
            return "\n"
        if event.key == "escape":
            return "\x1b"
        if event.key in {"backspace", "delete_left", "delete"}:
            return "\x7f"
        if event.key == "ctrl+c":
            return "^C"
        if event.character is not None and len(event.character) == 1:
            return event.character
        return None


class AddSlotModal(ModalScreen[dict[str, str] | None]):
    """Modal form for adding a new slot."""

    CSS = """
    AddSlotModal {
        align: center middle;
    }

    #add-slot-dialog {
        width: 80;
        max-width: 95%;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    #add-slot-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .add-slot-row {
        height: 3;
    }

    .add-slot-label {
        width: 18;
        content-align: left middle;
    }

    .add-slot-input {
        width: 1fr;
    }

    #add-slot-actions {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    _FIELD_ORDER = (
        "slot-profile",
        "slot-port",
    )

    def compose(self) -> ComposeResult:
        with Container(id="add-slot-dialog"):
            yield Label("Add Slot", id="add-slot-title")

            with Horizontal(classes="add-slot-row"):
                yield Label("Profile", classes="add-slot-label")
                yield Input(
                    value="summary-balanced",
                    placeholder="summary-balanced | summary-fast | qwen35",
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
                yield Button("Cancel", id="cancel-slot", variant="default")
                yield Button("Add Slot", id="submit-slot", variant="success")

    def on_mount(self) -> None:
        self.query_one("#slot-profile", Input).focus()

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
        if input_id is None:
            return
        try:
            idx = self._FIELD_ORDER.index(input_id)
        except ValueError:
            return

        next_idx = idx + 1
        if next_idx < len(self._FIELD_ORDER):
            self.query_one(f"#{self._FIELD_ORDER[next_idx]}", Input).focus()
            return
        self.dismiss(self._collect_values())

    def _collect_values(self) -> dict[str, str]:
        return {
            "profile": self.query_one("#slot-profile", Input).value,
            "port": self.query_one("#slot-port", Input).value,
        }
