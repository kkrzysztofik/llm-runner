"""AddSlotModal — modal form for adding a new server slot."""

from __future__ import annotations

from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select


class AddSlotModal(ModalScreen[dict[str, str] | None]):
    """Modal form for adding a new slot.

    Accepts a list of ``(label, value)`` tuples for the profile dropdown and
    returns a ``dict`` with ``"profile"`` and ``"port"`` keys on submit, or
    ``None`` on cancel.
    """

    DEFAULT_CSS = """
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

    #cancel-slot {
        border: tall $panel;
    }

    #submit-slot {
        border: tall $success;
        background: $success 20%;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(self, profile_options: list[tuple[str, str]]) -> None:
        super().__init__()
        self._profile_options = profile_options

    def compose(self):  # type: ignore[override]
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
        if event.input.id == "slot-port":
            self.dismiss(self._collect_values())

    def _collect_values(self) -> dict[str, str]:
        selected_profile = self.query_one("#slot-profile", Select).value
        return {
            "profile": "" if selected_profile == Select.BLANK else str(selected_profile),
            "port": self.query_one("#slot-port", Input).value,
        }
