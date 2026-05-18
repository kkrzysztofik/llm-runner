"""AddSlotModal — modal form for adding a new server slot."""

from textual.app import ComposeResult
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

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(self, profile_options: list[tuple[str, str]]) -> None:
        super().__init__()
        if not profile_options:
            raise ValueError("profile_options must not be empty")
        self._profile_options = profile_options

    def compose(self) -> ComposeResult:
        yield Container(
            Label("Add Slot", id="add-slot-title", classes="modal-title"),
            Horizontal(
                Label("Profile", classes="form-label add-slot-label"),
                Select(
                    options=self._profile_options,
                    allow_blank=False,
                    value=self._profile_options[0][1],
                    prompt="Choose a profile",
                    id="slot-profile",
                    classes="form-input add-slot-input",
                ),
                classes="form-row add-slot-row",
            ),
            Horizontal(
                Label("Port override", classes="form-label add-slot-label"),
                Input(
                    value="",
                    placeholder="optional; leave blank for profile default",
                    id="slot-port",
                    classes="form-input add-slot-input",
                ),
                classes="form-row add-slot-row",
            ),
            Horizontal(
                Button("Cancel", id="cancel-slot", classes="modal-button-cancel"),
                Button("Add Slot", id="submit-slot", classes="modal-button-success"),
                id="add-slot-actions",
                classes="modal-actions",
            ),
            id="add-slot-dialog",
            classes="modal-dialog add-slot-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#slot-profile", Select).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-slot":
            self.dismiss(None)
            return
        if event.button.id == "submit-slot":
            values = self._collect_values()
            if values is not None:
                self.dismiss(values)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "slot-port":
            values = self._collect_values()
            if values is not None:
                self.dismiss(values)

    def _collect_values(self) -> dict[str, str] | None:
        selected_profile = self.query_one("#slot-profile", Select).value
        port_raw = self.query_one("#slot-port", Input).value.strip()
        # Validate numeric port input
        if port_raw:
            try:
                port_val = int(port_raw)
                if not (1 <= port_val <= 65535):
                    self.notify("Port must be 1-65535", severity="error")
                    return None
            except ValueError:
                self.notify("Port must be a number", severity="error")
                return None
        return {
            "profile": "" if selected_profile == Select.BLANK else str(selected_profile),
            "port": port_raw,
        }
