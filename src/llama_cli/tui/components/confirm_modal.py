"""ConfirmModal — a simple yes/no confirmation dialog."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from .form_widgets import MODAL_CANCEL_BINDINGS


class ConfirmModal(ModalScreen[bool]):
    """Generic yes/no confirmation dialog.

    Args:
        title: Dialog title shown as the heading.
        message: The question/prompt for the user.
    """

    BINDINGS = MODAL_CANCEL_BINDINGS

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self._title, id="confirm-title", classes="modal-title confirm-title"),
            Label(self._message, id="confirm-message", classes="confirm-message"),
            Horizontal(
                Button("Cancel", id="cancel-confirm", classes="modal-button-cancel"),
                Button("Confirm", id="submit-confirm", classes="modal-button-success"),
                id="confirm-actions",
                classes="modal-actions confirm-actions",
            ),
            id="confirm-dialog",
            classes="modal-dialog confirm-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#submit-confirm", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-confirm":
            self.dismiss(False)
        elif event.button.id == "submit-confirm":
            self.dismiss(True)
