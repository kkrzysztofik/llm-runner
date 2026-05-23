"""Profiles screen showing all configured run profiles with CRUD actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

if TYPE_CHECKING:
    from llama_manager.config.profiles import RunProfileSpec


class ProfilesScreen(ModalScreen[dict[str, Any] | None]):
    """Screen listing all run profiles with add/edit/delete actions.

    Returns a dict with action key (``add``, ``edit``, ``delete``) or ``None``
    to close.  ``edit`` and ``delete`` also include a ``profile_id`` key.
    """

    def __init__(
        self,
        profiles: list[tuple[RunProfileSpec, str]],
        in_use_ids: set[str] | None = None,
    ) -> None:
        """Initialize the profiles screen.

        Args:
            profiles: List of ``(RunProfileSpec, source)`` tuples where source is
                ``"builtin"`` or ``"custom"``.
            in_use_ids: Set of profile IDs currently in use by running slots.
                Deletion is blocked for these.
        """
        super().__init__()
        self._profiles = profiles
        self._in_use_ids = in_use_ids or set()

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("ctrl+c", "cancel", "Close"),
    ]

    CSS = """
    ProfilesScreen {
        align: center middle;
    }
    .profiles-dialog {
        width: 80%;
        max-width: 95%;
        max-height: 90%;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    .profiles-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    .profiles-scroll-body {
        width: 100%;
        height: 1fr;
        min-height: 10;
        overflow-y: auto;
        margin-bottom: 1;
    }
    .profile-card {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
        border: solid $panel;
    }
    .profile-card:hover {
        border: solid $accent;
    }
    .profile-card-info {
        width: 1fr;
        height: auto;
    }
    .profile-card-actions {
        width: auto;
        height: auto;
        align-horizontal: right;
    }
    .profile-card-actions Button {
        margin-left: 1;
        min-width: 10;
    }
    .profile-id-text {
        text-style: bold;
        color: $text;
    }
    .profile-model-text {
        color: $text-muted;
    }
    .profile-meta-text {
        color: $text-muted;
    }
    .profiles-actions {
        height: 3;
        align-horizontal: right;
    }
    .profiles-actions Button {
        margin-left: 1;
        min-width: 12;
    }
    .badge-builtin {
        color: $success;
    }
    .badge-custom {
        color: $warning;
    }
    .empty-profiles {
        color: $text-muted;
        text-style: italic;
        margin-top: 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Container(
            Label("Run Profiles", classes="profiles-title"),
            Container(
                *(self._profile_card(spec, source) for spec, source in self._profiles)
                if self._profiles
                else [
                    Label(
                        "No profiles configured. Add one below.",
                        classes="empty-profiles",
                    )
                ],
                classes="profiles-scroll-body",
                id="profiles-list",
            ),
            Horizontal(
                Button("+ Add Profile", id="add-profile", classes="modal-button-success"),
                Button("Close", id="close-profiles", classes="modal-button-cancel"),
                classes="profiles-actions",
            ),
            classes="profiles-dialog",
        )

    def _profile_card(self, spec: RunProfileSpec, source: str) -> Horizontal:
        """Build a profile card row with info and action buttons."""
        in_use = spec.profile_id in self._in_use_ids
        source_label = "built-in" if source == "builtin" else "custom"
        source_badge_class = "badge-builtin" if source == "builtin" else "badge-custom"

        return Horizontal(
            Vertical(
                Label(
                    f"{spec.profile_id} — {spec.description or spec.alias or ''}",
                    classes="profile-id-text",
                ),
                Label(
                    f"Model: {spec.model or '(not set)'}",
                    classes="profile-model-text",
                ),
                Label(
                    f"Device: {spec.device or '(default)'}  |  "
                    f"Port: {spec.port}  |  "
                    f"Source: [{source_badge_class}]{source_label}[/]  |  "
                    f"Ctx: {spec.ctx_size}  |  Threads: {spec.threads}",
                    classes="profile-meta-text",
                ),
                classes="profile-card-info",
            ),
            Horizontal(
                Button("Edit", id=f"edit-{spec.profile_id}", variant="primary"),
                Button(
                    "Delete" if not in_use else "In Use",
                    id=f"delete-{spec.profile_id}",
                    variant="error" if not in_use else "default",
                    disabled=in_use,
                ),
                classes="profile-card-actions",
            ),
            classes="profile-card",
        )

    def action_cancel(self) -> None:
        """Close the screen."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button presses to dismiss with action data."""
        btn_id = event.button.id or ""
        if btn_id == "add-profile":
            self.dismiss({"action": "add"})
        elif btn_id == "close-profiles":
            self.dismiss(None)
        elif btn_id.startswith("edit-"):
            pid = btn_id[len("edit-") :]
            self.dismiss({"action": "edit", "profile_id": pid})
        elif btn_id.startswith("delete-"):
            pid = btn_id[len("delete-") :]
            self.dismiss({"action": "delete", "profile_id": pid})
