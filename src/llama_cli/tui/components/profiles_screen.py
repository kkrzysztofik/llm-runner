"""Profiles screen showing all configured run profiles with CRUD actions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

if TYPE_CHECKING:
    from llama_manager.config.profiles import RunProfileSpec
    from llama_manager.model_index import ModelIndexEntry


class ProfilesScreen(ModalScreen[dict[str, Any] | None]):
    """Screen listing all run profiles with add/edit/delete actions.

    Returns a dict with action key (``add``, ``edit``, ``delete``) or ``None``
    to close.  ``edit`` and ``delete`` also include a ``profile_id`` key.
    """

    def __init__(
        self,
        profiles: list[tuple[RunProfileSpec, str]],
        in_use_ids: set[str] | None = None,
        model_index: list[ModelIndexEntry] | None = None,
    ) -> None:
        """Initialize the profiles screen.

        Args:
            profiles: List of ``(RunProfileSpec, source)`` tuples where source is
                ``"builtin"`` or ``"custom"``.
            in_use_ids: Set of profile IDs currently in use by running slots.
                Deletion is blocked for these.
            model_index: Cached model index entries for GGUF metadata enrichment.
        """
        super().__init__()
        self._profiles = profiles
        self._in_use_ids = in_use_ids or set()
        self._model_index = model_index or []

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
        model_details = _format_model_details(spec, self._model_index or [])
        detail_widgets = (
            [Label(model_details, classes="profile-model-text")] if model_details else []
        )

        return Horizontal(
            Vertical(
                Label(
                    f"{spec.profile_id} — {spec.description or spec.alias or ''}",
                    classes="profile-id-text",
                ),
                Label(
                    _format_model_line(spec, self._model_index or []),
                    classes="profile-model-text",
                ),
                *detail_widgets,
                Label(
                    f"Device: {spec.device or '(default)'}  |  "
                    f"Port: {spec.port}  |  "
                    f"Source: [{source_badge_class}]{source_label}[/]  |  "
                    f"Ctx: {spec.ctx_size if spec.ctx_size else '?'}  |  "
                    f"Threads: {spec.threads}",
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


def _format_model_line(
    spec: RunProfileSpec,
    model_index: list[ModelIndexEntry] | None = None,
) -> str:
    """Format model display line with filename and detected quantization."""
    path = spec.model or ""
    if not path:
        return "Model: (not set)"
    filename = Path(path).name
    quantization = None
    for entry in model_index or []:
        if entry.path == path or Path(entry.path).name == filename:
            quantization = entry.quantization_type
            break
    if quantization:
        return f"Model: {filename}  [{quantization}]"
    quant_match = re.search(r"(IQ\d_[A-Z]+|Q\d_[A-Z_]+|F16|F32)", filename)
    if quant_match:
        return f"Model: {filename}  [{quant_match.group(1)}]"
    return f"Model: {filename}"


def _format_model_details(
    spec: RunProfileSpec,
    model_index: list[ModelIndexEntry] | None = None,
) -> str:
    """Format indexed model metadata using the choose-dialog detail fields."""
    entry = _find_model_index_entry(spec, model_index)
    if entry is None:
        return ""

    parts = []
    if entry.architecture:
        parts.append(f"Arch: {entry.architecture}")
    if entry.quantization_type:
        parts.append(f"Quant: {entry.quantization_type}")
    max_context_length = entry.max_context_length or entry.context_length
    if max_context_length:
        parts.append(f"Max Ctx: {max_context_length}")
    if entry.file_size_bytes:
        size_gib = entry.file_size_bytes / (1024**3)
        parts.append(f"Size: {size_gib:.1f} GiB")
    if entry.parse_error:
        parts.append(f"Metadata: {_short_parse_error(entry.parse_error)}")
    return "  |  ".join(parts)


def _find_model_index_entry(
    spec: RunProfileSpec,
    model_index: list[ModelIndexEntry] | None = None,
) -> ModelIndexEntry | None:
    """Return the model index entry matching a profile model path or filename."""
    path = spec.model or ""
    if not path:
        return None

    filename = Path(path).name
    for entry in model_index or []:
        if entry.path == path or Path(entry.path).name == filename:
            return entry
    return None


def _short_parse_error(error: str) -> str:
    """Keep parse error display compact in profile rows."""
    first_line = error.splitlines()[0] if error else ""
    if len(first_line) <= 64:
        return first_line
    return f"{first_line[:61]}..."
