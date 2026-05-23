"""Run profile creation and editing modal for the TUI dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Select

from llama_manager.config.profiles import RunProfileSpec
from llama_manager.model_index import ModelIndexEntry


@dataclass
class RunProfilePayload:
    """Payload from the run profile modal."""

    profile_id: str = ""
    label: str = ""
    server_bin: str = ""
    model: str = ""
    port: int = 8080
    ctx_size: int = 4096
    ubatch_size: int = 512
    n_gpu_layers: int | str = "all"
    threads: int = 8
    chat_template_kwargs: str = ""
    device: str = "CUDA:0"
    save_and_add_slot: bool = False
    original_profile_id: str = ""  # filled for edits


class RunProfileModal(ModalScreen[RunProfilePayload | None]):
    """Modal for creating or editing a run profile.

    Returns a ``RunProfilePayload`` on save, or ``None`` on cancel.
    """

    def __init__(
        self,
        profile: RunProfileSpec | None = None,
        edit_source: str | None = None,
        model_index: list[ModelIndexEntry] | None = None,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._edit_source = edit_source or ""
        self._model_index = model_index or []
        self._selected_model_path: str = profile.model if profile else ""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    CSS = """
    .profile-dialog {
        width: 70%;
        max-width: 95%;
        max-height: 90%;
    }

    .profile-scroll-body {
        width: 100%;
        padding-right: 1;
        height: 1fr;
    }

    .profile-field-label {
        width: 28;
        height: 3;
        color: $text;
        padding-right: 1;
        content-align: right middle;
    }

    .profile-input {
        width: 1fr;
        height: 3;
    }

    .profile-input Select {
        height: 3;
    }

    .profile-row.profile-model-row {
        height: auto;
        min-height: 10;
    }

    .profile-model-picker {
        width: 1fr;
        height: auto;
    }

    .profile-model-list {
        height: 8;
        max-height: 15;
        border: solid $panel;
        margin-bottom: 1;
    }

    .profile-selection-summary {
        height: auto;
        min-height: 1;
        color: $text;
        margin-top: 0;
        margin-bottom: 1;
    }

    .profile-model-option {
        color: $text;
    }

    #profile-model-details {
        color: $text-muted;
        margin-bottom: 1;
        display: none;
    }

    #profile-device > SelectCurrent {
        background: $surface-lighten-1;
        color: $text;
    }

    #profile-device > SelectCurrent Static#label {
        color: $text;
    }

    #profile-device-summary {
        width: 28;
        color: $text-muted;
        content-align: left middle;
    }

    .profile-row {
        width: 100%;
        height: 3;
        margin-bottom: 1;
    }

    .profile-actions {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }

    .profile-actions Button {
        margin-left: 1;
    }

    .profile-title {
        color: $accent;
    }
    """

    def compose(self) -> ComposeResult:
        title = "Edit Run Profile" if self._profile else "Create Run Profile"
        yield Container(
            Label(
                title,
                id="profile-title",
                classes="modal-title profile-title",
            ),
            _build_form_fields(
                prefill=self._profile_to_prefill(self._profile) if self._profile else {},
                model_index=self._model_index,
            ),
            Horizontal(
                Button("Cancel", id="cancel-profile", classes="modal-button-cancel"),
                Button(
                    "Save Changes" if self._profile else "Save Profile",
                    id="save-profile",
                    classes="modal-button-success",
                ),
                Button(
                    "Save & Add Slot",
                    id="save-add-profile",
                    classes="modal-button-warning",
                ),
                id="profile-actions",
                classes="modal-actions profile-actions",
            ),
            id="profile-dialog",
            classes="modal-dialog profile-dialog",
        )

    def on_mount(self) -> None:
        self.query_one("#profile-profile-id", Input).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle model selection from the list view."""
        item = event.item
        label_widget = item.query_one(Label)

        for entry in self._model_index:
            entry_label = f"{Path(entry.path).name} ({entry.quantization_type or '?'})"
            if label_widget.content == entry_label:
                self._selected_model_path = entry.path
                self.query_one("#profile-model-search", Input).value = entry.path
                self._update_selected_model(entry)
                break

    def on_select_changed(self, event: Select.Changed) -> None:
        """Keep a plain visible device summary beside the Select widget."""
        if event.select.id != "profile-device":
            return
        value = str(event.value) if event.value else "CUDA:0"
        self.query_one("#profile-device-summary", Label).update(_device_label(value))

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter model list as user types in the search input."""
        if event.input.id != "profile-model-search":
            return

        query = event.value.strip().lower()
        lst = self.query_one("#profile-model-list", ListView)
        self.query_one("#profile-selected-model", Label).update(
            _selected_model_text(event.value.strip())
        )

        lst.remove_children()

        if not self._model_index:
            self._selected_model_path = event.value.strip()
            return

        for entry in self._model_index:
            filename = Path(entry.path).name
            label = f"{filename} ({entry.quantization_type or '?'})"
            search_text = (
                f"{entry.path} {filename} {entry.quantization_type or ''} "
                f"{entry.architecture or ''}"
            )
            if query in search_text.lower():
                lst.mount(ListItem(Label(label, classes="profile-model-option")))

        if query and ("/" in query or "\\" in query or query.endswith(".gguf")):
            self._selected_model_path = event.value

    def _update_selected_model(self, entry: ModelIndexEntry) -> None:
        """Show model selection and metadata in stable plain labels."""
        self.query_one("#profile-selected-model", Label).update(_selected_model_text(entry.path))
        details = self.query_one("#profile-model-details", Label)
        parts = _model_detail_parts(entry)
        details.update("  |  ".join(parts))
        details.styles.display = "block" if parts else "none"

    def _profile_to_prefill(self, spec: RunProfileSpec) -> dict[str, str]:
        """Convert RunProfileSpec to prefill dict for form fields."""
        return {
            "profile-id": spec.profile_id,
            "label": spec.alias if spec.alias != spec.profile_id else "",
            "model": spec.model,
            "server-bin": spec.server_bin,
            "port": str(spec.port),
            "ctx-size": str(spec.ctx_size),
            "ubatch-size": str(spec.ubatch_size),
            "n-gpu-layers": str(spec.n_gpu_layers),
            "threads": str(spec.threads),
            "chat-template-kwargs": (
                spec.chat_template_kwargs if spec.chat_template_kwargs else "{}"
            ),
            "device": spec.device or "CUDA:0",
        }

    def action_cancel(self) -> None:
        """Close the modal."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button presses to collect values and dismiss."""
        if event.button.id == "cancel-profile":
            self.dismiss(None)
        elif event.button.id == "save-profile":
            self.dismiss(self._collect_values(save_and_add_slot=False))
        elif event.button.id == "save-add-profile":
            self.dismiss(self._collect_values(save_and_add_slot=True))

    def _collect_values(self, *, save_and_add_slot: bool = False) -> RunProfilePayload:
        """Read all Input widgets and return a typed payload."""
        ngl_raw = self.query_one("#profile-n-gpu-layers", Input).value.strip()
        if ngl_raw.lower() == "all":
            ngl_val: int | str = "all"
        else:
            try:
                ngl_val = int(ngl_raw) if ngl_raw else 0
            except ValueError:
                ngl_val = 0

        device_select = self.query_one("#profile-device", Select)
        device_val = str(device_select.value) if device_select.value else "CUDA:0"

        return RunProfilePayload(
            profile_id=self.query_one("#profile-profile-id", Input).value.strip(),
            label=self.query_one("#profile-label", Input).value.strip(),
            server_bin=self.query_one("#profile-server-bin", Input).value.strip(),
            model=self._selected_model_path or "",
            port=self._parse_int("profile-port", 8080),
            ctx_size=self._parse_int("profile-ctx-size", 4096),
            ubatch_size=self._parse_int("profile-ubatch-size", 512),
            n_gpu_layers=ngl_val,
            threads=self._parse_int("profile-threads", 8),
            chat_template_kwargs=self.query_one(
                "#profile-chat-template-kwargs", Input
            ).value.strip(),
            device=device_val,
            save_and_add_slot=save_and_add_slot,
            original_profile_id=(self._profile.profile_id if self._profile else ""),
        )

    def _parse_int(self, field_id: str, default: int) -> int:
        """Parse an integer from an Input widget, falling back to *default*."""
        raw = self.query_one(f"#{field_id}", Input).value.strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default


def _build_form_fields(
    prefill: dict[str, str] | None = None,
    model_index: list[ModelIndexEntry] | None = None,
) -> Container:
    """Build the scrollable form body for the profile modal."""
    p = prefill or {}
    return Container(
        _field_row("Profile ID", "profile-id", p.get("profile-id", "")),
        _field_row("Display Label", "label", p.get("label", "")),
        _model_row(model_index or [], p.get("model", "")),
        _field_row("Server Binary (optional)", "server-bin", p.get("server-bin", "")),
        _device_row(p.get("device", "CUDA:0")),
        _field_row("Port", "port", p.get("port", ""), type="number"),
        _field_row("Context Size", "ctx-size", p.get("ctx-size", ""), type="number"),
        _field_row(
            "Ubatch Size",
            "ubatch-size",
            p.get("ubatch-size", ""),
            type="number",
        ),
        _field_row(
            "GPU Layers (int or 'all')",
            "n-gpu-layers",
            p.get("n-gpu-layers", "all"),
        ),
        _field_row("Threads", "threads", p.get("threads", ""), type="number"),
        _field_row(
            "Chat Template Kwargs (JSON, optional)",
            "chat-template-kwargs",
            p.get("chat-template-kwargs", "{}"),
        ),
        classes="modal-scroll-body profile-scroll-body",
    )


def _field_row(
    label: str,
    field_id: str,
    value: str = "",
    *,
    type: Literal["text", "number", "integer"] = "text",
) -> Horizontal:
    """Build a labelled input row for one profile field."""
    return Horizontal(
        Label(f"{label}:", classes="form-label profile-field-label"),
        Input(
            value=value,
            id=f"profile-{field_id}",
            type=type,
            classes="form-input profile-input",
        ),
        classes="form-row profile-row",
    )


def _device_row(current_value: str = "CUDA:0") -> Horizontal:
    """Build a labelled Select row for device backend assignment."""
    if not current_value:
        current_value = "CUDA:0"
    return Horizontal(
        Label("Device:", classes="form-label profile-field-label"),
        Select(
            value=current_value,
            allow_blank=False,
            options=[
                ("CUDA:0 — NVIDIA GPU", "CUDA:0"),
                ("CUDA:0,1 — NVIDIA GPUs", "CUDA:0,1"),
                ("CUDA:1 — NVIDIA GPU", "CUDA:1"),
                ("SYCL0 — Intel Arc GPU", "SYCL0"),
            ],
            id="profile-device",
            classes="form-input profile-input",
        ),
        Label(_device_label(current_value), id="profile-device-summary"),
        classes="form-row profile-row",
    )


def _model_row(
    index: list[ModelIndexEntry],
    prefill_path: str = "",
) -> Horizontal:
    """Build the model selection row with search input and indexed model list."""
    search_placeholder = (
        "No indexed models found. Type path manually..."
        if not index
        else "Search indexed models or type path..."
    )
    items: list[ListItem] = []
    if index:
        for entry in index:
            filename = Path(entry.path).name
            label = f"{filename} ({entry.quantization_type or '?'})"
            items.append(ListItem(Label(label, classes="profile-model-option")))

    return Horizontal(
        Label("Model Path:", classes="form-label profile-field-label"),
        Vertical(
            Input(
                value=prefill_path,
                id="profile-model-search",
                placeholder=search_placeholder,
                classes="form-input profile-input",
            ),
            Label(
                _selected_model_text(prefill_path),
                id="profile-selected-model",
                classes="profile-selection-summary",
            ),
            ListView(*items, id="profile-model-list", classes="profile-model-list"),
            Label(id="profile-model-details", classes="profile-model-details"),
            classes="profile-model-picker",
        ),
        classes="form-row profile-row profile-model-row",
    )


def _selected_model_text(path: str) -> str:
    """Format the currently selected model for visible display."""
    if not path:
        return "Selected: (none)"
    return f"Selected: {Path(path).name}"


def _device_label(value: str) -> str:
    """Format device value for a visible summary label."""
    labels = {
        "CUDA:0": "CUDA:0 NVIDIA",
        "CUDA:0,1": "CUDA:0,1 NVIDIA",
        "CUDA:1": "CUDA:1 NVIDIA",
        "SYCL0": "SYCL0 Intel Arc",
    }
    return labels.get(value, value or "CUDA:0 NVIDIA")


def _model_detail_parts(entry: ModelIndexEntry) -> list[str]:
    """Return compact model metadata parts for the modal details label."""
    parts = []
    if entry.architecture:
        parts.append(f"Arch: {entry.architecture}")
    if entry.quantization_type:
        parts.append(f"Quant: {entry.quantization_type}")
    if entry.context_length:
        parts.append(f"Ctx: {entry.context_length}")
    if entry.file_size_bytes:
        size_gib = entry.file_size_bytes / (1024**3)
        parts.append(f"Size: {size_gib:.1f} GiB")
    if entry.parse_error:
        parts.append(f"Metadata: {_short_parse_error(entry.parse_error)}")
    return parts


def _short_parse_error(error: str) -> str:
    """Convert long parse exceptions into a UI-sized message."""
    if "timed out after" in error:
        return "parse timed out; using filename/cache fallback"
    return error.split(" for ", maxsplit=1)[0]
