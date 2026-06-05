"""Slot profile creation and editing modal for the TUI dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Collapsible, Input, Label, ListItem, ListView, Select

from llama_manager.config import Config
from llama_manager.config.profiles import SlotProfileSpec
from llama_manager.config.spec_decode import SpeculativeDecodingConfig
from llama_manager.model_index import ModelIndexEntry

from .form_widgets import (
    MODAL_CANCEL_BINDINGS,
    REASONING_FORMAT_CHOICES,
    REASONING_MODE_CHOICES,
    ROW_SELECT_CLASSES,
    SELECT_CLASSES,
    SPEC_TYPE_CHOICES,
    cache_type_row,
    checkbox_row,
    config_profile_prefill,
    field_row,
    select_row,
)


@dataclass
class SlotProfilePayload:
    """Payload from the slot profile modal."""

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
    bind_address: str = "127.0.0.1"
    tensor_split: str = ""
    reasoning_mode: str = "auto"
    reasoning_format: str = "none"
    reasoning_budget: str = ""
    use_jinja: bool = False
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    main_gpu: int = 0
    batch_size: int = 2048
    poll_ms: int = 50
    n_predict: int = 32768
    parallel: int = 4
    threads_batch: int = 0
    mmproj: str = ""
    spec_type: str = ""
    spec_ngram_size_n: int = 0
    draft_min: int = 0
    draft_max: int = 0
    spec_draft_n_max: int = 0
    spec_draft_p_min: float = 0.0
    spec_draft_cache_type_k: str = ""
    spec_draft_cache_type_v: str = ""
    spec_draft_device: str = ""
    save_and_add_slot: bool = False
    original_profile_id: str = ""  # filled for edits


def _parse_n_gpu_layers(raw: str) -> int | str:
    if raw.lower() == "all":
        return "all"
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return raw


class SlotProfileModal(ModalScreen[SlotProfilePayload | None]):
    """Modal for creating or editing a slot profile.

    Returns a ``SlotProfilePayload`` on save, or ``None`` on cancel.
    """

    def __init__(
        self,
        profile: SlotProfileSpec | None = None,
        edit_source: str | None = None,
        model_index: list[ModelIndexEntry] | None = None,
        config: Config | None = None,
    ) -> None:
        super().__init__()
        self._profile = profile
        self._edit_source = edit_source or ""
        self._model_index = model_index or []
        self._config = config
        self._selected_model_path: str = profile.model if profile else ""

    BINDINGS = MODAL_CANCEL_BINDINGS

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
        title = "Edit Slot Profile" if self._profile else "Create Slot Profile"
        yield Container(
            Label(
                title,
                id="profile-title",
                classes="modal-title profile-title",
            ),
            _build_form_fields(
                prefill=self._compose_prefill(),
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

    def _compose_prefill(self) -> dict[str, str]:
        if self._profile:
            return self._profile_to_prefill(self._profile)
        if self._config:
            return config_profile_prefill(self._config)
        return {}

    def _profile_to_prefill(self, spec: SlotProfileSpec) -> dict[str, str]:
        """Convert SlotProfileSpec to prefill dict for form fields."""
        spec_decode = spec.spec_decode
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
            "bind-address": spec.bind_address,
            "tensor-split": spec.tensor_split,
            "reasoning-mode": spec_decode.reasoning_mode,
            "reasoning-format": spec_decode.reasoning_format,
            "reasoning-budget": spec_decode.reasoning_budget,
            "use-jinja": "true" if spec.use_jinja else "false",
            "cache-type-k": spec.cache_type_k,
            "cache-type-v": spec.cache_type_v,
            "main-gpu": str(spec.main_gpu),
            "batch-size": str(spec.batch_size),
            "poll-ms": str(spec.poll_ms),
            "n-predict": str(spec.n_predict),
            "parallel": str(spec.parallel),
            "threads-batch": str(spec.threads_batch),
            "mmproj": spec.mmproj,
            "spec-type": spec_decode.spec_type,
            "spec-ngram-size-n": str(spec_decode.spec_ngram_size_n),
            "draft-min": str(spec_decode.draft_min),
            "draft-max": str(spec_decode.draft_max),
            "spec-draft-n-max": str(spec_decode.spec_draft_n_max),
            "spec-draft-p-min": str(spec_decode.spec_draft_p_min),
            "spec-draft-cache-type-k": spec_decode.spec_draft_cache_type_k,
            "spec-draft-cache-type-v": spec_decode.spec_draft_cache_type_v,
            "spec-draft-device": spec_decode.spec_draft_device,
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

    def _collect_values(self, *, save_and_add_slot: bool = False) -> SlotProfilePayload:
        """Read all Input widgets and return a typed payload."""
        ngl_raw = self.query_one("#profile-n-gpu-layers", Input).value.strip()
        ngl_val = _parse_n_gpu_layers(ngl_raw)

        device_select = self.query_one("#profile-device", Select)
        device_val = str(device_select.value) if device_select.value else "CUDA:0"

        return SlotProfilePayload(
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
            bind_address=self.query_one("#profile-bind-address", Input).value.strip()
            or "127.0.0.1",
            tensor_split=self.query_one("#profile-tensor-split", Input).value.strip(),
            reasoning_mode=str(self.query_one("#profile-reasoning-mode", Select).value or "auto"),
            reasoning_format=str(
                self.query_one("#profile-reasoning-format", Select).value or "none"
            ),
            reasoning_budget=self.query_one("#profile-reasoning-budget", Input).value.strip(),
            use_jinja=self.query_one("#profile-use-jinja", Checkbox).value,
            cache_type_k=str(self.query_one("#profile-cache-type-k", Select).value or "q8_0"),
            cache_type_v=str(self.query_one("#profile-cache-type-v", Select).value or "q8_0"),
            main_gpu=self._parse_int("profile-main-gpu", 0),
            batch_size=self._parse_int("profile-batch-size", 2048),
            poll_ms=self._parse_int("profile-poll-ms", 50),
            n_predict=self._parse_int("profile-n-predict", 32768),
            parallel=self._parse_int("profile-parallel", 4),
            threads_batch=self._parse_int("profile-threads-batch", 0),
            mmproj=self.query_one("#profile-mmproj", Input).value.strip(),
            spec_type=str(self.query_one("#profile-spec-type", Select).value or ""),
            spec_ngram_size_n=self._parse_int("profile-spec-ngram-size-n", 0),
            draft_min=self._parse_int("profile-draft-min", 0),
            draft_max=self._parse_int("profile-draft-max", 0),
            spec_draft_n_max=self._parse_int("profile-spec-draft-n-max", 0),
            spec_draft_p_min=self._parse_float("profile-spec-draft-p-min", 0.0),
            spec_draft_cache_type_k=str(
                self.query_one("#profile-spec-draft-cache-type-k", Select).value or ""
            ),
            spec_draft_cache_type_v=str(
                self.query_one("#profile-spec-draft-cache-type-v", Select).value or ""
            ),
            spec_draft_device=self.query_one("#profile-spec-draft-device", Input).value.strip(),
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

    def _parse_float(self, field_id: str, default: float) -> float:
        """Parse a float from an Input widget, falling back to *default*."""
        raw = self.query_one(f"#{field_id}", Input).value.strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default


def _build_form_fields(
    prefill: dict[str, str] | None = None,
    model_index: list[ModelIndexEntry] | None = None,
) -> Container:
    """Build the scrollable form body for the profile modal."""
    p = prefill or {}
    return Container(
        field_row("Profile ID", "profile-id", p.get("profile-id", "")),
        field_row("Display Label", "label", p.get("label", "")),
        _model_row(model_index or [], p.get("model", "")),
        _device_row(p.get("device", "CUDA:0")),
        field_row("Context Size", "ctx-size", p.get("ctx-size", ""), type="number"),
        _build_advanced_fields(p),
        _build_speculative_fields(p),
        field_row(
            "Chat Template Kwargs (JSON, optional)",
            "chat-template-kwargs",
            p.get("chat-template-kwargs", "{}"),
        ),
        classes="modal-scroll-body profile-scroll-body",
    )


def _build_advanced_fields(prefill: dict[str, str]) -> Collapsible:
    """Build the collapsed advanced section for optional profile tuning fields."""
    use_jinja = prefill.get("use-jinja", "false").lower() in ("1", "true", "yes", "on")
    return Collapsible(
        field_row("Server Binary (optional)", "server-bin", prefill.get("server-bin", "")),
        field_row("Port", "port", prefill.get("port", ""), type="number"),
        field_row("Ubatch Size", "ubatch-size", prefill.get("ubatch-size", ""), type="number"),
        field_row(
            "GPU Layers (int or 'all')",
            "n-gpu-layers",
            prefill.get("n-gpu-layers", "all"),
        ),
        field_row("Threads", "threads", prefill.get("threads", ""), type="number"),
        field_row("Bind Address", "bind-address", prefill.get("bind-address", "127.0.0.1")),
        field_row("Tensor Split", "tensor-split", prefill.get("tensor-split", "")),
        field_row("Main GPU", "main-gpu", prefill.get("main-gpu", "0"), type="number"),
        field_row("Batch Size", "batch-size", prefill.get("batch-size", ""), type="number"),
        field_row("Poll (ms)", "poll-ms", prefill.get("poll-ms", ""), type="number"),
        field_row("N Predict", "n-predict", prefill.get("n-predict", ""), type="number"),
        field_row(
            "Parallel (-1=unlimited)",
            "parallel",
            prefill.get("parallel", ""),
            type="number",
        ),
        field_row(
            "Threads Batch (0=omit)",
            "threads-batch",
            prefill.get("threads-batch", ""),
            type="number",
        ),
        cache_type_row("Cache Type K", "cache-type-k", prefill.get("cache-type-k", "q8_0")),
        cache_type_row("Cache Type V", "cache-type-v", prefill.get("cache-type-v", "q8_0")),
        select_row(
            "Reasoning Mode",
            "reasoning-mode",
            REASONING_MODE_CHOICES,
            prefill.get("reasoning-mode", "auto"),
        ),
        select_row(
            "Reasoning Format",
            "reasoning-format",
            REASONING_FORMAT_CHOICES,
            prefill.get("reasoning-format", "none"),
        ),
        field_row("Reasoning Budget", "reasoning-budget", prefill.get("reasoning-budget", "")),
        checkbox_row("Use Jinja", "use-jinja", use_jinja),
        field_row("MMProj (optional)", "mmproj", prefill.get("mmproj", "")),
        title="Advanced",
        collapsed=True,
        id="profile-advanced-collapsible",
        classes="profile-advanced-options",
    )


def _build_speculative_fields(prefill: dict[str, str]) -> Collapsible:
    """Build the collapsed speculative decoding section."""
    return Collapsible(
        select_row("Spec Type", "spec-type", SPEC_TYPE_CHOICES, prefill.get("spec-type", "")),
        field_row(
            "Ngram Size N",
            "spec-ngram-size-n",
            prefill.get("spec-ngram-size-n", "0"),
            type="number",
        ),
        field_row("Draft Min", "draft-min", prefill.get("draft-min", "0"), type="number"),
        field_row("Draft Max", "draft-max", prefill.get("draft-max", "0"), type="number"),
        field_row(
            "Draft N Max (MTP)",
            "spec-draft-n-max",
            prefill.get("spec-draft-n-max", "0"),
            type="number",
        ),
        field_row("Draft P Min (MTP)", "spec-draft-p-min", prefill.get("spec-draft-p-min", "0")),
        cache_type_row(
            "Draft Cache K",
            "spec-draft-cache-type-k",
            prefill.get("spec-draft-cache-type-k", ""),
            allow_empty=True,
        ),
        cache_type_row(
            "Draft Cache V",
            "spec-draft-cache-type-v",
            prefill.get("spec-draft-cache-type-v", ""),
            allow_empty=True,
        ),
        field_row("Draft Device", "spec-draft-device", prefill.get("spec-draft-device", "")),
        title="Speculative decoding",
        collapsed=True,
        classes="profile-advanced-options profile-speculative-options",
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
            classes=SELECT_CLASSES,
        ),
        classes=ROW_SELECT_CLASSES,
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


def _model_detail_parts(entry: ModelIndexEntry) -> list[str]:
    """Return compact model metadata parts for the modal details label."""
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
    return parts


def _short_parse_error(error: str) -> str:
    """Convert long parse exceptions into a UI-sized message."""
    if "timed out after" in error:
        return "parse timed out; using filename/cache fallback"
    return error.split(" for ", maxsplit=1)[0]


def payload_to_slot_profile_spec(profile_id: str, payload: SlotProfilePayload) -> SlotProfileSpec:
    """Build a ``SlotProfileSpec`` from modal form payload."""
    ngl = payload.n_gpu_layers
    ctk = payload.chat_template_kwargs
    return SlotProfileSpec(
        profile_id=profile_id,
        alias=payload.label or profile_id,
        device=payload.device,
        model=payload.model,
        port=payload.port,
        ctx_size=payload.ctx_size,
        ubatch_size=payload.ubatch_size,
        threads=payload.threads,
        description=payload.label or "",
        bind_address=payload.bind_address,
        tensor_split=payload.tensor_split,
        chat_template_kwargs=ctk if isinstance(ctk, str) else "",
        use_jinja=payload.use_jinja,
        cache_type_k=payload.cache_type_k,
        cache_type_v=payload.cache_type_v,
        n_gpu_layers=ngl if ngl == "all" else int(ngl),
        main_gpu=payload.main_gpu,
        server_bin=payload.server_bin,
        backend="llama_cpp",
        batch_size=payload.batch_size,
        poll_ms=payload.poll_ms,
        n_predict=payload.n_predict,
        parallel=payload.parallel,
        threads_batch=payload.threads_batch,
        mmproj=payload.mmproj,
        spec_decode=SpeculativeDecodingConfig(
            reasoning_mode=payload.reasoning_mode,
            reasoning_format=payload.reasoning_format,
            reasoning_budget=payload.reasoning_budget,
            spec_type=payload.spec_type,
            spec_ngram_size_n=payload.spec_ngram_size_n,
            draft_min=payload.draft_min,
            draft_max=payload.draft_max,
            spec_draft_n_max=payload.spec_draft_n_max,
            spec_draft_p_min=payload.spec_draft_p_min,
            spec_draft_cache_type_k=payload.spec_draft_cache_type_k,
            spec_draft_cache_type_v=payload.spec_draft_cache_type_v,
            spec_draft_device=payload.spec_draft_device,
        ),
    )
