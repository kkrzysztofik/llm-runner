"""ConfigModal — global config editor for the TUI dashboard."""

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Label, Select

from llama_manager.config import Config

from .form_widgets import build_config_profile_defaults_collapsible

_SECTION_LABEL_CLASSES = "form-section-label config-section-label"


@dataclass
class ConfigPayload:
    """Typed payload returned by the config modal on save."""

    llama_cpp_root: str = ""
    models_dir: str = ""
    llama_server_bin_intel: str = ""
    llama_server_bin_nvidia: str = ""
    host: str = ""
    build_git_remote: str = ""
    build_git_branch: str = ""
    smoke_listen_timeout_s: str = ""
    smoke_http_request_timeout_s: str = ""
    smoke_first_token_timeout_s: str = ""
    smoke_total_chat_timeout_s: str = ""
    log_file_level: str = ""
    log_stderr_level: str = ""
    default_profile_port: str = ""
    default_profile_ctx_size: str = ""
    default_profile_ubatch_size: str = ""
    default_profile_threads: str = ""
    default_profile_n_gpu_layers: str = ""
    default_bind_address: str = ""
    default_batch_size: str = ""
    default_poll_ms: str = ""
    default_n_predict: str = ""
    default_parallel: str = ""
    default_threads_batch: str = ""
    default_profile_cache_type_k: str = ""
    default_profile_cache_type_v: str = ""
    default_reasoning_mode: str = ""
    default_reasoning_format: str = ""
    default_reasoning_budget: str = ""
    default_use_jinja: bool = False
    default_profile_chat_template_kwargs: str = ""
    default_mmproj: str = ""
    default_spec_type: str = ""
    default_spec_ngram_size_n: str = ""
    default_draft_min: str = ""
    default_draft_max: str = ""
    default_spec_draft_n_max: str = ""
    default_spec_draft_p_min: str = ""
    default_spec_draft_cache_type_k: str = ""
    default_spec_draft_cache_type_v: str = ""
    default_spec_draft_device: str = ""
    restart: bool = False
    clean_cache: bool = False


class ConfigModal(ModalScreen[ConfigPayload | None]):
    """Full-screen modal for editing global Config settings.

    Returns a ``ConfigPayload`` dataclass with the edited values on save,
    with an explicit ``restart`` boolean when the caller should also restart
    all running server slots.  Returns ``None`` on cancel.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        c = self._config
        yield Container(
            Label(
                "⚙  Global Configuration",
                id="config-title",
                classes="modal-title config-title",
            ),
            VerticalScroll(
                Label("System Paths", classes=_SECTION_LABEL_CLASSES),
                self._field_row("llama-cpp root", "llama_cpp_root", c.llama_cpp_root),
                self._field_row("models directory", "models_dir", c.models_dir),
                Horizontal(
                    Label("Model Cache:", classes="form-label config-field-label"),
                    Button(
                        "Clean Model Cache",
                        id="clean-model-cache",
                        classes="modal-button-danger",
                    ),
                    classes="form-row config-row config-action-row",
                ),
                Label("Binary Paths", classes=_SECTION_LABEL_CLASSES),
                self._field_row(
                    "llama-server (Intel/SYCL)",
                    "llama_server_bin_intel",
                    c.llama_server_bin_intel,
                ),
                self._field_row(
                    "llama-server (NVIDIA/CUDA)",
                    "llama_server_bin_nvidia",
                    c.llama_server_bin_nvidia,
                ),
                Label("Network", classes=_SECTION_LABEL_CLASSES),
                self._field_row("bind host", "host", c.host),
                build_config_profile_defaults_collapsible(c),
                Label("Build", classes=_SECTION_LABEL_CLASSES),
                self._field_row("git remote", "build_git_remote", c.build_git_remote),
                self._field_row("git branch", "build_git_branch", c.build_git_branch),
                Label("Smoke Probes (seconds)", classes=_SECTION_LABEL_CLASSES),
                self._field_row(
                    "listen timeout",
                    "smoke_listen_timeout_s",
                    str(c.smoke_listen_timeout_s),
                ),
                self._field_row(
                    "http request timeout",
                    "smoke_http_request_timeout_s",
                    str(c.smoke_http_request_timeout_s),
                ),
                self._field_row(
                    "first token timeout",
                    "smoke_first_token_timeout_s",
                    str(c.smoke_first_token_timeout_s),
                ),
                self._field_row(
                    "total chat timeout",
                    "smoke_total_chat_timeout_s",
                    str(c.smoke_total_chat_timeout_s),
                ),
                Label("Logging", classes=_SECTION_LABEL_CLASSES),
                self._log_level_select("stderr level", "log_stderr_level", c.log_stderr_level),
                self._log_level_select("file level", "log_file_level", c.log_file_level),
                classes="modal-scroll-body config-scroll-body",
            ),
            Horizontal(
                Button("Cancel", id="cancel-config", classes="modal-button-cancel"),
                Button("Save", id="save-config", classes="modal-button-success"),
                Button(
                    "Save & Restart",
                    id="save-restart-config",
                    classes="modal-button-warning",
                ),
                id="config-actions",
                classes="modal-actions config-actions",
            ),
            id="config-dialog",
            classes="modal-dialog config-dialog",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _field_row(self, label: str, field_id: str, value: str) -> Widget:
        """Build a labelled input row for one config field."""
        return Horizontal(
            Label(f"{label}:", classes="form-label config-field-label"),
            Input(
                value=value,
                id=f"cfg-{field_id}",
                classes="form-input config-input",
            ),
            classes="form-row config-row",
        )

    def _log_level_select(self, label: str, select_id: str, value: str) -> Widget:
        """Build a labelled Select widget for log level selection."""
        choices = [
            ("DEBUG", "DEBUG"),
            ("INFO", "INFO"),
            ("WARNING", "WARNING"),
            ("ERROR", "ERROR"),
            ("CRITICAL", "CRITICAL"),
        ]
        return Horizontal(
            Label(f"{label}:", classes="form-label config-field-label"),
            Select(
                choices,
                value=value,
                id=f"cfg-{select_id}",
                classes="form-input config-input",
            ),
            classes="form-row config-row",
        )

    def _collect_values(self) -> ConfigPayload:
        """Read all Input widgets and return a typed payload."""
        return ConfigPayload(
            llama_cpp_root=self.query_one("#cfg-llama_cpp_root", Input).value.strip(),
            models_dir=self.query_one("#cfg-models_dir", Input).value.strip(),
            llama_server_bin_intel=self.query_one(
                "#cfg-llama_server_bin_intel", Input
            ).value.strip(),
            llama_server_bin_nvidia=self.query_one(
                "#cfg-llama_server_bin_nvidia", Input
            ).value.strip(),
            host=self.query_one("#cfg-host", Input).value.strip(),
            build_git_remote=self.query_one("#cfg-build_git_remote", Input).value.strip(),
            build_git_branch=self.query_one("#cfg-build_git_branch", Input).value.strip(),
            smoke_listen_timeout_s=self.query_one(
                "#cfg-smoke_listen_timeout_s", Input
            ).value.strip(),
            smoke_http_request_timeout_s=self.query_one(
                "#cfg-smoke_http_request_timeout_s", Input
            ).value.strip(),
            smoke_first_token_timeout_s=self.query_one(
                "#cfg-smoke_first_token_timeout_s", Input
            ).value.strip(),
            smoke_total_chat_timeout_s=self.query_one(
                "#cfg-smoke_total_chat_timeout_s", Input
            ).value.strip(),
            log_file_level=str(self.query_one("#cfg-log_file_level", Select).value or "DEBUG"),
            log_stderr_level=str(self.query_one("#cfg-log_stderr_level", Select).value or "INFO"),
            default_profile_port=self.query_one("#cfg-default_profile_port", Input).value.strip(),
            default_profile_ctx_size=self.query_one(
                "#cfg-default_profile_ctx_size", Input
            ).value.strip(),
            default_profile_ubatch_size=self.query_one(
                "#cfg-default_profile_ubatch_size", Input
            ).value.strip(),
            default_profile_threads=self.query_one(
                "#cfg-default_profile_threads", Input
            ).value.strip(),
            default_profile_n_gpu_layers=self.query_one(
                "#cfg-default_profile_n_gpu_layers", Input
            ).value.strip(),
            default_bind_address=self.query_one("#cfg-default_bind_address", Input).value.strip(),
            default_batch_size=self.query_one("#cfg-default_batch_size", Input).value.strip(),
            default_poll_ms=self.query_one("#cfg-default_poll_ms", Input).value.strip(),
            default_n_predict=self.query_one("#cfg-default_n_predict", Input).value.strip(),
            default_parallel=str(self.query_one("#cfg-default_parallel", Select).value or "4"),
            default_threads_batch=self.query_one("#cfg-default_threads_batch", Input).value.strip(),
            default_profile_cache_type_k=str(
                self.query_one("#cfg-default_profile_cache_type_k", Select).value or "q8_0"
            ),
            default_profile_cache_type_v=str(
                self.query_one("#cfg-default_profile_cache_type_v", Select).value or "q8_0"
            ),
            default_reasoning_mode=str(
                self.query_one("#cfg-default_reasoning_mode", Select).value or "auto"
            ),
            default_reasoning_format=str(
                self.query_one("#cfg-default_reasoning_format", Select).value or "none"
            ),
            default_reasoning_budget=self.query_one(
                "#cfg-default_reasoning_budget", Input
            ).value.strip(),
            default_use_jinja=self.query_one("#cfg-default_use_jinja", Checkbox).value,
            default_profile_chat_template_kwargs=self.query_one(
                "#cfg-default_profile_chat_template_kwargs", Input
            ).value.strip(),
            default_mmproj=self.query_one("#cfg-default_mmproj", Input).value.strip(),
            default_spec_type=str(self.query_one("#cfg-default_spec_type", Select).value or ""),
            default_spec_ngram_size_n=self.query_one(
                "#cfg-default_spec_ngram_size_n", Input
            ).value.strip(),
            default_draft_min=self.query_one("#cfg-default_draft_min", Input).value.strip(),
            default_draft_max=self.query_one("#cfg-default_draft_max", Input).value.strip(),
            default_spec_draft_n_max=self.query_one(
                "#cfg-default_spec_draft_n_max", Input
            ).value.strip(),
            default_spec_draft_p_min=self.query_one(
                "#cfg-default_spec_draft_p_min", Input
            ).value.strip(),
            default_spec_draft_cache_type_k=str(
                self.query_one("#cfg-default_spec_draft_cache_type_k", Select).value or ""
            ),
            default_spec_draft_cache_type_v=str(
                self.query_one("#cfg-default_spec_draft_cache_type_v", Select).value or ""
            ),
            default_spec_draft_device=self.query_one(
                "#cfg-default_spec_draft_device", Input
            ).value.strip(),
        )

    # ------------------------------------------------------------------
    # Actions & event handlers
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.query_one("#cfg-llama_cpp_root", Input).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-config":
            self.dismiss(None)
        elif event.button.id == "save-config":
            self.dismiss(self._collect_values())
        elif event.button.id == "save-restart-config":
            values = self._collect_values()
            values.restart = True
            self.dismiss(values)
        elif event.button.id == "clean-model-cache":
            values = self._collect_values()
            values.clean_cache = True
            self.dismiss(values)
