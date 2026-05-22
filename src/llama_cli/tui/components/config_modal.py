"""ConfigModal — global config editor for the TUI dashboard."""

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Input, Label

from llama_manager.config import Config

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
    restart: bool = False


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
