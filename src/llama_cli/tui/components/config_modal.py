"""ConfigModal — global config editor for the TUI dashboard."""

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Label, Select

from llama_manager.build_pipeline.models import SOURCE_FLAVOR_DEFAULTS
from llama_manager.config import Config

from .form_widgets import (
    CONFIG_ROW_SELECT_CLASSES,
    CONFIG_SELECT_CLASSES,
    MODAL_CANCEL_BINDINGS,
    build_config_profile_defaults_collapsible,
    field_row,
    select_row,
)

_SECTION_LABEL_CLASSES = "form-section-label config-section-label"
_FIELD_LABEL_CLASSES = "form-label config-field-label"
_FIELD_INPUT_CLASSES = "form-input config-input"
_FIELD_ROW_CLASSES = "form-row config-row"


@dataclass
class ConfigPayload:
    """Typed payload returned by the config modal on save."""

    llama_cpp_root: str = ""
    models_dir: str = ""
    llama_server_bin_intel: str = ""
    llama_server_bin_nvidia: str = ""
    host: str = ""
    build_source_flavor: str = ""
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
    default_spec_draft_model: str = ""
    default_spec_draft_hf: str = ""
    default_spec_draft_ngl: str = ""
    default_spec_dflash_cross_ctx: str = ""
    default_kv_unified: bool = False
    default_mmproj_offload: bool = True
    default_mmap: bool = True
    default_mlock: bool = False
    default_no_host_buffer: bool = False
    restart: bool = False
    clean_cache: bool = False

    def to_config_updates(self) -> dict[str, object]:
        return {
            "paths.llama_cpp_root": self.llama_cpp_root,
            "paths.models_dir": self.models_dir,
            "paths.llama_server_bin_intel": self.llama_server_bin_intel,
            "paths.llama_server_bin_nvidia": self.llama_server_bin_nvidia,
            "deployment.host": self.host,
            "build.source_flavor": self.build_source_flavor,
            "build.git_remote": self.build_git_remote,
            "build.git_branch": self.build_git_branch,
            "smoke.listen_timeout_s": self.smoke_listen_timeout_s,
            "smoke.http_request_timeout_s": self.smoke_http_request_timeout_s,
            "smoke.first_token_timeout_s": self.smoke_first_token_timeout_s,
            "smoke.total_chat_timeout_s": self.smoke_total_chat_timeout_s,
            "log_file_level": self.log_file_level,
            "log_stderr_level": self.log_stderr_level,
            "server_defaults.port": self.default_profile_port,
            "server_defaults.ctx_size": self.default_profile_ctx_size,
            "server_defaults.ubatch_size": self.default_profile_ubatch_size,
            "server_defaults.threads": self.default_profile_threads,
            "server_defaults.n_gpu_layers_profile": self.default_profile_n_gpu_layers,
            "server_defaults.bind_address": self.default_bind_address,
            "server_defaults.batch_size": self.default_batch_size,
            "server_defaults.poll_ms": self.default_poll_ms,
            "server_defaults.n_predict": self.default_n_predict,
            "server_defaults.parallel": self.default_parallel,
            "server_defaults.threads_batch": self.default_threads_batch,
            "server_defaults.cache_type_k": self.default_profile_cache_type_k,
            "server_defaults.cache_type_v": self.default_profile_cache_type_v,
            "server_defaults.reasoning_mode": self.default_reasoning_mode,
            "server_defaults.reasoning_format": self.default_reasoning_format,
            "server_defaults.reasoning_budget": self.default_reasoning_budget,
            "server_defaults.use_jinja": self.default_use_jinja,
            "server_defaults.chat_template_kwargs": self.default_profile_chat_template_kwargs,
            "server_defaults.mmproj": self.default_mmproj,
            "server_defaults.spec_type": self.default_spec_type,
            "server_defaults.spec_ngram_size_n": self.default_spec_ngram_size_n,
            "server_defaults.draft_min": self.default_draft_min,
            "server_defaults.draft_max": self.default_draft_max,
            "server_defaults.spec_draft_n_max": self.default_spec_draft_n_max,
            "server_defaults.spec_draft_p_min": self.default_spec_draft_p_min,
            "server_defaults.spec_draft_cache_type_k": self.default_spec_draft_cache_type_k,
            "server_defaults.spec_draft_cache_type_v": self.default_spec_draft_cache_type_v,
            "server_defaults.spec_draft_device": self.default_spec_draft_device,
            "server_defaults.spec_draft_model": self.default_spec_draft_model,
            "server_defaults.spec_draft_hf": self.default_spec_draft_hf,
            "server_defaults.spec_draft_ngl": self.default_spec_draft_ngl,
            "server_defaults.spec_dflash_cross_ctx": self.default_spec_dflash_cross_ctx,
            "server_defaults.kv_unified": self.default_kv_unified,
            "server_defaults.mmproj_offload": self.default_mmproj_offload,
            "server_defaults.mmap": self.default_mmap,
            "server_defaults.mlock": self.default_mlock,
            "server_defaults.no_host_buffer": self.default_no_host_buffer,
        }


class ConfigModal(ModalScreen[ConfigPayload | None]):
    """Full-screen modal for editing global Config settings.

    Returns a ``ConfigPayload`` dataclass with the edited values on save,
    with an explicit ``restart`` boolean when the caller should also restart
    all running server slots.  Returns ``None`` on cancel.
    """

    BINDINGS = MODAL_CANCEL_BINDINGS

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        c = self._config
        paths = c.paths
        deployment = c.deployment
        build = c.build
        smoke = c.smoke
        yield Container(
            Label(
                "⚙  Global Configuration",
                id="config-title",
                classes="modal-title config-title",
            ),
            VerticalScroll(
                Label("System Paths", classes=_SECTION_LABEL_CLASSES),
                field_row(
                    "llama-cpp root",
                    "llama_cpp_root",
                    paths.llama_cpp_root,
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                field_row(
                    "models directory",
                    "models_dir",
                    paths.models_dir,
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                Horizontal(
                    Label("Model Cache:", classes=_FIELD_LABEL_CLASSES),
                    Button(
                        "Clean Model Cache",
                        id="clean-model-cache",
                        classes="modal-button-danger",
                    ),
                    classes=f"{_FIELD_ROW_CLASSES} config-action-row",
                ),
                Label("Binary Paths", classes=_SECTION_LABEL_CLASSES),
                field_row(
                    "llama-server (Intel/SYCL)",
                    "llama_server_bin_intel",
                    paths.llama_server_bin_intel,
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                field_row(
                    "llama-server (NVIDIA/CUDA)",
                    "llama_server_bin_nvidia",
                    paths.llama_server_bin_nvidia,
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                Label("Network", classes=_SECTION_LABEL_CLASSES),
                field_row(
                    "bind host",
                    "host",
                    deployment.host,
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                build_config_profile_defaults_collapsible(c),
                Label("Build", classes=_SECTION_LABEL_CLASSES),
                self._source_flavor_select(build.source_flavor),
                field_row(
                    "git remote",
                    "build_git_remote",
                    build.git_remote,
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                field_row(
                    "git branch",
                    "build_git_branch",
                    build.git_branch,
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                Label("Smoke Probes (seconds)", classes=_SECTION_LABEL_CLASSES),
                field_row(
                    "listen timeout",
                    "smoke_listen_timeout_s",
                    str(smoke.listen_timeout_s),
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                field_row(
                    "http request timeout",
                    "smoke_http_request_timeout_s",
                    str(smoke.http_request_timeout_s),
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                field_row(
                    "first token timeout",
                    "smoke_first_token_timeout_s",
                    str(smoke.first_token_timeout_s),
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
                ),
                field_row(
                    "total chat timeout",
                    "smoke_total_chat_timeout_s",
                    str(smoke.total_chat_timeout_s),
                    id_prefix="cfg",
                    label_classes=_FIELD_LABEL_CLASSES,
                    input_classes=_FIELD_INPUT_CLASSES,
                    row_classes=_FIELD_ROW_CLASSES,
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

    def _source_flavor_select(self, value: str) -> Widget:
        """Build a labelled Select widget for llama.cpp source flavor."""
        flavors = list(SOURCE_FLAVOR_DEFAULTS)
        if value and value not in SOURCE_FLAVOR_DEFAULTS:
            flavors.append(value)
        choices = tuple((flavor, flavor) for flavor in flavors)
        return select_row(
            "source flavor",
            "build_source_flavor",
            choices,
            value or "upstream",
            id_prefix="cfg",
            allow_blank=False,
            label_classes=_FIELD_LABEL_CLASSES,
            input_classes=CONFIG_SELECT_CLASSES,
            row_classes=CONFIG_ROW_SELECT_CLASSES,
        )

    def _log_level_select(self, label: str, select_id: str, value: str) -> Widget:
        """Build a labelled Select widget for log level selection."""
        choices = (
            ("DEBUG", "DEBUG"),
            ("INFO", "INFO"),
            ("WARNING", "WARNING"),
            ("ERROR", "ERROR"),
            ("CRITICAL", "CRITICAL"),
        )
        return select_row(
            label,
            select_id,
            choices,
            value,
            id_prefix="cfg",
            allow_blank=False,
            label_classes=_FIELD_LABEL_CLASSES,
            input_classes=CONFIG_SELECT_CLASSES,
            row_classes=CONFIG_ROW_SELECT_CLASSES,
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
            build_source_flavor=str(
                self.query_one("#cfg-build_source_flavor", Select).value or "upstream"
            ),
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
            default_spec_draft_model=self.query_one(
                "#cfg-default_spec_draft_model", Input
            ).value.strip(),
            default_spec_draft_hf=self.query_one("#cfg-default_spec_draft_hf", Input).value.strip(),
            default_spec_draft_ngl=self.query_one(
                "#cfg-default_spec_draft_ngl", Input
            ).value.strip(),
            default_spec_dflash_cross_ctx=self.query_one(
                "#cfg-default_spec_dflash_cross_ctx", Input
            ).value.strip(),
            default_kv_unified=self.query_one("#cfg-default_kv_unified", Checkbox).value,
            default_mmproj_offload=self.query_one("#cfg-default_mmproj_offload", Checkbox).value,
            default_mmap=self.query_one("#cfg-default_mmap", Checkbox).value,
            default_mlock=self.query_one("#cfg-default_mlock", Checkbox).value,
            default_no_host_buffer=self.query_one("#cfg-default_no_host_buffer", Checkbox).value,
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
