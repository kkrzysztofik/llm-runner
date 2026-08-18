"""ConfigModal — global config editor for the TUI dashboard."""

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Label, Select

from llama_manager.build_pipeline.models import SOURCE_FLAVOR_DEFAULTS
from llama_manager.config import Config
from llama_manager.config.load_mode import LOAD_MODE_VALUES
from llama_manager.config.reasoning_effort import (
    REASONING_EFFORT_JSON_CONFLICT,
    REASONING_EFFORT_VALUES,
    chat_template_kwargs_has_reasoning_effort,
)
from llama_manager.config.server import SPLIT_MODE_VALUES
from llama_manager.config.tri_state import TRI_STATE_VALUES

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
    default_reasoning_preserve: str = "auto"
    default_reasoning_effort: str = "medium"
    default_reasoning_budget_message: str = ""
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
    default_load_mode: str = "auto"
    default_split_mode: str = "layer"
    default_nvidia_power_limit_watts: str = "290"
    default_no_host_buffer: bool = False
    default_ui: bool = False
    default_fit: str = "auto"
    default_ctx_checkpoints: str = ""
    default_temperature: str = ""
    default_top_k: str = ""
    default_top_p: str = ""
    default_min_p: str = ""
    default_presence_penalty: str = ""
    default_repeat_penalty: str = ""
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
            "server_defaults.reasoning_preserve": self.default_reasoning_preserve,
            "server_defaults.reasoning_effort": self.default_reasoning_effort,
            "server_defaults.reasoning_budget_message": self.default_reasoning_budget_message,
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
            "server_defaults.load_mode": self.default_load_mode,
            "server_defaults.split_mode": self.default_split_mode,
            "server_defaults.nvidia_power_limit_watts": self.default_nvidia_power_limit_watts,
            "server_defaults.no_host_buffer": self.default_no_host_buffer,
            "server_defaults.ui": self.default_ui,
            "server_defaults.fit": self.default_fit,
            "server_defaults.ctx_checkpoints": _optional_config_int(self.default_ctx_checkpoints),
            "server_defaults.temperature": _optional_config_float(self.default_temperature),
            "server_defaults.top_k": _optional_config_int(self.default_top_k),
            "server_defaults.top_p": _optional_config_float(self.default_top_p),
            "server_defaults.min_p": _optional_config_float(self.default_min_p),
            "server_defaults.presence_penalty": _optional_config_float(
                self.default_presence_penalty
            ),
            "server_defaults.repeat_penalty": _optional_config_float(self.default_repeat_penalty),
        }


def _optional_config_int(raw: str) -> int | None:
    """Return None for empty optional integer fields."""
    stripped = raw.strip()
    if not stripped:
        return None
    return int(stripped)


def _optional_config_float(raw: str) -> float | None:
    """Return None for empty optional float fields."""
    stripped = raw.strip()
    if not stripped:
        return None
    return float(stripped)


def _validate_optional_number_field(
    label: str, raw: str, *, is_int: bool, non_negative: bool = False
) -> str | None:
    """Return an error message for an optional numeric field, or None if valid."""
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        value = int(stripped) if is_int else float(stripped)
    except ValueError:
        return f"Invalid {label}: {raw!r}"
    if is_int and non_negative and value < 0:
        return f"Invalid {label}: {value} (must be >= 0)"
    return None


def _validate_config_payload(payload: ConfigPayload) -> list[str]:
    """Validate enum and optional numeric fields before save."""
    errors: list[str] = []
    enum_fields = (
        ("load mode", payload.default_load_mode, LOAD_MODE_VALUES),
        ("split mode", payload.default_split_mode, SPLIT_MODE_VALUES),
        ("reasoning preserve", payload.default_reasoning_preserve, TRI_STATE_VALUES),
        ("thinking level", payload.default_reasoning_effort, REASONING_EFFORT_VALUES),
        ("fit", payload.default_fit, TRI_STATE_VALUES),
    )
    for label, value, allowed in enum_fields:
        if value not in allowed:
            errors.append(f"Invalid {label}: {value!r}")
    if chat_template_kwargs_has_reasoning_effort(payload.default_profile_chat_template_kwargs):
        errors.append(REASONING_EFFORT_JSON_CONFLICT)

    numeric_fields = (
        ("ctx checkpoints", payload.default_ctx_checkpoints, True, True),
        ("top k", payload.default_top_k, True, False),
        ("temperature", payload.default_temperature, False, False),
        ("top p", payload.default_top_p, False, False),
        ("min p", payload.default_min_p, False, False),
        ("presence penalty", payload.default_presence_penalty, False, False),
        ("repeat penalty", payload.default_repeat_penalty, False, False),
        ("nvidia power limit", payload.default_nvidia_power_limit_watts, True, True),
    )
    if not payload.default_nvidia_power_limit_watts.strip():
        errors.append("Invalid nvidia power limit: must be a number (0 = disabled)")
    for label, raw, is_int, non_negative in numeric_fields:
        error = _validate_optional_number_field(
            label, raw, is_int=is_int, non_negative=non_negative
        )
        if error is not None:
            errors.append(error)

    return errors


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

    def _select_value(self, select_id: str, default: str) -> str:
        """Read a Select widget, falling back to a default when blank."""
        value = self.query_one(select_id, Select).value
        return str(value or default)

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
            build_source_flavor=self._select_value("#cfg-build_source_flavor", "upstream"),
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
            log_file_level=self._select_value("#cfg-log_file_level", "DEBUG"),
            log_stderr_level=self._select_value("#cfg-log_stderr_level", "INFO"),
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
            default_parallel=self._select_value("#cfg-default_parallel", "4"),
            default_threads_batch=self.query_one("#cfg-default_threads_batch", Input).value.strip(),
            default_profile_cache_type_k=self._select_value(
                "#cfg-default_profile_cache_type_k", "q8_0"
            ),
            default_profile_cache_type_v=self._select_value(
                "#cfg-default_profile_cache_type_v", "q8_0"
            ),
            default_reasoning_mode=self._select_value("#cfg-default_reasoning_mode", "auto"),
            default_reasoning_format=self._select_value("#cfg-default_reasoning_format", "none"),
            default_reasoning_budget=self.query_one(
                "#cfg-default_reasoning_budget", Input
            ).value.strip(),
            default_reasoning_preserve=self._select_value(
                "#cfg-default_reasoning_preserve", "auto"
            ),
            default_reasoning_effort=self._select_value("#cfg-default_reasoning_effort", "medium"),
            default_reasoning_budget_message=self.query_one(
                "#cfg-default_reasoning_budget_message", Input
            ).value.strip(),
            default_use_jinja=self.query_one("#cfg-default_use_jinja", Checkbox).value,
            default_profile_chat_template_kwargs=self.query_one(
                "#cfg-default_profile_chat_template_kwargs", Input
            ).value.strip(),
            default_mmproj=self.query_one("#cfg-default_mmproj", Input).value.strip(),
            default_spec_type=self._select_value("#cfg-default_spec_type", ""),
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
            default_spec_draft_cache_type_k=self._select_value(
                "#cfg-default_spec_draft_cache_type_k", ""
            ),
            default_spec_draft_cache_type_v=self._select_value(
                "#cfg-default_spec_draft_cache_type_v", ""
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
            default_load_mode=self._select_value("#cfg-default_load_mode", "auto"),
            default_split_mode=self._select_value("#cfg-default_split_mode", "layer"),
            default_nvidia_power_limit_watts=self.query_one(
                "#cfg-default_nvidia_power_limit_watts", Input
            ).value.strip(),
            default_no_host_buffer=self.query_one("#cfg-default_no_host_buffer", Checkbox).value,
            default_ui=self.query_one("#cfg-default_ui", Checkbox).value,
            default_fit=self._select_value("#cfg-default_fit", "auto"),
            default_ctx_checkpoints=self.query_one(
                "#cfg-default_ctx_checkpoints", Input
            ).value.strip(),
            default_temperature=self.query_one("#cfg-default_temperature", Input).value.strip(),
            default_top_k=self.query_one("#cfg-default_top_k", Input).value.strip(),
            default_top_p=self.query_one("#cfg-default_top_p", Input).value.strip(),
            default_min_p=self.query_one("#cfg-default_min_p", Input).value.strip(),
            default_presence_penalty=self.query_one(
                "#cfg-default_presence_penalty", Input
            ).value.strip(),
            default_repeat_penalty=self.query_one(
                "#cfg-default_repeat_penalty", Input
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
            self._dismiss_if_valid(restart=False)
        elif event.button.id == "save-restart-config":
            self._dismiss_if_valid(restart=True)
        elif event.button.id == "clean-model-cache":
            values = self._collect_values()
            values.clean_cache = True
            self.dismiss(values)

    def _dismiss_if_valid(self, *, restart: bool) -> None:
        values = self._collect_values()
        errors = _validate_config_payload(values)
        if errors:
            for error in errors:
                self.notify(error, severity="error")
            return
        if restart:
            values.restart = True
        self.dismiss(values)
