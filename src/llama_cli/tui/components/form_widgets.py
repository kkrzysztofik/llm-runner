"""Shared form row builders for Config and Run Profile modals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Checkbox, Collapsible, Input, Label, Select

if TYPE_CHECKING:
    from llama_manager.config import Config

MODAL_CANCEL_BINDINGS: list[Binding] = [
    Binding("escape", "cancel", "Cancel"),
    Binding("ctrl+c", "cancel", "Cancel"),
]

CACHE_TYPE_CHOICES: tuple[tuple[str, str], ...] = (
    ("q8_0", "q8_0"),
    ("f16", "f16"),
    ("q4_0", "q4_0"),
    ("q4_1", "q4_1"),
    ("q5_0", "q5_0"),
    ("q5_1", "q5_1"),
)

REASONING_MODE_CHOICES: tuple[tuple[str, str], ...] = (
    ("auto", "auto"),
    ("off", "off"),
    ("on", "on"),
)

REASONING_FORMAT_CHOICES: tuple[tuple[str, str], ...] = (
    ("none", "none"),
    ("deepseek", "deepseek"),
)

SPEC_TYPE_CHOICES: tuple[tuple[str, str], ...] = (
    ("(none)", ""),
    ("ngram-mod", "ngram-mod"),
    ("draft-mtp", "draft-mtp"),
    ("dflash", "dflash"),
)

DEFAULT_PARALLEL_CHOICES: tuple[tuple[str, str], ...] = (
    ("1", "1"),
    ("4", "4"),
    ("-1 (unlimited)", "-1"),
)

LABEL_CLASSES = "form-label profile-field-label"
INPUT_CLASSES = "form-input profile-input"
SELECT_CLASSES = "form-input profile-select"
CONFIG_SELECT_CLASSES = "form-input config-select"
ROW_CLASSES = "form-row profile-row"
ROW_SELECT_CLASSES = "form-row profile-row profile-row-select"
CONFIG_ROW_SELECT_CLASSES = "form-row config-row config-row-select"


def field_row(
    label: str,
    field_id: str,
    value: str = "",
    *,
    id_prefix: str = "profile",
    type: Literal["text", "number", "integer"] = "text",
    label_classes: str = LABEL_CLASSES,
    input_classes: str = INPUT_CLASSES,
    row_classes: str = ROW_CLASSES,
) -> Horizontal:
    """Build a labelled input row."""
    return Horizontal(
        Label(f"{label}:", classes=label_classes),
        Input(
            value=value,
            id=f"{id_prefix}-{field_id}",
            type=type,
            classes=input_classes,
        ),
        classes=row_classes,
    )


def select_row(
    label: str,
    field_id: str,
    choices: tuple[tuple[str, str], ...],
    value: str,
    *,
    id_prefix: str = "profile",
    allow_blank: bool = False,
    label_classes: str = LABEL_CLASSES,
    input_classes: str = SELECT_CLASSES,
    row_classes: str = ROW_SELECT_CLASSES,
) -> Horizontal:
    """Build a labelled Select row."""
    return Horizontal(
        Label(f"{label}:", classes=label_classes),
        Select(
            choices,
            value=value,
            allow_blank=allow_blank,
            id=f"{id_prefix}-{field_id}",
            classes=input_classes,
        ),
        classes=row_classes,
    )


def cache_type_row(
    label: str,
    field_id: str,
    value: str,
    *,
    id_prefix: str = "profile",
    allow_empty: bool = False,
    label_classes: str = LABEL_CLASSES,
    input_classes: str = SELECT_CLASSES,
    row_classes: str = ROW_SELECT_CLASSES,
) -> Horizontal:
    """Build a KV cache type Select row."""
    choices = CACHE_TYPE_CHOICES
    if allow_empty:
        choices = (("(omit)", ""),) + choices
    return select_row(
        label,
        field_id,
        choices,
        value or ("" if allow_empty else "q8_0"),
        id_prefix=id_prefix,
        allow_blank=False,
        label_classes=label_classes,
        input_classes=input_classes,
        row_classes=row_classes,
    )


def config_profile_prefill(config: Config) -> dict[str, str]:
    """Build run-profile form prefill values from global Config defaults."""
    defaults = config.server_defaults
    spec = defaults.spec_decode
    return {
        "port": str(defaults.port),
        "ctx-size": str(defaults.ctx_size),
        "ubatch-size": str(defaults.ubatch_size),
        "threads": str(defaults.threads),
        "n-gpu-layers": str(defaults.n_gpu_layers_profile),
        "bind-address": defaults.bind_address,
        "batch-size": str(defaults.batch_size),
        "poll-ms": str(defaults.poll_ms),
        "n-predict": str(defaults.n_predict),
        "parallel": str(defaults.parallel),
        "threads-batch": str(defaults.threads_batch),
        "cache-type-k": defaults.cache_type_k,
        "cache-type-v": defaults.cache_type_v,
        "reasoning-mode": spec.reasoning_mode,
        "reasoning-format": spec.reasoning_format,
        "reasoning-budget": spec.reasoning_budget,
        "use-jinja": "true" if defaults.use_jinja else "false",
        "chat-template-kwargs": defaults.chat_template_kwargs,
        "mmproj": defaults.mmproj,
        "spec-type": spec.spec_type,
        "spec-ngram-size-n": str(spec.spec_ngram_size_n),
        "draft-min": str(spec.draft_min),
        "draft-max": str(spec.draft_max),
        "spec-draft-n-max": str(spec.spec_draft_n_max),
        "spec-draft-p-min": str(spec.spec_draft_p_min),
        "spec-draft-cache-type-k": spec.spec_draft_cache_type_k,
        "spec-draft-cache-type-v": spec.spec_draft_cache_type_v,
        "spec-draft-device": spec.spec_draft_device,
        "spec-draft-model": spec.spec_draft_model,
        "spec-draft-hf": spec.spec_draft_hf,
        "spec-draft-ngl": str(spec.spec_draft_ngl),
        "spec-dflash-cross-ctx": str(spec.spec_dflash_cross_ctx),
        "kv-unified": "true" if defaults.kv_unified else "false",
        "mmproj-offload": "true" if defaults.mmproj_offload else "false",
        "mmap": "true" if defaults.mmap else "false",
        "mlock": "true" if defaults.mlock else "false",
        "no-host-buffer": "true" if defaults.no_host_buffer else "false",
    }


def build_config_profile_defaults_collapsible(config: Config) -> Collapsible:
    """Profile/server default fields for the global Config modal."""
    cfg_label = "form-label config-field-label"
    cfg_input = "form-input config-input"
    cfg_select = CONFIG_SELECT_CLASSES
    cfg_row = "form-row config-row"
    cfg_row_select = CONFIG_ROW_SELECT_CLASSES
    prefix = "cfg"
    defaults = config.server_defaults
    spec = defaults.spec_decode

    speculative = Collapsible(
        field_row(
            "Ngram Size N",
            "default_spec_ngram_size_n",
            str(spec.spec_ngram_size_n),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Draft Min",
            "default_draft_min",
            str(spec.draft_min),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Draft Max",
            "default_draft_max",
            str(spec.draft_max),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Draft N Max (MTP)",
            "default_spec_draft_n_max",
            str(spec.spec_draft_n_max),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Draft P Min (MTP)",
            "default_spec_draft_p_min",
            str(spec.spec_draft_p_min),
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        cache_type_row(
            "Draft Cache K",
            "default_spec_draft_cache_type_k",
            spec.spec_draft_cache_type_k,
            id_prefix=prefix,
            allow_empty=True,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        cache_type_row(
            "Draft Cache V",
            "default_spec_draft_cache_type_v",
            spec.spec_draft_cache_type_v,
            id_prefix=prefix,
            allow_empty=True,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        field_row(
            "Draft Device",
            "default_spec_draft_device",
            spec.spec_draft_device,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Draft Model (DFlash)",
            "default_spec_draft_model",
            spec.spec_draft_model,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Draft HF Repo (DFlash)",
            "default_spec_draft_hf",
            spec.spec_draft_hf,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Draft GPU Layers (DFlash)",
            "default_spec_draft_ngl",
            str(spec.spec_draft_ngl),
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "DFlash Cross Ctx",
            "default_spec_dflash_cross_ctx",
            str(spec.spec_dflash_cross_ctx),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        title="Speculative defaults",
        collapsed=True,
        classes="config-advanced-options",
    )

    return Collapsible(
        field_row(
            "Default port",
            "default_profile_port",
            str(defaults.port),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Default ctx size",
            "default_profile_ctx_size",
            str(defaults.ctx_size),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Default ubatch",
            "default_profile_ubatch_size",
            str(defaults.ubatch_size),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Default threads",
            "default_profile_threads",
            str(defaults.threads),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Default GPU layers",
            "default_profile_n_gpu_layers",
            str(defaults.n_gpu_layers_profile),
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Bind address",
            "default_bind_address",
            defaults.bind_address,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Batch size",
            "default_batch_size",
            str(defaults.batch_size),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "Poll (ms)",
            "default_poll_ms",
            str(defaults.poll_ms),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "N predict",
            "default_n_predict",
            str(defaults.n_predict),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        select_row(
            "Default parallel",
            "default_parallel",
            DEFAULT_PARALLEL_CHOICES,
            str(defaults.parallel),
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        field_row(
            "Threads batch",
            "default_threads_batch",
            str(defaults.threads_batch),
            id_prefix=prefix,
            type="number",
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        cache_type_row(
            "Cache K",
            "default_profile_cache_type_k",
            defaults.cache_type_k,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        cache_type_row(
            "Cache V",
            "default_profile_cache_type_v",
            defaults.cache_type_v,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        select_row(
            "Reasoning mode",
            "default_reasoning_mode",
            REASONING_MODE_CHOICES,
            spec.reasoning_mode,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        select_row(
            "Reasoning format",
            "default_reasoning_format",
            REASONING_FORMAT_CHOICES,
            spec.reasoning_format,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        field_row(
            "Reasoning budget",
            "default_reasoning_budget",
            spec.reasoning_budget,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        checkbox_row(
            "Use Jinja",
            "default_use_jinja",
            defaults.use_jinja,
            id_prefix=prefix,
            label_classes=cfg_label,
            row_classes=cfg_row,
        ),
        field_row(
            "Chat template kwargs",
            "default_profile_chat_template_kwargs",
            defaults.chat_template_kwargs,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        field_row(
            "MMProj",
            "default_mmproj",
            defaults.mmproj,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_input,
            row_classes=cfg_row,
        ),
        select_row(
            "Spec type",
            "default_spec_type",
            SPEC_TYPE_CHOICES,
            spec.spec_type,
            id_prefix=prefix,
            label_classes=cfg_label,
            input_classes=cfg_select,
            row_classes=cfg_row_select,
        ),
        speculative,
        checkbox_row(
            "Unified KV",
            "default_kv_unified",
            defaults.kv_unified,
            id_prefix=prefix,
            label_classes=cfg_label,
            row_classes=cfg_row,
        ),
        checkbox_row(
            "MMProj Offload",
            "default_mmproj_offload",
            defaults.mmproj_offload,
            id_prefix=prefix,
            label_classes=cfg_label,
            row_classes=cfg_row,
        ),
        checkbox_row(
            "MMap",
            "default_mmap",
            defaults.mmap,
            id_prefix=prefix,
            label_classes=cfg_label,
            row_classes=cfg_row,
        ),
        checkbox_row(
            "MLock",
            "default_mlock",
            defaults.mlock,
            id_prefix=prefix,
            label_classes=cfg_label,
            row_classes=cfg_row,
        ),
        checkbox_row(
            "No Host Buffer",
            "default_no_host_buffer",
            defaults.no_host_buffer,
            id_prefix=prefix,
            label_classes=cfg_label,
            row_classes=cfg_row,
        ),
        title="Profile / server defaults",
        collapsed=True,
        classes="config-advanced-options",
    )


def checkbox_row(
    label: str,
    field_id: str,
    value: bool,
    *,
    id_prefix: str = "profile",
    label_classes: str = LABEL_CLASSES,
    row_classes: str = ROW_CLASSES,
) -> Horizontal:
    """Build a labelled checkbox row."""
    return Horizontal(
        Label(f"{label}:", classes=label_classes),
        Checkbox("Enabled", value=value, id=f"{id_prefix}-{field_id}"),
        classes=row_classes,
    )
