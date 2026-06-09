"""Config file persistence — read/write $XDG_CONFIG_HOME/llm-runner/config.toml.

Load order: config file is the baseline; env vars always win.
Only the modal-exposed fields are written/read — the full Config dataclass
handles all remaining fields via its own defaults and env-var factories.
"""

import dataclasses
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .defaults import (
    BuildPipelineConfig,
    Config,
    DeploymentConfig,
    PathsConfig,
    ServerDefaultsConfig,
    SmokeConfig,
)

_PERSISTED_SECTIONS: dict[str, tuple[str, ...]] = {
    "paths": (
        "llama_cpp_root",
        "models_dir",
        "llama_server_bin_intel",
        "llama_server_bin_nvidia",
    ),
    "deployment": (
        "host",
        "model_summary_balanced",
        "model_summary_fast",
        "model_qwen35",
        "model_qwen35_both",
        "summary_balanced_port",
        "summary_fast_port",
        "qwen35_port",
        "summary_balanced_chat_template_kwargs",
        "summary_fast_chat_template_kwargs",
    ),
    "build": ("git_remote", "git_branch"),
    "smoke": (
        "listen_timeout_s",
        "http_request_timeout_s",
        "first_token_timeout_s",
        "total_chat_timeout_s",
    ),
    "server_defaults": (
        "port",
        "ctx_size",
        "ubatch_size",
        "threads",
        "n_gpu_layers_profile",
        "bind_address",
        "batch_size",
        "poll_ms",
        "n_predict",
        "parallel",
        "threads_batch",
        "cache_type_k",
        "cache_type_v",
        "reasoning_mode",
        "reasoning_format",
        "reasoning_budget",
        "use_jinja",
        "chat_template_kwargs",
        "mmproj",
        "spec_type",
        "spec_ngram_size_n",
        "draft_min",
        "draft_max",
        "spec_draft_n_max",
        "spec_draft_p_min",
        "spec_draft_cache_type_k",
        "spec_draft_cache_type_v",
        "spec_draft_device",
    ),
}

_TOP_LEVEL_FIELDS: tuple[str, ...] = ("log_file_level", "log_stderr_level")
_UPDATE_FIELDS: frozenset[str] = frozenset(
    f"{section}.{field}" for section, fields in _PERSISTED_SECTIONS.items() for field in fields
) | frozenset(_TOP_LEVEL_FIELDS)

# Fields that have a corresponding env var. When building a Config from the
# file, these env vars override whatever is in the file.
_ENV_OVERRIDES: dict[str, str] = {
    "llama_cpp_root": "LLAMA_CPP_ROOT",
    "models_dir": "MODELS_DIR",
}


def config_file_path() -> Path:
    """Return the canonical config file path.

    Returns:
        ``$XDG_CONFIG_HOME/llm-runner/config.toml`` when XDG_CONFIG_HOME is
        set, otherwise ``~/.config/llm-runner/config.toml``.
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg_config) / "llm-runner" / "config.toml"


def load_config_overrides_from_file(path: Path) -> dict[str, Any]:
    """Parse *path* as TOML and return recognised nested config sections.

    Returns an empty dict when the file does not exist or is empty.

    Args:
        path: Path to the TOML config file.

    Returns:
        Dict of Config constructor kwargs for recognised nested sections.
    """
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    overrides: dict[str, Any] = {}
    for section, fields in _PERSISTED_SECTIONS.items():
        section_data = raw.get(section)
        if isinstance(section_data, dict):
            values = {field: section_data[field] for field in fields if field in section_data}
            if values:
                overrides[section] = values
    for field in _TOP_LEVEL_FIELDS:
        if field in raw:
            overrides[field] = raw[field]
    return overrides


def save_config_to_file(config: Config, path: Path) -> None:
    """Write the modal-exposed fields from *config* to *path* as TOML.

    Creates parent directories as needed.

    Args:
        config: The Config instance whose values to persist.
        path: Destination file path (will be overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for field in _TOP_LEVEL_FIELDS:
        lines.append(f"{field} = {_toml_value(getattr(config, field))}")
    for section, fields in _PERSISTED_SECTIONS.items():
        lines.append("")
        lines.append(f"[{section}]")
        section_config = getattr(config, section)
        for field in fields:
            lines.append(f"{field} = {_toml_value(getattr(section_config, field))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_config() -> Config:
    """Construct a Config honoring both the config file and env vars.

    Resolution order (highest priority last wins):
    1. Config dataclass hard-coded defaults
    2. Values from the config file (``config_file_path()``)
    3. Env vars listed in ``_ENV_OVERRIDES`` (env always wins)

    Returns:
        A fully initialised Config instance.
    """
    file_overrides = load_config_overrides_from_file(config_file_path())

    # Env vars explicitly listed in _ENV_OVERRIDES override file values.
    paths_overrides = dict(file_overrides.get("paths", {}))
    for field_name, env_var in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            paths_overrides[field_name] = os.environ[env_var]
    if paths_overrides:
        file_overrides["paths"] = paths_overrides

    kwargs = dict(file_overrides)
    for section, cls in (
        ("paths", PathsConfig),
        ("build", BuildPipelineConfig),
        ("smoke", SmokeConfig),
        ("server_defaults", ServerDefaultsConfig),
        ("deployment", DeploymentConfig),
    ):
        if section in kwargs:
            kwargs[section] = cls(**kwargs[section])

    return Config(**kwargs)


def _toml_value(value: Any) -> str:
    """Serialize a scalar config value to a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        escaped = escaped.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        escaped = escaped.replace("\b", "\\b").replace("\f", "\\f")
        escaped = "".join(c if c.isprintable() else f"\\u{ord(c):04X}" for c in escaped)
        return f'"{escaped}"'
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


# Fields that require integer coercion from string input.
_INT_FIELDS: frozenset[str] = frozenset(
    {
        "smoke.listen_timeout_s",
        "smoke.http_request_timeout_s",
        "smoke.first_token_timeout_s",
        "smoke.total_chat_timeout_s",
        "server_defaults.port",
        "server_defaults.ctx_size",
        "server_defaults.ubatch_size",
        "server_defaults.threads",
        "server_defaults.batch_size",
        "server_defaults.poll_ms",
        "server_defaults.n_predict",
        "server_defaults.parallel",
        "server_defaults.threads_batch",
        "server_defaults.spec_ngram_size_n",
        "server_defaults.draft_min",
        "server_defaults.draft_max",
        "server_defaults.spec_draft_n_max",
        "server_defaults.spec_dflash_cross_ctx",
        "deployment.summary_balanced_port",
        "deployment.summary_fast_port",
        "deployment.qwen35_port",
    }
)

_FLOAT_FIELDS: frozenset[str] = frozenset({"server_defaults.spec_draft_p_min"})

_BOOL_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off"})

_BOOL_FIELDS: frozenset[str] = frozenset(
    {
        "server_defaults.use_jinja",
        "server_defaults.kv_unified",
        "server_defaults.mmproj_offload",
        "server_defaults.mmap",
        "server_defaults.mlock",
        "server_defaults.no_host_buffer",
    }
)


@dataclasses.dataclass
class ConfigUpdateResult:
    """Result of applying config updates."""

    success: bool
    updated_fields: list[str]
    errors: list[str]


def _coerce_config_field_value(
    field_name: str,
    raw_value: object,
) -> tuple[object | None, str | None]:
    """Return (coerced_value, error). value is None when coercion fails."""
    if field_name in _INT_FIELDS:
        try:
            return int(raw_value), None  # type: ignore[arg-type]
        except ValueError, TypeError:
            return None, _invalid_value_message(field_name, raw_value)
    if field_name in _FLOAT_FIELDS:
        try:
            return float(raw_value), None  # type: ignore[arg-type]
        except ValueError, TypeError:
            return None, _invalid_value_message(field_name, raw_value)
    if field_name in _BOOL_FIELDS:
        return _coerce_bool_field_value(field_name, raw_value)
    return raw_value, None


def _coerce_bool_field_value(field_name: str, raw_value: object) -> tuple[bool | None, str | None]:
    """Coerce *raw_value* into a bool for fields in ``_BOOL_FIELDS``."""
    if isinstance(raw_value, bool):
        return raw_value, None
    if isinstance(raw_value, int):
        if raw_value in (0, 1):
            return bool(raw_value), None
        return None, _invalid_value_message(field_name, raw_value)
    if isinstance(raw_value, str):
        token = raw_value.strip().lower()
        if token in _BOOL_TRUE_TOKENS:
            return True, None
        if token in _BOOL_FALSE_TOKENS:
            return False, None
    return None, _invalid_value_message(field_name, raw_value)


def _invalid_value_message(field_name: str, raw_value: object) -> str:
    return f"Invalid value '{raw_value}' for {field_name} — config not saved."


def apply_config_updates(
    config: Config,
    updates: Mapping[str, object],
    *,
    persist: bool = True,
) -> ConfigUpdateResult:
    """Apply configuration updates to a Config instance.

    Validates known fields against the Config dataclass, coerces
    integer fields from string input, and optionally persists to disk.

    Args:
        config: The Config instance to update.
        updates: Mapping of field name → value.
        persist: If True, write changes to the config file.

    Returns:
        ConfigUpdateResult with success status, updated fields, and errors.
    """
    updated_fields: list[str] = []
    errors: list[str] = []

    for field_name, raw_value in updates.items():
        # Skip unknown fields silently
        if field_name not in _UPDATE_FIELDS:
            continue

        value, error = _coerce_config_field_value(field_name, raw_value)
        if error is not None:
            errors.append(error)
            continue

        if "." in field_name:
            section, attr = field_name.split(".", 1)
            setattr(getattr(config, section), attr, value)
        else:
            setattr(config, field_name, value)
        updated_fields.append(field_name)

    # Persist if requested and no errors
    if persist and updated_fields and not errors:
        try:
            save_config_to_file(config, config_file_path())
        except OSError as exc:
            errors.append(f"Config save failed: {exc}")

    return ConfigUpdateResult(
        success=len(errors) == 0,
        updated_fields=updated_fields,
        errors=errors,
    )
