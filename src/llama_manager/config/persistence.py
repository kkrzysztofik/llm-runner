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

from .defaults import Config

# Fields that the config modal exposes and this module persists.
_PERSISTED_FIELDS: tuple[str, ...] = (
    "llama_cpp_root",
    "models_dir",
    "llama_server_bin_intel",
    "llama_server_bin_nvidia",
    "host",
    "build_git_remote",
    "build_git_branch",
    "smoke_listen_timeout_s",
    "smoke_http_request_timeout_s",
    "smoke_first_token_timeout_s",
    "smoke_total_chat_timeout_s",
    "log_file_level",
    "log_stderr_level",
    "default_profile_port",
    "default_profile_ctx_size",
    "default_profile_ubatch_size",
    "default_profile_threads",
    "default_profile_n_gpu_layers",
    "default_bind_address",
    "default_batch_size",
    "default_poll_ms",
    "default_n_predict",
    "default_parallel",
    "default_threads_batch",
    "default_profile_cache_type_k",
    "default_profile_cache_type_v",
    "default_reasoning_mode",
    "default_reasoning_format",
    "default_reasoning_budget",
    "default_use_jinja",
    "default_profile_chat_template_kwargs",
    "default_mmproj",
    "default_spec_type",
    "default_spec_ngram_size_n",
    "default_draft_min",
    "default_draft_max",
    "default_spec_draft_n_max",
    "default_spec_draft_p_min",
    "default_spec_draft_cache_type_k",
    "default_spec_draft_cache_type_v",
    "default_spec_draft_device",
)

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
    """Parse *path* as TOML and return only the recognised config keys.

    Returns an empty dict when the file does not exist or is empty.

    Args:
        path: Path to the TOML config file.

    Returns:
        Dict of field name → value for keys found in the file that match
        ``_PERSISTED_FIELDS``.
    """
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return {k: v for k, v in raw.items() if k in _PERSISTED_FIELDS}


def save_config_to_file(config: Config, path: Path) -> None:
    """Write the modal-exposed fields from *config* to *path* as TOML.

    Creates parent directories as needed.

    Args:
        config: The Config instance whose values to persist.
        path: Destination file path (will be overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {field: getattr(config, field) for field in _PERSISTED_FIELDS}
    lines = [f"{field} = {_toml_value(data[field])}" for field in _PERSISTED_FIELDS]
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
    env_overrides: dict[str, Any] = {}
    for field_name, env_var in _ENV_OVERRIDES.items():
        if env_var in os.environ:
            env_overrides[field_name] = os.environ[env_var]

    kwargs: dict[str, Any] = {**file_overrides, **env_overrides}
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
        "smoke_listen_timeout_s",
        "smoke_http_request_timeout_s",
        "smoke_first_token_timeout_s",
        "smoke_total_chat_timeout_s",
        "default_profile_port",
        "default_profile_ctx_size",
        "default_profile_ubatch_size",
        "default_profile_threads",
        "default_batch_size",
        "default_poll_ms",
        "default_n_predict",
        "default_parallel",
        "default_threads_batch",
        "default_spec_ngram_size_n",
        "default_draft_min",
        "default_draft_max",
        "default_spec_draft_n_max",
    }
)

_FLOAT_FIELDS: frozenset[str] = frozenset({"default_spec_draft_p_min"})

_BOOL_FIELDS: frozenset[str] = frozenset({"default_use_jinja"})


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
            return None, f"Invalid value '{raw_value}' for {field_name} — config not saved."
    if field_name in _FLOAT_FIELDS:
        try:
            return float(raw_value), None  # type: ignore[arg-type]
        except ValueError, TypeError:
            return None, f"Invalid value '{raw_value}' for {field_name} — config not saved."
    if field_name in _BOOL_FIELDS:
        if isinstance(raw_value, bool):
            return raw_value, None
        if isinstance(raw_value, int):
            if raw_value == 1:
                return True, None
            if raw_value == 0:
                return False, None
            return None, f"Invalid value '{raw_value}' for {field_name} — config not saved."
        if isinstance(raw_value, str):
            token = raw_value.strip().lower()
            if token in ("1", "true", "yes", "on"):
                return True, None
            if token in ("0", "false", "no", "off"):
                return False, None
            return None, f"Invalid value '{raw_value}' for {field_name} — config not saved."
        return None, f"Invalid value '{raw_value}' for {field_name} — config not saved."
    return raw_value, None


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

    config_fields = {f.name for f in dataclasses.fields(config)}

    for field_name, raw_value in updates.items():
        # Skip unknown fields silently
        if field_name not in config_fields:
            continue

        value, error = _coerce_config_field_value(field_name, raw_value)
        if error is not None:
            errors.append(error)
            continue

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
