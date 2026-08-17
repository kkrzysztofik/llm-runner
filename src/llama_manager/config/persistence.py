"""Config file persistence — read/write $XDG_CONFIG_HOME/llm-runner/config.toml.

Load order: config file is the baseline; env vars always win.
Only the modal-exposed fields are written/read — the full Config dataclass
handles all remaining fields via its own defaults and env-var factories.
"""

import dataclasses
import os
import tomllib
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from .defaults import (
    BuildPipelineConfig,
    Config,
    DeploymentConfig,
    PathsConfig,
    ServerDefaultsConfig,
    SmokeConfig,
)
from .load_mode import resolve_load_mode
from .reasoning_effort import resolve_reasoning_effort
from .tri_state import resolve_fit, resolve_reasoning_preserve

_SECTION_CLASSES: dict[str, type] = {
    "paths": PathsConfig,
    "deployment": DeploymentConfig,
    "build": BuildPipelineConfig,
    "smoke": SmokeConfig,
    "server_defaults": ServerDefaultsConfig,
}


def _section_field_names() -> dict[str, set[str]]:
    return {section: {f.name for f in fields(cls)} for section, cls in _SECTION_CLASSES.items()}


def _field_type(cls: type, name: str):
    return get_type_hints(cls)[name]


def _is_nullable(t) -> bool:
    origin = get_origin(t)
    if origin is Union or origin is UnionType:
        return NoneType in get_args(t)
    return False


def _coercion_kind(t) -> str | None:
    if _is_nullable(t):
        return _coercion_kind(next(a for a in get_args(t) if a is not NoneType))
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is bool:
        return "bool"
    return None


_TOP_LEVEL_FIELDS: tuple[str, ...] = ("log_file_level", "log_stderr_level")
_LEGACY_SERVER_DEFAULTS_KEYS: tuple[str, ...] = ("mmap", "mlock")
_UPDATE_FIELDS: frozenset[str] = frozenset(
    f"{section}.{field}" for section, fields in _section_field_names().items() for field in fields
) | frozenset(_TOP_LEVEL_FIELDS)

_COERCION_FIELDS: dict[str, set[str]] = {"int": set(), "float": set(), "bool": set()}
_NULLABLE_FIELDS: set[str] = set()
for _section, _cls in _SECTION_CLASSES.items():
    for _f in fields(_cls):
        _t = _field_type(_cls, _f.name)
        if _is_nullable(_t):
            _NULLABLE_FIELDS.add(f"{_section}.{_f.name}")
        _kind = _coercion_kind(_t)
        if _kind:
            _COERCION_FIELDS[_kind].add(f"{_section}.{_f.name}")

# Fields that have a corresponding env var. When building a Config from the
# file, these env vars override whatever is in the file.
_ENV_OVERRIDES: dict[str, str] = {
    "llama_cpp_root": "LLAMA_CPP_ROOT",
    "models_dir": "MODELS_DIR",
    "xdg_cache_base": "XDG_CACHE_HOME",
    "xdg_state_base": "XDG_STATE_HOME",
    "xdg_data_base": "XDG_DATA_HOME",
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
    for section, section_fields in _section_field_names().items():
        section_data = raw.get(section)
        if isinstance(section_data, dict):
            values = {
                field: section_data[field] for field in section_fields if field in section_data
            }
            if section == "server_defaults":
                for legacy_key in _LEGACY_SERVER_DEFAULTS_KEYS:
                    if legacy_key in section_data:
                        values[legacy_key] = section_data[legacy_key]
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
    for section, _cls in _SECTION_CLASSES.items():
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in dataclasses.asdict(getattr(config, section)).items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
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
    if "server_defaults" in kwargs:
        kwargs["server_defaults"] = _normalize_server_defaults_section(kwargs["server_defaults"])
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


def _normalize_server_defaults_section(section_data: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy keys and tri-state fields when loading server_defaults."""
    normalized = dict(section_data)
    normalized["load_mode"] = resolve_load_mode(section_data)
    normalized.pop("mmap", None)
    normalized.pop("mlock", None)
    normalized["reasoning_preserve"] = resolve_reasoning_preserve(section_data)
    normalized["reasoning_effort"] = resolve_reasoning_effort(section_data)
    normalized["fit"] = resolve_fit(section_data)
    return normalized


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


_BOOL_TRUE_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off"})


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
    if raw_value is None and field_name in _NULLABLE_FIELDS:
        return None, None
    if field_name in _COERCION_FIELDS["int"]:
        try:
            return int(raw_value), None  # type: ignore[arg-type]
        except ValueError, TypeError:
            return None, _invalid_value_message(field_name, raw_value)
    if field_name in _COERCION_FIELDS["float"]:
        try:
            return float(raw_value), None  # type: ignore[arg-type]
        except ValueError, TypeError:
            return None, _invalid_value_message(field_name, raw_value)
    if field_name in _COERCION_FIELDS["bool"]:
        return _coerce_bool_field_value(field_name, raw_value)
    return raw_value, None


def _coerce_bool_field_value(field_name: str, raw_value: object) -> tuple[bool | None, str | None]:
    """Coerce *raw_value* into a bool for boolean-typed config fields."""
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
