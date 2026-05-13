"""Config file persistence — read/write $XDG_CONFIG_HOME/llm-runner/config.toml.

Load order: config file is the baseline; env vars always win.
Only the modal-exposed fields are written/read — the full Config dataclass
handles all remaining fields via its own defaults and env-var factories.
"""

from __future__ import annotations

import os
import tomllib
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
