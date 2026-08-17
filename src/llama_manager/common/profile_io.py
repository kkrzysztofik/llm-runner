"""Shared TOML profile persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_profile_toml(path: Path) -> dict[str, Any]:
    """Read a profile TOML file, returning an empty dict when unavailable."""
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as file_obj:
            import tomllib

            return tomllib.load(file_obj)
    except Exception:  # noqa: BLE001
        return {}


def write_profile_toml(path: Path, data: dict[str, Any]) -> None:
    """Write the shared profile TOML shape used by slot/run profile stores."""
    lines: list[str] = []
    hidden_builtins = set(data.get("hidden_builtin_profiles", []))
    if hidden_builtins:
        lines.append(f"hidden_builtin_profiles = {json.dumps(sorted(hidden_builtins))}")
        lines.append("")
    for index, profile in enumerate(data.get("profiles", [])):
        if index > 0:
            lines.append("")
        lines.append("[[profiles]]")
        for key, value in profile.items():
            if value is None:
                continue
            if isinstance(value, bool):
                lines.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, (int, float, str, list)):
                lines.append(f"{key} = {json.dumps(value)}")
            elif isinstance(value, dict):
                lines.append(f"{key} = {json.dumps(json.dumps(value))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")
