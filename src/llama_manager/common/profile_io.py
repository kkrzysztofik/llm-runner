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
    profiles = data.get("profiles", [])

    if hidden_builtins:
        hidden_list = sorted(hidden_builtins)
        lines.append(f"hidden_builtin_profiles = {json.dumps(hidden_list)}")
        lines.append("")

    for index, profile in enumerate(profiles):
        if index > 0:
            lines.append("")
        lines.extend(_profile_lines(profile))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def profile_dir_path(base_dir: Path, profile_id: str) -> Path:
    """Return the directory path for a profile under a base directory."""
    return base_dir / profile_id


def _profile_lines(profile: dict[str, Any]) -> list[str]:
    lines = [
        "[[profiles]]",
        f"profile_id = {json.dumps(profile['profile_id'])}",
        f"alias = {json.dumps(profile.get('alias', ''))}",
        f"device = {json.dumps(profile.get('device', ''))}",
        f"model = {json.dumps(profile['model'])}",
        f"port = {profile['port']}",
        f"ctx_size = {profile['ctx_size']}",
        f"ubatch_size = {profile['ubatch_size']}",
        f"threads = {profile['threads']}",
        f"description = {json.dumps(profile.get('description', ''))}",
        f"bind_address = {json.dumps(profile.get('bind_address', '127.0.0.1'))}",
        f"tensor_split = {json.dumps(profile.get('tensor_split', ''))}",
        f"reasoning_mode = {json.dumps(profile.get('reasoning_mode', 'auto'))}",
        f"reasoning_format = {json.dumps(profile.get('reasoning_format', 'none'))}",
    ]
    chat_template_kwargs = profile.get("chat_template_kwargs", "")
    if isinstance(chat_template_kwargs, dict):
        lines.append(f"chat_template_kwargs = {json.dumps(json.dumps(chat_template_kwargs))}")
    else:
        lines.append(f"chat_template_kwargs = {json.dumps(str(chat_template_kwargs))}")
    lines.extend(
        [
            f"reasoning_budget = {json.dumps(profile.get('reasoning_budget', ''))}",
            f"use_jinja = {str(profile.get('use_jinja', False)).lower()}",
            f"cache_type_k = {json.dumps(profile.get('cache_type_k', 'q8_0'))}",
            f"cache_type_v = {json.dumps(profile.get('cache_type_v', 'q8_0'))}",
        ]
    )
    n_gpu_layers = profile.get("n_gpu_layers", 99)
    if isinstance(n_gpu_layers, str):
        lines.append(f"n_gpu_layers = {json.dumps(n_gpu_layers)}")
    else:
        lines.append(f"n_gpu_layers = {int(n_gpu_layers)}")
    lines.extend(
        [
            f"main_gpu = {int(profile.get('main_gpu', 0))}",
            f"server_bin = {json.dumps(profile.get('server_bin', ''))}",
            f"backend = {json.dumps(profile.get('backend', 'llama_cpp'))}",
        ]
    )
    risky_acknowledged = profile.get("risky_acknowledged", [])
    if risky_acknowledged:
        lines.append(f"risky_acknowledged = {json.dumps(risky_acknowledged)}")
    lines.extend(
        [
            f"batch_size = {int(profile.get('batch_size', 2048))}",
            f"poll_ms = {int(profile.get('poll_ms', 50))}",
            f"n_predict = {int(profile.get('n_predict', 32768))}",
            f"parallel = {int(profile.get('parallel', 4))}",
            f"threads_batch = {int(profile.get('threads_batch', 0))}",
            f"mmproj = {json.dumps(profile.get('mmproj', ''))}",
            f"spec_type = {json.dumps(profile.get('spec_type', ''))}",
            f"spec_ngram_size_n = {int(profile.get('spec_ngram_size_n', 0))}",
            f"draft_min = {int(profile.get('draft_min', 0))}",
            f"draft_max = {int(profile.get('draft_max', 0))}",
            f"spec_draft_n_max = {int(profile.get('spec_draft_n_max', 0))}",
            f"spec_draft_p_min = {float(profile.get('spec_draft_p_min', 0.0))}",
            f"spec_draft_cache_type_k = {json.dumps(profile.get('spec_draft_cache_type_k', ''))}",
            f"spec_draft_cache_type_v = {json.dumps(profile.get('spec_draft_cache_type_v', ''))}",
            f"spec_draft_device = {json.dumps(profile.get('spec_draft_device', ''))}",
        ]
    )
    return lines
