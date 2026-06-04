"""Persistent store for custom slot profiles saved to XDG config TOML."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from .config.profiles import SlotProfileSpec

logger = logging.getLogger(__name__)


def slot_profiles_file_path() -> Path:
    """Return the path to the custom slot profiles TOML file."""
    xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return xdg_config / "llm-runner" / "slot_profiles.toml"


def load_custom_slot_profiles() -> list[SlotProfileSpec]:
    """Load custom slot profiles from disk."""
    profiles_dicts, _ = _load_toml_data(slot_profiles_file_path())
    result: list[SlotProfileSpec] = []
    for p in profiles_dicts:
        try:
            result.append(_profile_from_dict(p))
        except Exception:  # noqa: BLE001, S112
            continue
    return result


def save_custom_slot_profile(profile: SlotProfileSpec) -> None:
    """Save a single custom slot profile to disk."""
    path = slot_profiles_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_dicts, hidden_builtins = _load_toml_data(path)
    ids = {p["profile_id"] for p in existing_dicts}
    if profile.profile_id in ids:
        raise ValueError(f"Duplicate profile_id: {profile.profile_id}")

    profiles_dict = [*existing_dicts, _profile_to_dict(profile)]
    _write_toml_data(profiles_dict, hidden_builtins, path)


def upsert_custom_slot_profile(original_profile_id: str, profile: SlotProfileSpec) -> None:
    """Upsert a custom slot profile."""
    path = slot_profiles_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_dicts, hidden_builtins = _load_toml_data(path)
    ids_to_check = {
        p["profile_id"] for p in existing_dicts if p["profile_id"] != original_profile_id
    }
    if profile.profile_id in ids_to_check:
        raise ValueError(f"Duplicate profile_id: {profile.profile_id}")

    filtered_dicts = [p for p in existing_dicts if p["profile_id"] != original_profile_id]
    filtered_dicts.append(_profile_to_dict(profile))
    _write_toml_data(filtered_dicts, hidden_builtins, path)


def delete_custom_slot_profile(
    profile_id: str, builtin_profile_ids: set[str] | None = None
) -> bool:
    """Delete or hide a slot profile."""
    path = slot_profiles_file_path()

    if path.exists():
        existing_dicts, hidden_builtins = _load_toml_data(path)
        custom_entries = [p for p in existing_dicts if p["profile_id"] == profile_id]
        if custom_entries:
            filtered_dicts = [p for p in existing_dicts if p["profile_id"] != profile_id]
            _write_toml_data(filtered_dicts, hidden_builtins, path)
            logger.info("slot profile store: deleted custom profile %s", profile_id)
            return True
    else:
        existing_dicts = []
        hidden_builtins: set[str] = set()

    if builtin_profile_ids is not None and profile_id in builtin_profile_ids:
        if profile_id not in hidden_builtins:
            hidden_builtins.add(profile_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_toml_data(existing_dicts, hidden_builtins, path)
            logger.info("slot profile store: hidden built-in profile %s", profile_id)
        return True

    return False


def load_hidden_builtin_profile_ids() -> set[str]:
    """Load hidden built-in slot profile IDs."""
    path = slot_profiles_file_path()
    if not path.exists():
        return set()
    try:
        with open(path, "rb") as f:
            data = __load_toml(f)
        return set(data.get("hidden_builtin_profiles", []))
    except Exception:  # noqa: BLE001
        return set()


def custom_slot_profile_exists(profile_id: str) -> bool:
    """Return True if a custom slot profile exists."""
    profiles_dicts, _ = _load_toml_data(slot_profiles_file_path())
    return any(p["profile_id"] == profile_id for p in profiles_dicts)


def _profile_to_dict(profile: SlotProfileSpec) -> dict[str, Any]:
    """Convert a slot profile to a TOML-serializable dict."""
    return {
        "profile_id": profile.profile_id,
        "alias": profile.alias,
        "device": profile.device,
        "model": profile.model,
        "port": profile.port,
        "ctx_size": profile.ctx_size,
        "ubatch_size": profile.ubatch_size,
        "threads": profile.threads,
        "description": profile.description,
        "bind_address": profile.bind_address,
        "tensor_split": profile.tensor_split,
        "reasoning_mode": profile.reasoning_mode,
        "reasoning_format": profile.reasoning_format,
        "chat_template_kwargs": profile.chat_template_kwargs,
        "reasoning_budget": profile.reasoning_budget,
        "use_jinja": profile.use_jinja,
        "cache_type_k": profile.cache_type_k,
        "cache_type_v": profile.cache_type_v,
        "n_gpu_layers": profile.n_gpu_layers,
        "main_gpu": profile.main_gpu,
        "server_bin": profile.server_bin,
        "backend": profile.backend,
        "risky_acknowledged": list(profile.risky_acknowledged),
        "batch_size": profile.batch_size,
        "poll_ms": profile.poll_ms,
        "n_predict": profile.n_predict,
        "parallel": profile.parallel,
        "threads_batch": profile.threads_batch,
        "mmproj": profile.mmproj,
        "spec_type": profile.spec_type,
        "spec_ngram_size_n": profile.spec_ngram_size_n,
        "draft_min": profile.draft_min,
        "draft_max": profile.draft_max,
        "spec_draft_n_max": profile.spec_draft_n_max,
        "spec_draft_p_min": profile.spec_draft_p_min,
        "spec_draft_cache_type_k": profile.spec_draft_cache_type_k,
        "spec_draft_cache_type_v": profile.spec_draft_cache_type_v,
        "spec_draft_device": profile.spec_draft_device,
    }


def _profile_from_dict(data: dict[str, Any]) -> SlotProfileSpec:
    """Reconstruct a slot profile from a TOML dict."""
    return SlotProfileSpec(
        profile_id=data["profile_id"],
        alias=data.get("alias", ""),
        device=data.get("device", ""),
        model=data["model"],
        port=int(data["port"]),
        ctx_size=int(data["ctx_size"]),
        ubatch_size=int(data["ubatch_size"]),
        threads=int(data["threads"]),
        description=data.get("description", ""),
        bind_address=data.get("bind_address", "127.0.0.1"),
        tensor_split=data.get("tensor_split", ""),
        reasoning_mode=data.get("reasoning_mode", "auto"),
        reasoning_format=data.get("reasoning_format", "none"),
        chat_template_kwargs=data.get("chat_template_kwargs", ""),
        reasoning_budget=data.get("reasoning_budget", ""),
        use_jinja=data.get("use_jinja", False),
        cache_type_k=data.get("cache_type_k", "q8_0"),
        cache_type_v=data.get("cache_type_v", "q8_0"),
        n_gpu_layers=data.get("n_gpu_layers", 99),
        main_gpu=int(data.get("main_gpu", 0)),
        server_bin=data.get("server_bin", ""),
        backend=data.get("backend", "llama_cpp"),
        risky_acknowledged=tuple(data.get("risky_acknowledged", [])),
        batch_size=int(data.get("batch_size", 2048)),
        poll_ms=int(data.get("poll_ms", 50)),
        n_predict=int(data.get("n_predict", 32768)),
        parallel=int(data.get("parallel", 4)),
        threads_batch=int(data.get("threads_batch", 0)),
        mmproj=data.get("mmproj", ""),
        spec_type=data.get("spec_type", ""),
        spec_ngram_size_n=int(data.get("spec_ngram_size_n", 0)),
        draft_min=int(data.get("draft_min", 0)),
        draft_max=int(data.get("draft_max", 0)),
        spec_draft_n_max=int(data.get("spec_draft_n_max", 0)),
        spec_draft_p_min=float(data.get("spec_draft_p_min", 0.0)),
        spec_draft_cache_type_k=data.get("spec_draft_cache_type_k", ""),
        spec_draft_cache_type_v=data.get("spec_draft_cache_type_v", ""),
        spec_draft_device=data.get("spec_draft_device", ""),
    )


def _load_toml_data(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load slot profile TOML data."""
    if not path.exists():
        return [], set()
    try:
        with open(path, "rb") as f:
            data = __load_toml(f)
        profiles_data = data.get("profiles", [])
        hidden = set(data.get("hidden_builtin_profiles", []))
        return profiles_data, hidden
    except Exception:  # noqa: BLE001
        return [], set()


def _write_toml_data(
    profiles: list[dict[str, Any]],
    hidden_builtins: set[str],
    path: Path,
) -> None:
    """Write slot profiles and hidden builtins as a TOML file."""
    lines: list[str] = []

    if hidden_builtins:
        hidden_list = sorted(hidden_builtins)
        lines.append(f"hidden_builtin_profiles = {json.dumps(hidden_list)}")
        lines.append("")

    for i, p in enumerate(profiles):
        if i > 0:
            lines.append("")
        lines.append("[[profiles]]")
        lines.append(f"profile_id = {json.dumps(p['profile_id'])}")
        lines.append(f"alias = {json.dumps(p.get('alias', ''))}")
        lines.append(f"device = {json.dumps(p.get('device', ''))}")
        lines.append(f"model = {json.dumps(p['model'])}")
        lines.append(f"port = {p['port']}")
        lines.append(f"ctx_size = {p['ctx_size']}")
        lines.append(f"ubatch_size = {p['ubatch_size']}")
        lines.append(f"threads = {p['threads']}")
        lines.append(f"description = {json.dumps(p.get('description', ''))}")
        lines.append(f"bind_address = {json.dumps(p.get('bind_address', '127.0.0.1'))}")
        lines.append(f"tensor_split = {json.dumps(p.get('tensor_split', ''))}")
        lines.append(f"reasoning_mode = {json.dumps(p.get('reasoning_mode', 'auto'))}")
        lines.append(f"reasoning_format = {json.dumps(p.get('reasoning_format', 'none'))}")
        ctk = p.get("chat_template_kwargs", "")
        if isinstance(ctk, dict):
            lines.append(f"chat_template_kwargs = {json.dumps(json.dumps(ctk))}")
        else:
            lines.append(f"chat_template_kwargs = {json.dumps(str(ctk))}")
        lines.append(f"reasoning_budget = {json.dumps(p.get('reasoning_budget', ''))}")
        lines.append(f"use_jinja = {str(p.get('use_jinja', False)).lower()}")
        lines.append(f"cache_type_k = {json.dumps(p.get('cache_type_k', 'q8_0'))}")
        lines.append(f"cache_type_v = {json.dumps(p.get('cache_type_v', 'q8_0'))}")
        ngl = p.get("n_gpu_layers", 99)
        if isinstance(ngl, str):
            lines.append(f"n_gpu_layers = {json.dumps(ngl)}")
        else:
            lines.append(f"n_gpu_layers = {int(ngl)}")
        lines.append(f"main_gpu = {int(p.get('main_gpu', 0))}")
        lines.append(f"server_bin = {json.dumps(p.get('server_bin', ''))}")
        lines.append(f"backend = {json.dumps(p.get('backend', 'llama_cpp'))}")
        ra = p.get("risky_acknowledged", [])
        if ra:
            lines.append(f"risky_acknowledged = {json.dumps(ra)}")
        lines.append(f"batch_size = {int(p.get('batch_size', 2048))}")
        lines.append(f"poll_ms = {int(p.get('poll_ms', 50))}")
        lines.append(f"n_predict = {int(p.get('n_predict', 32768))}")
        lines.append(f"parallel = {int(p.get('parallel', 4))}")
        lines.append(f"threads_batch = {int(p.get('threads_batch', 0))}")
        lines.append(f"mmproj = {json.dumps(p.get('mmproj', ''))}")
        lines.append(f"spec_type = {json.dumps(p.get('spec_type', ''))}")
        lines.append(f"spec_ngram_size_n = {int(p.get('spec_ngram_size_n', 0))}")
        lines.append(f"draft_min = {int(p.get('draft_min', 0))}")
        lines.append(f"draft_max = {int(p.get('draft_max', 0))}")
        lines.append(f"spec_draft_n_max = {int(p.get('spec_draft_n_max', 0))}")
        lines.append(f"spec_draft_p_min = {float(p.get('spec_draft_p_min', 0.0))}")
        lines.append(
            f"spec_draft_cache_type_k = {json.dumps(p.get('spec_draft_cache_type_k', ''))}"
        )
        lines.append(
            f"spec_draft_cache_type_v = {json.dumps(p.get('spec_draft_cache_type_v', ''))}"
        )
        lines.append(f"spec_draft_device = {json.dumps(p.get('spec_draft_device', ''))}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def __load_toml(file_obj: Any) -> dict[str, Any]:
    """Load TOML data from a binary file handle."""
    import tomllib

    return tomllib.load(file_obj)
