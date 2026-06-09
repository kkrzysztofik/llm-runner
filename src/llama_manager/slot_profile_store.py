"""Persistent store for custom slot profiles saved to XDG config TOML."""

import logging
import os
from pathlib import Path
from typing import Any

from .common.profile_io import read_profile_toml, write_profile_toml
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
    data = read_profile_toml(path)
    return set(data.get("hidden_builtin_profiles", []))


def custom_slot_profile_exists(profile_id: str) -> bool:
    """Return True if a custom slot profile exists."""
    profiles_dicts, _ = _load_toml_data(slot_profiles_file_path())
    return any(p["profile_id"] == profile_id for p in profiles_dicts)


def _profile_to_dict(profile: SlotProfileSpec) -> dict[str, Any]:
    """Convert a slot profile to a TOML-serializable dict."""
    spec = profile.spec_decode
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
        "reasoning_mode": spec.reasoning_mode,
        "reasoning_format": spec.reasoning_format,
        "chat_template_kwargs": profile.chat_template_kwargs,
        "reasoning_budget": spec.reasoning_budget,
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
        "spec_type": spec.spec_type,
        "spec_ngram_size_n": spec.spec_ngram_size_n,
        "draft_min": spec.draft_min,
        "draft_max": spec.draft_max,
        "spec_draft_n_max": spec.spec_draft_n_max,
        "spec_draft_p_min": spec.spec_draft_p_min,
        "spec_draft_cache_type_k": spec.spec_draft_cache_type_k,
        "spec_draft_cache_type_v": spec.spec_draft_cache_type_v,
        "spec_draft_device": spec.spec_draft_device,
        "spec_draft_model": spec.spec_draft_model,
        "spec_draft_hf": spec.spec_draft_hf,
        "spec_draft_ngl": spec.spec_draft_ngl,
        "spec_dflash_cross_ctx": spec.spec_dflash_cross_ctx,
        "kv_unified": profile.kv_unified,
        "mmproj_offload": profile.mmproj_offload,
        "mmap": profile.mmap,
        "mlock": profile.mlock,
        "no_host_buffer": profile.no_host_buffer,
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
        chat_template_kwargs=data.get("chat_template_kwargs", ""),
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
        reasoning_mode=data.get("reasoning_mode", "auto"),
        reasoning_format=data.get("reasoning_format", "none"),
        reasoning_budget=data.get("reasoning_budget", ""),
        spec_type=data.get("spec_type", ""),
        spec_ngram_size_n=int(data.get("spec_ngram_size_n", 0)),
        draft_min=int(data.get("draft_min", 0)),
        draft_max=int(data.get("draft_max", 0)),
        spec_draft_n_max=int(data.get("spec_draft_n_max", 0)),
        spec_draft_p_min=float(data.get("spec_draft_p_min", 0.0)),
        spec_draft_cache_type_k=data.get("spec_draft_cache_type_k", ""),
        spec_draft_cache_type_v=data.get("spec_draft_cache_type_v", ""),
        spec_draft_device=data.get("spec_draft_device", ""),
        spec_draft_model=data.get("spec_draft_model", ""),
        spec_draft_hf=data.get("spec_draft_hf", ""),
        spec_draft_ngl=data.get("spec_draft_ngl", ""),
        spec_dflash_cross_ctx=int(data.get("spec_dflash_cross_ctx", 0)),
        kv_unified=data.get("kv_unified", False),
        mmproj_offload=data.get("mmproj_offload", True),
        mmap=data.get("mmap", True),
        mlock=data.get("mlock", False),
        no_host_buffer=data.get("no_host_buffer", False),
    )


def _load_toml_data(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load slot profile TOML data."""
    data = read_profile_toml(path)
    profiles_data = data.get("profiles", [])
    hidden = set(data.get("hidden_builtin_profiles", []))
    return profiles_data, hidden


def _write_toml_data(
    profiles: list[dict[str, Any]],
    hidden_builtins: set[str],
    path: Path,
) -> None:
    """Write slot profiles and hidden builtins as a TOML file."""
    write_profile_toml(
        path,
        {"profiles": profiles, "hidden_builtin_profiles": sorted(hidden_builtins)},
    )
