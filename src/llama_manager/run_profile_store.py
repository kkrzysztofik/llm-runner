"""Persistent store for custom run profiles saved to XDG config TOML."""

import json
import os
from pathlib import Path
from typing import Any

from .config.profiles import RunProfileSpec


def run_profiles_file_path() -> Path:
    """Return the path to the custom run profiles TOML file.

    Returns:
        Path to ``$XDG_CONFIG_HOME/llm-runner/run_profiles.toml``.
    """
    xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return xdg_config / "llm-runner" / "run_profiles.toml"


def load_custom_run_profiles() -> list[RunProfileSpec]:
    """Load custom run profiles from disk.

    Returns an empty list if the file doesn't exist or is invalid.
    Skips malformed entries with missing required keys.

    Returns:
        List of ``RunProfileSpec`` instances parsed from TOML.
    """
    profiles_dicts, _ = _load_toml_data(run_profiles_file_path())
    result: list[RunProfileSpec] = []
    for p in profiles_dicts:
        try:
            result.append(_profile_from_dict(p))
        except Exception:  # noqa: BLE001, S112
            continue
    return result


def save_custom_run_profile(profile: RunProfileSpec) -> None:
    """Save a single custom run profile to disk.

    Creates the parent directory if needed. Rejects duplicate ``profile_id``
    — raises ``ValueError`` if a profile with the same id already exists.

    Args:
        profile: The profile to persist.

    Raises:
        ValueError: If *profile* has a duplicate ``profile_id``.
    """
    path = run_profiles_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_dicts, hidden_builtins = _load_toml_data(path)
    ids = {p["profile_id"] for p in existing_dicts}
    if profile.profile_id in ids:
        raise ValueError(f"Duplicate profile_id: {profile.profile_id}")

    profiles_dict: list[dict[str, Any]] = []
    for p in existing_dicts:
        profiles_dict.append(p)
    profiles_dict.append(_profile_to_dict(profile))

    _write_toml_data(profiles_dict, hidden_builtins, path)


def upsert_custom_run_profile(original_profile_id: str, profile: RunProfileSpec) -> None:
    """Upsert a custom run profile.

    If *original_profile_id* matches an existing custom profile entry in
    ``[[profiles]]``, that entry is replaced. If no match (e.g. overriding a
    built-in for the first time), a new entry is appended.

    When ``profile.profile_id != original_profile_id`` (rename):
      - For existing custom entries: remove old entry, add new.
        Checks new profile_id doesn't conflict with other entries.
      - For non-existing (built-in edit with rename): add new entry.
        Checks new profile_id doesn't conflict with other entries.

    Raises ValueError on duplicate profile_id for new entries or renames.
    """
    path = run_profiles_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_dicts, hidden_builtins = _load_toml_data(path)

    new_id = profile.profile_id
    # Check for conflicts with existing entries (excluding the original if it exists)
    ids_to_check = {
        p["profile_id"] for p in existing_dicts if p["profile_id"] != original_profile_id
    }
    if new_id in ids_to_check:
        raise ValueError(f"Duplicate profile_id: {new_id}")

    # Find and remove the original entry if it exists
    filtered_dicts = [p for p in existing_dicts if p["profile_id"] != original_profile_id]

    # Add the new/updated entry
    filtered_dicts.append(_profile_to_dict(profile))

    _write_toml_data(filtered_dicts, hidden_builtins, path)


def delete_custom_run_profile(profile_id: str, builtin_profile_ids: set[str] | None = None) -> bool:
    """Delete/hide a run profile.

    - If *profile_id* is in ``[[profiles]]`` (custom entry): remove it from TOML.
      This restores the built-in if the original is a built-in override.
    - If *profile_id* is NOT in ``[[profiles]]`` but IS in *builtin_profile_ids*:
      add it to hidden_builtin_profiles header in TOML.
    - Otherwise: return False (not found).

    Args:
        profile_id: The profile identifier to delete/hide.
        builtin_profile_ids: If provided, profiles in this set that have no custom
            entry will be added to hidden_builtin_profiles instead of returning False.

    Returns:
        True if the profile was found and acted on.
    """
    path = run_profiles_file_path()

    if path.exists():
        existing_dicts, hidden_builtins = _load_toml_data(path)

        # Check if it's a custom entry — remove it from [[profiles]]
        custom_entries = [p for p in existing_dicts if p["profile_id"] == profile_id]
        if custom_entries:
            filtered_dicts = [p for p in existing_dicts if p["profile_id"] != profile_id]
            _write_toml_data(filtered_dicts, hidden_builtins, path)
            return True
    else:
        existing_dicts = []
        hidden_builtins: set[str] = set()

    # Check if it's a built-in that should be hidden
    if builtin_profile_ids is not None and profile_id in builtin_profile_ids:
        if profile_id not in hidden_builtins:
            hidden_builtins.add(profile_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_toml_data(existing_dicts, hidden_builtins, path)
        return True

    return False


def load_hidden_builtin_profile_ids() -> set[str]:
    """Load the set of hidden built-in profile IDs from the TOML file.

    Returns:
        Set of hidden built-in profile ID strings.
    """
    path = run_profiles_file_path()
    if not path.exists():
        return set()
    try:
        with open(path, "rb") as f:
            data = __load_toml(f)
        return set(data.get("hidden_builtin_profiles", []))
    except Exception:  # noqa: BLE001
        return set()


def custom_profile_exists(profile_id: str) -> bool:
    """Check if a profile_id exists in the ``[[profiles]]`` section of the TOML.

    Args:
        profile_id: The profile identifier to check.

    Returns:
        True if a custom entry with this profile_id exists.
    """
    profiles_dicts, _ = _load_toml_data(run_profiles_file_path())
    return any(p["profile_id"] == profile_id for p in profiles_dicts)


def _profile_to_dict(profile: RunProfileSpec) -> dict[str, Any]:
    """Convert a ``RunProfileSpec`` to a TOML-serializable dict."""
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
    }


def _profile_from_dict(data: dict[str, Any]) -> RunProfileSpec:
    """Reconstruct a ``RunProfileSpec`` from a TOML dict."""
    return RunProfileSpec(
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
    )


def _load_toml_data(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load profiles and hidden builtins from the TOML file.

    Args:
        path: Path to the TOML file.

    Returns:
        Tuple of (list of profile dicts, set of hidden builtin IDs).
        Returns empty list and empty set if file doesn't exist or is invalid.
    """
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
    """Write profiles and hidden builtins as a TOML file.

    Uses ``json.dumps`` for string values and simple formatting for scalars.

    Args:
        profiles: List of profile dicts to serialize.
        hidden_builtins: Set of hidden built-in profile IDs to write as header.
        path: Destination file path.
    """
    lines: list[str] = []

    # Write hidden_builtin_profiles header if non-empty
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
            lines.append(f"chat_template_kwargs = {json.dumps(ctk)}")
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

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def __load_toml(file_obj: Any) -> dict[str, Any]:
    """Load TOML data from a binary file handle.

    Args:
        file_obj: Open file in binary mode.

    Returns:
        Parsed TOML dict.
    """
    import tomllib

    return tomllib.load(file_obj)
