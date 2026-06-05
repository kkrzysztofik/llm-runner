"""Tests for persistent run-profile TOML store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llama_manager.config.profiles import SlotProfileSpec as RunProfileSpec
from llama_manager.slot_profile_store import (
    _profile_from_dict,
    _profile_to_dict,
)
from llama_manager.slot_profile_store import (
    load_custom_slot_profiles as load_custom_run_profiles,
)
from llama_manager.slot_profile_store import (
    save_custom_slot_profile as save_custom_run_profile,
)
from llama_manager.slot_profile_store import (
    slot_profiles_file_path as run_profiles_file_path,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def xdg_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set XDG_CONFIG_HOME to a temp dir and return it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture()
def sample_profile() -> RunProfileSpec:
    """Return a minimal valid RunProfileSpec for testing."""
    return RunProfileSpec(
        profile_id="my-custom-profile",
        model="/path/to/model.gguf",
        alias="my-custom",
        device="CUDA:0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        description="A test profile",
        backend="llama_cpp",
    )


# ---------------------------------------------------------------------------
# run_profiles_file_path
# ---------------------------------------------------------------------------


def test_run_profiles_file_path_uses_xdg_config(xdg_config_home: Path) -> None:
    """run_profiles_file_path should use XDG_CONFIG_HOME when set."""
    expected = xdg_config_home / "llm-runner" / "slot_profiles.toml"
    assert run_profiles_file_path() == expected


def test_run_profiles_file_path_falls_back_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When XDG_CONFIG_HOME is unset, path should be ~/.config/llm-runner/slot_profiles.toml."""
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    def fake_home() -> Path:
        return tmp_path

    monkeypatch.setattr(Path, "home", staticmethod(fake_home))
    result = run_profiles_file_path()
    assert result == tmp_path / ".config" / "llm-runner" / "slot_profiles.toml"


# ---------------------------------------------------------------------------
# load_custom_run_profiles
# ---------------------------------------------------------------------------


def test_load_custom_run_profiles_empty_when_no_file(xdg_config_home: Path) -> None:
    """load_custom_run_profiles should return empty list when TOML does not exist."""
    profiles = load_custom_run_profiles()
    assert profiles == []


def test_load_custom_run_profiles_parses_valid_file(xdg_config_home: Path) -> None:
    """load_custom_run_profiles should parse a valid TOML file correctly."""
    toml_path = xdg_config_home / "llm-runner" / "slot_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        '[["profiles"]]\n'
        'profile_id = "test-profile"\n'
        'model = "/models/test.gguf"\n'
        "port = 9090\n"
        "ctx_size = 8192\n"
        "ubatch_size = 1024\n"
        "threads = 4\n"
        'alias = "test"\n'
        'device = "SYCL0"\n'
        'backend = "llama_cpp"\n',
        encoding="utf-8",
    )
    profiles = load_custom_run_profiles()
    assert len(profiles) == 1
    p = profiles[0]
    assert p.profile_id == "test-profile"
    assert p.model == "/models/test.gguf"
    assert p.port == 9090
    assert p.ctx_size == 8192
    assert p.ubatch_size == 1024
    assert p.threads == 4
    assert p.alias == "test"
    assert p.device == "SYCL0"
    assert p.backend == "llama_cpp"


def test_load_custom_run_profiles_returns_empty_on_invalid_toml(xdg_config_home: Path) -> None:
    """load_custom_run_profiles should return empty list for invalid TOML."""
    toml_path = xdg_config_home / "llm-runner" / "slot_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text("this is not valid toml {{{", encoding="utf-8")
    profiles = load_custom_run_profiles()
    assert profiles == []


def test_load_custom_run_profiles_returns_empty_on_missing_keys(xdg_config_home: Path) -> None:
    """load_custom_run_profiles should return empty list when required keys are missing."""
    toml_path = xdg_config_home / "llm-runner" / "slot_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        '[["profiles"]]\n'
        'profile_id = "broken-profile"\n'
        "port = 8080\n",  # missing required 'model' key
        encoding="utf-8",
    )
    profiles = load_custom_run_profiles()
    assert profiles == []


# ---------------------------------------------------------------------------
# save_custom_run_profile
# ---------------------------------------------------------------------------


def test_save_custom_run_profile_creates_file(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """save_custom_run_profile should create the TOML file."""
    save_custom_run_profile(sample_profile)
    toml_path = run_profiles_file_path()
    assert toml_path.exists()
    assert toml_path.is_file()


def test_save_custom_run_profile_rejects_duplicate_id(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """save_custom_run_profile should raise ValueError on duplicate profile_id."""
    save_custom_run_profile(sample_profile)
    with pytest.raises(ValueError, match="Duplicate profile_id"):
        save_custom_run_profile(sample_profile)


def test_save_custom_run_profile_appends_to_existing(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """Saving a second profile should append it and both should be loadable."""
    profile_2 = RunProfileSpec(
        profile_id="another-profile",
        model="/models/another.gguf",
        alias="another",
        device="CUDA:1",
        port=9090,
        ctx_size=2048,
        ubatch_size=256,
        threads=4,
        backend="llama_cpp",
    )
    save_custom_run_profile(sample_profile)
    save_custom_run_profile(profile_2)

    loaded = load_custom_run_profiles()
    ids = {p.profile_id for p in loaded}
    assert "my-custom-profile" in ids
    assert "another-profile" in ids


def test_save_custom_run_profile_writes_valid_toml(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """The written TOML should be parseable by tomllib."""
    import tomllib

    save_custom_run_profile(sample_profile)
    toml_path = run_profiles_file_path()
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    assert "profiles" in data
    assert len(data["profiles"]) == 1
    assert data["profiles"][0]["profile_id"] == "my-custom-profile"


# ---------------------------------------------------------------------------
# Round-trip serialization
# ---------------------------------------------------------------------------


def test_roundtrip_serialization(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """save → load should produce a profile with matching fields."""
    save_custom_run_profile(sample_profile)
    loaded = load_custom_run_profiles()
    assert len(loaded) == 1
    original = sample_profile
    restored = loaded[0]

    assert restored.profile_id == original.profile_id
    assert restored.model == original.model
    assert restored.alias == original.alias
    assert restored.device == original.device
    assert restored.port == original.port
    assert restored.ctx_size == original.ctx_size
    assert restored.ubatch_size == original.ubatch_size
    assert restored.threads == original.threads
    assert restored.description == original.description
    assert restored.backend == original.backend


def test_roundtrip_serialization_with_all_fields(xdg_config_home: Path) -> None:
    """Round-trip should preserve all optional fields."""
    full_profile = RunProfileSpec(
        profile_id="full-profile",
        model="/models/full.gguf",
        alias="full",
        device="SYCL0",
        port=8888,
        ctx_size=16384,
        ubatch_size=2048,
        threads=16,
        description="A full-featured profile",
        bind_address="0.0.0.0",
        tensor_split="2,2",
        reasoning_mode="off",
        reasoning_format="none",
        chat_template_kwargs='{"key": "value"}',
        reasoning_budget="512",
        use_jinja=True,
        cache_type_k="q4_0",
        cache_type_v="q4_0",
        n_gpu_layers=99,
        main_gpu=1,
        server_bin="/usr/local/bin/llama-server",
        backend="llama_cpp",
        risky_acknowledged=("risky1", "risky2"),
        batch_size=1024,
        poll_ms=25,
        n_predict=16384,
        parallel=2,
        threads_batch=4,
        mmproj="/models/mmproj.gguf",
        spec_type="ngram-mod",
        spec_ngram_size_n=10,
        draft_min=2,
        draft_max=6,
        spec_draft_n_max=0,
        spec_draft_p_min=0.0,
        spec_draft_cache_type_k="",
        spec_draft_cache_type_v="",
        spec_draft_device="",
    )
    save_custom_run_profile(full_profile)
    loaded = load_custom_run_profiles()
    assert len(loaded) == 1
    r = loaded[0]

    assert r.profile_id == full_profile.profile_id
    assert r.model == full_profile.model
    assert r.alias == full_profile.alias
    assert r.device == full_profile.device
    assert r.port == full_profile.port
    assert r.ctx_size == full_profile.ctx_size
    assert r.ubatch_size == full_profile.ubatch_size
    assert r.threads == full_profile.threads
    assert r.description == full_profile.description
    assert r.bind_address == full_profile.bind_address
    assert r.tensor_split == full_profile.tensor_split
    assert r.reasoning_mode == full_profile.reasoning_mode
    assert r.reasoning_format == full_profile.reasoning_format
    assert r.chat_template_kwargs == full_profile.chat_template_kwargs
    assert r.reasoning_budget == full_profile.reasoning_budget
    assert r.use_jinja == full_profile.use_jinja
    assert r.cache_type_k == full_profile.cache_type_k
    assert r.cache_type_v == full_profile.cache_type_v
    assert r.n_gpu_layers == full_profile.n_gpu_layers
    assert r.main_gpu == full_profile.main_gpu
    assert r.server_bin == full_profile.server_bin
    assert r.backend == full_profile.backend
    assert r.risky_acknowledged == full_profile.risky_acknowledged
    assert r.batch_size == full_profile.batch_size
    assert r.poll_ms == full_profile.poll_ms
    assert r.n_predict == full_profile.n_predict
    assert r.parallel == full_profile.parallel
    assert r.threads_batch == full_profile.threads_batch
    assert r.mmproj == full_profile.mmproj
    assert r.spec_type == full_profile.spec_type
    assert r.spec_ngram_size_n == full_profile.spec_ngram_size_n
    assert r.draft_min == full_profile.draft_min
    assert r.draft_max == full_profile.draft_max


# ---------------------------------------------------------------------------
# _profile_to_dict / _profile_from_dict
# ---------------------------------------------------------------------------


def test_profile_to_dict_roundtrips(sample_profile: RunProfileSpec) -> None:
    """_profile_to_dict + _profile_from_dict should preserve all fields."""
    d = _profile_to_dict(sample_profile)
    restored = _profile_from_dict(d)
    assert restored.profile_id == sample_profile.profile_id
    assert restored.model == sample_profile.model
    assert restored.alias == sample_profile.alias
    assert restored.device == sample_profile.device
    assert restored.port == sample_profile.port
    assert restored.ctx_size == sample_profile.ctx_size


def test_profile_to_dict_handles_string_ngl() -> None:
    """_profile_to_dict should serialize n_gpu_layers as a quoted string when it's 'all'."""
    profile = RunProfileSpec(
        profile_id="ngl-all",
        model="/models/test.gguf",
        alias="ngl",
        device="CUDA:0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        n_gpu_layers="all",
        backend="llama_cpp",
    )
    d = _profile_to_dict(profile)
    assert d["n_gpu_layers"] == "all"


def test_profile_from_dict_applies_defaults() -> None:
    """_profile_from_dict should fill in defaults for missing keys."""
    minimal: dict[str, Any] = {
        "profile_id": "minimal",
        "model": "/models/min.gguf",
        "port": 8080,
        "ctx_size": 4096,
        "ubatch_size": 512,
        "threads": 4,
        "alias": "min",
        "device": "",
        "backend": "llama_cpp",
    }
    p = _profile_from_dict(minimal)
    assert p.bind_address == "127.0.0.1"
    assert p.reasoning_mode == "auto"
    assert p.reasoning_format == "none"
    assert p.use_jinja is False
    assert p.cache_type_k == "q8_0"
    assert p.cache_type_v == "q8_0"
    assert p.n_gpu_layers == 99
    assert p.main_gpu == 0
    assert p.server_bin == ""
    assert p.backend == "llama_cpp"
    assert p.risky_acknowledged == ()


# ---------------------------------------------------------------------------
# upsert_custom_run_profile
# ---------------------------------------------------------------------------


def test_upsert_custom_run_profile_appends_new(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """upsert with non-existing original_profile_id should append a new entry."""
    from llama_manager.slot_profile_store import (
        upsert_custom_slot_profile as upsert_custom_run_profile,
    )

    upsert_custom_run_profile("nonexistent", sample_profile)

    loaded = load_custom_run_profiles()
    assert len(loaded) == 1
    assert loaded[0].profile_id == "my-custom-profile"


def test_upsert_custom_run_profile_replaces_existing(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """upsert with existing original_profile_id should replace that entry."""
    from llama_manager.slot_profile_store import (
        upsert_custom_slot_profile as upsert_custom_run_profile,
    )

    # First save the profile
    save_custom_run_profile(sample_profile)

    # Now upsert with modified values
    updated = RunProfileSpec(
        profile_id="my-custom-profile",
        model="/models/updated.gguf",
        alias="my-custom-updated",
        device="CUDA:0",
        port=9999,
        ctx_size=8192,
        ubatch_size=1024,
        threads=16,
        backend="llama_cpp",
    )
    upsert_custom_run_profile("my-custom-profile", updated)

    loaded = load_custom_run_profiles()
    assert len(loaded) == 1
    assert loaded[0].profile_id == "my-custom-profile"
    assert loaded[0].model == "/models/updated.gguf"
    assert loaded[0].port == 9999


def test_upsert_custom_run_profile_rename(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """upsert with different profile.profile_id should rename the entry."""
    from llama_manager.slot_profile_store import (
        upsert_custom_slot_profile as upsert_custom_run_profile,
    )

    save_custom_run_profile(sample_profile)

    renamed = RunProfileSpec(
        profile_id="renamed-profile",
        model="/models/renamed.gguf",
        alias="renamed",
        device="CUDA:0",
        port=9090,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )
    upsert_custom_run_profile("my-custom-profile", renamed)

    loaded = load_custom_run_profiles()
    assert len(loaded) == 1
    assert loaded[0].profile_id == "renamed-profile"
    assert loaded[0].model == "/models/renamed.gguf"


def test_upsert_custom_run_profile_rename_checks_conflict(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """Rename that conflicts with existing entry should raise ValueError."""
    from llama_manager.slot_profile_store import (
        upsert_custom_slot_profile as upsert_custom_run_profile,
    )

    profile_2 = RunProfileSpec(
        profile_id="target-profile",
        model="/models/target.gguf",
        alias="target",
        device="CUDA:0",
        port=9090,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )
    save_custom_run_profile(sample_profile)
    save_custom_run_profile(profile_2)

    renamed = RunProfileSpec(
        profile_id="target-profile",  # conflicts with profile_2
        model="/models/other.gguf",
        alias="other",
        device="CUDA:0",
        port=9090,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )
    with pytest.raises(ValueError, match="Duplicate profile_id"):
        upsert_custom_run_profile("my-custom-profile", renamed)


# ---------------------------------------------------------------------------
# delete_custom_run_profile
# ---------------------------------------------------------------------------


def test_delete_custom_run_profile_removes_custom_entry(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """delete should remove a custom profile from [[profiles]]."""
    from llama_manager.slot_profile_store import (
        delete_custom_slot_profile as delete_custom_run_profile,
    )
    from llama_manager.slot_profile_store import (
        load_custom_slot_profiles as load_custom_run_profiles,
    )

    save_custom_run_profile(sample_profile)

    result = delete_custom_run_profile("my-custom-profile")
    assert result is True

    loaded = load_custom_run_profiles()
    assert len(loaded) == 0


def test_delete_custom_run_profile_hides_builtin(
    xdg_config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_profile: RunProfileSpec,
) -> None:
    """delete should mark a built-in as hidden when no custom entry exists."""
    from llama_manager.slot_profile_store import (
        delete_custom_slot_profile as delete_custom_run_profile,
    )
    from llama_manager.slot_profile_store import (
        load_hidden_builtin_profile_ids,
    )

    toml_path = xdg_config_home / "llm-runner" / "slot_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "llama_manager.slot_profile_store.slot_profiles_file_path", lambda: toml_path
    )

    builtin_ids: set[str] = {"summary-balanced", "summary-fast", "qwen35"}
    result = delete_custom_run_profile("summary-fast", builtin_ids)
    assert result is True

    hidden = load_hidden_builtin_profile_ids()
    assert "summary-fast" in hidden


def test_delete_custom_run_profile_not_found(
    xdg_config_home: Path,
) -> None:
    """delete should return False for unknown profile_id."""
    from llama_manager.slot_profile_store import (
        delete_custom_slot_profile as delete_custom_run_profile,
    )

    result = delete_custom_run_profile("unknown-profile")
    assert result is False


def test_delete_custom_run_profile_removes_custom_and_restores_builtin(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """delete custom entry should remove it, hiding the builtin."""
    from llama_manager.slot_profile_store import (
        delete_custom_slot_profile as delete_custom_run_profile,
    )
    from llama_manager.slot_profile_store import (
        load_hidden_builtin_profile_ids,
    )

    save_custom_run_profile(sample_profile)
    builtin_ids: set[str] = {"summary-balanced", "summary-fast", "qwen35"}

    # Delete a built-in that has no custom entry
    result = delete_custom_run_profile("summary-fast", builtin_ids)
    assert result is True

    hidden = load_hidden_builtin_profile_ids()
    assert "summary-fast" in hidden


# ---------------------------------------------------------------------------
# custom_profile_exists
# ---------------------------------------------------------------------------


def test_custom_profile_exists_true(
    xdg_config_home: Path,
    sample_profile: RunProfileSpec,
) -> None:
    """custom_profile_exists should return True for a saved custom profile."""
    from llama_manager.slot_profile_store import custom_slot_profile_exists as custom_profile_exists

    save_custom_run_profile(sample_profile)
    assert custom_profile_exists("my-custom-profile") is True


def test_custom_profile_exists_false(
    xdg_config_home: Path,
) -> None:
    """custom_profile_exists should return False for unknown profile_id."""
    from llama_manager.slot_profile_store import custom_slot_profile_exists as custom_profile_exists

    assert custom_profile_exists("nonexistent") is False


# ---------------------------------------------------------------------------
# load_hidden_builtin_profile_ids
# ---------------------------------------------------------------------------


def test_load_hidden_builtin_profile_ids_returns_set(
    xdg_config_home: Path,
) -> None:
    """load_hidden_builtin_profile_ids should load from TOML."""
    from llama_manager.slot_profile_store import load_hidden_builtin_profile_ids

    toml_path = xdg_config_home / "llm-runner" / "slot_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        'hidden_builtin_profiles = ["summary-fast", "qwen35"]\n',
        encoding="utf-8",
    )

    hidden = load_hidden_builtin_profile_ids()
    assert hidden == {"summary-fast", "qwen35"}


def test_load_hidden_builtin_profile_ids_empty_when_no_file(
    xdg_config_home: Path,
) -> None:
    """load_hidden_builtin_profile_ids should return empty set when no file."""
    from llama_manager.slot_profile_store import load_hidden_builtin_profile_ids

    hidden = load_hidden_builtin_profile_ids()
    assert hidden == set()
