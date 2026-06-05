"""Tests for create_tui_profile_registry — built-in + custom profile merging."""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_manager.config import Config, SlotProfileSpec
from llama_manager.config.builder import create_tui_profile_registry
from llama_manager.slot_profile_store import save_custom_slot_profile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def xdg_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set XDG_CONFIG_HOME to a temp dir and return it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture()
def config(xdg_config_home: Path) -> Config:
    """Return a Config pointing at the temp XDG dir."""
    return Config()


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------


def test_tui_registry_includes_builtins(config: Config) -> None:
    """create_tui_profile_registry should include all built-in profiles."""
    registry = create_tui_profile_registry(config)

    profile_ids = registry.profile_ids
    assert "summary-balanced" in profile_ids
    assert "summary-fast" in profile_ids
    assert "qwen35" in profile_ids


def test_tui_registry_includes_custom(
    config: Config,
    xdg_config_home: Path,
) -> None:
    """create_tui_profile_registry should include custom profiles from disk."""
    custom_profile = SlotProfileSpec(
        profile_id="my-custom",
        model="/models/custom.gguf",
        alias="my-custom",
        device="CUDA:0",
        port=9090,
        ctx_size=8192,
        ubatch_size=1024,
        threads=4,
        backend="llama_cpp",
    )
    save_custom_slot_profile(custom_profile)

    registry = create_tui_profile_registry(config)
    profile_ids = registry.profile_ids
    assert "my-custom" in profile_ids


def test_tui_registry_custom_overrides_builtin(
    config: Config,
    xdg_config_home: Path,
) -> None:
    """Custom profile with same profile_id as a built-in should override it."""
    # Save a custom profile that shadows the built-in 'summary-balanced'
    override_profile = SlotProfileSpec(
        profile_id="summary-balanced",
        model="/models/override.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=7777,  # different port
        ctx_size=1024,
        ubatch_size=128,
        threads=2,
        backend="llama_cpp",
    )
    save_custom_slot_profile(override_profile)

    registry = create_tui_profile_registry(config)
    profile = registry.get_profile("summary-balanced")

    # Custom profile should win
    assert profile.model == "/models/override.gguf"
    assert profile.port == 7777
    assert profile.ctx_size == 1024
    assert profile.threads == 2


def test_tui_registry_custom_profile_keeps_builtin_profiles(
    config: Config,
    xdg_config_home: Path,
) -> None:
    """Custom profiles should not affect built-in slot profiles."""
    save_custom_slot_profile(
        SlotProfileSpec(
            profile_id="my-custom",
            model="/models/custom.gguf",
            alias="my-custom",
            device="CUDA:0",
            port=9090,
            ctx_size=4096,
            ubatch_size=512,
            threads=8,
            backend="llama_cpp",
        ),
    )

    registry = create_tui_profile_registry(config)
    profile_ids = registry.profile_ids
    assert "summary-balanced" in profile_ids
    assert "qwen35" in profile_ids


def test_tui_registry_multiple_custom_profiles(
    config: Config,
    xdg_config_home: Path,
) -> None:
    """Multiple custom profiles should all be included."""
    save_custom_slot_profile(
        SlotProfileSpec(
            profile_id="profile-a",
            model="/models/a.gguf",
            alias="profile-a",
            device="CUDA:0",
            port=9090,
            ctx_size=4096,
            ubatch_size=512,
            threads=8,
            backend="llama_cpp",
        ),
    )
    save_custom_slot_profile(
        SlotProfileSpec(
            profile_id="profile-b",
            model="/models/b.gguf",
            alias="profile-b",
            device="CUDA:1",
            port=9091,
            ctx_size=8192,
            ubatch_size=1024,
            threads=16,
            backend="llama_cpp",
        ),
    )

    registry = create_tui_profile_registry(config)
    profile_ids = registry.profile_ids
    assert "profile-a" in profile_ids
    assert "profile-b" in profile_ids
    # Built-ins still present
    assert "summary-balanced" in profile_ids
    assert "qwen35" in profile_ids


def test_tui_registry_preserves_builtin_profile_fields_when_custom_no_override(
    config: Config,
    xdg_config_home: Path,
) -> None:
    """When custom profile doesn't shadow a built-in, built-in fields are preserved."""
    save_custom_slot_profile(
        SlotProfileSpec(
            profile_id="extra-profile",
            model="/models/extra.gguf",
            alias="extra",
            device="CUDA:0",
            port=9090,
            ctx_size=4096,
            ubatch_size=512,
            threads=8,
            backend="llama_cpp",
        ),
    )

    registry = create_tui_profile_registry(config)
    builtin = registry.get_profile("summary-balanced")

    assert builtin.profile_id == "summary-balanced"
    assert builtin.device == "SYCL0"
    assert builtin.backend == "llama_cpp"
