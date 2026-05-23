"""Tests for create_tui_profile_registry hidden built-in filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_manager.config.builder import create_tui_profile_registry
from llama_manager.config.defaults import Config
from llama_manager.config.profiles import RunProfileSpec
from llama_manager.run_profile_store import save_custom_run_profile


@pytest.fixture()
def xdg_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set XDG_CONFIG_HOME to a temp dir and return it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def test_tui_registry_filters_hidden_builtin(
    xdg_config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TUI registry should exclude hidden built-in profiles."""
    toml_path = xdg_config_home / "llm-runner" / "run_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        'hidden_builtin_profiles = ["summary-fast"]\n',
        encoding="utf-8",
    )

    # Patch run_profiles_file_path to point to our temp TOML
    monkeypatch.setattr(
        "llama_manager.run_profile_store.run_profiles_file_path",
        lambda: toml_path,
    )

    registry = create_tui_profile_registry(Config())
    profile_ids = {p.profile_id for p in registry.profiles}

    assert "summary-fast" not in profile_ids
    assert "summary-balanced" in profile_ids
    assert "qwen35" in profile_ids


def test_tui_registry_includes_all_when_no_hidden(
    xdg_config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TUI registry should include all built-ins when no hidden profile."""
    toml_path = xdg_config_home / "llm-runner" / "run_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "llama_manager.run_profile_store.run_profiles_file_path",
        lambda: toml_path,
    )

    registry = create_tui_profile_registry(Config())
    profile_ids = {p.profile_id for p in registry.profiles}

    assert "summary-balanced" in profile_ids
    assert "summary-fast" in profile_ids
    assert "qwen35" in profile_ids


def test_tui_registry_custom_overrides_hidden_builtin(
    xdg_config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom profile with same profile_id as hidden built-in should appear."""
    toml_path = xdg_config_home / "llm-runner" / "run_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        'hidden_builtin_profiles = ["summary-fast"]\n',
        encoding="utf-8",
    )

    custom_profile = RunProfileSpec(
        profile_id="summary-fast",
        model="/models/custom-summary.gguf",
        alias="summary-fast-custom",
        device="SYCL0",
        port=8081,
        ctx_size=8192,
        ubatch_size=1024,
        threads=4,
        backend="llama_cpp",
    )
    save_custom_run_profile(custom_profile)

    monkeypatch.setattr(
        "llama_manager.run_profile_store.run_profiles_file_path",
        lambda: toml_path,
    )

    registry = create_tui_profile_registry(Config())
    profile_ids = {p.profile_id for p in registry.profiles}

    # Custom profile overrides hidden built-in
    assert "summary-fast" in profile_ids


def test_tui_registry_multiple_hidden(
    xdg_config_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TUI registry should filter multiple hidden built-ins."""
    toml_path = xdg_config_home / "llm-runner" / "run_profiles.toml"
    toml_path.parent.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(
        'hidden_builtin_profiles = ["summary-fast", "qwen35"]\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "llama_manager.run_profile_store.run_profiles_file_path",
        lambda: toml_path,
    )

    registry = create_tui_profile_registry(Config())
    profile_ids = {p.profile_id for p in registry.profiles}

    assert "summary-fast" not in profile_ids
    assert "qwen35" not in profile_ids
    assert "summary-balanced" in profile_ids
