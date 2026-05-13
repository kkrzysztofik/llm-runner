"""Unit tests for llama_manager.config.persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_manager.config import Config
from llama_manager.config.persistence import (
    _ENV_OVERRIDES,
    _PERSISTED_FIELDS,
    build_config,
    config_file_path,
    load_config_overrides_from_file,
    save_config_to_file,
)

# ---------------------------------------------------------------------------
# config_file_path
# ---------------------------------------------------------------------------


def test_config_file_path_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = config_file_path()
    assert result == tmp_path / "llm-runner" / "config.toml"


def test_config_file_path_fallback_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = config_file_path()
    assert result == tmp_path / ".config" / "llm-runner" / "config.toml"


# ---------------------------------------------------------------------------
# load_config_overrides_from_file
# ---------------------------------------------------------------------------


def test_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    result = load_config_overrides_from_file(tmp_path / "nonexistent.toml")
    assert result == {}


def test_load_returns_only_known_keys(tmp_path: Path) -> None:
    toml_path = tmp_path / "config.toml"
    toml_path.write_bytes(
        b'llama_cpp_root = "/my/llama"\nunknown_field = "ignored"\nhost = "0.0.0.0"\n'
    )
    result = load_config_overrides_from_file(toml_path)
    assert result == {"llama_cpp_root": "/my/llama", "host": "0.0.0.0"}
    assert "unknown_field" not in result


def test_load_integer_fields_preserve_type(tmp_path: Path) -> None:
    toml_path = tmp_path / "config.toml"
    toml_path.write_bytes(b"smoke_listen_timeout_s = 60\n")
    result = load_config_overrides_from_file(toml_path)
    assert result["smoke_listen_timeout_s"] == 60
    assert isinstance(result["smoke_listen_timeout_s"], int)


# ---------------------------------------------------------------------------
# save_config_to_file
# ---------------------------------------------------------------------------


def test_roundtrip_save_load(tmp_path: Path) -> None:
    cfg = Config(
        llama_cpp_root="/roundtrip/llama",
        models_dir="/roundtrip/models",
        host="10.0.0.1",
        build_git_branch="dev",
        smoke_listen_timeout_s=99,
    )
    path = tmp_path / "config.toml"
    save_config_to_file(cfg, path)

    loaded = load_config_overrides_from_file(path)
    assert loaded["llama_cpp_root"] == "/roundtrip/llama"
    assert loaded["models_dir"] == "/roundtrip/models"
    assert loaded["host"] == "10.0.0.1"
    assert loaded["build_git_branch"] == "dev"
    assert loaded["smoke_listen_timeout_s"] == 99


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "config.toml"
    save_config_to_file(Config(), path)
    assert path.exists()


def test_save_writes_all_persisted_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    save_config_to_file(Config(), path)
    loaded = load_config_overrides_from_file(path)
    for field in _PERSISTED_FIELDS:
        assert field in loaded, f"Field '{field}' missing from saved TOML"


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


def test_build_config_uses_file_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)

    cfg_to_save = Config(llama_cpp_root="/from/file", host="192.168.1.1")
    save_config_to_file(cfg_to_save, config_file_path())

    result = build_config()
    assert result.llama_cpp_root == "/from/file"
    assert result.host == "192.168.1.1"


def test_env_var_overrides_file_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("LLAMA_CPP_ROOT", "/from/env")

    cfg_to_save = Config(llama_cpp_root="/from/file")
    save_config_to_file(cfg_to_save, config_file_path())

    result = build_config()
    assert result.llama_cpp_root == "/from/env"


def test_build_config_missing_file_uses_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty"))
    monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)
    monkeypatch.delenv("MODELS_DIR", raising=False)

    result = build_config()
    # Should produce a valid Config with hard-coded defaults
    assert isinstance(result, Config)
    assert result.build_git_remote == "https://github.com/ggerganov/llama.cpp.git"


def test_models_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("MODELS_DIR", "/env/models")

    cfg_to_save = Config(models_dir="/file/models")
    save_config_to_file(cfg_to_save, config_file_path())

    result = build_config()
    assert result.models_dir == "/env/models"


def test_env_overrides_dict_is_consistent() -> None:
    """Every env override key must be a persisted field."""
    for field_name in _ENV_OVERRIDES:
        assert field_name in _PERSISTED_FIELDS, (
            f"_ENV_OVERRIDES key '{field_name}' is not in _PERSISTED_FIELDS"
        )
