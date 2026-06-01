"""Tests for config update application."""

from pathlib import Path
from unittest.mock import patch

from llama_manager import Config, apply_config_updates


class TestApplyConfigUpdates:
    """Tests for apply_config_updates function."""

    def test_valid_updates_applied(self) -> None:
        """Valid updates should be applied successfully."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {"host": "0.0.0.0"},
            persist=False,
        )

        assert result.success is True
        assert "host" in result.updated_fields
        assert cfg.host == "0.0.0.0"
        assert result.errors == []

    def test_integer_field_coercion(self) -> None:
        """Integer fields should be coerced from string input."""
        cfg = Config()
        original = cfg.smoke_listen_timeout_s

        result = apply_config_updates(
            cfg,
            {"smoke_listen_timeout_s": "300"},
            persist=False,
        )

        assert result.success is True
        assert cfg.smoke_listen_timeout_s == 300
        assert cfg.smoke_listen_timeout_s != original

    def test_invalid_integer_field_errors(self) -> None:
        """Invalid integer fields should produce errors."""
        cfg = Config()
        original_timeout = cfg.smoke_listen_timeout_s

        result = apply_config_updates(
            cfg,
            {"smoke_listen_timeout_s": "not-a-number"},
            persist=False,
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "Invalid value" in result.errors[0]
        assert cfg.smoke_listen_timeout_s == original_timeout  # unchanged

    def test_unknown_fields_ignored(self) -> None:
        """Unknown fields should be silently ignored."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {"nonexistent_field": "value"},
            persist=False,
        )

        assert result.success is True
        assert result.updated_fields == []
        assert result.errors == []

    def test_multiple_valid_updates(self) -> None:
        """Multiple valid updates should all be applied."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {
                "host": "0.0.0.0",
                "smoke_listen_timeout_s": "600",
                "models_dir": "/custom/models",
            },
            persist=False,
        )

        assert result.success is True
        assert len(result.updated_fields) == 3
        assert cfg.host == "0.0.0.0"
        assert cfg.smoke_listen_timeout_s == 600
        assert cfg.models_dir == "/custom/models"

    def test_persist_false_skips_file_write(self, tmp_path: Path) -> None:
        """persist=False should not write to disk."""
        cfg = Config()

        with patch("llama_manager.config.persistence.config_file_path") as mock_path:
            mock_path.return_value = tmp_path / "config.toml"
            result = apply_config_updates(cfg, {"host": "0.0.0.0"}, persist=False)

        assert result.success is True
        assert not (tmp_path / "config.toml").exists()

    def test_persist_true_writes_to_disk(self, tmp_path: Path) -> None:
        """persist=True should write to the config file."""
        cfg = Config()

        with patch("llama_manager.config.persistence.config_file_path") as mock_path:
            mock_path.return_value = tmp_path / "config.toml"
            result = apply_config_updates(cfg, {"host": "0.0.0.0"}, persist=True)

        assert result.success is True
        assert (tmp_path / "config.toml").exists()

    def test_empty_updates_returns_success(self) -> None:
        """Empty updates should return success with no changes."""
        cfg = Config()

        result = apply_config_updates(cfg, {}, persist=False)

        assert result.success is True
        assert result.updated_fields == []
        assert result.errors == []

    def test_string_field_preserved_as_string(self) -> None:
        """String fields should be preserved as strings."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {"host": "192.168.1.1"},
            persist=False,
        )

        assert result.success is True
        assert isinstance(cfg.host, str)
        assert cfg.host == "192.168.1.1"

    def test_profile_default_fields_coerced(self) -> None:
        """Profile launch defaults should coerce numeric and bool fields."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {
                "default_batch_size": "1024",
                "default_poll_ms": "0",
                "default_parallel": "4",
                "default_use_jinja": True,
                "default_spec_draft_p_min": "0.25",
            },
            persist=False,
        )

        assert result.success is True
        assert cfg.default_batch_size == 1024
        assert cfg.default_poll_ms == 0
        assert cfg.default_parallel == 4
        assert cfg.default_use_jinja is True
        assert cfg.default_spec_draft_p_min == 0.25
