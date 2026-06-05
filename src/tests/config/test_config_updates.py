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
            {"deployment.host": "0.0.0.0"},
            persist=False,
        )

        assert result.success is True
        assert "deployment.host" in result.updated_fields
        assert cfg.deployment.host == "0.0.0.0"
        assert result.errors == []

    def test_integer_field_coercion(self) -> None:
        """Integer fields should be coerced from string input."""
        cfg = Config()
        original = cfg.smoke.listen_timeout_s

        result = apply_config_updates(
            cfg,
            {"smoke.listen_timeout_s": "300"},
            persist=False,
        )

        assert result.success is True
        assert cfg.smoke.listen_timeout_s == 300
        assert cfg.smoke.listen_timeout_s != original

    def test_invalid_integer_field_errors(self) -> None:
        """Invalid integer fields should produce errors."""
        cfg = Config()
        original_timeout = cfg.smoke.listen_timeout_s

        result = apply_config_updates(
            cfg,
            {"smoke.listen_timeout_s": "not-a-number"},
            persist=False,
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "Invalid value" in result.errors[0]
        assert cfg.smoke.listen_timeout_s == original_timeout  # unchanged

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
                "deployment.host": "0.0.0.0",
                "smoke.listen_timeout_s": "600",
                "paths.models_dir": "/custom/models",
            },
            persist=False,
        )

        assert result.success is True
        assert len(result.updated_fields) == 3
        assert cfg.deployment.host == "0.0.0.0"
        assert cfg.smoke.listen_timeout_s == 600
        assert cfg.paths.models_dir == "/custom/models"

    def test_persist_false_skips_file_write(self, tmp_path: Path) -> None:
        """persist=False should not write to disk."""
        cfg = Config()

        with patch("llama_manager.config.persistence.config_file_path") as mock_path:
            mock_path.return_value = tmp_path / "config.toml"
            result = apply_config_updates(cfg, {"deployment.host": "0.0.0.0"}, persist=False)

        assert result.success is True
        assert not (tmp_path / "config.toml").exists()

    def test_persist_true_writes_to_disk(self, tmp_path: Path) -> None:
        """persist=True should write to the config file."""
        cfg = Config()

        with patch("llama_manager.config.persistence.config_file_path") as mock_path:
            mock_path.return_value = tmp_path / "config.toml"
            result = apply_config_updates(cfg, {"deployment.host": "0.0.0.0"}, persist=True)

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
            {"deployment.host": "192.168.1.1"},
            persist=False,
        )

        assert result.success is True
        assert isinstance(cfg.deployment.host, str)
        assert cfg.deployment.host == "192.168.1.1"

    def test_profile_default_fields_coerced(self) -> None:
        """Profile launch defaults should coerce numeric and bool fields."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {
                "server_defaults.batch_size": "1024",
                "server_defaults.poll_ms": "0",
                "server_defaults.parallel": "4",
                "server_defaults.use_jinja": True,
                "server_defaults.spec_draft_p_min": "0.25",
            },
            persist=False,
        )

        assert result.success is True
        assert cfg.server_defaults.batch_size == 1024
        assert cfg.server_defaults.poll_ms == 0
        assert cfg.server_defaults.parallel == 4
        assert cfg.server_defaults.use_jinja is True
        assert cfg.server_defaults.spec_draft_p_min == 0.25

    def test_invalid_float_field_errors(self) -> None:
        """Invalid float fields should produce errors."""
        cfg = Config()
        original = cfg.server_defaults.spec_draft_p_min

        result = apply_config_updates(
            cfg,
            {"server_defaults.spec_draft_p_min": "not-a-float"},
            persist=False,
        )

        assert result.success is False
        assert len(result.errors) > 0
        assert "Invalid value" in result.errors[0]
        assert cfg.server_defaults.spec_draft_p_min == original

    def test_bool_field_from_string(self) -> None:
        """Bool fields should coerce common truthy strings."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {"server_defaults.use_jinja": "yes"},
            persist=False,
        )

        assert result.success is True
        assert cfg.server_defaults.use_jinja is True

    def test_bool_field_from_non_string(self) -> None:
        """Bool fields should coerce non-string truthy values."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {"server_defaults.use_jinja": 1},
            persist=False,
        )

        assert result.success is True
        assert cfg.server_defaults.use_jinja is True

    def test_bool_field_rejects_invalid_token(self) -> None:
        """Bool fields should reject unrecognized string tokens."""
        cfg = Config()

        result = apply_config_updates(
            cfg,
            {"server_defaults.use_jinja": "tru"},
            persist=False,
        )

        assert result.success is False
        assert any("server_defaults.use_jinja" in err for err in result.errors)
        assert cfg.server_defaults.use_jinja is False

    def test_persist_oserror_appends_error(self, tmp_path: Path) -> None:
        """OSError during persist should append an error message."""
        cfg = Config()

        with patch("llama_manager.config.persistence.config_file_path") as mock_path:
            mock_path.return_value = tmp_path / "config.toml"
            with patch(
                "llama_manager.config.persistence.save_config_to_file",
                side_effect=OSError("disk full"),
            ):
                result = apply_config_updates(cfg, {"deployment.host": "0.0.0.0"}, persist=True)

        assert result.success is False
        assert any("Config save failed" in err for err in result.errors)
