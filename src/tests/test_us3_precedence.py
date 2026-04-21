from unittest.mock import patch

import pytest

from llama_manager.config import Config
from llama_manager.config_builder import merge_config_overrides


def test_precedence_overrides_win() -> None:
    """Overrides should take highest precedence over all other sources."""
    defaults = Config()
    slot_cfg: dict[str, object] = {"port": 8080}
    profile_cfg: dict[str, object] = {"port": 8081}
    override_cfg: dict[str, object] = {"port": 8082}

    result = merge_config_overrides(
        defaults, slot_config=slot_cfg, profile_config=profile_cfg, override_config=override_cfg
    )
    assert result.port == 8082


def test_precedence_profile_wins() -> None:
    """Profile should win over slot and workstation configurations."""
    defaults = Config()
    slot_cfg: dict[str, object] = {"threads": 4}
    profile_cfg: dict[str, object] = {"threads": 24}

    result = merge_config_overrides(defaults, slot_config=slot_cfg, profile_config=profile_cfg)
    assert result.threads == 24


def test_precedence_slot_wins_over_defaults() -> None:
    """Slot/Workstation should win over defaults."""
    defaults = Config()
    # Default port is 8080
    slot_cfg: dict[str, object] = {"port": 9000}

    result = merge_config_overrides(defaults, slot_config=slot_cfg)
    assert result.port == 9000


def test_scalar_field_precedence() -> None:
    """Scalar fields in profile_config take precedence over slot_config."""
    defaults = Config()
    slot_cfg: dict[str, object] = {"threads": 4}
    profile_cfg: dict[str, object] = {"threads": 12}

    result = merge_config_overrides(defaults, slot_config=slot_cfg, profile_config=profile_cfg)

    assert result.threads == 12


def test_list_fields_concatenate_across_layers() -> None:
    """List fields should concatenate across merge layers (slot + override)."""
    defaults = Config()
    slot_cfg: dict[str, list[str]] = {"risky_acknowledged": ["slot-risk"]}
    override_cfg: dict[str, list[str]] = {"risky_acknowledged": ["override-risk"]}

    result = merge_config_overrides(
        defaults,
        slot_config=slot_cfg,
        override_config=override_cfg,
    )
    assert result.risky_acknowledged == ["slot-risk", "override-risk"]


def test_non_whitelisted_profile_keys_ignored() -> None:
    """Non-whitelisted keys in profile_config should be silently ignored."""
    defaults = Config()
    profile_cfg: dict[str, object] = {
        "port": 9999,  # NOT in PROFILE_OVERRIDE_FIELDS
        "n_gpu_layers": 100,  # NOT in PROFILE_OVERRIDE_FIELDS
        "threads": 32,  # IS in PROFILE_OVERRIDE_FIELDS
    }

    result = merge_config_overrides(defaults, profile_config=profile_cfg)

    # threads should be applied (whitelisted)
    assert result.threads == 32
    # port should NOT be overridden by profile (not whitelisted)
    assert result.port == defaults.summary_balanced_port
    # n_gpu_layers should NOT be overridden by profile (not whitelisted)
    assert result.n_gpu_layers == defaults.default_n_gpu_layers


def test_merge_validates_port_range() -> None:
    defaults = Config()
    with pytest.raises(ValueError) as exc:
        merge_config_overrides(defaults, override_config={"port": 1023})
    assert "port must be between 1024 and 65535" in str(exc.value)


def test_merge_validates_threads_positive() -> None:
    defaults = Config()
    with pytest.raises(ValueError) as exc:
        merge_config_overrides(defaults, override_config={"threads": 0})
    assert "threads must be greater than 0" in str(exc.value)


def test_model_path_validation_only_when_model_is_overridden() -> None:
    defaults = Config()
    with patch("llama_manager.config_builder.os.path.exists", return_value=False):
        # Should not validate defaults-only model path
        merge_config_overrides(defaults, override_config={"port": 8088})

    # Should validate when model path is explicitly overridden
    with pytest.raises(ValueError) as exc:
        merge_config_overrides(defaults, override_config={"port": 8088, "model": "/missing.gguf"})
    assert "model path not found" in str(exc.value)
