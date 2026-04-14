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
    slot_cfg: dict[str, object] = {"port": 8080}
    profile_cfg: dict[str, object] = {"port": 8081}

    result = merge_config_overrides(defaults, slot_config=slot_cfg, profile_config=profile_cfg)
    assert result.port == 8081


def test_precedence_slot_wins_over_defaults() -> None:
    """Slot/Workstation should win over defaults."""
    defaults = Config()
    # Default is defaults.summary_balanced_port (8080)
    slot_cfg: dict[str, object] = {"port": 9000}

    result = merge_config_overrides(defaults, slot_config=slot_cfg)
    assert result.port == 9000


def test_deep_merge_dict_fields() -> None:
    """Dict fields in config overrides should be deep merged into the base dict."""
    defaults = Config()
    # List fields concatenate across merge layers
    slot_cfg: dict[str, list[str]] = {"risky_acknowledged": ["slot-risk"]}
    profile_cfg: dict[str, list[str]] = {"risky_acknowledged": ["profile-risk"]}

    result = merge_config_overrides(defaults, slot_config=slot_cfg, profile_config=profile_cfg)

    assert result.risky_acknowledged == ["slot-risk", "profile-risk"]


def test_list_fields_concatenate_across_layers() -> None:
    """List fields should concatenate in precedence merge."""
    defaults = Config()
    slot_cfg: dict[str, list[str]] = {"risky_acknowledged": ["slot-risk"]}
    profile_cfg: dict[str, list[str]] = {"risky_acknowledged": ["profile-risk"]}
    override_cfg: dict[str, list[str]] = {"risky_acknowledged": ["override-risk"]}

    result = merge_config_overrides(
        defaults,
        slot_config=slot_cfg,
        profile_config=profile_cfg,
        override_config=override_cfg,
    )
    assert result.risky_acknowledged == ["slot-risk", "profile-risk", "override-risk"]


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


def test_model_path_validation_only_when_model_is_overridden(monkeypatch):
    defaults = Config()
    monkeypatch.setattr("os.path.exists", lambda _: False)

    # Should not validate defaults-only model path
    merge_config_overrides(defaults, override_config={"port": 8088})

    # Should validate when model path is explicitly overridden
    with pytest.raises(ValueError) as exc:
        merge_config_overrides(defaults, override_config={"port": 8088, "model": "/missing.gguf"})
    assert "model path not found" in str(exc.value)
