from llama_manager.config import Config
from llama_manager.config_builder import merge_config_overrides


def test_precedence_overrides_win():
    """Overrides should take highest precedence over all other sources."""
    defaults = Config()
    slot_cfg = {"port": 8080}
    profile_cfg = {"port": 8081}
    override_cfg = {"port": 8082}

    result = merge_config_overrides(
        defaults, slot_config=slot_cfg, profile_config=profile_cfg, override_config=override_cfg
    )
    assert result.port == 8082


def test_precedence_profile_wins():
    """Profile should win over slot and workstation configurations."""
    defaults = Config()
    slot_cfg = {"port": 8080}
    profile_cfg = {"port": 8081}

    result = merge_config_overrides(defaults, slot_config=slot_cfg, profile_config=profile_cfg)
    assert result.port == 8081


def test_precedence_slot_wins_over_defaults():
    """Slot/Workstation should win over defaults."""
    defaults = Config()
    # Default is defaults.summary_balanced_port (8080)
    slot_cfg = {"port": 9000}

    result = merge_config_overrides(defaults, slot_config=slot_cfg)
    assert result.port == 9000


def test_deep_merge_dict_fields():
    """Dict fields like chat_template_kwargs should be deep merged."""
    defaults = Config()
    # Assume defaults.summary_balanced_chat_template_kwargs has some base values
    slot_cfg = {"chat_template_kwargs": {"b": "slot_val", "c": "slot_val"}}
    profile_cfg = {"chat_template_kwargs": {"c": "profile_val", "d": "profile_val"}}

    result = merge_config_overrides(defaults, slot_config=slot_cfg, profile_config=profile_cfg)

    # b is only in slot -> should remain
    assert result.chat_template_kwargs["b"] == "slot_val"
    # c is in both -> profile wins
    assert result.chat_template_kwargs["c"] == "profile_val"
    # d is only in profile -> should be present
    assert result.chat_template_kwargs["d"] == "profile_val"
