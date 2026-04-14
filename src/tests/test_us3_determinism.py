import json

from llama_manager.config import Config
from llama_manager.config_builder import merge_config_overrides
from llama_manager.server import build_server_cmd


def test_resolution_is_deterministic() -> None:
    """Repeated calls to resolve_config (via merge_config_overrides) with identical inputs must produce identical results."""
    defaults = Config()
    slot_cfg: dict[str, object] = {"port": 8080, "ctx_size": 32000}
    profile_cfg: dict[str, object] = {"threads": 12}
    override_cfg: dict[str, object] = {"port": 9000}

    # First resolution
    res1 = merge_config_overrides(
        defaults, slot_config=slot_cfg, profile_config=profile_cfg, override_config=override_cfg
    )

    # Second resolution
    res2 = merge_config_overrides(
        defaults, slot_config=slot_cfg, profile_config=profile_cfg, override_config=override_cfg
    )

    # Verify exact equality of ServerConfig fields
    assert res1 == res2

    # Verify JSON serialization is identical
    def to_dict(cfg: object) -> dict[str, object]:
        return dict(vars(cfg))

    assert json.dumps(to_dict(res1), sort_keys=True) == json.dumps(to_dict(res2), sort_keys=True)


def test_command_generation_is_deterministic() -> None:
    """Repeated command generation for the same config must be identical."""
    defaults = Config()
    cfg = merge_config_overrides(defaults, slot_config={"port": 8080})

    cmd1 = build_server_cmd(cfg)
    cmd2 = build_server_cmd(cfg)

    assert cmd1 == cmd2
