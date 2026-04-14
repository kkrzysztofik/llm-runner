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
    # Use a helper to convert ServerConfig to dict for JSON dumping
    def to_dict(cfg: object) -> dict[str, object]:
        return {k: getattr(cfg, k) for k in cfg.__dict__}  # type: ignore[attr-defined]

    assert json.dumps(to_dict(res1), sort_keys=True) == json.dumps(to_dict(res2), sort_keys=True)


def test_command_generation_is_deterministic() -> None:
    """Repeated command generation for the same config must be identical."""
    defaults = Config()
    cfg = merge_config_overrides(defaults, slot_config={"port": 8080})

    cmd1 = build_server_cmd(cfg)
    cmd2 = build_server_cmd(cfg)

    assert cmd1 == cmd2


def test_repeated_runs_produce_identical_artifacts() -> None:
    """Repeated runs produce matching artifact fields."""
    # This would test the artifact generation logic
    # For now, we check if the config resolution (the source of artifact fields) is stable
    defaults = Config()
    args: dict[str, dict[str, object]] = {"slot_config": {"port": 8080}}

    res1 = merge_config_overrides(defaults, **args)
    res2 = merge_config_overrides(defaults, **args)

    assert res1.port == res2.port
    assert res1.model == res2.model
