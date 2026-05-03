"""Characterization tests for apply_profile_overrides.

Locks in the behavior of the extracted _apply_profile_overrides logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import Config, ServerConfig
from llama_manager.config.builder import apply_profile_overrides
from tests.support.factories import make_server_config


def _make_config(alias: str = "test", backend: str = "llama_cpp") -> ServerConfig:
    """Helper to create a ServerConfig for tests."""
    return make_server_config(
        alias=alias,
        backend=backend,
        port=8080,
        model="/models/test.gguf",
        device="SYCL0",
        bind_address="127.0.0.1",
        server_bin="dummy-server",
        tensor_split="",
        reasoning_mode="off",
        reasoning_format="deepseek",
        chat_template_kwargs="{}",
        reasoning_budget="",
        use_jinja=True,
        n_gpu_layers=99,
        risky_acknowledged=[],
    )


@patch("llama_manager.config.builder.get_gpu_identifier")
@patch("llama_manager.config.builder.load_profile_with_staleness")
@patch("llama_manager.config.builder.profile_to_override_dict")
def test_exception_during_profile_load(
    mock_profile_to_override_dict: MagicMock,
    mock_load_profile: MagicMock,
    mock_get_gpu_id: MagicMock,
) -> None:
    """Unexpected exceptions during profile loading are re-raised."""
    mock_get_gpu_id.return_value = "test-gpu"
    mock_load_profile.side_effect = RuntimeError("disk error")

    cfg = _make_config(alias="summary-balanced")
    base = Config()
    get_driver = MagicMock(return_value="1.0")

    with pytest.raises(RuntimeError, match="disk error"):
        apply_profile_overrides([cfg], base, get_driver)

    mock_profile_to_override_dict.assert_not_called()


@patch("llama_manager.config.builder.get_gpu_identifier")
@patch("llama_manager.config.builder.load_profile_with_staleness")
@patch("llama_manager.config.builder.profile_to_override_dict")
def test_no_profile_record(
    mock_profile_to_override_dict: MagicMock,
    mock_load_profile: MagicMock,
    mock_get_gpu_id: MagicMock,
) -> None:
    """When record is None, message indicates no profile found."""
    mock_get_gpu_id.return_value = "test-gpu"
    mock_load_profile.return_value = (None, None)

    cfg = _make_config(alias="summary-balanced")
    base = Config()
    get_driver = MagicMock(return_value="1.0")

    updated, messages = apply_profile_overrides([cfg], base, get_driver)

    assert len(updated) == 1
    assert updated[0] is cfg
    assert messages == ["No profile found for summary-balanced; using defaults"]
    mock_profile_to_override_dict.assert_not_called()


@patch("llama_manager.config.builder.get_gpu_identifier")
@patch("llama_manager.config.builder.load_profile_with_staleness")
@patch("llama_manager.config.builder.profile_to_override_dict")
def test_stale_profile(
    mock_profile_to_override_dict: MagicMock,
    mock_load_profile: MagicMock,
    mock_get_gpu_id: MagicMock,
) -> None:
    """When profile is stale, message includes reasons and config is unchanged."""
    from llama_manager.config.profile_cache import StalenessReason, StalenessResult

    mock_get_gpu_id.return_value = "test-gpu"
    mock_load_profile.return_value = (
        MagicMock(),
        StalenessResult(
            is_stale=True,
            reasons=[StalenessReason.DRIVER_CHANGED, StalenessReason.AGE_EXCEEDED],
        ),
    )

    cfg = _make_config(alias="summary-balanced")
    base = Config()
    get_driver = MagicMock(return_value="1.0")

    updated, messages = apply_profile_overrides([cfg], base, get_driver)

    assert len(updated) == 1
    assert updated[0] is cfg
    assert len(messages) == 1
    assert "Profile stale for summary-balanced" in messages[0]
    assert "Driver Changed" in messages[0]
    assert "Age Exceeded" in messages[0]
    mock_profile_to_override_dict.assert_not_called()


@patch("llama_manager.config.builder.get_gpu_identifier")
@patch("llama_manager.config.builder.load_profile_with_staleness")
@patch("llama_manager.config.builder.profile_to_override_dict")
def test_empty_profile_overrides(
    mock_profile_to_override_dict: MagicMock,
    mock_load_profile: MagicMock,
    mock_get_gpu_id: MagicMock,
) -> None:
    """When profile overrides dict is empty, message indicates empty profile."""
    from llama_manager.config.profile_cache import StalenessResult

    mock_get_gpu_id.return_value = "test-gpu"
    mock_load_profile.return_value = (MagicMock(), StalenessResult(is_stale=False))
    mock_profile_to_override_dict.return_value = {}

    cfg = _make_config(alias="summary-balanced")
    base = Config()
    get_driver = MagicMock(return_value="1.0")

    updated, messages = apply_profile_overrides([cfg], base, get_driver)

    assert len(updated) == 1
    assert updated[0] is cfg
    assert messages == ["Profile empty for summary-balanced; using defaults"]


@patch("llama_manager.config.builder.get_gpu_identifier")
@patch("llama_manager.config.builder.load_profile_with_staleness")
@patch("llama_manager.config.builder.profile_to_override_dict")
def test_fresh_profile_applied(
    mock_profile_to_override_dict: MagicMock,
    mock_load_profile: MagicMock,
    mock_get_gpu_id: MagicMock,
) -> None:
    """When profile is fresh, overrides are applied and identity fields preserved."""
    from llama_manager.config.profile_cache import StalenessResult

    mock_get_gpu_id.return_value = "test-gpu"
    mock_load_profile.return_value = (MagicMock(), StalenessResult(is_stale=False))
    mock_profile_to_override_dict.return_value = {"threads": 16, "ctx_size": 8192}

    cfg = _make_config(alias="summary-balanced")
    base = Config()
    get_driver = MagicMock(return_value="1.0")

    updated, messages = apply_profile_overrides([cfg], base, get_driver)

    assert len(updated) == 1
    merged = updated[0]
    assert merged is not cfg
    assert merged.threads == 16
    assert merged.ctx_size == 8192

    # Identity fields preserved from original config
    assert merged.model == cfg.model
    assert merged.alias == cfg.alias
    assert merged.device == cfg.device
    assert merged.port == cfg.port
    assert merged.bind_address == cfg.bind_address
    assert merged.server_bin == cfg.server_bin
    assert merged.backend == cfg.backend
    assert merged.tensor_split == cfg.tensor_split
    assert merged.reasoning_mode == cfg.reasoning_mode
    assert merged.reasoning_format == cfg.reasoning_format
    assert merged.chat_template_kwargs == cfg.chat_template_kwargs
    assert merged.reasoning_budget == cfg.reasoning_budget
    assert merged.use_jinja == cfg.use_jinja
    assert merged.n_gpu_layers == cfg.n_gpu_layers
    assert merged.risky_acknowledged == cfg.risky_acknowledged

    assert len(messages) == 1
    assert "Applied profile: summary-balanced (balanced)" in messages[0]
    assert "threads=16" in messages[0]
    assert "ctx=8192" in messages[0]


@patch("llama_manager.config.builder.get_gpu_identifier")
@patch("llama_manager.config.builder.load_profile_with_staleness")
@patch("llama_manager.config.builder.profile_to_override_dict")
def test_mixed_configs(
    mock_profile_to_override_dict: MagicMock,
    mock_load_profile: MagicMock,
    mock_get_gpu_id: MagicMock,
) -> None:
    """Multiple configs with different profile outcomes are handled independently."""
    from llama_manager.config.profile_cache import StalenessResult

    mock_get_gpu_id.return_value = "test-gpu"

    def _load_side_effect(
        *,
        profiles_dir,
        gpu_identifier,
        backend,
        flavor,
        current_driver_version,
        current_binary_version,
        staleness_days,
    ):
        # Return different results based on backend to simulate mixed outcomes
        if backend == "llama_cpp":
            return (MagicMock(), StalenessResult(is_stale=False))
        return (None, None)

    mock_load_profile.side_effect = _load_side_effect
    mock_profile_to_override_dict.return_value = {"threads": 12}

    cfg1 = _make_config(alias="summary-balanced", backend="llama_cpp")
    cfg2 = _make_config(alias="qwen35-coding", backend="cuda")
    base = Config()
    get_driver = MagicMock(return_value="1.0")

    updated, messages = apply_profile_overrides([cfg1, cfg2], base, get_driver)

    assert len(updated) == 2
    assert updated[0].threads == 12
    assert updated[1] is cfg2
    assert any("Applied profile" in m for m in messages)
    assert any("No profile found" in m for m in messages)
