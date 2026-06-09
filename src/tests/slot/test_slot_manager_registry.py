"""Tests for add_slot_from_form with registry parameter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from llama_manager.config import Config, SlotProfileRegistry, SlotProfileSpec
from llama_manager.config.builder import create_default_profile_registry
from llama_manager.orchestration import ServerManager
from llama_manager.slot_manager import add_slot_from_form

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> Config:
    return Config()


@pytest.fixture()
def default_registry(config: Config) -> SlotProfileRegistry:
    return create_default_profile_registry(config)


@pytest.fixture()
def mock_server_manager() -> MagicMock:
    return MagicMock(spec=ServerManager)


@pytest.fixture()
def mock_make_collector() -> Any:
    return MagicMock(return_value=MagicMock())


@pytest.fixture()
def base_state() -> dict[str, Any]:
    """Return a complete runtime-state dict matching the slot_manager expectations."""
    return {
        "log_buffers": {},
        "server_processes": {},
        "slot_states": {},
        "unsaved_slots": set(),
        "slots": [],
    }


@pytest.fixture()
def mock_server_manager_with_upsert(
    mock_server_manager: MagicMock,
    base_state: dict[str, Any],
) -> MagicMock:
    """Server manager that returns a successful upsert result."""
    mock_server_manager.upsert_profile_slot.return_value = (
        True,
        ["Slot added successfully"],
        base_state,
    )
    return mock_server_manager


# ---------------------------------------------------------------------------
# add_slot_from_form — with custom registry
# ---------------------------------------------------------------------------


def test_add_slot_with_custom_registry(
    mock_server_manager: MagicMock,
    mock_make_collector: Any,
    base_state: dict[str, Any],
    config: Config,
) -> None:
    """add_slot_from_form should use the provided custom registry."""
    custom_registry = SlotProfileRegistry(
        profiles=(
            SlotProfileSpec(
                profile_id="custom-profile",
                model="/models/custom.gguf",
                alias="custom",
                device="CUDA:0",
                port=9090,
                ctx_size=4096,
                ubatch_size=512,
                threads=8,
                backend="llama_cpp",
            ),
        ),
    )

    values = {"profile": "custom-profile", "port": ""}
    success, messages, state = add_slot_from_form(
        values=values,
        config=config,
        configs=[],
        gpu_indices=[],
        gpu_stats=[],
        server_manager=mock_server_manager,
        state=base_state,
        registry=custom_registry,
    )

    # The custom profile should be found and used
    assert success is True
    assert any("added" in m.lower() for m in messages)


def test_add_slot_with_custom_registry_rejects_unknown_profile(
    mock_server_manager: MagicMock,
    mock_make_collector: Any,
    base_state: dict[str, Any],
    config: Config,
) -> None:
    """add_slot_from_form should reject profile IDs not in the custom registry."""
    custom_registry = SlotProfileRegistry(
        profiles=(
            SlotProfileSpec(
                profile_id="only-this-one",
                model="/models/only.gguf",
                alias="only",
                device="CUDA:0",
                port=9090,
                ctx_size=4096,
                ubatch_size=512,
                threads=8,
                backend="llama_cpp",
            ),
        ),
    )

    values = {"profile": "unknown-profile", "port": ""}
    success, messages, state = add_slot_from_form(
        values=values,
        config=config,
        configs=[],
        gpu_indices=[],
        gpu_stats=[],
        server_manager=mock_server_manager,
        state=base_state,
        registry=custom_registry,
    )

    assert success is False
    assert any("Unknown profile" in m for m in messages)


def test_add_slot_with_custom_registry_port_override(
    mock_server_manager: MagicMock,
    mock_make_collector: Any,
    base_state: dict[str, Any],
    config: Config,
) -> None:
    """add_slot_from_form should apply port override when provided."""
    custom_registry = SlotProfileRegistry(
        profiles=(
            SlotProfileSpec(
                profile_id="custom-profile",
                model="/models/custom.gguf",
                alias="custom",
                device="CUDA:0",
                port=9090,
                ctx_size=4096,
                ubatch_size=512,
                threads=8,
                backend="llama_cpp",
            ),
        ),
    )

    values = {"profile": "custom-profile", "port": "9999"}
    success, messages, state = add_slot_from_form(
        values=values,
        config=config,
        configs=[],
        gpu_indices=[],
        gpu_stats=[],
        server_manager=mock_server_manager,
        state=base_state,
        registry=custom_registry,
    )

    assert success is True


# ---------------------------------------------------------------------------
# add_slot_from_form — without registry (fallback)
# ---------------------------------------------------------------------------


def test_add_slot_without_registry_uses_defaults(
    mock_server_manager: MagicMock,
    mock_make_collector: Any,
    base_state: dict[str, Any],
    config: Config,
) -> None:
    """add_slot_from_form should fall back to built-in registry when registry is None."""
    values = {"profile": "summary-balanced", "port": ""}
    success, messages, state = add_slot_from_form(
        values=values,
        config=config,
        configs=[],
        gpu_indices=[],
        gpu_stats=[],
        server_manager=mock_server_manager,
        state=base_state,
        registry=None,
    )

    assert success is True


def test_add_slot_without_registry_rejects_unknown_builtin(
    mock_server_manager: MagicMock,
    mock_make_collector: Any,
    base_state: dict[str, Any],
    config: Config,
) -> None:
    """add_slot_from_form should reject unknown profiles even with built-in fallback."""
    values = {"profile": "nonexistent-profile", "port": ""}
    success, messages, state = add_slot_from_form(
        values=values,
        config=config,
        configs=[],
        gpu_indices=[],
        gpu_stats=[],
        server_manager=mock_server_manager,
        state=base_state,
        registry=None,
    )

    assert success is False
    assert any("Unknown profile" in m for m in messages)


def test_add_slot_empty_profile_requires_registry(
    mock_server_manager: MagicMock,
    mock_make_collector: Any,
    base_state: dict[str, Any],
    config: Config,
) -> None:
    """Empty profile value should return failure regardless of registry."""
    values = {"profile": "", "port": ""}
    success, messages, state = add_slot_from_form(
        values=values,
        config=config,
        configs=[],
        gpu_indices=[],
        gpu_stats=[],
        server_manager=mock_server_manager,
        state=base_state,
        registry=None,
    )

    assert success is False
    assert any("Profile is required" in m for m in messages)


def test_add_slot_whitespace_profile_requires_registry(
    mock_server_manager: MagicMock,
    mock_make_collector: Any,
    base_state: dict[str, Any],
    config: Config,
) -> None:
    """Whitespace-only profile value should return failure."""
    values = {"profile": "   ", "port": ""}
    success, messages, state = add_slot_from_form(
        values=values,
        config=config,
        configs=[],
        gpu_indices=[],
        gpu_stats=[],
        server_manager=mock_server_manager,
        state=base_state,
        registry=None,
    )

    assert success is False
    assert any("Profile is required" in m for m in messages)
