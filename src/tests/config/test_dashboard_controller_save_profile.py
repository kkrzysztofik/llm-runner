"""Tests for DashboardController.save_run_profile_from_form."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.dashboard_controller import DashboardController, RunProfilePayload


def _make_payload(**overrides: object) -> RunProfilePayload:
    """Build a valid RunProfilePayload with optional overrides."""
    defaults: dict[str, Any] = {
        "profile_id": "my-profile",
        "label": "My Profile",
        "server_bin": "/usr/local/bin/llama-server",
        "model": "/models/model.gguf",
        "port": 8080,
        "ctx_size": 4096,
        "ubatch_size": 512,
        "n_gpu_layers": 99,
        "threads": 8,
        "chat_template_kwargs": "",
    }
    defaults.update(overrides)
    return RunProfilePayload(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def controller() -> DashboardController:
    return DashboardController()


@pytest.fixture()
def valid_payload() -> RunProfilePayload:
    return RunProfilePayload(
        profile_id="test-dc-profile",
        label="My Profile",
        server_bin="/usr/local/bin/llama-server",
        model="/models/model.gguf",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers=99,
        threads=8,
        chat_template_kwargs="",
    )


@pytest.fixture()
def clean_xdg_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure no existing run_profiles.toml interferes with tests."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_save_custom_run_profile() -> Generator[MagicMock]:
    """Mock save_custom_run_profile at its source module.

    The DashboardController.save_run_profile_from_form method does a local import:
        from .run_profile_store import save_custom_run_profile
    Patching the source module ensures the local import picks up the mock.
    Uses yield so the patch context persists through the test body.
    """
    mock = MagicMock(return_value=None)
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        mock,
    ):
        yield mock


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_save_valid_profile(
    controller: DashboardController,
    valid_payload: RunProfilePayload,
    mock_save_custom_run_profile: MagicMock,
    clean_xdg_config: Path,
) -> None:
    """save_run_profile_from_form should return True for valid payload."""
    result = controller.save_run_profile_from_form(valid_payload)
    assert result is True


def test_save_valid_profile_calls_save(
    controller: DashboardController,
    valid_payload: RunProfilePayload,
    mock_save_custom_run_profile: MagicMock,
    clean_xdg_config: Path,
) -> None:
    """save_run_profile_from_form should call save_custom_run_profile."""
    controller.save_run_profile_from_form(valid_payload)
    mock_save_custom_run_profile.assert_called_once()


# ---------------------------------------------------------------------------
# Profile ID validation
# ---------------------------------------------------------------------------


def test_save_empty_profile_id(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for empty profile_id."""
    payload = _make_payload(profile_id="", label="Label")
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_whitespace_profile_id(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for whitespace-only profile_id."""
    payload = _make_payload(profile_id="   ", label="Label")
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_profile_id_normalized(controller: DashboardController) -> None:
    """profile_id should be normalized: spaces→hyphens, upper→lower."""
    payload = _make_payload(profile_id="My Custom Profile", label="My Custom Profile")
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ) as mock_save:
        controller.save_run_profile_from_form(payload)
        call_args = mock_save.call_args
        saved_profile = call_args[0][0]
        assert saved_profile.profile_id == "my-custom-profile"


# ---------------------------------------------------------------------------
# Port validation
# ---------------------------------------------------------------------------


def test_save_invalid_port_too_low(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for port < 1024."""
    payload = _make_payload(profile_id="test", label="Test", port=80)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_invalid_port_zero(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for port == 0."""
    payload = _make_payload(profile_id="test", label="Test", port=0)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_invalid_port_too_high(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for port > 65535."""
    payload = _make_payload(profile_id="test", label="Test", port=70000)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_port_at_boundary(controller: DashboardController) -> None:
    """save_run_profile_from_form should accept port == 1024."""
    payload = _make_payload(profile_id="test", label="Test", port=1024)
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ):
        result = controller.save_run_profile_from_form(payload)
        assert result is True


def test_save_port_at_max(controller: DashboardController) -> None:
    """save_run_profile_from_form should accept port == 65535."""
    payload = _make_payload(profile_id="test", label="Test", port=65535)
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ):
        result = controller.save_run_profile_from_form(payload)
        assert result is True


# ---------------------------------------------------------------------------
# Size validations
# ---------------------------------------------------------------------------


def test_save_nonpositive_ctx_size(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for ctx_size <= 0."""
    payload = _make_payload(profile_id="test", label="Test", ctx_size=0)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_nonpositive_ctx_size_negative(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for negative ctx_size."""
    payload = _make_payload(profile_id="test", label="Test", ctx_size=-100)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_nonpositive_ubatch_size(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for ubatch_size <= 0."""
    payload = _make_payload(profile_id="test", label="Test", ubatch_size=0)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_nonpositive_threads(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for threads <= 0."""
    payload = _make_payload(profile_id="test", label="Test", threads=0)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


# ---------------------------------------------------------------------------
# n_gpu_layers validation
# ---------------------------------------------------------------------------


def test_save_ngpu_layers_all(controller: DashboardController) -> None:
    """save_run_profile_from_form should accept n_gpu_layers == 'all'."""
    payload = _make_payload(profile_id="test", label="Test", n_gpu_layers="all")
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ) as mock_save:
        result = controller.save_run_profile_from_form(payload)
        assert result is True
        call_args = mock_save.call_args
        saved_profile = call_args[0][0]
        assert saved_profile.n_gpu_layers == "all"


def test_save_ngpu_layers_negative(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for negative n_gpu_layers."""
    payload = _make_payload(profile_id="test", label="Test", n_gpu_layers=-1)
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_ngpu_layers_zero(controller: DashboardController) -> None:
    """save_run_profile_from_form should accept n_gpu_layers == 0."""
    payload = _make_payload(profile_id="test", label="Test", n_gpu_layers=0)
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ):
        result = controller.save_run_profile_from_form(payload)
        assert result is True


def test_save_ngpu_layers_string_invalid(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for invalid n_gpu_layers string."""
    payload = _make_payload(profile_id="test", label="Test", n_gpu_layers="invalid")
    result = controller.save_run_profile_from_form(payload)
    assert result is False


# ---------------------------------------------------------------------------
# chat_template_kwargs validation
# ---------------------------------------------------------------------------


def test_save_invalid_json_chat_template(controller: DashboardController) -> None:
    """save_run_profile_from_form should return False for invalid JSON chat_template_kwargs."""
    payload = _make_payload(profile_id="test", label="Test", chat_template_kwargs="{invalid json")
    result = controller.save_run_profile_from_form(payload)
    assert result is False


def test_save_valid_json_chat_template(controller: DashboardController) -> None:
    """save_run_profile_from_form should accept valid JSON chat_template_kwargs."""
    payload = _make_payload(
        profile_id="test", label="Test", chat_template_kwargs='{"key": "value"}'
    )
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ) as mock_save:
        result = controller.save_run_profile_from_form(payload)
        assert result is True
        call_args = mock_save.call_args
        saved_profile = call_args[0][0]
        assert saved_profile.chat_template_kwargs == '{"key": "value"}'


def test_save_dict_chat_template(controller: DashboardController) -> None:
    """save_run_profile_from_form should accept dict chat_template_kwargs."""
    payload = _make_payload(profile_id="test", label="Test", chat_template_kwargs={"key": "value"})
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ) as mock_save:
        result = controller.save_run_profile_from_form(payload)
        assert result is True
        call_args = mock_save.call_args
        saved_profile = call_args[0][0]
        # Dict should be serialized to JSON string
        assert saved_profile.chat_template_kwargs == json.dumps({"key": "value"})


def test_save_empty_chat_template(controller: DashboardController) -> None:
    """save_run_profile_from_form should accept empty chat_template_kwargs."""
    payload = _make_payload(profile_id="test", label="Test", chat_template_kwargs="")
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        return_value=None,
    ):
        result = controller.save_run_profile_from_form(payload)
        assert result is True


# ---------------------------------------------------------------------------
# Duplicate save (ValueError from store)
# ---------------------------------------------------------------------------


def test_save_duplicate_profile_returns_false(
    controller: DashboardController,
    valid_payload: RunProfilePayload,
) -> None:
    """save_run_profile_from_form should return False when store raises ValueError."""
    with patch(
        "llama_manager.run_profile_store.save_custom_run_profile",
        side_effect=ValueError("Duplicate profile_id: my-profile"),
    ):
        result = controller.save_run_profile_from_form(valid_payload)
        assert result is False
