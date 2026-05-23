"""Tests for DashboardController run-profile CRUD methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llama_cli.tui import DashboardController
from llama_cli.tui.components.run_profile_modal import RunProfilePayload
from llama_manager.config.profiles import RunProfileSpec
from tests.support.helpers import make_server_config


@pytest.fixture()
def mock_controller() -> DashboardController:
    """Create a minimal DashboardController for profile CRUD testing."""
    configs = [make_server_config(alias="summary-balanced")]
    return DashboardController(
        configs=configs,
        gpu_indices=[0],
        slots=None,
        register_signals=False,
    )


# ---------------------------------------------------------------------------
# list_run_profiles
# ---------------------------------------------------------------------------


def test_list_run_profiles_returns_tuples(mock_controller: DashboardController) -> None:
    """list_run_profiles should return list of (spec, source) tuples."""
    with patch.object(mock_controller, "_build_tui_registry") as mock_registry:
        mock_reg = MagicMock()
        mock_reg.profiles = (
            RunProfileSpec(
                profile_id="summary-balanced",
                model="/models/test.gguf",
                alias="summary-balanced",
                device="SYCL0",
                port=8080,
                ctx_size=4096,
                ubatch_size=512,
                threads=8,
                backend="llama_cpp",
            ),
        )
        mock_registry.return_value = mock_reg

        with patch("llama_manager.run_profile_store.custom_profile_exists", return_value=False):
            result = mock_controller.list_run_profiles()

    assert isinstance(result, list)
    assert len(result) == 1
    spec, source = result[0]
    assert isinstance(spec, RunProfileSpec)
    assert source == "builtin"


def test_list_run_profiles_marks_custom() -> None:
    """list_run_profiles should mark custom profiles as 'custom'."""
    configs = [make_server_config(alias="test")]
    ctrl = DashboardController(
        configs=configs,
        gpu_indices=[0],
        register_signals=False,
    )
    with patch.object(ctrl, "_build_tui_registry") as mock_registry:
        mock_reg = MagicMock()
        mock_reg.profiles = (
            RunProfileSpec(
                profile_id="my-custom",
                model="/models/custom.gguf",
                alias="my-custom",
                device="CUDA:0",
                port=9090,
                ctx_size=8192,
                ubatch_size=256,
                threads=4,
                backend="llama_cpp",
            ),
        )
        mock_registry.return_value = mock_reg

        with patch("llama_manager.run_profile_store.custom_profile_exists", return_value=True):
            result = ctrl.list_run_profiles()

    assert len(result) == 1
    _, source = result[0]
    assert source == "custom"


# ---------------------------------------------------------------------------
# _builtin_profile_ids
# ---------------------------------------------------------------------------


def test_builtin_profile_ids_returns_known_ids(mock_controller: DashboardController) -> None:
    """_builtin_profile_ids should return the 3 built-in IDs."""
    ids = mock_controller._builtin_profile_ids()
    assert ids == {"summary-balanced", "summary-fast", "qwen35"}


# ---------------------------------------------------------------------------
# is_profile_in_use
# ---------------------------------------------------------------------------


def test_is_profile_in_use_true(mock_controller: DashboardController) -> None:
    """is_profile_in_use should return True for matching alias."""
    assert mock_controller.is_profile_in_use("summary-balanced") is True


def test_is_profile_in_use_false(mock_controller: DashboardController) -> None:
    """is_profile_in_use should return False for non-matching alias."""
    assert mock_controller.is_profile_in_use("nonexistent") is False


def test_is_profile_in_use_coding_alias(mock_controller: DashboardController) -> None:
    """is_profile_in_use should match qwen35-coding alias."""
    configs = [make_server_config(alias="qwen35-coding")]
    ctrl = DashboardController(
        configs=configs,
        gpu_indices=[0],
        register_signals=False,
    )
    assert ctrl.is_profile_in_use("qwen35") is True


# ---------------------------------------------------------------------------
# update_run_profile
# ---------------------------------------------------------------------------


def test_update_run_profile_valid(mock_controller: DashboardController) -> None:
    """update_run_profile should call upsert with validated spec."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="/models/test.gguf",
        port=9090,
        ctx_size=8192,
        ubatch_size=1024,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
        device="SYCL0",
    )

    with patch(
        "llama_manager.run_profile_store.upsert_custom_run_profile",
    ) as mock_upsert:
        result = mock_controller.update_run_profile("original-id", payload)

    assert result is True
    mock_upsert.assert_called_once()
    call_args = mock_upsert.call_args
    assert call_args[0][0] == "original-id"
    assert call_args[0][1].profile_id == "my-profile"
    assert call_args[0][1].model == "/models/test.gguf"
    assert call_args[0][1].port == 9090
    assert call_args[0][1].device == "SYCL0"


def test_update_run_profile_empty_device(mock_controller: DashboardController) -> None:
    """update_run_profile should return False for empty device."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="/models/test.gguf",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
        device="",
    )

    result = mock_controller.update_run_profile("original-id", payload)
    assert result is False


def test_update_run_profile_device_passed_through(mock_controller: DashboardController) -> None:
    """update_run_profile should pass device through to the spec."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="/models/test.gguf",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
        device="CUDA:1",
    )

    with patch(
        "llama_manager.run_profile_store.upsert_custom_run_profile",
    ) as mock_upsert:
        result = mock_controller.update_run_profile("original-id", payload)

    assert result is True
    call_args = mock_upsert.call_args
    assert call_args[0][1].device == "CUDA:1"


def test_update_run_profile_invalid_port(mock_controller: DashboardController) -> None:
    """update_run_profile should return False for invalid port."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="/models/test.gguf",
        port=70000,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
    )

    result = mock_controller.update_run_profile("original-id", payload)
    assert result is False


def test_update_run_profile_port_too_low(mock_controller: DashboardController) -> None:
    """update_run_profile should return False for port below 1024."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="/models/test.gguf",
        port=500,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
    )

    result = mock_controller.update_run_profile("original-id", payload)
    assert result is False


def test_update_run_profile_empty_profile_id(mock_controller: DashboardController) -> None:
    """update_run_profile should return False for empty profile_id."""
    payload = RunProfilePayload(
        profile_id="",
        label="My Profile",
        model="/models/test.gguf",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
    )

    result = mock_controller.update_run_profile("original-id", payload)
    assert result is False


def test_update_run_profile_empty_model(mock_controller: DashboardController) -> None:
    """update_run_profile should return False for empty model."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
    )

    result = mock_controller.update_run_profile("original-id", payload)
    assert result is False


def test_update_run_profile_invalid_ctx_size(mock_controller: DashboardController) -> None:
    """update_run_profile should return False for non-positive ctx_size."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="/models/test.gguf",
        port=8080,
        ctx_size=0,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
    )

    result = mock_controller.update_run_profile("original-id", payload)
    assert result is False


def test_update_run_profile_upsert_raises_returns_false(
    mock_controller: DashboardController,
) -> None:
    """update_run_profile should return False when upsert raises ValueError."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        model="/models/test.gguf",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        n_gpu_layers="all",
        threads=8,
        chat_template_kwargs="",
    )

    with patch(
        "llama_manager.run_profile_store.upsert_custom_run_profile",
        side_effect=ValueError("Duplicate profile_id: my-profile"),
    ):
        result = mock_controller.update_run_profile("original-id", payload)

    assert result is False


# ---------------------------------------------------------------------------
# delete_run_profile
# ---------------------------------------------------------------------------


def test_delete_run_profile_success(mock_controller: DashboardController) -> None:
    """delete_run_profile should call store delete and return True."""
    with (
        patch.object(mock_controller, "is_profile_in_use", return_value=False),
        patch(
            "llama_manager.run_profile_store.delete_custom_run_profile",
            return_value=True,
        ) as mock_delete,
    ):
        result = mock_controller.delete_run_profile("summary-balanced")

    assert result is True
    mock_delete.assert_called_once_with(
        "summary-balanced", {"summary-balanced", "summary-fast", "qwen35"}
    )


def test_delete_run_profile_in_use(mock_controller: DashboardController) -> None:
    """delete_run_profile should return False when profile is in use."""
    with (
        patch.object(mock_controller, "is_profile_in_use", return_value=True),
        patch("llama_manager.run_profile_store.delete_custom_run_profile"),
    ):
        result = mock_controller.delete_run_profile("summary-balanced")

    assert result is False


def test_delete_run_profile_not_found(mock_controller: DashboardController) -> None:
    """delete_run_profile should return False when store returns False."""
    with (
        patch.object(mock_controller, "is_profile_in_use", return_value=False),
        patch(
            "llama_manager.run_profile_store.delete_custom_run_profile",
            return_value=False,
        ),
    ):
        result = mock_controller.delete_run_profile("nonexistent")

    assert result is False


def test_delete_run_profile_exception_returns_false(
    mock_controller: DashboardController,
) -> None:
    """delete_run_profile should return False on exception."""
    with (
        patch.object(mock_controller, "is_profile_in_use", return_value=False),
        patch(
            "llama_manager.run_profile_store.delete_custom_run_profile",
            side_effect=RuntimeError("store error"),
        ),
    ):
        result = mock_controller.delete_run_profile("summary-balanced")

    assert result is False


# ---------------------------------------------------------------------------
# Model index delegation
# ---------------------------------------------------------------------------


def test_load_model_index_delegates(mock_controller: DashboardController) -> None:
    """load_model_index should delegate to llama_manager.load_model_index."""
    with patch(
        "llama_cli.tui.controller.load_model_index",
    ) as mock_load:
        mock_load.return_value = []
        result = mock_controller.load_model_index()

    mock_load.assert_called_once_with(mock_controller.config)
    assert result == []


def test_refresh_model_index_delegates(mock_controller: DashboardController) -> None:
    """refresh_model_index should delegate to llama_manager.refresh_model_index."""
    with patch(
        "llama_cli.tui.controller.refresh_model_index",
    ) as mock_refresh:
        mock_refresh.return_value = ([], 0, 0)
        result = mock_controller.refresh_model_index()

    mock_refresh.assert_called_once_with(mock_controller.config)
    assert result == ([], 0, 0)


def test_model_index_path_delegates(mock_controller: DashboardController) -> None:
    """model_index_path should delegate and return a string."""
    with patch(
        "llama_cli.tui.controller.model_index_path",
    ) as mock_path:
        mock_path.return_value = "/tmp/idx.json"
        result = mock_controller.model_index_path()

    mock_path.assert_called_once_with(mock_controller.config)
    assert isinstance(result, str)
    assert result == "/tmp/idx.json"
