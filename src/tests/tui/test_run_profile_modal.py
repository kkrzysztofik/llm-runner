"""Tests for RunProfilePayload dataclass and RunProfileModal value collection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App
from textual.widgets import Checkbox, Collapsible, Input, ListView, Select

from llama_cli.tui.components.run_profile_modal import (
    RunProfileModal,
    RunProfilePayload,
)
from llama_manager.config import Config
from llama_manager.config.profiles import RunProfileSpec
from llama_manager.model_index import ModelIndexEntry

# ---------------------------------------------------------------------------
# RunProfilePayload defaults
# ---------------------------------------------------------------------------


def test_payload_defaults() -> None:
    """RunProfilePayload should have correct default values."""
    payload = RunProfilePayload()

    assert payload.profile_id == ""
    assert payload.label == ""
    assert payload.server_bin == ""
    assert payload.model == ""
    assert payload.port == 8080
    assert payload.ctx_size == 4096
    assert payload.ubatch_size == 512
    assert payload.n_gpu_layers == "all"
    assert payload.threads == 8
    assert payload.chat_template_kwargs == ""
    assert payload.save_and_add_slot is False


def test_payload_custom_values() -> None:
    """RunProfilePayload should preserve custom values."""
    payload = RunProfilePayload(
        profile_id="my-profile",
        label="My Profile",
        server_bin="/usr/local/bin/llama-server",
        model="/models/model.gguf",
        port=9090,
        ctx_size=8192,
        ubatch_size=1024,
        n_gpu_layers=99,
        threads=16,
        chat_template_kwargs='{"key": "val"}',
        save_and_add_slot=True,
    )

    assert payload.profile_id == "my-profile"
    assert payload.label == "My Profile"
    assert payload.server_bin == "/usr/local/bin/llama-server"
    assert payload.model == "/models/model.gguf"
    assert payload.port == 9090
    assert payload.ctx_size == 8192
    assert payload.ubatch_size == 1024
    assert payload.n_gpu_layers == 99
    assert payload.threads == 16
    assert payload.chat_template_kwargs == '{"key": "val"}'
    assert payload.save_and_add_slot is True


def test_payload_ngpu_layers_all() -> None:
    """RunProfilePayload should accept 'all' as n_gpu_layers."""
    payload = RunProfilePayload(n_gpu_layers="all")
    assert payload.n_gpu_layers == "all"


# ---------------------------------------------------------------------------
# RunProfileModal — _collect_values
# ---------------------------------------------------------------------------


def _make_mock_modal() -> MagicMock:
    """Create a mock RunProfileModal with mocked Input widgets."""
    modal = MagicMock(spec=RunProfileModal)

    def make_input(value: str) -> MagicMock:
        inp = MagicMock()
        inp.value = value
        return inp

    modal.query_one = MagicMock(side_effect=lambda selector, cls: make_input(""))
    return modal


def test_modal_collects_values_defaults() -> None:
    """_collect_values should return payload with defaults when all inputs are empty.

    Note: empty n_gpu_layers → 0 (not 'all'), empty int fields → defaults.
    """

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        inp = MagicMock()
        inp.value = ""
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal()
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.n_gpu_layers == 0


# ---------------------------------------------------------------------------
# RunProfileModal — edit mode
# ---------------------------------------------------------------------------


def test_modal_edit_mode_title() -> None:
    """RunProfileModal should show 'Edit Run Profile' when profile is provided."""
    spec = RunProfileSpec(
        profile_id="summary-balanced",
        model="/models/test.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )
    modal = RunProfileModal(profile=spec)
    assert modal._profile is not None


def test_modal_edit_prefills_values() -> None:
    """_profile_to_prefill should return dict with profile values."""
    spec = RunProfileSpec(
        profile_id="my-profile",
        model="/models/custom.gguf",
        alias="My Profile",
        device="CUDA:0",
        port=9090,
        ctx_size=8192,
        ubatch_size=1024,
        threads=16,
        n_gpu_layers=99,
        server_bin="/usr/bin/llama-server",
        chat_template_kwargs='{"key": "val"}',
        backend="llama_cpp",
    )
    modal = RunProfileModal(profile=spec)
    prefill = modal._profile_to_prefill(spec)

    assert prefill["profile-id"] == "my-profile"
    assert prefill["label"] == "My Profile"
    assert prefill["model"] == "/models/custom.gguf"
    assert prefill["server-bin"] == "/usr/bin/llama-server"
    assert prefill["port"] == "9090"
    assert prefill["ctx-size"] == "8192"
    assert prefill["ubatch-size"] == "1024"
    assert prefill["n-gpu-layers"] == "99"
    assert prefill["threads"] == "16"
    assert prefill["chat-template-kwargs"] == '{"key": "val"}'


def test_modal_edit_prefills_empty_label() -> None:
    """_profile_to_prefill should set empty label when alias equals profile_id."""
    spec = RunProfileSpec(
        profile_id="my-profile",
        model="/models/test.gguf",
        alias="my-profile",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )
    modal = RunProfileModal(profile=spec)
    prefill = modal._profile_to_prefill(spec)

    assert prefill["label"] == ""


def test_modal_collects_values_edit_mode() -> None:
    """_collect_values in edit mode should include original_profile_id."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        inp = MagicMock()
        value_map: dict[str, str] = {
            "#profile-profile-id": "updated-profile",
            "#profile-label": "Updated Label",
            "#profile-model": "/models/updated.gguf",
            "#profile-server-bin": "",
            "#profile-port": "8080",
            "#profile-ctx-size": "4096",
            "#profile-ubatch-size": "512",
            "#profile-n-gpu-layers": "all",
            "#profile-threads": "8",
            "#profile-chat-template-kwargs": "{}",
        }
        inp.value = value_map.get(selector, "")
        return inp

    spec = RunProfileSpec(
        profile_id="original-id",
        model="/models/original.gguf",
        alias="original",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal(profile=spec)
        instance._selected_model_path = "/models/updated.gguf"
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.original_profile_id == "original-id"
    assert payload.profile_id == "updated-profile"
    assert payload.model == "/models/updated.gguf"


def test_modal_collects_values_edit_mode_empty_profile() -> None:
    """_collect_values when profile is None should set original_profile_id to empty."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        inp = MagicMock()
        value_map: dict[str, str] = {
            "#profile-profile-id": "new-profile",
            "#profile-label": "New",
            "#profile-model": "/models/new.gguf",
            "#profile-server-bin": "",
            "#profile-port": "8080",
            "#profile-ctx-size": "4096",
            "#profile-ubatch-size": "512",
            "#profile-n-gpu-layers": "all",
            "#profile-threads": "8",
            "#profile-chat-template-kwargs": "{}",
        }
        inp.value = value_map.get(selector, "")
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal(profile=None)
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.original_profile_id == ""
    assert payload.profile_id == "new-profile"


def test_modal_collects_values_integer_fields_fallback() -> None:
    """_parse_int should fall back to defaults for invalid integer inputs."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        inp = MagicMock()
        value_map: dict[str, str] = {
            "#profile-profile-id": "test",
            "#profile-label": "Test",
            "#profile-model": "/models/test.gguf",
            "#profile-server-bin": "",
            "#profile-port": "not_a_port",
            "#profile-ctx-size": "abc",
            "#profile-ubatch-size": "",
            "#profile-n-gpu-layers": "all",
            "#profile-threads": "xyz",
            "#profile-chat-template-kwargs": "",
        }
        inp.value = value_map.get(selector, "")
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal()
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.port == 8080  # default
    assert payload.ctx_size == 4096  # default
    assert payload.ubatch_size == 512  # default
    assert payload.threads == 8  # default


def test_modal_collects_values_empty_ngpu_layers_fallback() -> None:
    """_collect_values should fall back to 0 when n_gpu_layers input is empty."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        inp = MagicMock()
        value_map: dict[str, str] = {
            "#profile-profile-id": "test",
            "#profile-label": "Test",
            "#profile-model": "/models/test.gguf",
            "#profile-server-bin": "",
            "#profile-port": "8080",
            "#profile-ctx-size": "4096",
            "#profile-ubatch-size": "512",
            "#profile-n-gpu-layers": "",  # empty
            "#profile-threads": "8",
            "#profile-chat-template-kwargs": "",
        }
        inp.value = value_map.get(selector, "")
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal()
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.n_gpu_layers == 0


# ---------------------------------------------------------------------------
# RunProfileModal — device field
# ---------------------------------------------------------------------------


def test_modal_device_default() -> None:
    """_collect_values should return device='CUDA:0' when Select shows default."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        if selector == "#profile-device":
            sel = MagicMock()
            sel.value = "CUDA:0"
            return sel
        inp = MagicMock()
        inp.value = ""
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal()
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.device == "CUDA:0"


@pytest.mark.anyio
async def test_modal_device_options() -> None:
    """The device Select should have the four expected device options."""
    from textual.app import ComposeResult

    from llama_cli.tui.components.run_profile_modal import _device_row

    class _DeviceRowApp(App[None]):
        def compose(self) -> ComposeResult:
            yield _device_row()

    app = _DeviceRowApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        select_widget = app.query_one("#profile-device", Select)
        # _options is the private attribute holding Select choices (includes NULL placeholder)
        options = [opt[1] for opt in select_widget._options]

        assert "CUDA:0" in options
        assert "CUDA:0,1" in options
        assert "CUDA:1" in options
        assert "SYCL0" in options
        # 4 entries: blank selection is disabled so the device never renders empty.
        assert len(options) == 4
        assert select_widget._allow_blank is False


def test_modal_collects_values_device_selection() -> None:
    """_collect_values should pass through the selected device value."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        if selector == "#profile-device":
            sel = MagicMock()
            sel.value = "SYCL0"
            return sel
        inp = MagicMock()
        inp.value = ""
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal()
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.device == "SYCL0"


def test_modal_collects_values_device_empty_fallback() -> None:
    """_collect_values should fall back to 'CUDA:0' when Select value is empty."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        if selector == "#profile-device":
            sel = MagicMock()
            sel.value = ""
            return sel
        inp = MagicMock()
        inp.value = ""
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        instance = RunProfileModal()
        payload = instance._collect_values(save_and_add_slot=False)

    assert payload.device == "CUDA:0"


def test_payload_has_device_field() -> None:
    """RunProfilePayload should have a device field with default 'CUDA:0'."""
    payload = RunProfilePayload()
    assert payload.device == "CUDA:0"


def test_payload_custom_device() -> None:
    """RunProfilePayload should accept a custom device value."""
    payload = RunProfilePayload(device="SYCL0")
    assert payload.device == "SYCL0"


# ---------------------------------------------------------------------------
# RunProfileModal — model_index parameter
# ---------------------------------------------------------------------------


def test_modal_accepts_model_index() -> None:
    """RunProfileModal should accept and store model_index."""
    from llama_manager.model_index import ModelIndexEntry

    entries = [
        ModelIndexEntry(
            path="/models/test.gguf",
            normalized_stem="test",
            general_name=None,
            architecture="llama",
            file_type=None,
            quantization_type="Q4_K_M",
            context_length=None,
            embedding_length=None,
            block_count=None,
            file_size_bytes=1000,
            parse_error=None,
            mtime_iso="2024-01-01T00:00:00+00:00",
        )
    ]
    modal = RunProfileModal(model_index=entries)
    assert len(modal._model_index) == 1
    assert modal._model_index[0].path == "/models/test.gguf"


def test_modal_handles_empty_model_index() -> None:
    """RunProfileModal should handle empty model_index list."""
    modal = RunProfileModal(model_index=[])
    assert modal._model_index == []


def test_modal_model_index_defaults_to_empty() -> None:
    """RunProfileModal should default model_index to empty list."""
    modal = RunProfileModal()
    assert modal._model_index == []


def test_modal_selected_model_path_empty_initially() -> None:
    """_selected_model_path should be empty string initially."""
    modal = RunProfileModal()
    assert modal._selected_model_path == ""


def test_modal_collects_values_with_model_index() -> None:
    """_collect_values should use _selected_model_path when model_index is present."""

    def query_one_side_effect(selector: str, cls: type) -> MagicMock:
        inp = MagicMock()
        inp.value = ""
        return inp

    with patch.object(
        RunProfileModal,
        "query_one",
        side_effect=query_one_side_effect,
    ):
        modal = RunProfileModal(
            model_index=[
                ModelIndexEntry(
                    path="/models/selected.gguf",
                    normalized_stem="selected",
                    general_name=None,
                    architecture="llama",
                    file_type=None,
                    quantization_type="Q4_K_M",
                    context_length=None,
                    embedding_length=None,
                    block_count=None,
                    file_size_bytes=1000,
                    parse_error=None,
                    mtime_iso="2024-01-01T00:00:00+00:00",
                )
            ],
        )
        modal._selected_model_path = "/models/selected.gguf"
        payload = modal._collect_values(save_and_add_slot=False)

    assert payload.model == "/models/selected.gguf"


# ---------------------------------------------------------------------------
# Model picker — edit prefill
# ---------------------------------------------------------------------------


def _make_model_index_entry(path: str = "/models/selected.gguf") -> ModelIndexEntry:
    """Create a model index entry for modal tests."""
    return ModelIndexEntry(
        path=path,
        normalized_stem="selected",
        general_name=None,
        architecture="llama",
        file_type=None,
        quantization_type="Q4_K_M",
        context_length=32768,
        embedding_length=None,
        block_count=None,
        file_size_bytes=4 * 1024**3,
        parse_error=None,
        mtime_iso="2024-01-01T00:00:00+00:00",
    )


def test_modal_edit_prefills_selected_model_path() -> None:
    """Edit mode should initialize _selected_model_path from profile.model."""
    spec = RunProfileSpec(
        profile_id="my-profile",
        model="/models/existing.gguf",
        alias="My Profile",
        device="CUDA:0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )
    modal = RunProfileModal(profile=spec)
    assert modal._selected_model_path == "/models/existing.gguf"


def test_modal_create_has_empty_selected_model_path() -> None:
    """Create mode should have empty _selected_model_path."""
    modal = RunProfileModal()
    assert modal._selected_model_path == ""


@pytest.mark.anyio
async def test_modal_advanced_section_collapsed_by_default() -> None:
    """Advanced profile fields should live in a collapsed Collapsible section."""
    modal = RunProfileModal()
    app = App[None]()

    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        advanced = modal.query_one(".profile-advanced-options", Collapsible)
        assert advanced.title == "Advanced"
        assert advanced.collapsed is True

        modal.query_one("#profile-server-bin", Input)
        modal.query_one("#profile-port", Input)
        modal.query_one("#profile-ubatch-size", Input)
        modal.query_one("#profile-n-gpu-layers", Input)
        modal.query_one("#profile-threads", Input)


@pytest.mark.anyio
async def test_modal_model_picker_mounts_search_and_list() -> None:
    """The model picker should mount a visible search input and indexed model list."""
    modal = RunProfileModal(model_index=[_make_model_index_entry()])
    app = App[None]()

    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        search = modal.query_one("#profile-model-search", Input)
        model_list = modal.query_one("#profile-model-list", ListView)
        model_row = modal.query_one(".profile-model-row")
        picker = modal.query_one(".profile-model-picker")

        assert search.placeholder == "Search indexed models or type path..."
        assert len(list(model_list.children)) == 1
        assert model_row.has_class("profile-row")
        assert picker.has_class("profile-model-picker")


@pytest.mark.anyio
async def test_modal_model_selection_updates_visible_input() -> None:
    """Selecting an indexed model should mirror the path into the visible input."""
    entry = _make_model_index_entry()
    modal = RunProfileModal(model_index=[entry])
    app = App[None]()

    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        model_list = modal.query_one("#profile-model-list", ListView)
        item = list(model_list.children)[0]
        event = cast(Any, SimpleNamespace(item=item))

        modal.on_list_view_selected(event)

        search = modal.query_one("#profile-model-search", Input)
        assert modal._selected_model_path == entry.path
        assert search.value == entry.path


@pytest.mark.anyio
async def test_modal_model_selection_keeps_selected_path_in_filter() -> None:
    """Selecting a model should not clear the visible list when the full path is shown."""
    entry = _make_model_index_entry()
    modal = RunProfileModal(model_index=[entry])
    app = App[None]()

    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        search = modal.query_one("#profile-model-search", Input)
        search.value = entry.path
        await pilot.pause()

        model_list = modal.query_one("#profile-model-list", ListView)
        assert len(list(model_list.children)) == 1


@pytest.mark.anyio
async def test_create_modal_prefills_from_config() -> None:
    """Create mode should prefill advanced fields from Config defaults."""
    config = Config(
        default_batch_size=1024,
        default_poll_ms=0,
        default_n_predict=8192,
        default_parallel=2,
        default_profile_cache_type_k="f16",
        default_reasoning_mode="off",
        default_use_jinja=True,
    )
    modal = RunProfileModal(config=config)
    app = App[None]()

    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        batch = modal.query_one("#profile-batch-size", Input)
        poll = modal.query_one("#profile-poll-ms", Input)
        n_predict = modal.query_one("#profile-n-predict", Input)
        parallel = modal.query_one("#profile-parallel", Input)
        cache_k = modal.query_one("#profile-cache-type-k", Select)
        jinja = modal.query_one("#profile-use-jinja", Checkbox)

    assert batch.value == "1024"
    assert poll.value == "0"
    assert n_predict.value == "8192"
    assert parallel.value == "2"
    assert cache_k.value == "f16"
    assert jinja.value is True


@pytest.mark.anyio
async def test_select_displays_current_value_in_control() -> None:
    """Select widgets should render the chosen option inside SelectCurrent."""
    from textual.widgets import Static
    from textual.widgets._select import SelectCurrent

    config = Config(default_profile_cache_type_k="f16", default_reasoning_mode="off")
    modal = RunProfileModal(config=config)
    app = App[None]()

    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        cache_k = modal.query_one("#profile-cache-type-k", Select)
        assert cache_k.value == "f16"
        cache_label = cache_k.query_one(SelectCurrent).query_one("#label", Static)
        assert "f16" in str(cache_label.content)

        reasoning = modal.query_one("#profile-reasoning-mode", Select)
        assert reasoning.value == "off"
        reasoning_label = reasoning.query_one(SelectCurrent).query_one("#label", Static)
        assert "off" in str(reasoning_label.content)


# ---------------------------------------------------------------------------
# Model index — case-insensitive scan
# ---------------------------------------------------------------------------


def test_format_model_line_uses_model_index_quantization() -> None:
    """_format_model_line should use GGUF metadata from model_index when available."""
    from llama_cli.tui.components.profiles_screen import _format_model_line
    from llama_manager.model_index import ModelIndexEntry

    spec = RunProfileSpec(
        profile_id="test",
        model="/models/custom.gguf",
        alias="test",
        device="CUDA:0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        backend="llama_cpp",
    )
    model_index = [
        ModelIndexEntry(
            path="/models/custom.gguf",
            normalized_stem="custom",
            general_name=None,
            architecture="llama",
            file_type=None,
            quantization_type="Q5_K_M",
            context_length=None,
            embedding_length=None,
            block_count=None,
            file_size_bytes=4000000000,
            parse_error=None,
            mtime_iso="2024-01-01T00:00:00+00:00",
        )
    ]
    result = _format_model_line(spec, model_index)
    assert result == "Model: custom.gguf  [Q5_K_M]"
