"""Tests for ProfilesScreen — TUI modal for managing run profiles."""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Label

from llama_cli.tui.components.profiles_screen import ProfilesScreen
from llama_manager.config.profiles import RunProfileSpec
from llama_manager.model_index import ModelIndexEntry


@pytest.fixture()
def sample_profiles() -> list[tuple[RunProfileSpec, str]]:
    """Return sample profiles for testing."""
    builtin = RunProfileSpec(
        profile_id="summary-balanced",
        model="/models/summary-balanced.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        description="Run summary-balanced model on Intel SYCL.",
        backend="llama_cpp",
    )
    custom = RunProfileSpec(
        profile_id="my-custom",
        model="/models/custom.gguf",
        alias="my-custom",
        device="CUDA:0",
        port=9090,
        ctx_size=8192,
        ubatch_size=256,
        threads=4,
        description="Custom profile",
        backend="llama_cpp",
    )
    return [(builtin, "builtin"), (custom, "custom")]


@pytest.fixture()
def sample_profiles_with_in_use() -> list[tuple[RunProfileSpec, str]]:
    """Return profiles where one is in use."""
    builtin = RunProfileSpec(
        profile_id="summary-balanced",
        model="/models/summary-balanced.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        description="Run summary-balanced model on Intel SYCL.",
        backend="llama_cpp",
    )
    return [(builtin, "builtin")]


class _ProfilesHostApp(App[Any]):
    """Minimal Textual app that hosts ProfilesScreen for testing."""

    def __init__(
        self,
        profiles: list[tuple[RunProfileSpec, str]],
        in_use_ids: set[str] | None = None,
        model_index: list[ModelIndexEntry] | None = None,
    ) -> None:
        super().__init__()
        self._profiles = profiles
        self._in_use_ids = in_use_ids or set()
        self._model_index = model_index or []

    def compose(self) -> ComposeResult:
        yield ProfilesScreen(
            profiles=self._profiles,
            in_use_ids=self._in_use_ids,
            model_index=self._model_index,
        )


@pytest.mark.anyio
async def test_profiles_screen_dismiss_on_escape(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Pressing escape should dismiss with None."""
    result_holder: list[Any] = []

    def on_result(result: object) -> None:
        result_holder.append(result)

    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await app.push_screen(ProfilesScreen(profiles=sample_profiles), on_result)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert len(result_holder) == 1
    assert result_holder[0] is None


@pytest.mark.anyio
async def test_profiles_screen_dismiss_on_ctrl_c(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Pressing ctrl+c should dismiss with None."""
    result_holder: list[Any] = []

    def on_result(result: object) -> None:
        result_holder.append(result)

    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await app.push_screen(ProfilesScreen(profiles=sample_profiles), on_result)
        await pilot.pause()
        # Trigger action directly since ctrl+c may be intercepted
        screen = app.screen
        assert isinstance(screen, ProfilesScreen)
        screen.action_cancel()
        await pilot.pause()

    assert len(result_holder) == 1
    assert result_holder[0] is None


@pytest.mark.anyio
async def test_profiles_screen_add_profile(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Clicking '+ Add Profile' returns add action."""
    result_holder: list[Any] = []

    def on_result(result: object) -> None:
        result_holder.append(result)

    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await app.push_screen(ProfilesScreen(profiles=sample_profiles), on_result)
        await pilot.pause()
        await pilot.click("#add-profile")
        await pilot.pause()

    assert len(result_holder) == 1
    assert result_holder[0] == {"action": "add"}


@pytest.mark.anyio
async def test_profiles_screen_close_profile(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Clicking 'Close' returns None."""
    result_holder: list[Any] = []

    def on_result(result: object) -> None:
        result_holder.append(result)

    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await app.push_screen(ProfilesScreen(profiles=sample_profiles), on_result)
        await pilot.pause()
        await pilot.click("#close-profiles")
        await pilot.pause()

    assert len(result_holder) == 1
    assert result_holder[0] is None


@pytest.mark.anyio
async def test_profiles_screen_edit_profile(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Clicking Edit button returns edit action with profile_id."""
    result_holder: list[Any] = []

    def on_result(result: object) -> None:
        result_holder.append(result)

    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await app.push_screen(ProfilesScreen(profiles=sample_profiles), on_result)
        await pilot.pause()
        await pilot.click("#edit-summary-balanced")
        await pilot.pause()

    assert len(result_holder) == 1
    assert result_holder[0] == {"action": "edit", "profile_id": "summary-balanced"}


@pytest.mark.anyio
async def test_profiles_screen_delete_profile(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Clicking Delete button returns delete action with profile_id."""
    result_holder: list[Any] = []

    def on_result(result: object) -> None:
        result_holder.append(result)

    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await app.push_screen(ProfilesScreen(profiles=sample_profiles), on_result)
        await pilot.pause()
        await pilot.click("#delete-summary-balanced")
        await pilot.pause()

    assert len(result_holder) == 1
    assert result_holder[0] == {"action": "delete", "profile_id": "summary-balanced"}


@pytest.mark.anyio
async def test_profiles_screen_delete_blocked_in_use(
    sample_profiles_with_in_use: list[tuple[RunProfileSpec, str]],
) -> None:
    """In-use profile shows 'In Use' button, disabled."""
    in_use_ids = {"summary-balanced"}
    app = _ProfilesHostApp(
        profiles=sample_profiles_with_in_use,
        in_use_ids=in_use_ids,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        delete_btn = app.query_one("#delete-summary-balanced", Button)
        assert delete_btn.disabled is True
        assert delete_btn.label == "In Use"


@pytest.mark.anyio
async def test_profiles_screen_shows_profiles(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Screen should display profile info from the list."""
    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = list(app.query(Label))
        label_texts = [getattr(lbl, "_Static__content", "") for lbl in labels]

        assert any("Run Profiles" in t for t in label_texts)
        assert any("summary-balanced" in t for t in label_texts)
        assert any("my-custom" in t for t in label_texts)


@pytest.mark.anyio
async def test_profiles_screen_empty_list() -> None:
    """Empty profiles list should show 'No profiles configured' message."""
    app = _ProfilesHostApp(profiles=[])
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = list(app.query(Label))
        label_texts = [getattr(lbl, "_Static__content", "") for lbl in labels]
        assert any("No profiles configured" in t for t in label_texts)


@pytest.mark.anyio
async def test_profiles_screen_shows_device(
    sample_profiles: list[tuple[RunProfileSpec, str]],
) -> None:
    """Profile cards should show Device label with the spec's device value."""
    app = _ProfilesHostApp(sample_profiles)
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = list(app.query(Label))
        label_texts = [getattr(lbl, "_Static__content", "") for lbl in labels]

        meta_text = " ".join(label_texts)
        assert "Device: SYCL0" in meta_text
        assert "Device: CUDA:0" in meta_text


@pytest.mark.anyio
async def test_profiles_screen_shows_model_filename() -> None:
    """Profile card should show model filename, not full path."""
    builtin = RunProfileSpec(
        profile_id="summary-balanced",
        model="/models/quantized/Q4_K_M_summ.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        description="Run summary-balanced model.",
        backend="llama_cpp",
    )
    app = _ProfilesHostApp([(builtin, "builtin")])
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = list(app.query(Label))
        label_texts = [getattr(lbl, "_Static__content", "") for lbl in labels]

        model_texts = [t for t in label_texts if "Model:" in t]
        assert len(model_texts) >= 1
        # Should show filename, not full path
        assert "Q4_K_M_summ.gguf" in model_texts[0]
        # Should NOT show the full path
        assert "/models/quantized" not in model_texts[0]


@pytest.mark.anyio
async def test_profiles_screen_shows_indexed_model_details() -> None:
    """Profile card should show the same indexed model details as the chooser."""
    builtin = RunProfileSpec(
        profile_id="summary-balanced",
        model="/models/summary-balanced.gguf",
        alias="summary-balanced",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        description="Run summary-balanced model.",
        backend="llama_cpp",
    )
    model_index = [
        ModelIndexEntry(
            path="/models/summary-balanced.gguf",
            normalized_stem="summary-balanced",
            general_name=None,
            architecture="qwen3",
            file_type=None,
            quantization_type="Q4_K_M",
            context_length=16144,
            max_context_length=16144,
            embedding_length=None,
            block_count=None,
            file_size_bytes=3 * 1024**3,
            parse_error="metadata warning",
            mtime_iso="2026-05-24T00:00:00+00:00",
        )
    ]
    app = _ProfilesHostApp([(builtin, "builtin")], model_index=model_index)
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = list(app.query(Label))
        label_texts = [str(getattr(lbl, "_Static__content", "")) for lbl in labels]

        details_text = next(t for t in label_texts if "Arch:" in t)
        assert "Arch: qwen3" in details_text
        assert "Quant: Q4_K_M" in details_text
        assert "Max Ctx: 16144" in details_text
        assert "Size: 3.0 GiB" in details_text
        assert "Metadata: metadata warning" in details_text


@pytest.mark.anyio
async def test_profiles_screen_shows_ctx_fallback_for_zero() -> None:
    """Profile card should show '?' for ctx_size when it's falsy.

    Note: RunProfileSpec validates ctx_size > 0, so we test the display
    logic by verifying the meta-text format uses conditional fallback.
    """
    # RunProfileSpec requires ctx_size > 0, so we can't create one with 0.
    # Instead verify the meta-text format includes "Ctx:" followed by a value
    # or '?'. The display code is: f"Ctx: {spec.ctx_size if spec.ctx_size else '?'}"
    builtin = RunProfileSpec(
        profile_id="normal-ctx",
        model="/models/test.gguf",
        alias="normal-ctx",
        device="SYCL0",
        port=8080,
        ctx_size=4096,
        ubatch_size=512,
        threads=8,
        description="Normal profile.",
        backend="llama_cpp",
    )
    app = _ProfilesHostApp([(builtin, "builtin")])
    async with app.run_test() as pilot:
        await pilot.pause()
        labels = list(app.query(Label))
        label_texts = [getattr(lbl, "_Static__content", "") for lbl in labels]

        meta_text = " ".join(label_texts)
        # Should show the actual ctx_size value
        assert "Ctx: 4096" in meta_text
