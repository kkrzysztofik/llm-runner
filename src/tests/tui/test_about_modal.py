"""Tests for AboutModal and version helper."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest
from textual.app import App
from textual.widgets import Static

from llama_cli.tui.components.about_modal import AboutModal, _app_version


def test_app_version_returns_package_version() -> None:
    with patch("llama_cli.tui.components.about_modal.version", return_value="9.9.9"):
        assert _app_version() == "9.9.9"


def test_app_version_falls_back_to_dev() -> None:
    with patch(
        "llama_cli.tui.components.about_modal.version",
        side_effect=PackageNotFoundError("llm-runner"),
    ):
        assert _app_version() == "dev"


@pytest.mark.anyio
async def test_about_modal_renders_content() -> None:
    modal = AboutModal()
    app = App[None]()

    async with app.run_test() as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        content = modal.query_one("#about-content", Static)
        rendered = str(content.render())
        assert "llm-runner" in rendered
        assert "Key Bindings" in rendered


@pytest.mark.anyio
async def test_about_modal_dismisses_on_key() -> None:
    modal = AboutModal()
    result_holder: list[object] = []

    def on_result(result: object) -> None:
        result_holder.append(result)

    app = App[None]()
    async with app.run_test() as pilot:
        await app.push_screen(modal, on_result)
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()

    assert len(result_holder) == 1
    assert result_holder[0] is None
