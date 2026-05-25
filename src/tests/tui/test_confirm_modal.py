"""Tests for ConfirmModal — simple yes/no confirmation dialog.

Covers:
- compose() renders title and message labels
- on_mount() focuses the submit (confirm) button
- action_cancel() dismisses with False
- on_button_pressed() with #cancel-confirm dismisses with False
- on_button_pressed() with #submit-confirm dismisses with True
- Escape binding triggers action_cancel → dismiss(False)
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Button, Label

from llama_cli.tui.components.confirm_modal import ConfirmModal


class ConfirmModalHostApp(App[None]):
    """Minimal Textual app that hosts ConfirmModal for testing."""

    pass


class TestConfirmModalCompose:
    """Tests for ConfirmModal screen composition and mount state."""

    @pytest.mark.anyio
    async def test_renders_title_label(self) -> None:
        """compose() should render a #confirm-title label."""
        modal = ConfirmModal(title="Delete slot?", message="This cannot be undone.")
        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            title_label = modal.query_one("#confirm-title", Label)
            assert title_label is not None

    @pytest.mark.anyio
    async def test_renders_message_label(self) -> None:
        """compose() should render a #confirm-message label."""
        modal = ConfirmModal(title="Delete slot?", message="This cannot be undone.")
        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            msg_label = modal.query_one("#confirm-message", Label)
            assert msg_label is not None

    @pytest.mark.anyio
    async def test_on_mount_focuses_submit_button(self) -> None:
        """on_mount() should give focus to the #submit-confirm button."""
        modal = ConfirmModal(title="Delete slot?", message="This cannot be undone.")
        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            submit_btn = modal.query_one("#submit-confirm", Button)
            assert submit_btn.has_focus

    @pytest.mark.anyio
    async def test_both_buttons_present(self) -> None:
        """compose() should render both #cancel-confirm and #submit-confirm buttons."""
        modal = ConfirmModal(title="Test", message="Confirm?")
        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal)
            await pilot.pause()
            # Both buttons must exist in the DOM
            modal.query_one("#cancel-confirm", Button)
            modal.query_one("#submit-confirm", Button)


class TestConfirmModalDismiss:
    """Tests for ConfirmModal dismiss behaviour on user interactions."""

    @pytest.mark.anyio
    async def test_action_cancel_dismisses_false(self) -> None:
        """action_cancel() should dismiss the modal with False."""
        modal = ConfirmModal(title="Test", message="Are you sure?")
        result_holder: list[object] = []

        def on_result(result: object) -> None:
            result_holder.append(result)

        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal, on_result)
            await pilot.pause()
            modal.action_cancel()
            await pilot.pause()

        assert len(result_holder) == 1
        assert result_holder[0] is False

    @pytest.mark.anyio
    async def test_cancel_button_click_dismisses_false(self) -> None:
        """Clicking #cancel-confirm should dismiss with False."""
        modal = ConfirmModal(title="Test", message="Are you sure?")
        result_holder: list[object] = []

        def on_result(result: object) -> None:
            result_holder.append(result)

        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal, on_result)
            for _ in range(5):
                await pilot.pause()
            cancel_btn = modal.query_one("#cancel-confirm", Button)
            await pilot.click(cancel_btn)
            await pilot.pause()

        assert len(result_holder) == 1
        assert result_holder[0] is False

    @pytest.mark.anyio
    async def test_confirm_button_click_dismisses_true(self) -> None:
        """Clicking #submit-confirm should dismiss with True."""
        modal = ConfirmModal(title="Test", message="Are you sure?")
        result_holder: list[object] = []

        def on_result(result: object) -> None:
            result_holder.append(result)

        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal, on_result)
            for _ in range(5):
                await pilot.pause()
            submit_btn = modal.query_one("#submit-confirm", Button)
            await pilot.click(submit_btn)
            await pilot.pause()

        assert len(result_holder) == 1
        assert result_holder[0] is True

    @pytest.mark.anyio
    async def test_escape_key_dismisses_false(self) -> None:
        """Escape key should trigger action_cancel and dismiss with False."""
        modal = ConfirmModal(title="Test", message="Are you sure?")
        result_holder: list[object] = []

        def on_result(result: object) -> None:
            result_holder.append(result)

        app = ConfirmModalHostApp()
        async with app.run_test() as pilot:
            await app.push_screen(modal, on_result)
            for _ in range(5):
                await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert len(result_holder) == 1
        assert result_holder[0] is False
