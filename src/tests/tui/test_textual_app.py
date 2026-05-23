"""Tests for DashboardApp — Textual app shell and profile options caching.

Covers:
- Profile options caching logic
- Reconcile server log panels
- Build modal dismisses on cancel
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llama_cli.tui.components.build import BuildModalScreen
from llama_cli.tui.textual_app import (
    DashboardApp,
    _profile_options_cached,
)
from tests.support.helpers import make_server_config


def _make_controller() -> MagicMock:
    """Create a mocked DashboardController for DashboardApp tests."""
    from llama_cli.tui.model import DashboardModel
    from llama_cli.tui.viewmodel import DashboardViewModel
    from llama_manager.config import Config

    config = Config()
    model = DashboardModel(configs=[make_server_config(alias="test")], gpu_indices=[])
    view_model = DashboardViewModel(model)

    controller = MagicMock()
    controller.running = True
    controller.config = config
    controller.configs = [make_server_config(alias="test")]
    controller.view_model = view_model
    controller.server_manager = MagicMock()
    controller.model = model
    controller.model.build_cancel_event = None
    # Provide valid return for model-index worker called on mount
    controller.refresh_model_index.return_value = ([], 0, 0)
    return controller


class TestProfileOptionsCached:
    """Tests for _profile_options_cached — extracted cache helper."""

    def test_caching_returns_cached_on_same_config(self) -> None:
        """When config_id matches, cached options should be returned."""
        view_model = MagicMock()
        config = make_server_config(alias="cached")
        cache = [("cached_model", "cached_alias")]
        cache_config_id = id(config)

        result_options, result_id = _profile_options_cached(
            view_model, config, cache, cache_config_id
        )

        assert result_options == cache
        assert result_id == cache_config_id
        # view_model.profile_options should NOT be called
        view_model.profile_options.assert_not_called()

    def test_cache_invalidated_on_new_config(self) -> None:
        """When config_id differs, cache should be invalidated."""
        view_model = MagicMock()
        view_model.profile_options.return_value = [("new_model", "new_alias")]
        config = make_server_config(alias="new")
        old_cache: list[tuple[str, str]] = [("old", "old")]
        old_config_id = id(make_server_config(alias="old"))

        result_options, result_id = _profile_options_cached(
            view_model, config, old_cache, old_config_id
        )

        assert result_options == [("new_model", "new_alias")]
        view_model.profile_options.assert_called_once()

    def test_none_cache_always_recomputes(self) -> None:
        """When cache is None, options should always be recomputed."""
        view_model = MagicMock()
        view_model.profile_options.return_value = [("recomputed", "alias")]
        config = make_server_config(alias="test")

        result_options, result_id = _profile_options_cached(view_model, config, None, None)

        assert result_options == [("recomputed", "alias")]
        view_model.profile_options.assert_called_once()

    def test_returns_config_id(self) -> None:
        """Result should include the config_id."""
        view_model = MagicMock()
        view_model.profile_options.return_value = []
        config = make_server_config(alias="test")

        _, result_id = _profile_options_cached(view_model, config, None, None)

        assert result_id == id(config)


class TestBuildModalScreenCancel:
    """Tests for BuildModalScreen cancel behavior."""

    @pytest.mark.anyio
    async def test_cancel_dismisses_none(self) -> None:
        """Cancel button should dismiss the modal with None."""
        from llama_cli.tui.textual_app import DashboardApp

        controller = _make_controller()
        screen = BuildModalScreen()
        result_holder: list[object] = []

        def on_result(result: object) -> None:
            result_holder.append(result)

        app = DashboardApp(controller)
        async with app.run_test() as pilot:
            await app.push_screen(screen, on_result)
            # Wait for the screen to fully render and the status worker to complete
            for _ in range(10):
                await pilot.pause()
            # Use action_cancel instead of button click to avoid OutOfBounds
            screen.action_cancel()
            await pilot.pause()

        assert len(result_holder) == 1
        assert result_holder[0] is None

    @pytest.mark.anyio
    async def test_escape_dismisses_none(self) -> None:
        """Escape key should dismiss the modal."""
        from llama_cli.tui.textual_app import DashboardApp

        controller = _make_controller()
        screen = BuildModalScreen()
        result_holder: list[object] = []

        def on_result(result: object) -> None:
            result_holder.append(result)

        app = DashboardApp(controller)
        async with app.run_test() as pilot:
            await app.push_screen(screen, on_result)
            for _ in range(5):
                await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert len(result_holder) == 1
        assert result_holder[0] is None


class TestDashboardAppInit:
    """Tests for DashboardApp initialization."""

    @pytest.mark.anyio
    async def test_app_instantiation(self) -> None:
        """DashboardApp should be instantiable with a controller."""
        controller = _make_controller()
        app = DashboardApp(controller)
        assert app.controller is controller
        assert app.view_model is controller.view_model

    @pytest.mark.anyio
    async def test_profile_cache_initialized(self) -> None:
        """DashboardApp should initialize cache attributes."""
        controller = _make_controller()
        app = DashboardApp(controller)
        assert app._profile_options_cache is None
        assert app._profile_cache_config_id is None


class TestDashboardAppProfileOptions:
    """Tests for DashboardApp._build_profile_options caching."""

    @pytest.mark.anyio
    async def test_first_call_recomputes(self) -> None:
        """First call should recompute options from view_model."""
        controller = _make_controller()
        # Replace view_model with a MagicMock for this test
        controller.view_model = MagicMock()
        controller.view_model.profile_options.return_value = [
            ("model_a", "alias_a"),
        ]
        app = DashboardApp(controller)

        options = app._build_profile_options()

        assert options == [("model_a", "alias_a")]

    @pytest.mark.anyio
    async def test_second_call_returns_cached(self) -> None:
        """Second call with same config should return cached options."""
        controller = _make_controller()
        # Replace view_model with a MagicMock for this test
        controller.view_model = MagicMock()
        controller.view_model.profile_options.return_value = [
            ("cached_model", "cached_alias"),
        ]
        app = DashboardApp(controller)

        # First call
        options1 = app._build_profile_options()
        # Second call
        options2 = app._build_profile_options()

        assert options1 == options2
        # profile_options should only be called once
        assert controller.view_model.profile_options.call_count == 1
