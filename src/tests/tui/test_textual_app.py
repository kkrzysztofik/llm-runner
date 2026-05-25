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


class TestDashboardAppModelIndex:
    """Tests for DashboardApp model-index callback methods.

    These methods are called from background threads. Testing them directly
    (with app.notify mocked) avoids needing a running event loop while still
    covering every branch.
    """

    def _make_app(self) -> tuple[DashboardApp, MagicMock]:
        """Create a DashboardApp and a MagicMock wired in as app.notify."""
        controller = _make_controller()
        app = DashboardApp(controller)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[assignment]
        return app, notify_mock

    def test_handle_progress_zero_total_no_notify(self) -> None:
        """_handle_model_index_progress is a no-op when total <= 0."""
        app, notify_mock = self._make_app()

        app._handle_model_index_progress(0, 0, 0)

        notify_mock.assert_not_called()

    def test_handle_progress_negative_total_no_notify(self) -> None:
        """_handle_model_index_progress is a no-op when total is negative."""
        app, notify_mock = self._make_app()

        app._handle_model_index_progress(5, -1, 0)

        notify_mock.assert_not_called()

    def test_handle_progress_below_threshold_no_notify(self) -> None:
        """No notification when scanned is below the 1/5 progress threshold."""
        app, notify_mock = self._make_app()
        # notify_every = max(1, 10 // 5) = 2; scanned=1 < notify_every=2 → no notify
        app._handle_model_index_progress(1, 10, 0)

        notify_mock.assert_not_called()

    def test_handle_progress_at_threshold_notifies(self) -> None:
        """Notification is sent when scanned reaches the 1/5 threshold."""
        app, notify_mock = self._make_app()
        # notify_every = max(1, 10 // 5) = 2; scanned=2 >= notify_every=2 → notify
        app._handle_model_index_progress(2, 10, 0)

        notify_mock.assert_called_once_with(
            "Indexed 2/10 models",
            title="Models",
            severity="information",
        )

    def test_handle_progress_with_errors_severity_warning(self) -> None:
        """Notification uses severity='warning' when errors > 0."""
        app, notify_mock = self._make_app()
        app._handle_model_index_progress(2, 10, 1)

        call_kwargs = notify_mock.call_args
        assert call_kwargs.kwargs["severity"] == "warning"
        assert "(1 errors)" in call_kwargs.args[0]

    def test_handle_progress_complete_notifies(self) -> None:
        """Notification is always sent when scanned == total."""
        app, notify_mock = self._make_app()
        # scanned == total should bypass the threshold guard
        app._handle_model_index_progress(10, 10, 0)

        notify_mock.assert_called_once()
        assert "Indexed 10/10" in notify_mock.call_args.args[0]

    def test_handle_complete_no_errors(self) -> None:
        """_handle_model_index_complete notifies with total models found."""
        app, notify_mock = self._make_app()

        app._handle_model_index_complete(5, 0)

        notify_mock.assert_called_once_with(
            "Indexing complete: 5 models found",
            title="Models",
            severity="information",
        )

    def test_handle_complete_with_errors_severity_warning(self) -> None:
        """_handle_model_index_complete uses severity='warning' when errors > 0."""
        app, notify_mock = self._make_app()

        app._handle_model_index_complete(5, 2)

        call_kwargs = notify_mock.call_args
        assert call_kwargs.kwargs["severity"] == "warning"
        assert "5 models found" in call_kwargs.args[0]
        assert "(2 errors)" in call_kwargs.args[0]

    def test_index_models_notifies_when_refresh_starts(self) -> None:
        """_index_models notifies 'Indexing models...' when a refresh is started."""
        controller = _make_controller()
        controller.refresh_model_index_async.return_value = True
        app = DashboardApp(controller)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[assignment]

        app._index_models()

        notify_mock.assert_called_once_with("Indexing models...", title="Models")

    def test_index_models_no_notify_when_already_running(self) -> None:
        """_index_models does not notify when refresh_model_index_async returns False."""
        controller = _make_controller()
        controller.refresh_model_index_async.return_value = False
        app = DashboardApp(controller)
        notify_mock = MagicMock()
        app.notify = notify_mock  # type: ignore[assignment]

        app._index_models()

        notify_mock.assert_not_called()


class TestDashboardAppCheckAction:
    """Tests for DashboardApp.check_action — binding visibility gating."""

    def _make_app(
        self,
        *,
        build_request: bool = False,
        risk_prompt: object = None,
    ) -> DashboardApp:
        controller = _make_controller()
        controller.model.build_request = build_request
        controller.model.risk_prompt = risk_prompt  # type: ignore[assignment]
        return DashboardApp(controller)

    def test_normal_state_hides_confirm_and_reject(self) -> None:
        """In normal mode 'confirm' and 'reject' bindings are hidden."""
        app = self._make_app()

        assert app.check_action("confirm", ()) is False
        assert app.check_action("reject", ()) is False

    def test_normal_state_shows_build_action(self) -> None:
        """In normal mode 'build' binding is visible."""
        app = self._make_app()

        assert app.check_action("build", ()) is True

    def test_build_request_only_shows_cancel(self) -> None:
        """During build_request only 'cancel_pending_prompt' returns True."""
        app = self._make_app(build_request=True)

        assert app.check_action("cancel_pending_prompt", ()) is True
        assert app.check_action("build", ()) is False
        assert app.check_action("add_slot", ()) is False

    def test_risk_prompt_hides_refresh_add_build_config(self) -> None:
        """During a risk prompt, refresh/add_slot/build/open_config are hidden."""
        from llama_cli.tui.types import RiskPromptState

        risk = RiskPromptState(kind="hardware", acknowledged=False)
        app = self._make_app(risk_prompt=risk)

        assert app.check_action("refresh_dashboard", ()) is False
        assert app.check_action("add_slot", ()) is False
        assert app.check_action("build", ()) is False
        assert app.check_action("open_config", ()) is False

    def test_hardware_risk_prompt_allows_quit(self) -> None:
        """During hardware risk prompt 'quit_dashboard' remains visible."""
        from llama_cli.tui.types import RiskPromptState

        risk = RiskPromptState(kind="hardware", acknowledged=False)
        app = self._make_app(risk_prompt=risk)

        assert app.check_action("quit_dashboard", ()) is True

    def test_vram_risk_prompt_hides_quit(self) -> None:
        """During vram risk prompt 'quit_dashboard' is hidden."""
        from llama_cli.tui.types import RiskPromptState

        risk = RiskPromptState(kind="vram", acknowledged=False)
        app = self._make_app(risk_prompt=risk)

        assert app.check_action("quit_dashboard", ()) is False
