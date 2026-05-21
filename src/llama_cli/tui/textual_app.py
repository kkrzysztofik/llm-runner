"""Textual App shell for the llm-runner TUI dashboard."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.css.query import NoMatches
from textual.widgets import Footer

if TYPE_CHECKING:
    from .controller import DashboardController

from .components.build import BuildModalScreen
from .components.config_modal import ConfigModal, ConfigPayload
from .components.modal import AddSlotModal
from .components.server_log import ServerLogPanel
from .components.system_health import (
    CPUUsageWidget,
    MemorySwapWidget,
    SystemInfoWidget,
)
from .components.system_status import SystemStatusWidget
from .types import BuildWizardResult

# ---------------------------------------------------------------------------
# Extracted pure helper: profile options caching
# ---------------------------------------------------------------------------


def _profile_options_cached(
    view_model: object,
    config: object,
    cache: list[tuple[str, str]] | None,
    cache_config_id: int | None,
) -> tuple[list[tuple[str, str]], int | None]:
    """Return (options, config_id) with caching logic.

    Extracted from ``DashboardApp._build_profile_options`` for testability.
    """
    config_id = id(config)
    if cache is not None and cache_config_id == config_id:
        return cache, config_id
    options = view_model.profile_options(config)  # type: ignore[union-attr]
    return options, config_id


_RISK_HIDDEN_ACTIONS = frozenset(
    {"refresh_dashboard", "add_slot", "build", "open_config"},
)
_NORMAL_HIDDEN_ACTIONS = frozenset({"confirm", "reject"})


class DashboardApp(App[None]):
    """Textual shell for the llm-runner dashboard."""

    TITLE = "llm-runner"
    CSS_PATH = [
        "textual_app.tcss",
        "system_status.tcss",
        "dashboard_panels.tcss",
        "modals.tcss",
    ]
    BINDINGS = [
        Binding("q", "quit_dashboard", "Quit", priority=True),
        Binding(
            "ctrl+c",
            "cancel_pending_prompt",
            "Cancel",
            key_display="^C",
        ),
        Binding("escape", "cancel_pending_prompt", "Cancel", show=False),
        Binding("r", "refresh_dashboard", "Refresh"),
        Binding("b", "build", "Build"),
        Binding("a", "add_slot", "Add Slot"),
        Binding("c", "open_config", "Config"),
        Binding("y", "confirm", "Confirm"),
        Binding("n", "reject", "Abort"),
    ]

    def __init__(self, controller: DashboardController) -> None:
        super().__init__()
        self.controller = controller
        self.view_model = controller.view_model
        self._last_notified_status_ts: float = 0.0
        self._active_notice_toasts: set[str] = set()
        self._profile_options_cache: list[tuple[str, str]] | None = None
        self._profile_cache_config_id: int | None = None
        self.last_build_backend: str = "sycl"

    def compose(self) -> ComposeResult:
        with Container(id="dashboard"):
            yield SystemStatusWidget(self.view_model)
            with Container(id="content"):
                for i in range(self.view_model.server_column_count()):
                    yield ServerLogPanel(i, self.view_model)
        yield Footer(show_command_palette=False)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Control which bindings appear in the footer for the current mode."""
        state = self.view_model.command_menu()

        if state.build_request:
            return action == "cancel_pending_prompt"

        if state.risk_prompt is not None:
            if action in _RISK_HIDDEN_ACTIONS:
                return False
            return not (action == "quit_dashboard" and state.risk_prompt.kind == "vram")

        return action not in _NORMAL_HIDDEN_ACTIONS

    def on_mount(self) -> None:
        self.refresh_dashboard()
        self.set_interval(0.25, self.refresh_dashboard)

    def action_quit_dashboard(self) -> None:
        self.controller.request_quit()
        if not self.controller.running:
            self.exit()

    def action_interrupt_dashboard(self) -> None:
        self.controller.interrupt()
        if not self.controller.running:
            self.exit()

    def action_refresh_dashboard(self) -> None:
        self.controller.refresh_display()
        self.refresh_dashboard()

    def action_add_slot(self) -> None:
        self.push_screen(
            AddSlotModal(profile_options=self._build_profile_options()),
            self._handle_add_slot_modal_result,
        )

    def action_open_config(self) -> None:
        self.push_screen(
            ConfigModal(config=self.controller.config),
            self._handle_config_modal_result,
        )

    def _handle_config_modal_result(self, result: ConfigPayload | None) -> None:
        if result is not None:
            self.controller.save_config(result)
        self.refresh_dashboard()

    def _handle_add_slot_modal_result(self, result: dict[str, str] | None) -> None:
        if result is None:
            self.controller.cancel_add_slot_form()
        else:
            self.controller.add_slot_from_form(result)
            self._reconcile_server_log_panels()
        self.refresh_dashboard()

    def _handle_build_modal_result(self, result: BuildWizardResult | None) -> None:
        if result is None:
            self.controller.cancel_pending_prompt()
        else:
            self.last_build_backend = result.backends[0] if result.backends else "sycl"
            self.controller.handle_build_selection(result.backends, result.options)
        self.refresh_dashboard()

    def _reconcile_server_log_panels(self) -> None:
        """Ensure ServerLogPanel widgets match the current slot count."""
        container = self.query_one("#content", Container)
        current_panels = list(container.query(ServerLogPanel))
        needed = self.view_model.server_column_count()
        if len(current_panels) > needed:
            for panel in current_panels[needed:]:
                panel.remove()
        for i in range(len(current_panels), needed):
            container.mount(ServerLogPanel(i, self.view_model))

    def _build_profile_options(self) -> list[tuple[str, str]]:
        options, config_id = _profile_options_cached(
            self.view_model,
            self.controller.config,
            self._profile_options_cache,
            self._profile_cache_config_id,
        )
        self._profile_options_cache = options
        self._profile_cache_config_id = config_id
        return options

    def action_build(self) -> None:
        screen = BuildModalScreen(last_backend=self.last_build_backend)
        self.push_screen(screen, self._handle_build_modal_result)

    def action_confirm(self) -> None:
        self.controller.acknowledge_risk()
        self.refresh_dashboard()

    def action_reject(self) -> None:
        self.controller.reject_risk()
        if not self.controller.running:
            self.exit()
        self.refresh_dashboard()

    def action_cancel_pending_prompt(self) -> None:
        cancelled = self.controller.cancel_pending_prompt()
        if not cancelled:
            self.controller.interrupt()
        self.refresh_dashboard()

    def refresh_dashboard(self) -> None:
        if not self.controller.running:
            self.exit()
            return

        self._emit_status_toasts()

        # Refresh each leaf widget.  SystemStatusWidget and SystemHealthWidget
        # use compose(), so their children own their own repaints.
        for widget_type in (CPUUsageWidget, MemorySwapWidget, SystemInfoWidget):
            with contextlib.suppress(NoMatches):
                self.query_one(widget_type).refresh(recompose=True)
        for panel in self.query(ServerLogPanel):
            panel.refresh(recompose=True)
        self.refresh_bindings()

    def _emit_status_toasts(self) -> None:
        notices = self.view_model.system_notices()
        current_notices = set(notices)
        for notice in notices:
            if notice not in self._active_notice_toasts:
                self.notify(notice, title="Alert", severity="warning")
        self._active_notice_toasts = current_notices

        updates = self.controller.get_status_messages_since(self._last_notified_status_ts)
        if not updates:
            return
        for ts, message in updates:
            self.notify(message, title="Status", severity="information")
            self._last_notified_status_ts = max(self._last_notified_status_ts, ts)
