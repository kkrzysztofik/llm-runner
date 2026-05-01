"""Command menu builder and CommandMenu widget for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget

if TYPE_CHECKING:
    from llama_cli.tui.controller import TUIApp


def build_command_menu(
    profile_request: str | None,
    risk_panel: Panel | None,
    active_risk_kind: str | None,
) -> Text:
    """Build an htop-style bottom command menu.

    Context-aware: shows different commands based on active TUI state.
    """
    menu = Text()

    def _add_item(key: str, desc: str) -> None:
        menu.append(f" {key} ", style="bold cyan reverse")
        menu.append(f" {desc} ", style="white")

    if profile_request is not None:
        _add_item("1", "Balanced")
        _add_item("2", "Fast")
        _add_item("3", "Quality")
        _add_item("^C", "Cancel")
    elif risk_panel is not None:
        _add_item("y", "Confirm")
        _add_item("n", "Abort")
        if active_risk_kind != "vram":
            _add_item("q", "Quit")
    else:
        _add_item("q", "Quit")
        _add_item("r", "Refresh")
        _add_item("a", "Add slot")
        _add_item("b", "Build")
        _add_item("s", "Smoke")
        _add_item("P", "Profile")
        _add_item("^C", "Stop")

    return menu


# ---------------------------------------------------------------------------
# Compound Widget
# ---------------------------------------------------------------------------


class CommandMenu(Widget):
    """Bottom command menu bar — context-aware hotkey display.

    Replaces the anonymous ``Static(id="menu")`` widget.  Renders different
    key hints depending on the current TUI mode (normal / profile-select /
    risk-acknowledgement).
    """

    DEFAULT_CSS = """
    CommandMenu {
        height: 1;
    }
    """

    def __init__(self, controller: TUIApp) -> None:
        super().__init__(id="menu")
        self._controller = controller

    def render(self) -> RenderResult:
        ctrl = self._controller
        return build_command_menu(
            ctrl.profile_request,
            ctrl.risk_panel,
            ctrl.active_risk_kind,
        )
