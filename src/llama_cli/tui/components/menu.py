"""Command menu renderer and widget for the TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget

if TYPE_CHECKING:
    from llama_cli.tui.types import CommandMenuState
    from llama_cli.tui.viewmodel import DashboardViewModel


class CommandMenuRenderer:
    """Builds the htop-style bottom command menu."""

    def render(self, state: CommandMenuState) -> Text:
        """Render the command menu for the current UI state."""
        menu = Text()

        def add_item(key: str, desc: str) -> None:
            menu.append(f" {key} ", style="bold cyan reverse")
            menu.append(f" {desc} ", style="white")

        if state.build_request:
            add_item("1", "SYCL")
            add_item("2", "CUDA")
            add_item("3", "Both")
            add_item("^C", "Cancel")
        elif state.smoke_request:
            add_item("1", "Both")
            add_item("2", "Active Slot")
            add_item("^C", "Cancel")
        elif state.profile_request is not None:
            add_item("1", "Balanced")
            add_item("2", "Fast")
            add_item("3", "Quality")
            add_item("^C", "Cancel")
        elif state.risk_prompt is not None:
            add_item("y", "Confirm")
            add_item("n", "Abort")
            if state.risk_prompt.kind != "vram":
                add_item("q", "Quit")
        else:
            add_item("q", "Quit")
            add_item("r", "Refresh")
            add_item("a", "Add slot")
            add_item("b", "Build")
            add_item("s", "Smoke")
            add_item("P", "Profile")
            add_item("^C", "Stop")

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

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(id="menu")
        self._view_model = view_model
        self._renderer = CommandMenuRenderer()

    def render(self) -> RenderResult:
        return self._renderer.render(self._view_model.command_menu())
