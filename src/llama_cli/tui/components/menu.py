"""Command menu renderer and widget for the TUI."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult, RenderResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from llama_cli.tui.types import CommandMenuState
    from llama_cli.tui.viewmodel import DashboardViewModel


def _command_menu_items(state: CommandMenuState) -> list[tuple[str, str]]:
    """Return the hotkey items for the current command menu state."""
    items: list[tuple[str, str]] = []

    def add_item(key: str, desc: str) -> None:
        items.append((key, desc))

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
        add_item("c", "Config")
        add_item("^C", "Stop")

    return items


# ---------------------------------------------------------------------------
# Compound Widget
# ---------------------------------------------------------------------------


class CommandMenu(Widget):
    """Bottom command menu bar — context-aware hotkey display.

    Replaces the anonymous ``Static(id="menu")`` widget.  Renders different
    key hints depending on the current TUI mode (normal / profile-select /
    risk-acknowledgement).
    """

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(id="menu", classes="command-menu")
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        for key, desc in _command_menu_items(self._view_model.command_menu()):
            yield Horizontal(
                Static(key, classes="command-menu-key"),
                Static(desc, classes="command-menu-description"),
                classes="command-menu-item",
            )

    def render(self) -> RenderResult:
        return ""
