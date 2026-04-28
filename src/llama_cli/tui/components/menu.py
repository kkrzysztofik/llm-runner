"""Bottom command menu builder for the TUI."""

from rich.panel import Panel
from rich.text import Text


def build_command_menu(
    profile_request: str | None,
    slot_config_state: dict[str, str],
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
    elif slot_config_state:
        _add_item("Enter", "Next")
        _add_item("Backspace", "Edit")
        _add_item("Esc", "Cancel")
    elif risk_panel is not None:
        _add_item("y", "Confirm")
        _add_item("n", "Abort")
        if active_risk_kind != "vram":
            _add_item("q", "Quit")
    else:
        _add_item("q", "Quit")
        _add_item("r", "Refresh")
        _add_item("a", "Add slot")
        _add_item("P", "Profile")
        _add_item("^C", "Stop")

    return menu
