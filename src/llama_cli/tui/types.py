"""Dataclass types shared across the TUI submodule."""

from dataclasses import dataclass

from rich.panel import Panel
from rich.text import Text


@dataclass(frozen=True)
class DashboardSnapshot:
    """Current dashboard renderables for Textual widgets."""

    alerts: Panel
    left: Panel | None
    right: Panel
    menu: Text
