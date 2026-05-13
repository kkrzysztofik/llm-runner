"""Top system status renderer and compound widget."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget

from .system_health import SystemHealthWidget


class SystemStatusWidget(Widget):
    """Top status bar composed from focused child widgets."""

    def __init__(self) -> None:
        super().__init__(id="alerts", classes="system-status")

    def compose(self) -> ComposeResult:
        yield SystemHealthWidget()
