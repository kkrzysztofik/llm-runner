"""Top system status renderer and compound widget."""

from textual.app import ComposeResult
from textual.widget import Widget

from .system_health import SystemHealthProvider, SystemHealthWidget


class SystemStatusWidget(Widget):
    """Top status bar composed from focused child widgets."""

    def __init__(self, provider: SystemHealthProvider | None = None) -> None:
        super().__init__(id="alerts", classes="system-status")
        self._provider = provider

    def compose(self) -> ComposeResult:
        yield SystemHealthWidget(self._provider)
