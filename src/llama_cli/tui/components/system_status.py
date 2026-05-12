"""Top system status renderer and compound widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.widget import Widget

from .gpu_telemetry import GPUTelemetryWidget
from .system_health import SystemHealthWidget

if TYPE_CHECKING:
    from llama_cli.tui.viewmodel import DashboardViewModel


class SystemStatusWidget(Widget):
    """Top status bar composed from focused child widgets."""

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(id="alerts", classes="system-status")
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        yield SystemHealthWidget()
        yield GPUTelemetryWidget(self._view_model)
