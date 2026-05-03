"""Top system status renderer and compound widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.panel import Panel
from textual.app import ComposeResult
from textual.widget import Widget

from .gpu_telemetry import GPUTelemetryLineRenderer, GPUTelemetryWidget
from .notices import NoticesRenderer, NoticesWidget
from .system_health import SystemHealthRenderer, SystemHealthWidget

if TYPE_CHECKING:
    from llama_cli.tui.types import SystemStatusState
    from llama_cli.tui.viewmodel import DashboardViewModel


class SystemStatusPanelRenderer:
    """Builds the combined top system status panel."""

    def __init__(self) -> None:
        self._health_renderer = SystemHealthRenderer()
        self._gpu_renderer = GPUTelemetryLineRenderer()
        self._notices_renderer = NoticesRenderer()

    def render_panel(self, state: SystemStatusState) -> Panel:
        text = self._health_renderer.render()

        if state.gpu_lines:
            text.append("\n")
            text.append_text(self._gpu_renderer.render(state.gpu_lines))

        notice_text = self._notices_renderer.render(state.notices)
        text.append_text(notice_text)

        return Panel(
            text,
            title="",
            box=box.SQUARE,
            border_style="black",
            padding=(0, 0),
        )


class SystemStatusWidget(Widget):
    """Top status bar composed from focused child widgets."""

    DEFAULT_CSS = """
    SystemStatusWidget {
        height: auto;
        max-height: 35%;
    }
    """

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(id="alerts")
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        yield SystemHealthWidget()
        yield GPUTelemetryWidget(self._view_model)
        yield NoticesWidget(self._view_model)
