"""GPU telemetry renderers and widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget

if TYPE_CHECKING:
    from llama_cli.tui.viewmodel import DashboardViewModel


class GPUTelemetryPanelRenderer:
    """Builds the GPU telemetry panel from pre-formatted stat lines."""

    def render_panel(self, lines: list[str]) -> Panel | None:
        if not lines:
            return None

        text = Text("\n".join(lines))
        return Panel(text, title="GPU Telemetry", border_style="yellow")


class GPUTelemetryLineRenderer:
    """Builds one-line GPU telemetry text."""

    def render(self, gpu_lines: list[str]) -> Text:
        text = Text()
        if not gpu_lines:
            return text
        text.append("GPU ", style="bold yellow")
        text.append(
            " | ".join(line.replace("\n", " ").strip() for line in gpu_lines),
            style="yellow",
        )
        text.append("\n")
        return text


class GPUTelemetryWidget(Widget):
    """One-line GPU stats for all configured GPUs."""

    DEFAULT_CSS = """
    GPUTelemetryWidget {
        height: 1;
    }
    GPUTelemetryWidget.hidden {
        display: none;
    }
    """

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__()
        self._view_model = view_model
        self._renderer = GPUTelemetryLineRenderer()

    def _check_visibility(self) -> None:
        gpu_lines = self._view_model.gpu_telemetry_lines()
        if not gpu_lines:
            self.add_class("hidden")
        else:
            self.remove_class("hidden")

    def render(self) -> RenderResult:
        self._check_visibility()
        gpu_lines = self._view_model.gpu_telemetry_lines()
        return self._renderer.render(gpu_lines)
