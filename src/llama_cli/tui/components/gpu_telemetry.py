"""GPU telemetry renderers and widgets."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from llama_cli.tui.viewmodel import DashboardViewModel


def _flatten_gpu_lines(gpu_lines: list[str]) -> str:
    return " | ".join(line.replace("\n", " ").strip() for line in gpu_lines)


class GPUTelemetryWidget(Widget):
    """One-line GPU stats for all configured GPUs."""

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(classes="gpu-telemetry")
        self._view_model = view_model

    def _sync_visibility(self, gpu_lines: list[str]) -> None:
        if not gpu_lines:
            self.add_class("hidden")
        else:
            self.remove_class("hidden")

    def compose(self) -> ComposeResult:
        gpu_lines = self._view_model.gpu_telemetry_lines()
        self._sync_visibility(gpu_lines)
        if not gpu_lines:
            return

        yield Horizontal(
            Static("GPU", classes="gpu-telemetry-label"),
            Static(_flatten_gpu_lines(gpu_lines), classes="gpu-telemetry-value"),
            classes="gpu-telemetry-row",
        )
