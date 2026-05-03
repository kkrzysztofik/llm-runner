"""Server log column widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget

from llama_cli.tui.types import ServerColumnState

from .server_column import ServerColumnPanel
from .slot_status import SlotStatusPanel

if TYPE_CHECKING:
    from llama_cli.tui.viewmodel import DashboardViewModel


class ServerLogPanel(Widget):
    """Per-slot column widget: logs + GPU stats."""

    DEFAULT_CSS = """
    ServerLogPanel {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, slot_index: int, view_model: DashboardViewModel) -> None:
        super().__init__(classes="column")
        self._slot_index = slot_index
        self._view_model = view_model

    def render(self) -> RenderResult:
        state = self._view_model.column(
            self._slot_index,
            stale_warning=None,
        )
        if state is None:
            if self._slot_index == 0:
                return SlotStatusPanel(self._view_model.slot_status(configs=[])).render()
            return Panel(
                Text("No secondary config", style="dim"),
                title="Status",
                border_style="dim",
            )

        stale_warning = self._view_model.stale_warning(state.config)
        return ServerColumnPanel(
            ServerColumnState(
                config=state.config,
                buffer=state.buffer,
                gpu=state.gpu,
                host=state.host,
                stale_warning=stale_warning,
                slot_states=state.slot_states,
                server_processes=state.server_processes,
                is_unsaved=state.is_unsaved,
            )
        ).render()
