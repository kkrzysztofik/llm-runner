"""Server log column widget."""

import logging
import time
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Static

from .server_column import ServerColumnPanel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_cli.tui.viewmodel import DashboardViewModel


class ServerLogPanel(Widget):
    """Per-slot column widget: logs + GPU stats."""

    def __init__(self, slot_index: int, view_model: DashboardViewModel) -> None:
        super().__init__(classes="column server-log-panel")
        self._slot_index = slot_index
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        start = time.perf_counter()
        state = self._view_model.column(self._slot_index)
        if state is None:
            logger.debug(
                "ServerLogPanel.compose: placeholder slot_index=%d duration_ms=%.1f",
                self._slot_index,
                (time.perf_counter() - start) * 1000,
            )
            message = (
                "No slots configured.\n\nPress 'a' to add a new slot."
                if self._slot_index == 0
                else "No secondary config"
            )
            yield Container(
                Static("Status", classes="panel-title server-placeholder-title"),
                Static(message, classes="server-placeholder-body"),
                classes="server-placeholder",
            )
            return

        logger.debug(
            "ServerLogPanel.compose: server slot_index=%d alias=%s duration_ms=%.1f",
            self._slot_index,
            state.alias,
            (time.perf_counter() - start) * 1000,
        )
        yield ServerColumnPanel(state)
