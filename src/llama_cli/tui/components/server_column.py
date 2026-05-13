"""Per-server column widget."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Static

from llama_cli.tui.types import ServerColumnState

from .gpu_stats import GPUStatsPanel


class ServerColumnPanel(Widget):
    """Per-server column: header, telemetry, and logs."""

    def __init__(
        self,
        state: ServerColumnState,
    ) -> None:
        super().__init__(classes="server-column")
        self._state = state

    def compose(self) -> ComposeResult:
        yield self._build_header()
        yield GPUStatsPanel(self._state.gpu_stats)
        yield Container(
            Static("Logs", classes="panel-title server-log-title"),
            Static(self._state.logs_text, classes="server-log-content"),
            classes="server-logs",
        )

    def _build_header(self) -> Container:
        header_children: list[Widget] = [
            Horizontal(
                Static(f"[{self._state.alias}]", classes="server-column-alias"),
                Static("UNSAVED", classes="server-column-unsaved")
                if self._state.is_unsaved
                else Static("", classes="hidden"),
                Static(
                    self._state.status.upper(),
                    classes=f"server-column-status {self._state.status_class}",
                ),
                Static(self._state.backend_label, classes="server-column-backend"),
                Static(
                    self._state.url,
                    classes="server-column-url",
                ),
                classes="server-column-header-row",
            ),
            Static(self._state.config_summary, classes="server-column-config"),
        ]
        if self._state.stale_warning:
            header_children.append(
                Static(self._state.stale_warning, classes="server-column-warning")
            )
        return Container(*header_children, classes="server-column-header")
