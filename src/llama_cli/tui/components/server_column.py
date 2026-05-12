"""Per-server column widget."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Static

from llama_cli.tui.types import ServerColumnState

from .gpu_stats import GPUStatsPanel
from .slot_status import BACKEND_LABELS, SlotStatusResolver


class ServerColumnPanel(Widget):
    """Per-server column: header, telemetry, and logs."""

    def __init__(
        self,
        state: ServerColumnState,
    ) -> None:
        super().__init__(classes="server-column")
        self._state = state
        self._resolver = SlotStatusResolver()

    def compose(self) -> ComposeResult:
        logs_text = self._state.buffer.get_text(empty_message="Waiting for output...")
        yield self._build_header()
        yield GPUStatsPanel(self._state.gpu)
        yield Container(
            Static("Logs", classes="panel-title server-log-title"),
            Static(logs_text, classes="server-log-content"),
            classes="server-logs",
        )

    def _build_header(self) -> Container:
        cfg = self._state.config
        status = self._resolver.resolve(
            cfg.alias,
            self._state.slot_states,
            self._state.server_processes,
        )
        backend_label = BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"])
        status_class = f"server-column-status-{status.replace('_', '-')}"

        header_children: list[Widget] = [
            Horizontal(
                Static(f"[{cfg.alias}]", classes="server-column-alias"),
                Static("UNSAVED", classes="server-column-unsaved")
                if self._state.is_unsaved
                else Static("", classes="hidden"),
                Static(status.upper(), classes=f"server-column-status {status_class}"),
                Static(backend_label, classes="server-column-backend"),
                Static(
                    f"http://{self._state.host}:{cfg.port}",
                    classes="server-column-url",
                ),
                classes="server-column-header-row",
            ),
            Static(
                f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}",
                classes="server-column-config",
            ),
        ]
        if self._state.stale_warning:
            header_children.append(
                Static(self._state.stale_warning, classes="server-column-warning")
            )
        return Container(*header_children, classes="server-column-header")
