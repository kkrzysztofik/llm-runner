"""Per-server column panel."""

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.widget import Widget

from llama_cli.colors import Colors
from llama_cli.tui.types import ServerColumnState

from .gpu_stats import GPUStatsPanel
from .slot_status import BACKEND_LABELS, STATUS_COLORS, SlotStatusResolver


class ServerColumnPanel(Widget):
    """Per-server column panel: header + GPU stats + logs."""

    def __init__(
        self,
        state: ServerColumnState,
    ) -> None:
        super().__init__()
        self._state = state
        self._resolver = SlotStatusResolver()

    def render(self) -> Panel:  # type: ignore[override]
        cfg = self._state.config
        color_code = Colors.get_code(cfg.alias)
        color_style = color_code if color_code else "white"

        header = self._build_header(color_style)
        gpu_panel = GPUStatsPanel(self._state.gpu).render()
        logs_text = self._state.buffer.get_text(empty_message="Waiting for output...")
        logs = Panel(Text(logs_text), title="Logs", border_style="dim")
        return Panel(Group(header, gpu_panel, logs), border_style=color_style)

    def _build_header(self, color_style: str) -> Text:
        cfg = self._state.config
        status = self._resolver.resolve(
            cfg.alias,
            self._state.slot_states,
            self._state.server_processes,
        )
        backend_label = BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"])
        status_color = STATUS_COLORS.get(status, "white")

        header = Text()
        header.append(f"[{cfg.alias}] ", style=f"bold {color_style}")
        if self._state.is_unsaved:
            header.append("UNSAVED ", style="bold yellow")
        header.append(f"{status.upper()} ", style=status_color)
        header.append(f"| {backend_label} ", style="cyan")
        header.append(f"| http://{self._state.host}:{cfg.port}", style="dim")
        header.append("\n")
        header.append(
            f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}",
            style="cyan",
        )
        if self._state.stale_warning:
            header.append("\n")
            header.append(self._state.stale_warning, style="yellow")
        header.append("\n\n")
        return header
