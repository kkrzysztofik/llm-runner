"""Widget classes for TUI column areas (logs + GPU stats)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import psutil
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget

from llama_cli.colors import Colors
from llama_manager import GPUStats, LogBuffer, ServerConfig, SlotState

if TYPE_CHECKING:
    from llama_cli.tui.controller import TUIApp

# ---------------------------------------------------------------------------
# Module-level lookup tables
# ---------------------------------------------------------------------------

BACKEND_LABELS: dict[str, str] = {
    "sycl": "SYCL",
    "cuda": "CUDA",
    "llama_cpp": "CPU",
}

STATUS_COLORS: dict[str, str] = {
    SlotState.RUNNING.value: "green",
    SlotState.LAUNCHING.value: "yellow",
    SlotState.DEGRADED.value: "yellow",
    SlotState.CRASHED.value: "red",
    SlotState.OFFLINE.value: "dim",
    SlotState.IDLE.value: "dim",
}


def _resolve_slot_status(
    alias: str,
    slot_states: dict[str, str],
    server_processes: dict[str, Any],
) -> str:
    """Resolve final slot status from tracked state + process liveness."""
    state = slot_states.get(alias, SlotState.OFFLINE.value)
    status = state
    if state == SlotState.RUNNING.value:
        proc = server_processes.get(alias)
        if not proc:
            status = SlotState.CRASHED.value
        elif hasattr(proc, "poll"):
            # subprocess.Popen — poll() returns None while running
            if proc.poll() is not None:
                status = SlotState.CRASHED.value
        elif not (proc.pid and psutil.pid_exists(proc.pid)):
            status = SlotState.CRASHED.value
    return status


# ---------------------------------------------------------------------------
# GPUStatsPanel
# ---------------------------------------------------------------------------


class GPUStatsPanel(Widget):
    """Compact htop-style GPU telemetry panel (Rich Panel renderable).

    Constructed with a ``GPUStats`` snapshot (or ``None`` when unavailable)
    and used via ``render()`` as an embedded Rich renderable inside a larger
    column panel — it is never yielded in ``compose()``.
    """

    def __init__(self, gpu: GPUStats | None) -> None:
        super().__init__()
        self._gpu = gpu

    def render(self) -> Panel:  # type: ignore[override]
        if self._gpu is None:
            return Panel(
                Text("GPU stats unavailable", style="dim"), title="GPU", border_style="dim"
            )
        return self._build_panel(self._gpu)

    @staticmethod
    def _build_panel(gpu: GPUStats) -> Panel:
        stats = gpu.get_stats_snapshot()
        gpu_pct = GPUStatsPanel._parse_percent(stats.get("gpu_util"))
        mem_pct = GPUStatsPanel._parse_percent(stats.get("mem_util"))
        cpu_pct = GPUStatsPanel._parse_percent(stats.get("cpu"))
        sys_mem_pct = GPUStatsPanel._parse_percent(stats.get("mem"))

        text = Text()
        text.append("Device: ", style="bright_white")
        text.append(str(stats.get("device", "N/A")), style="cyan")
        text.append("\n")

        if gpu_pct is not None or mem_pct is not None:
            GPUStatsPanel._append_usage_line(
                text, "GPU", gpu_pct, str(stats.get("gpu_util", "N/A"))
            )
            text.append("  ", style="dim")
            GPUStatsPanel._append_usage_line(
                text, "VRAM", mem_pct, str(stats.get("mem_util", "N/A"))
            )
        else:
            GPUStatsPanel._append_usage_line(text, "CPU", cpu_pct, str(stats.get("cpu", "N/A")))
            text.append("  ", style="dim")
            GPUStatsPanel._append_usage_line(text, "Mem", sys_mem_pct, str(stats.get("mem", "N/A")))
        text.append("\n")

        temp = str(stats.get("temp", "N/A"))
        power = str(stats.get("power", "N/A"))
        if temp != "N/A" or power != "N/A":
            text.append("Temp:", style="bright_cyan")
            text.append(f" {temp}", style="bright_white" if temp != "N/A" else "dim")
            text.append("  ", style="dim")
            text.append("Power:", style="bright_cyan")
            text.append(f" {power}", style="bright_white" if power != "N/A" else "dim")

        return Panel(text, title="GPU", border_style="yellow")

    @staticmethod
    def _append_usage_line(text: Text, label: str, percent: float | None, raw: str) -> None:
        text.append(f"{label}", style="bright_blue")
        text.append("[", style="white")
        if percent is None:
            text.append("?" * 10, style="dim")
        else:
            filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * 10))
            color = GPUStatsPanel._usage_color(percent)
            text.append("|" * filled, style=color)
            if filled < 10:
                text.append(" " * (10 - filled), style="dim")
        text.append("]", style="white")
        text.append(f" {raw}", style="bright_white" if raw != "N/A" else "dim")

    @staticmethod
    def _usage_color(percent: float) -> str:
        if percent >= 85:
            return "red"
        if percent >= 60:
            return "yellow"
        return "green"

    @staticmethod
    def _parse_percent(value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.upper() == "N/A":
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            return float(text)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# ServerColumnPanel
# ---------------------------------------------------------------------------


class ServerColumnPanel(Widget):
    """Per-server column panel: header + GPU stats + logs (Rich Panel renderable).

    Encapsulates all rendering for a single running slot.  Used via
    ``render()`` as an embedded Rich renderable inside ``ServerLogPanel``
    and the controller's ``_build_column_panel`` helper — never yielded in
    ``compose()``.
    """

    def __init__(
        self,
        cfg: ServerConfig,
        buffer: LogBuffer,
        gpu: GPUStats | None,
        host: str,
        stale_warning: str | None = None,
        slot_states: dict[str, str] | None = None,
        server_processes: dict[str, Any] | None = None,
        is_unsaved: bool = False,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._buffer = buffer
        self._gpu = gpu
        self._host = host
        self._stale_warning = stale_warning
        self._slot_states = slot_states or {}
        self._server_processes = server_processes or {}
        self._is_unsaved = is_unsaved

    def render(self) -> Panel:  # type: ignore[override]
        cfg = self._cfg
        color_code = Colors.get_code(cfg.alias)
        color_style = color_code if color_code else "white"

        header = self._build_header(color_style)
        gpu_panel = GPUStatsPanel(self._gpu).render()
        logs_text = self._buffer.get_text(empty_message="Waiting for output...")
        logs = Panel(Text(logs_text), title="Logs", border_style="dim")
        return Panel(Group(header, gpu_panel, logs), border_style=color_style)

    def _build_header(self, color_style: str) -> Text:
        cfg = self._cfg
        status = _resolve_slot_status(cfg.alias, self._slot_states, self._server_processes)
        backend_label = BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"])
        status_color = STATUS_COLORS.get(status, "white")

        header = Text()
        header.append(f"[{cfg.alias}] ", style=f"bold {color_style}")
        if self._is_unsaved:
            header.append("UNSAVED ", style="bold yellow")
        header.append(f"{status.upper()} ", style=status_color)
        header.append(f"| {backend_label} ", style="cyan")
        header.append(f"| http://{self._host}:{cfg.port}", style="dim")
        header.append("\n")
        header.append(
            f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}",
            style="cyan",
        )
        if self._stale_warning:
            header.append("\n")
            header.append(self._stale_warning, style="yellow")
        header.append("\n\n")
        return header


# ---------------------------------------------------------------------------
# SlotStatusPanel
# ---------------------------------------------------------------------------


class SlotStatusPanel(Widget):
    """Multi-slot status panel listing every configured slot (Rich Panel renderable).

    Shown in the primary column when no slots are running yet, or when the
    primary slot index is out of range.  Used via ``render()`` — never
    yielded in ``compose()``.
    """

    def __init__(
        self,
        configs: list[ServerConfig],
        slot_states: dict[str, str],
        server_processes: dict[str, Any],
        log_buffers: dict[str, LogBuffer],
        host: str,
        unsaved_slots: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._configs = configs
        self._slot_states = slot_states
        self._server_processes = server_processes
        self._log_buffers = log_buffers
        self._host = host
        self._unsaved_slots = unsaved_slots or set()

    def render(self) -> Panel:  # type: ignore[override]
        sections: list[Any] = [self._render_slot_section(cfg) for cfg in self._configs]
        if not sections:
            sections = [
                Text(
                    "No slots configured.\n\n"
                    "Press 'a' to add a new slot\n"
                    "or run with a mode:\n"
                    "  llm-runner tui both",
                    style="dim",
                )
            ]
        return Panel(Group(*sections), title="Slot Status", border_style="blue")

    def _render_slot_section(self, cfg: ServerConfig) -> Text:
        alias = cfg.alias
        status = _resolve_slot_status(alias, self._slot_states, self._server_processes)
        backend_label = BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"])
        color = STATUS_COLORS.get(status, "white")

        section = Text()
        section.append(f"[{alias}] ", style="bold")
        if alias in self._unsaved_slots:
            section.append("UNSAVED ", style="bold yellow")
        section.append(f"{status.upper()} ", style=color)
        section.append(f"| {backend_label} ", style="cyan")
        section.append(f"| http://{self._host}:{cfg.port}", style="dim")
        section.append("\n")

        buffer = self._log_buffers.get(alias)
        if buffer is not None:
            log_lines = buffer.get_lines()[-3:] if buffer.get_lines() else []
            log_text = "\n".join(log_lines) if log_lines else "  (no logs yet)"
            section.append(Text(log_text + "\n", style="dim"))

        return section


# ---------------------------------------------------------------------------
# ServerLogPanel — Textual widget used in compose()
# ---------------------------------------------------------------------------


class ServerLogPanel(Widget):
    """Per-slot column widget: logs + GPU stats.

    Accepts ``slot_index`` (0 = primary, 1 = secondary) and renders the
    appropriate column content.  When the slot index is out of range the
    widget falls back gracefully so it is always visible rather than hidden.
    """

    DEFAULT_CSS = """
    ServerLogPanel {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, slot_index: int, controller: TUIApp) -> None:
        super().__init__(classes="column")
        self._slot_index = slot_index
        self._controller = controller

    def render(self) -> RenderResult:
        ctrl = self._controller
        configs = ctrl.configs
        if self._slot_index >= len(configs):
            if self._slot_index == 0:
                # Primary slot absent → show the "no slots" panel
                return SlotStatusPanel(
                    [],
                    ctrl.slot_states,
                    ctrl.server_processes,
                    ctrl.log_buffers,
                    ctrl.config.host,
                ).render()
            return Panel(
                Text("No secondary config", style="dim"),
                title="Status",
                border_style="dim",
            )

        cfg = configs[self._slot_index]
        buffer = ctrl.log_buffers[cfg.alias]
        gpu: GPUStats | None = (
            ctrl.gpu_stats[self._slot_index] if self._slot_index < len(ctrl.gpu_stats) else None
        )
        return ServerColumnPanel(
            cfg,
            buffer,
            gpu,
            ctrl.config.host,
            stale_warning=ctrl.get_stale_warning(cfg),
            slot_states=ctrl.slot_states,
            server_processes=ctrl.server_processes,
            is_unsaved=cfg.alias in ctrl.unsaved_slots,
        ).render()
