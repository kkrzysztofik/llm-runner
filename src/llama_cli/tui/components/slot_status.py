"""Slot status resolution and fallback panel."""

from __future__ import annotations

from typing import Any

import psutil
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.widget import Widget

from llama_cli.tui.types import SlotStatusState
from llama_manager import ServerConfig, SlotState

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


class SlotStatusResolver:
    """Resolves final slot status from tracked state and process liveness."""

    def resolve(
        self,
        alias: str,
        slot_states: dict[str, str],
        server_processes: dict[str, Any],
    ) -> str:
        state = slot_states.get(alias, SlotState.OFFLINE.value)
        status = state
        if state == SlotState.RUNNING.value:
            proc = server_processes.get(alias)
            if not proc:
                status = SlotState.CRASHED.value
            elif hasattr(proc, "poll"):
                if proc.poll() is not None:
                    status = SlotState.CRASHED.value
            elif not (proc.pid and psutil.pid_exists(proc.pid)):
                status = SlotState.CRASHED.value
        return status


class SlotStatusPanel(Widget):
    """Multi-slot status panel listing every configured slot."""

    def __init__(
        self,
        state: SlotStatusState,
    ) -> None:
        super().__init__()
        self._state = state
        self._resolver = SlotStatusResolver()

    def render(self) -> Panel:  # type: ignore[override]
        sections: list[Any] = [self._render_slot_section(cfg) for cfg in self._state.configs]
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
        status = self._resolver.resolve(
            alias,
            self._state.slot_states,
            self._state.server_processes,
        )
        backend_label = BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"])
        color = STATUS_COLORS.get(status, "white")

        section = Text()
        section.append(f"[{alias}] ", style="bold")
        if alias in self._state.unsaved_slots:
            section.append("UNSAVED ", style="bold yellow")
        section.append(f"{status.upper()} ", style=color)
        section.append(f"| {backend_label} ", style="cyan")
        section.append(f"| http://{self._state.host}:{cfg.port}", style="dim")
        section.append("\n")

        buffer = self._state.log_buffers.get(alias)
        if buffer is not None:
            log_lines = buffer.get_lines()[-3:] if buffer.get_lines() else []
            log_text = "\n".join(log_lines) if log_lines else "  (no logs yet)"
            section.append(Text(log_text + "\n", style="dim"))

        return section
