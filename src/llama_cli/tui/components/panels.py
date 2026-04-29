"""Pure panel-builder functions for TUI column and slot widgets."""

from typing import Any

import psutil
from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from llama_cli.colors import Colors
from llama_manager import GPUStats, LogBuffer, ServerConfig, SlotState

# ---- Lookup tables (formerly TUIApp class variables) ----

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


def build_column_panel(
    cfg: ServerConfig,
    buffer: LogBuffer,
    gpu: GPUStats | None,
    host: str,
    stale_warning: str | None = None,
) -> Panel:
    """Build the per-server column panel with logs and GPU stats."""
    color_code = Colors.get_code(cfg.alias)
    color_style = color_code if color_code else "white"

    header = Text()
    header.append(f"{cfg.alias.upper()} ", style=f"bold {color_style}")
    header.append(f"http://{host}:{cfg.port}/v1", style="dim")
    header.append("\n")
    header.append(
        f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}", style="cyan"
    )

    if stale_warning:
        header.append("\n")
        header.append(stale_warning, style="yellow")

    header.append("\n\n")

    logs_text = buffer.get_text(empty_message="Waiting for output...")
    logs = Panel(Text(logs_text), title="Logs", border_style="dim")
    gpu_renderable = (
        Panel(Text(gpu.format_stats_text()), title="GPU", border_style="yellow")
        if gpu is not None
        else Panel(Text("GPU stats unavailable", style="dim"), title="GPU", border_style="dim")
    )
    return Panel(Group(header, gpu_renderable, logs), border_style=color_style)


def build_placeholder_panel() -> Panel:
    """Build a placeholder panel for the right column when only one config exists."""
    return Panel(
        Text("No secondary config", style="dim"),
        title="Status",
        border_style="dim",
    )


def build_slot_section(
    cfg: ServerConfig,
    slot_states: dict[str, str],
    server_processes: dict[str, Any],
    log_buffers: dict[str, LogBuffer],
    host: str,
) -> Text:
    """Build the status Text for a single slot."""
    alias = cfg.alias
    state = slot_states.get(alias, SlotState.OFFLINE.value)

    status = state
    if state == SlotState.RUNNING.value:
        proc = server_processes.get(alias)
        if not proc or not (proc.pid and psutil.pid_exists(proc.pid)):
            status = SlotState.CRASHED.value

    backend_label = BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"])
    color = STATUS_COLORS.get(status, "white")

    header = Text()
    header.append(f"[{alias}] ", style="bold")
    header.append(f"{status.upper()} ", style=color)
    header.append(f"| {backend_label} ", style="cyan")
    header.append(f"| http://{host}:{cfg.port}", style="dim")
    header.append("\n")

    buffer = log_buffers.get(alias)
    if buffer is not None:
        log_lines = buffer.get_lines()[-3:] if buffer.get_lines() else []
        log_text = "\n".join(log_lines) if log_lines else "  (no logs yet)"
        header.append(Text(log_text + "\n", style="dim"))

    return header


def build_slot_status_panel(
    configs: list[ServerConfig],
    slot_states: dict[str, str],
    server_processes: dict[str, Any],
    log_buffers: dict[str, LogBuffer],
    host: str,
) -> Panel:
    """Build a panel showing per-slot status (health, logs, GPU stats, backend label)."""
    sections: list[Any] = [
        build_slot_section(cfg, slot_states, server_processes, log_buffers, host) for cfg in configs
    ]

    if not sections:
        empty_msg = Text(
            "No slots configured.\n\n"
            "Press 'a' to add a new slot\n"
            "or run with a mode:\n"
            "  llm-runner tui both",
            style="dim",
        )
        sections = [empty_msg]

    return Panel(Group(*sections), title="Slot Status", border_style="blue")
