"""Pure alert/status panel builders for the TUI."""

import time

import psutil
from rich import box
from rich.panel import Panel
from rich.text import Text

from llama_manager import LaunchResult

from ..constants import STATUS_PREFIX, STYLE_BOLD_RED, STYLE_BOLD_YELLOW


def build_status_panel(launch_result: LaunchResult) -> Panel | None:
    """Build a panel describing a launch result.

    Returns None if the launch succeeded (no panel needed).
    """
    if launch_result.is_success():
        return None

    status_text = Text()
    if launch_result.is_blocked():
        status_text.append(STATUS_PREFIX, style=STYLE_BOLD_RED)
        status_text.append("BLOCKED", style="bold red reverse")
        status_text.append("\n\n")
        if launch_result.errors is not None:
            status_text.append("FR-005 Error Details:\n", style=STYLE_BOLD_YELLOW)
            for error_detail in launch_result.errors.errors:
                status_text.append(f"  - {error_detail.error_code}\n", style="red")
                status_text.append(
                    f"    failed_check: {error_detail.failed_check}\n",
                    style="dim",
                )
                status_text.append(
                    f"    why_blocked: {error_detail.why_blocked}\n",
                    style="dim",
                )
                status_text.append(
                    f"    how_to_fix: {error_detail.how_to_fix}\n\n",
                    style="dim",
                )
        return Panel(
            status_text,
            title="[red]Launch Failed[/red]",
            border_style="red",
        )

    status_text.append(STATUS_PREFIX, style=STYLE_BOLD_YELLOW)
    status_text.append("DEGRADED", style=STYLE_BOLD_YELLOW)
    status_text.append(" (partial success)\n\n", style="dim")
    launched = launch_result.launched or []
    if launched:
        status_text.append("Launched slots:\n", style="bold green")
        for slot_id in launched:
            status_text.append(f"  + {slot_id}\n", style="green")
        status_text.append("\n")
    for warning in launch_result.warnings or []:
        status_text.append(f"  ! {warning}\n", style="yellow")
    return Panel(
        status_text,
        title="[yellow]Launch Degraded[/yellow]",
        border_style="yellow",
    )


def build_risk_panel_required(kind: str = "hardware") -> Panel:
    """Build the risk acknowledgement-required panel."""
    text = Text()
    text.append("RISK STATUS: ", style="bold")
    text.append(" ACKNOWLEDGEMENT REQUIRED ", style="bold red reverse")
    if kind == "vram":
        text.append("\nVRAM heuristics indicate a high risk of Out-Of-Memory errors.")
    else:
        text.append("\nHardware validation warnings detected. Launch is blocked.")

    text.append("\n\nPress ", style="dim")
    text.append("[y]", style="bold green")
    text.append(" to acknowledge and continue, or ", style="dim")
    text.append("[n]", style="bold red")
    text.append(" to abort.", style="dim")

    return Panel(text, title="Risk Management", border_style="red")


def build_risk_panel_acknowledged() -> Panel:
    """Build the risk acknowledged panel."""
    text = Text()
    text.append("RISK STATUS: ", style="bold")
    text.append(" ACKNOWLEDGED ", style="bold green reverse")
    text.append("\nRisky operations (privileged ports, non-loopback bind) were acknowledged.")
    return Panel(text, title="Risk Management", border_style="green")


def build_profile_status_panel(
    profile_status: dict[str, str],
    profile_flavor: dict[str, str],
) -> Panel | None:
    """Build a panel showing active profile operations.

    ``profile_status`` should already be filtered to non-idle entries.
    """
    if not profile_status:
        return None

    text = Text()
    for alias, status in profile_status.items():
        flavor = profile_flavor.get(alias, "unknown")
        if status == "running":
            text.append("\u25b6 ", style="yellow")
            text.append(f"Profiling {alias}: {flavor} ", style="yellow")
            text.append("[running...]", style="dim")
        elif status == "done":
            text.append("\u2713 ", style="green")
            text.append(f"Profile {alias}: {flavor} ", style="green")
            text.append("[done]", style="dim")
        elif status == "failed":
            text.append("\u2717 ", style="red")
            text.append(f"Profile {alias}: {flavor} ", style="red")
            text.append("[failed]", style="dim")
        text.append("\n")

    return Panel(text, title="Profile Status", border_style="yellow")


def build_status_messages_panel(messages: list[str]) -> Text | None:
    """Build inline alert text from a list of status messages.

    Returns None if the list is empty.
    """
    if not messages:
        return None

    text = Text()
    text.append("ALERTS\n", style="bold yellow")
    for msg in messages:
        text.append(f"• {msg}\n", style="green")
    return text


def build_gpu_telemetry_panel(lines: list[str]) -> Panel | None:
    """Build the GPU telemetry panel from pre-formatted stat lines.

    Returns None if no lines are provided.
    """
    if not lines:
        return None

    text = Text("\n".join(lines))
    return Panel(text, title="GPU Telemetry", border_style="yellow")


def build_system_status_panel(
    gpu_lines: list[str],
    notices: list[str] | None = None,
) -> Panel:
    """Build a single htop-style system status panel for the top area."""
    notices = notices or []
    content_width = 116
    right_block_width = min(44, max(36, content_width // 3))
    left_block_width = max(28, content_width - right_block_width - 3)

    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    uptime_s = int(time.time() - psutil.boot_time())
    tasks, threads, running = _get_task_stats()

    try:
        load_1, load_5, load_15 = psutil.getloadavg()
    except (AttributeError, OSError):
        load_1 = load_5 = load_15 = -1.0

    text = Text()
    text.append("Date/Time: ", style="bright_cyan")
    text.append(time.strftime("%Y-%m-%d %H:%M:%S"), style="bright_white")
    text.append("\n")

    for line in _build_core_grid_lines(cpu_per_core, left_width=left_block_width):
        text.append(line)
        text.append("\n")

    mem_bar_width = max(20, left_block_width - 6)
    mem_left = Text()
    mem_left.append("Mem", style="bright_cyan")
    mem_left.append("[", style="bright_cyan")
    _append_segmented_bar(mem_left, mem.percent, mem_bar_width)
    mem_left.append("]", style="bright_cyan")

    mem_right = Text()
    _append_aligned_value(
        mem_right,
        f"{_format_bytes(mem.used)}/{_format_bytes(mem.total)}",
        width=14,
        style="bright_white",
    )
    mem_right.append("  Tasks:", style="bright_cyan")
    mem_right.append(f" {tasks:>3}", style="bright_white")
    mem_right.append("  Thr:", style="cyan")
    mem_right.append(f" {threads:>4}", style="bright_white")
    mem_right.append("  Run:", style="green")
    mem_right.append(f" {running:>2}", style="bright_white")
    _append_two_column_line(text, mem_left, mem_right, left_width=left_block_width)

    swp_left = Text()
    swp_left.append("Swp", style="bright_cyan")
    swp_left.append("[", style="bright_cyan")
    _append_segmented_bar(swp_left, swap.percent, mem_bar_width)
    swp_left.append("]", style="bright_cyan")

    swp_right = Text()
    _append_aligned_value(
        swp_right,
        f"{_format_bytes(swap.used)}/{_format_bytes(swap.total)}",
        width=14,
        style="dim",
    )
    if load_1 >= 0:
        swp_right.append("  Load:", style="bright_cyan")
        swp_right.append(f" {load_1:.2f}", style="bright_white")
        swp_right.append(f" {load_5:.2f}", style="cyan")
        swp_right.append(f" {load_15:.2f}", style="bright_blue")
    else:
        swp_right.append("  Load: n/a", style="dim")
    _append_two_column_line(text, swp_left, swp_right, left_width=left_block_width)

    uptime_left = Text()
    uptime_right = Text()
    _append_aligned_value(uptime_right, "", width=14, style="dim")
    uptime_right.append("  Uptime:", style="bright_cyan")
    uptime_right.append(f" {_format_uptime(uptime_s)}", style="bright_white")
    _append_two_column_line(text, uptime_left, uptime_right, left_width=left_block_width)

    if gpu_lines:
        text.append("\n")
        text.append("GPU ", style="bold yellow")
        text.append(" | ".join(line.replace("\n", " ").strip() for line in gpu_lines), style="yellow")
        text.append("\n")

    for notice in notices[-2:]:
        text.append(f"! {notice}\n", style="bold yellow")

    return Panel(
        text,
        title="",
        box=box.SQUARE,
        border_style="black",
        padding=(0, 0),
    )


def _usage_bar(percent: float, width: int = 10) -> str:
    filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * width))
    return "|" * filled + " " * (width - filled)


def _usage_color(percent: float) -> str:
    if percent >= 85:
        return "red"
    if percent >= 60:
        return "yellow"
    return "green"


def _append_segmented_bar(text: Text, percent: float, width: int) -> None:
    """Append an htop-like multi-color bar segment to ``text``."""
    filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * width))
    if filled <= 0:
        text.append(" " * width, style="dim")
        return

    palette = ["green", "green", "green", "green", "cyan", "magenta", "yellow", "yellow"]
    for i in range(filled):
        text.append("|", style=palette[i % len(palette)])
    if filled < width:
        text.append(" " * (width - filled), style="dim")


def _format_uptime(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _build_core_grid_lines(cpu_per_core: list[float], left_width: int) -> list[Text]:
    if not cpu_per_core:
        return [Text("No CPU data", style="dim")]

    rows = 3
    cols = (len(cpu_per_core) + rows - 1) // rows
    cell_width = max(14, left_width // max(1, cols))
    bar_width = 5
    lines: list[Text] = []
    for row in range(rows):
        line = Text()
        for col in range(cols):
            idx = col * rows + row
            if idx >= len(cpu_per_core):
                continue
            pct = cpu_per_core[idx]
            cell = Text()
            cell.append(f"{idx:>2}", style="bright_blue")
            cell.append("[", style="bright_white")
            cell.append(_usage_bar(pct, width=bar_width), style=_usage_color(pct))
            cell.append("]", style="bright_white")
            cell.append(" ")
            cell.append(f"{pct:5.1f}%", style="bright_white" if pct > 0 else "dim")
            line.append_text(cell)
            pad = max(1, cell_width - len(cell.plain))
            line.append(" " * pad)
        lines.append(line)
    return lines


def _append_two_column_line(text: Text, left: Text, right: Text, left_width: int = 66) -> None:
    left_plain = left.plain
    pad = max(3, left_width - len(left_plain))
    text.append_text(left)
    text.append(" " * pad)
    text.append_text(right)
    text.append("\n")


def _append_aligned_value(text: Text, value: str, width: int, style: str) -> None:
    text.append(value.rjust(width), style=style)


def _format_bytes(num_bytes: int) -> str:
    gib = num_bytes / (1024**3)
    if gib >= 10:
        return f"{gib:,.1f}G"
    return f"{gib:,.2f}G"


def _get_task_stats() -> tuple[int, int, int]:
    task_count = 0
    thread_count = 0
    running_count = 0
    for proc in psutil.process_iter(attrs=["status", "num_threads"]):
        task_count += 1
        info = proc.info
        thread_count += int(info.get("num_threads") or 0)
        if info.get("status") == psutil.STATUS_RUNNING:
            running_count += 1
    return task_count, thread_count, running_count
