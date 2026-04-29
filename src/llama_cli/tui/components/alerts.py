"""Pure alert/status panel builders for the TUI."""

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


def build_status_messages_panel(messages: list[str]) -> Panel | None:
    """Build a panel from a list of status messages.

    Returns None if the list is empty.
    """
    if not messages:
        return None

    text = Text()
    for msg in messages:
        text.append(msg + "\n", style="green")
    return Panel(text, title="Status", border_style="green")


def build_gpu_telemetry_panel(lines: list[str]) -> Panel | None:
    """Build the GPU telemetry panel from pre-formatted stat lines.

    Returns None if no lines are provided.
    """
    if not lines:
        return None

    text = Text("\n".join(lines))
    return Panel(text, title="GPU Telemetry", border_style="yellow")
