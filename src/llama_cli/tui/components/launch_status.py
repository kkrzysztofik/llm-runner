"""Launch status panel renderer."""

from rich.panel import Panel
from rich.text import Text

from llama_manager import LaunchResult

from ..constants import STATUS_PREFIX, STYLE_BOLD_RED, STYLE_BOLD_YELLOW


class LaunchStatusPanelRenderer:
    """Builds a panel describing a launch result."""

    def render(self, launch_result: LaunchResult) -> Panel | None:
        """Return None if the launch succeeded and no panel is needed."""
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
