"""Profile status panel renderer."""

from rich.panel import Panel
from rich.text import Text


class ProfileStatusPanelRenderer:
    """Builds the active profile operations panel."""

    def render(
        self,
        profile_status: dict[str, str],
        profile_flavor: dict[str, str],
    ) -> Panel | None:
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
