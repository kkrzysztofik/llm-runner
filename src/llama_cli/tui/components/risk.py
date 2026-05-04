"""Risk acknowledgement panel renderer."""

from typing import Literal

from rich.panel import Panel
from rich.text import Text


class RiskPanelRenderer:
    """Builds risk acknowledgement panels."""

    def required(self, kind: Literal["vram", "hardware"] = "hardware") -> Panel:
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

    def acknowledged(self, risk_description: str | None = None) -> Panel:
        text = Text()
        text.append("RISK STATUS: ", style="bold")
        text.append(" ACKNOWLEDGED ", style="bold green reverse")
        if risk_description:
            text.append(f"\nRisky operation(s) acknowledged: {risk_description}")
        else:
            text.append("\nRisky operation(s) were acknowledged.")
        return Panel(text, title="Risk Management", border_style="green")
