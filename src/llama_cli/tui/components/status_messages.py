"""Status message renderer."""

from rich.text import Text


class StatusMessagesRenderer:
    """Builds inline alert text from status messages."""

    def render(self, messages: list[str]) -> Text | None:
        if not messages:
            return None

        text = Text()
        text.append("ALERTS\n", style="bold yellow")
        for msg in messages:
            text.append(f"• {msg}\n", style="green")
        return text
