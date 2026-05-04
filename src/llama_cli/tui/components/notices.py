"""System notice renderer and widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget

if TYPE_CHECKING:
    from llama_cli.tui.viewmodel import DashboardViewModel


class NoticesRenderer:
    """Builds active system notice text."""

    def render(self, notices: list[str]) -> Text:
        text = Text()
        for notice in notices[-2:]:
            text.append(f"! {notice}\n", style="bold yellow")
        return text


class NoticesWidget(Widget):
    """Active system notices."""

    DEFAULT_CSS = """
    NoticesWidget {
        height: auto;
    }
    NoticesWidget.hidden {
        display: none;
    }
    """

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__()
        self._view_model = view_model
        self._renderer = NoticesRenderer()
        self._has_notices = False

    def on_mount(self) -> None:
        self.set_interval(0.25, self._check_notices)

    def _check_notices(self) -> None:
        notices = self._view_model.system_notices()
        messages = self._view_model.status_messages()
        if (notices or messages) and not self._has_notices:
            self._has_notices = True
            self.remove_class("hidden")
        elif not notices and not messages and self._has_notices:
            self._has_notices = False
            self.add_class("hidden")

    def render(self) -> RenderResult:
        notices = self._view_model.system_notices()
        messages = self._view_model.status_messages()
        if not notices and not messages:
            return Text()
        text = self._renderer.render(notices)
        for msg in messages:
            text.append(f"• {msg}\n", style="green")
        return text
