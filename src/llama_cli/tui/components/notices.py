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

    def render(self) -> RenderResult:
        notices = self._view_model.system_notices()
        if not notices:
            self.add_class("hidden")
            return Text()
        self.remove_class("hidden")
        return self._renderer.render(notices)
