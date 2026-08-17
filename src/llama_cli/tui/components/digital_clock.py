"""llm-runner logo and digital clock for the datetime header row."""

from __future__ import annotations

import re
from datetime import datetime
from itertools import zip_longest

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Digits

_LLM_BLOCK = [
    "[ansi_bright_red]██╗[/ansi_bright_red]       [ansi_bright_green]██╗[/ansi_bright_green]       [ansi_bright_magenta]███╗   ███╗[/ansi_bright_magenta]",
    "[ansi_bright_red]██║[/ansi_bright_red]       [ansi_bright_green]██║[/ansi_bright_green]       [ansi_bright_magenta]████╗ ████║[/ansi_bright_magenta]",
    "[ansi_bright_red]██║[/ansi_bright_red]       [ansi_bright_green]██║[/ansi_bright_green]       [ansi_bright_magenta]██╔███╗██║ [/ansi_bright_magenta]",
    "[ansi_bright_red]██║[/ansi_bright_red]       [ansi_bright_green]██║[/ansi_bright_green]       [ansi_bright_magenta]██║╚██╔╝██║[/ansi_bright_magenta]",
    "[ansi_bright_red]██║[/ansi_bright_red]       [ansi_bright_green]██║[/ansi_bright_green]       [ansi_bright_magenta]██║ ╚═╝ ██║[/ansi_bright_magenta]",
    "[ansi_bright_red]██╚═══╗[/ansi_bright_red]   [ansi_bright_green]██╚═══╗[/ansi_bright_green]   [ansi_bright_magenta]██║     ██║[/ansi_bright_magenta]",
    "[ansi_bright_red]╚█████╝[/ansi_bright_red]   [ansi_bright_green]╚█████╝[/ansi_bright_green]   [ansi_bright_magenta]╚═╝     ╚═╝[/ansi_bright_magenta]",
]

_ROBOT_BLOCK = [
    "       [ansi_white]╭───╮[/ansi_white]",
    "      [ansi_white]/ [ansi_bright_red]■[/ansi_bright_red] [ansi_blue]■[/ansi_blue][/ansi_white] [ansi_white]\\ [/ansi_white]",
    "    [ansi_white]╭┴───────┴╮[/ansi_white]",
    "   [ansi_white]╭┤[/ansi_white] [ansi_blue]█[/ansi_blue] [ansi_bright_white]███[/ansi_bright_white] [ansi_blue]█[/ansi_blue] [ansi_white]├╮[/ansi_white]",
    "   [ansi_white]││[/ansi_white] [ansi_blue]█[/ansi_blue] [ansi_bright_white]░░░[/ansi_bright_white] [ansi_blue]█[/ansi_blue] [ansi_white]││[/ansi_white]",
    "   [ansi_white]██▄       ▄██[/ansi_white]",
    "      [ansi_white]▀█████▀[/ansi_white]",
]

_LOGO_GAP = "  "


def _pad_markup_line(s: str, width: int) -> str:
    needed = width - len(re.sub(r"\[[^\]]*\]", "", s))
    return s + " " * max(0, needed)


_ROBOT_WIDTH = max(len(re.sub(r"\[[^\]]*\]", "", r)) for r in _ROBOT_BLOCK)

LLM_RUNNER_LOGO = "\n".join(
    llm + _LOGO_GAP + _pad_markup_line(robot, _ROBOT_WIDTH)
    for llm, robot in zip_longest(_LLM_BLOCK, _ROBOT_BLOCK, fillvalue="")
)


class DigitalClockWidget(Widget):
    """Block digital time (HH:MM:SS) for the far-right datetime cluster."""

    def __init__(self) -> None:
        super().__init__(classes="datetime-digits-wrap")

    def compose(self) -> ComposeResult:
        yield Digits("", classes="datetime-digits")

    def on_mount(self) -> None:
        self._tick()
        self.set_interval(1, self._tick, name="digital-clock-tick")

    def _tick(self) -> None:
        self.query_one(Digits).update(datetime.now().strftime("%H:%M:%S"))
