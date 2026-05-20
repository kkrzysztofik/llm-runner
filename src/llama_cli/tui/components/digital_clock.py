"""llm-runner logo and digital clock for the datetime header row."""

from __future__ import annotations

import re
from datetime import datetime

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
_LOGO_ROWS = max(len(_LLM_BLOCK), len(_ROBOT_BLOCK))


def _clean_markup(s: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", s)


_LLM_WIDTH = max(len(_clean_markup(line)) for line in _LLM_BLOCK)
_ROBOT_WIDTH = max(len(_clean_markup(line)) for line in _ROBOT_BLOCK)


def _pad_markup_line(s: str, width: int) -> str:
    clean = _clean_markup(s)
    needed = width - len(clean)
    if needed > 0:
        return s + " " * needed
    return s


def _build_logo_rows() -> list[str]:
    llm_rows = _LLM_BLOCK + [" " * _LLM_WIDTH] * (_LOGO_ROWS - len(_LLM_BLOCK))
    robot_rows = _ROBOT_BLOCK + [" " * _ROBOT_WIDTH] * (_LOGO_ROWS - len(_ROBOT_BLOCK))
    return [
        _pad_markup_line(llm_rows[i], _LLM_WIDTH)
        + _LOGO_GAP
        + _pad_markup_line(robot_rows[i], _ROBOT_WIDTH)
        for i in range(_LOGO_ROWS)
    ]


LLM_RUNNER_LOGO = "\n".join(_build_logo_rows())


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
