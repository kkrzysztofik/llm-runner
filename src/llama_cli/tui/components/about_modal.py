"""AboutModal — read-only app info dialog."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Static

_REPO_URL = "https://github.com/kkrzysztofik/llm-runner"


def _app_version() -> str:
    try:
        return version("llm-runner")
    except PackageNotFoundError:
        return "dev"


_CONTENT = """\
[bold $accent]llm-runner[/] [dim]v{version}[/]

[dim]TUI for managing llama.cpp server instances (Intel SYCL + NVIDIA CUDA)[/]

[bold]License[/]   MIT · Copyright © 2026 Krzysztof Krzysztofik
[bold]Repo[/]      {repo}

─────────────────────────────────────────────
[bold]Key Bindings[/]
─────────────────────────────────────────────
  [bold $accent]q[/]   Quit          [bold $accent]r[/]   Refresh
  [bold $accent]b[/]   Build         [bold $accent]a[/]   Add Slot
  [bold $accent]c[/]   Config        [bold $accent]p[/]   Profiles
  [bold $accent]h[/]   About
─────────────────────────────────────────────

[dim italic]Press any key to close[/]
"""


class AboutModal(ModalScreen[None]):
    """Read-only About dialog.  Dismissed by pressing any key."""

    def compose(self) -> ComposeResult:
        content = _CONTENT.format(version=_app_version(), repo=_REPO_URL)
        yield Vertical(
            Static(content, id="about-content"),
            id="about-dialog",
            classes="about-dialog",
        )

    def on_key(self, event: Key) -> None:
        event.stop()
        self.dismiss()
