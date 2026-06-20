"""Read-only modal showing aggregate token stats by run profile."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label

if TYPE_CHECKING:
    from llama_manager.config.profiles import SlotProfileSpec
    from llama_manager.slot_stats import ProfileStatsAggregate


class ProfileStatsScreen(ModalScreen[None]):
    """Read-only profile aggregate stats screen."""

    BINDINGS = [
        Binding("escape", "cancel", "Close"),
        Binding("ctrl+c", "cancel", "Close"),
    ]

    CSS = """
    ProfileStatsScreen {
        align: center middle;
    }
    .profile-stats-dialog {
        width: 80%;
        max-width: 95%;
        max-height: 90%;
        height: auto;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }
    .profile-stats-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    .profile-stats-header {
        color: $text-muted;
        text-style: bold;
        height: 1;
    }
    .profile-stats-body {
        width: 100%;
        height: 1fr;
        min-height: 8;
        overflow-y: auto;
        margin-bottom: 1;
    }
    .profile-stats-row {
        width: 100%;
        height: 1;
    }
    .profile-stats-row:hover {
        background: $boost;
    }
    .profile-stats-profile {
        width: 2fr;
    }
    .profile-stats-number {
        width: 1fr;
        text-align: right;
    }
    .profile-stats-updated {
        width: 1fr;
        text-align: right;
    }
    .profile-stats-empty {
        color: $text-muted;
        text-style: italic;
        margin-top: 2;
    }
    .profile-stats-actions {
        height: 3;
        align-horizontal: right;
    }
    .profile-stats-actions Button {
        margin-left: 1;
        min-width: 12;
    }
    """

    def __init__(
        self,
        stats_by_profile: dict[str, ProfileStatsAggregate],
        profiles: list[tuple[SlotProfileSpec, str]],
    ) -> None:
        super().__init__()
        self._stats_by_profile = stats_by_profile
        self._profile_labels = {
            spec.profile_id: spec.description or spec.alias or spec.profile_id
            for spec, _source in profiles
        }

    def compose(self) -> ComposeResult:
        rows = list(self._stats_rows())
        yield Container(
            Label("Profile Stats", classes="profile-stats-title"),
            Horizontal(
                Label("Profile", classes="profile-stats-profile profile-stats-header"),
                Label("Input", classes="profile-stats-number profile-stats-header"),
                Label("Output", classes="profile-stats-number profile-stats-header"),
                Label("Sessions", classes="profile-stats-number profile-stats-header"),
                Label("Updated", classes="profile-stats-updated profile-stats-header"),
                classes="profile-stats-row",
            ),
            Vertical(
                *(
                    rows
                    if rows
                    else [
                        Label(
                            "No aggregate profile stats recorded yet.",
                            classes="profile-stats-empty",
                        )
                    ]
                ),
                classes="profile-stats-body",
            ),
            Horizontal(
                Button("Close", id="close-profile-stats", classes="modal-button-cancel"),
                classes="profile-stats-actions",
            ),
            classes="profile-stats-dialog",
        )

    def _stats_rows(self) -> list[Horizontal]:
        rows: list[Horizontal] = []
        for profile_id, stats in sorted(self._stats_by_profile.items()):
            label = self._profile_labels.get(profile_id, profile_id)
            rows.append(
                Horizontal(
                    Label(f"{profile_id} - {label}", classes="profile-stats-profile"),
                    Label(str(max(0, stats.tokens_in)), classes="profile-stats-number"),
                    Label(str(max(0, stats.tokens_out)), classes="profile-stats-number"),
                    Label(str(max(0, stats.sessions_count)), classes="profile-stats-number"),
                    Label(_format_updated_at(stats.updated_at), classes="profile-stats-updated"),
                    classes="profile-stats-row",
                )
            )
        return rows

    def action_cancel(self) -> None:
        """Close the screen."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Close on the modal close button."""
        if event.button.id == "close-profile-stats":
            self.dismiss(None)


def _format_updated_at(value: float) -> str:
    if value <= 0:
        return "--"
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")
    except ValueError, OSError, OverflowError:
        return "--"
