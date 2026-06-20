"""Per-server column widget."""

import logging

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import Log, Static

from llama_cli.tui.types import ServerColumnState

from .gpu_stats import GPUStatsPanel


class ServerColumnPanel(Widget):
    """Per-server column: header, telemetry, and logs."""

    def __init__(
        self,
        state: ServerColumnState,
    ) -> None:
        super().__init__(classes="server-column")
        self._state = state

    def compose(self) -> ComposeResult:
        yield self._build_header()
        yield GPUStatsPanel(self._state.gpu_stats)
        yield self._build_runtime_stats()
        yield self._build_logs()

    def _build_header(self) -> Container:
        header_children: list[Widget] = [
            Horizontal(
                Static(self._state.profile_name, classes="server-column-profile-name"),
                Static(
                    self._state.status_label,
                    classes=f"server-column-status {self._state.status_class}",
                ),
                classes="server-column-title-row",
            ),
            Horizontal(
                Static(self._state.backend_label, classes="server-column-backend"),
                Static(self._state.config_summary, classes="server-column-config"),
                classes="server-column-meta-row",
            ),
            Horizontal(
                Static("URL", classes="server-column-url-label"),
                Static(self._state.url, classes="server-column-url"),
                classes="server-column-url-row",
            ),
        ]
        if self._state.stale_warning:
            header_children.append(
                Static(self._state.stale_warning, classes="server-column-warning")
            )
        return Container(*header_children, classes="server-column-header")

    def _build_runtime_stats(self) -> Container:
        stats = self._state.runtime_stats
        return Container(
            Static("Stats", classes="panel-title slot-stats-title"),
            Horizontal(
                self._stat_cell("TPS", stats.tps),
                self._stat_cell("PP", stats.pp),
                self._stat_cell("Tok In", stats.tokens_in),
                self._stat_cell("Tok Out", stats.tokens_out),
                classes="slot-stats-row",
            ),
            classes="slot-stats",
        )

    @staticmethod
    def _stat_cell(label: str, value: str) -> Container:
        return Container(
            Static(label, classes="slot-stats-label"),
            Static(value, classes="slot-stats-value"),
            classes="slot-stats-cell",
        )

    def _build_logs(self) -> Container:
        log = Log(max_lines=500, auto_scroll=True, classes="server-log-content")
        log._llm_runner_lines = self._state.log_lines  # type: ignore[attr-defined]
        return Container(
            Static("Logs", classes="panel-title server-log-title"),
            log,
            classes="server-logs",
        )

    def on_mount(self) -> None:
        try:
            log = self.query_one(".server-log-content", Log)
            log.write_lines(list(self._state.log_lines))
        except NoMatches:
            # Log widget not yet in the tree; lines will be applied on next refresh.
            pass
        except Exception:
            logging.exception(
                "server_column: failed to write initial log lines for %s", self._state.profile_name
            )
