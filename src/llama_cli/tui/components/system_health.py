"""System health widgets."""

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Static

from llama_cli.tui.types import (
    CPU_CORE_BAR_WIDTH,
    DEFAULT_CONTENT_WIDTH,
    MAX_CONTENT_WIDTH,
    MIN_CONTENT_WIDTH,
    CPUCoreSnapshot,
)

from .digital_clock import LLM_RUNNER_LOGO, DigitalClockWidget
from .gpu_stats import usage_fill

if TYPE_CHECKING:
    from llama_cli.tui.viewmodel import DashboardViewModel

_SYSTEM_INFO_LABEL = "system-health-label system-info-label"
_SYSTEM_INFO_PRIMARY_VALUE = "system-info-value system-info-primary-value"
_SYSTEM_INFO_ROW = "system-health-inline-row system-info-row"

_NO_CPU_DATA = "No CPU data"


def _content_width(width: int | None) -> int:
    if width is None or width <= 0:
        return DEFAULT_CONTENT_WIDTH
    return min(MAX_CONTENT_WIDTH, max(MIN_CONTENT_WIDTH, width))


def _memory_bar_width(content_width: int) -> int:
    label_width = len("Mem[]")
    value_width = 14
    column_gap = 3
    return max(4, content_width - label_width - column_gap - value_width)


def _usage_color(percent: float) -> str:
    if percent >= 85:
        return "red"
    if percent >= 60:
        return "yellow"
    return "green"


class SystemHealthWidget(Widget):
    """Container for focused system health sections."""

    def __init__(self, provider: DashboardViewModel | None = None, **kwargs: Any) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"system-health {classes}".strip(), **kwargs)
        self._view_model = provider

    def compose(self) -> ComposeResult:
        if self._view_model is None:
            return
        yield DateTimeWidget(self._view_model)
        yield CPUUsageWidget(self._view_model)
        yield Horizontal(
            MemorySwapWidget(self._view_model),
            SystemInfoWidget(self._view_model),
            classes="system-health-resource-row",
        )


class DateTimeWidget(Widget):
    """Date/time section for the system health area."""

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(classes="system-health-section system-health-datetime")
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        snapshot = self._view_model.current_datetime_snapshot()
        yield Horizontal(
            Static(LLM_RUNNER_LOGO, markup=True, classes="llm-runner-logo"),
            Static("", classes="datetime-header-spacer"),
            Horizontal(
                Static(snapshot.date_text, classes="datetime-date"),
                DigitalClockWidget(),
                classes="datetime-far-right",
            ),
            classes="system-health-datetime-row",
        )


class CPUUsageWidget(Widget):
    """CPU per-core usage section for the system health area."""

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(classes="system-health-section system-health-cpu")
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        rows = self._view_model.cpu_usage_rows(width=self.size.width)
        if not rows:
            yield Static(_NO_CPU_DATA, classes="system-health-muted-value")
            return

        for row in rows:
            yield Horizontal(
                *(self._core_cell(core) for core in row),
                classes="system-health-cpu-row",
            )

    def _core_cell(self, core: CPUCoreSnapshot) -> Container:
        value_class = (
            "system-health-value system-health-muted-value"
            if core.percent <= 0
            else "system-health-value"
        )
        return Container(
            Static(f"{core.index:>2}", classes="cpu-core-index"),
            Static(
                usage_fill(core.percent, CPU_CORE_BAR_WIDTH),
                classes=f"cpu-core-bar system-health-meter-{_usage_color(core.percent)}",
            ),
            Static(f"{core.percent:5.1f}%", classes=f"cpu-core-percent {value_class}"),
            classes="cpu-core-cell",
        )


class MemorySwapWidget(Widget):
    """Memory and swap usage section."""

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(classes="system-health-section system-health-memory-swap")
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        for row in self._view_model.memory_usage_rows():
            value_class = (
                "system-health-value system-health-muted-value"
                if row.label == "Swp" and row.percent <= 0
                else "system-health-value"
            )
            yield Horizontal(
                Static(row.label, classes="memory-swap-label"),
                Static(
                    usage_fill(
                        row.percent,
                        _memory_bar_width(_content_width(self.size.width)),
                    ),
                    classes=f"memory-swap-bar system-health-meter-{_usage_color(row.percent)}",
                ),
                Static(row.value_text, classes=value_class),
                classes="memory-swap-row",
            )


class SystemInfoWidget(Widget):
    """Task, load, and uptime section."""

    def __init__(self, view_model: DashboardViewModel) -> None:
        super().__init__(classes="system-health-section system-health-system-info")
        self._view_model = view_model

    def compose(self) -> ComposeResult:
        snapshot = self._view_model.system_info_snapshot()

        yield Horizontal(
            Static("Tasks:", classes=_SYSTEM_INFO_LABEL),
            Static(f"{snapshot.tasks:>3}", classes=_SYSTEM_INFO_PRIMARY_VALUE),
            Static("Thr:", classes="system-info-label system-info-secondary-label"),
            Static(
                f"{snapshot.threads:>4}",
                classes=_SYSTEM_INFO_PRIMARY_VALUE,
            ),
            Static("Run:", classes="system-info-label system-info-running-label"),
            Static(f"{snapshot.running:>2}", classes=_SYSTEM_INFO_PRIMARY_VALUE),
            classes=_SYSTEM_INFO_ROW,
        )

        if snapshot.load_values is None:
            yield Horizontal(
                Static("Load:", classes=_SYSTEM_INFO_LABEL),
                Static("n/a", classes="system-info-value system-info-muted-value"),
                classes=_SYSTEM_INFO_ROW,
            )
        else:
            load_1, load_5, load_15 = snapshot.load_values
            yield Horizontal(
                Static("Load:", classes=_SYSTEM_INFO_LABEL),
                Static(
                    f"{load_1:.2f}",
                    classes=_SYSTEM_INFO_PRIMARY_VALUE,
                ),
                Static(
                    f"{load_5:.2f}",
                    classes="system-info-value system-info-secondary-value",
                ),
                Static(
                    f"{load_15:.2f}",
                    classes="system-info-value system-info-tertiary-value",
                ),
                classes=_SYSTEM_INFO_ROW,
            )

        yield Horizontal(
            Static("Uptime:", classes=_SYSTEM_INFO_LABEL),
            Static(snapshot.uptime, classes=_SYSTEM_INFO_PRIMARY_VALUE),
            classes=_SYSTEM_INFO_ROW,
        )
