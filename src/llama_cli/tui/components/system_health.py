"""System health widgets."""

from typing import Protocol

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Static

from llama_cli.tui.types import (
    CPUCoreSnapshot,
    DateTimeSnapshot,
    MemoryUsageSnapshot,
    SystemInfoSnapshot,
)

from .digital_clock import LLM_RUNNER_LOGO, DigitalClockWidget

_SYSTEM_INFO_LABEL = "system-health-label system-info-label"
_SYSTEM_INFO_PRIMARY_VALUE = "system-info-value system-info-primary-value"
_SYSTEM_INFO_ROW = "system-health-inline-row system-info-row"


class SystemHealthProvider(Protocol):
    """View-model protocol for system-health display state."""

    def cpu_usage_rows(self, width: int | None = None) -> list[list[CPUCoreSnapshot]]: ...

    def memory_usage_rows(self) -> list[MemoryUsageSnapshot]: ...

    def system_info_snapshot(self) -> SystemInfoSnapshot: ...

    def current_datetime_snapshot(self) -> DateTimeSnapshot: ...


_NO_CPU_DATA = "No CPU data"


class _EmptySystemHealthProvider:
    """Fallback provider for isolated widget tests."""

    def cpu_usage_rows(self, _width: int | None = None) -> list[list[CPUCoreSnapshot]]:
        return []

    def memory_usage_rows(self) -> list[MemoryUsageSnapshot]:
        return []

    def system_info_snapshot(self) -> SystemInfoSnapshot:
        return SystemInfoSnapshot(
            tasks=0,
            threads=0,
            running=0,
            load_values=None,
            uptime="00:00:00",
        )

    def current_datetime_snapshot(self) -> DateTimeSnapshot:
        return DateTimeSnapshot(date_text="")


class SystemHealthRenderer:
    """Builds system-health snapshots for the Textual widgets."""

    MIN_CONTENT_WIDTH = 40
    MAX_CONTENT_WIDTH = 240
    CPU_CORE_BAR_WIDTH = 5
    CPU_CORE_CELL_WIDTH = 16

    def __init__(self, provider: SystemHealthProvider | None = None) -> None:
        self._provider = provider or _EmptySystemHealthProvider()

    def render_cpu_usage(self, width: int | None = None) -> str:
        rows = self.cpu_usage_rows(width)
        if not rows:
            return _NO_CPU_DATA
        cpu_per_core: list[float] = [core.percent for row in rows for core in row]
        return "\n".join(self._format_core_grid_lines(cpu_per_core, self._content_width(width)))

    def cpu_usage_rows(self, width: int | None = None) -> list[list[CPUCoreSnapshot]]:
        return self._provider.cpu_usage_rows(width)

    def render_memory_swap_usage(self, width: int | None = None) -> str:
        content_width = self._content_width(width)
        rows = self.memory_usage_rows()
        return "\n".join(self._format_memory_row(row, content_width) for row in rows)

    def memory_usage_rows(self) -> list[MemoryUsageSnapshot]:
        return self._provider.memory_usage_rows()

    def render_system_info(self) -> str:
        snapshot = self.system_info_snapshot()

        lines = [self._task_summary(snapshot.tasks, snapshot.threads, snapshot.running)]
        if snapshot.load_values is None:
            lines.append(self._load_summary(-1.0, -1.0, -1.0))
        else:
            lines.append(self._load_summary(*snapshot.load_values))
        lines.append(f"Uptime: {snapshot.uptime}")
        return "\n".join(lines)

    def system_info_snapshot(self) -> SystemInfoSnapshot:
        return self._provider.system_info_snapshot()

    def current_datetime_snapshot(self) -> DateTimeSnapshot:
        return self._provider.current_datetime_snapshot()

    def _content_width(self, width: int | None) -> int:
        if width is None or width <= 0:
            return 116
        return min(self.MAX_CONTENT_WIDTH, max(self.MIN_CONTENT_WIDTH, width))

    def _memory_bar_width(self, content_width: int) -> int:
        label_width = len("Mem[]")
        value_width = 14
        column_gap = 3
        return max(4, content_width - label_width - column_gap - value_width)

    def _task_summary(self, tasks: int, threads: int, running: int) -> str:
        return f"  Tasks: {tasks:>3}  Thr: {threads:>4}  Run: {running:>2}"

    def _load_summary(self, load_1: float, load_5: float, load_15: float) -> str:
        if load_1 >= 0:
            return f"  Load: {load_1:.2f} {load_5:.2f} {load_15:.2f}"
        return "  Load: n/a"

    def _usage_bar(self, percent: float, width: int) -> str:
        filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * width))
        return "|" * filled + " " * (width - filled)

    def _usage_color(self, percent: float) -> str:
        if percent >= 85:
            return "red"
        if percent >= 60:
            return "yellow"
        return "green"

    def _format_uptime(self, seconds: int) -> str:
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _build_core_grid_rows(
        self, cpu_per_core: list[float], content_width: int
    ) -> list[list[CPUCoreSnapshot]]:
        if not cpu_per_core:
            return []

        max_cols = max(1, content_width // self.CPU_CORE_CELL_WIDTH)
        rows = max(1, (len(cpu_per_core) + max_cols - 1) // max_cols)
        cols = (len(cpu_per_core) + rows - 1) // rows
        snapshot_rows: list[list[CPUCoreSnapshot]] = []
        for row in range(rows):
            snapshot_row: list[CPUCoreSnapshot] = []
            for col in range(cols):
                idx = col * rows + row
                if idx >= len(cpu_per_core):
                    continue
                snapshot_row.append(CPUCoreSnapshot(index=idx, percent=cpu_per_core[idx]))
            snapshot_rows.append(snapshot_row)
        return snapshot_rows

    def _build_core_grid_lines(self, cpu_per_core: list[float], content_width: int) -> list[str]:
        return self._format_core_grid_lines(cpu_per_core, content_width)

    def _format_core_grid_lines(self, cpu_per_core: list[float], content_width: int) -> list[str]:
        rows = self._build_core_grid_rows(cpu_per_core, content_width)
        if not rows:
            return [_NO_CPU_DATA]
        flat_count = sum(len(row) for row in rows)
        cols = max(1, (flat_count + len(rows) - 1) // len(rows))
        cell_width = max(self.CPU_CORE_CELL_WIDTH, content_width // max(1, cols))
        lines: list[str] = []
        for row in rows:
            parts: list[str] = []
            for cell in row:
                text = self._build_core_cell_text(cell.index, cell.percent)
                parts.append(text.ljust(cell_width))
            lines.append("".join(parts).rstrip())
        return lines

    def _build_core_cell_text(self, idx: int, pct: float) -> str:
        bar_width = self.CPU_CORE_BAR_WIDTH
        meter = self._usage_bar(pct, width=bar_width)
        return f"{idx:>2} {meter} {pct:5.1f}%"

    def _format_memory_row(self, row: MemoryUsageSnapshot, content_width: int) -> str:
        meter = self._usage_bar(row.percent, width=self._memory_bar_width(content_width))
        left = f"{row.label}[{meter}]"
        value = row.value_text.rjust(14)
        pad = max(1, content_width - len(left) - len(value))
        return f"{left}{' ' * pad}{value}"


class SystemHealthWidget(Widget):
    """Container for focused system health sections."""

    def __init__(self, provider: SystemHealthProvider | None = None) -> None:
        super().__init__(classes="system-health")
        self._renderer = SystemHealthRenderer(provider)

    def compose(self) -> ComposeResult:
        yield DateTimeWidget(self._renderer)
        yield CPUUsageWidget(self._renderer)
        yield Horizontal(
            MemorySwapWidget(self._renderer),
            SystemInfoWidget(self._renderer),
            classes="system-health-resource-row",
        )


class DateTimeWidget(Widget):
    """Date/time section for the system health area."""

    def __init__(self, renderer: SystemHealthRenderer) -> None:
        super().__init__(classes="system-health-section system-health-datetime")
        self._renderer = renderer

    def compose(self) -> ComposeResult:
        snapshot = self._renderer.current_datetime_snapshot()
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

    def __init__(self, renderer: SystemHealthRenderer) -> None:
        super().__init__(classes="system-health-section system-health-cpu")
        self._renderer = renderer

    def compose(self) -> ComposeResult:
        rows = self._renderer.cpu_usage_rows(width=self.size.width)
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
                self._renderer._usage_bar(
                    core.percent,
                    width=self._renderer.CPU_CORE_BAR_WIDTH,
                ),
                classes=(
                    f"cpu-core-bar system-health-meter-{self._renderer._usage_color(core.percent)}"
                ),
            ),
            Static(f"{core.percent:5.1f}%", classes=f"cpu-core-percent {value_class}"),
            classes="cpu-core-cell",
        )


class MemorySwapWidget(Widget):
    """Memory and swap usage section."""

    def __init__(self, renderer: SystemHealthRenderer) -> None:
        super().__init__(classes="system-health-section system-health-memory-swap")
        self._renderer = renderer

    def compose(self) -> ComposeResult:
        for row in self._renderer.memory_usage_rows():
            value_class = (
                "system-health-value system-health-muted-value"
                if row.label == "Swp" and row.percent <= 0
                else "system-health-value"
            )
            yield Horizontal(
                Static(row.label, classes="memory-swap-label"),
                Static(
                    self._renderer._usage_bar(
                        row.percent,
                        width=self._renderer._memory_bar_width(
                            self._renderer._content_width(self.size.width)
                        ),
                    ),
                    classes=(
                        "memory-swap-bar "
                        f"system-health-meter-{self._renderer._usage_color(row.percent)}"
                    ),
                ),
                Static(row.value_text, classes=value_class),
                classes="memory-swap-row",
            )


class SystemInfoWidget(Widget):
    """Task, load, and uptime section."""

    def __init__(self, renderer: SystemHealthRenderer) -> None:
        super().__init__(classes="system-health-section system-health-system-info")
        self._renderer = renderer

    def compose(self) -> ComposeResult:
        snapshot = self._renderer.system_info_snapshot()

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
