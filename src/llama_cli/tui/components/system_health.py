from __future__ import annotations

"""System health widgets."""

import time
from dataclasses import dataclass

import psutil
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Static


@dataclass(frozen=True)
class CPUCoreSnapshot:
    """Structured CPU core usage cell."""

    index: int
    percent: float


@dataclass(frozen=True)
class MemoryUsageSnapshot:
    """Structured memory or swap usage row."""

    label: str
    percent: float
    value_text: str


class SystemHealthRenderer:
    """Builds system-health snapshots for the Textual widgets."""

    MIN_CONTENT_WIDTH = 40
    MAX_CONTENT_WIDTH = 160

    def __init__(self) -> None:
        self._task_cache: tuple[int, int, int] | None = None
        self._task_cache_ts: float = 0.0
        self._task_cache_ttl: float = 1.5  # 1.5s TTL
        self._cpu_primed = False
        # Prime CPU percent once so the first render call already has data
        _ = psutil.cpu_percent(interval=0.1, percpu=True)
        self._cpu_primed = True

    def render_cpu_usage(self, width: int | None = None) -> str:
        content_width = self._content_width(width)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        return "\n".join(self._build_core_grid_lines(cpu_per_core, content_width=content_width))

    def cpu_usage_rows(self, width: int | None = None) -> list[list[CPUCoreSnapshot]]:
        content_width = self._content_width(width)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        return self._build_core_grid_rows(cpu_per_core, content_width=content_width)

    def render_memory_swap_usage(self, width: int | None = None) -> str:
        content_width = self._content_width(width)
        rows = self.memory_usage_rows(width)
        return "\n".join(self._format_memory_row(row, content_width) for row in rows)

    def memory_usage_rows(self, width: int | None = None) -> list[MemoryUsageSnapshot]:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return [
            MemoryUsageSnapshot(
                label="Mem",
                percent=mem.percent,
                value_text=f"{self._format_bytes(mem.used)}/{self._format_bytes(mem.total)}",
            ),
            MemoryUsageSnapshot(
                label="Swp",
                percent=swap.percent,
                value_text=f"{self._format_bytes(swap.used)}/{self._format_bytes(swap.total)}",
            ),
        ]

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
        uptime_s = int(time.time() - psutil.boot_time())
        tasks, threads, running = self._get_task_stats()

        try:
            load_values: tuple[float, float, float] | None = psutil.getloadavg()
        except (AttributeError, OSError):
            load_values = None

        return SystemInfoSnapshot(
            tasks=tasks,
            threads=threads,
            running=running,
            load_values=load_values,
            uptime=self._format_uptime(uptime_s),
        )

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

        min_cell_width = len(self._build_core_cell_text(99, 100.0)) + 1
        max_cols = max(1, content_width // min_cell_width)
        rows = max(3, (len(cpu_per_core) + max_cols - 1) // max_cols)
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
        rows = self._build_core_grid_rows(cpu_per_core, content_width)
        if not rows:
            return ["No CPU data"]

        flat_count = sum(len(row) for row in rows)
        cols = max(1, (flat_count + len(rows) - 1) // len(rows))
        min_cell_width = len(self._build_core_cell_text(99, 100.0)) + 1
        cell_width = max(min_cell_width, content_width // max(1, cols))
        lines: list[str] = []
        for row in rows:
            parts: list[str] = []
            for cell in row:
                text = self._build_core_cell_text(cell.index, cell.percent)
                parts.append(text.ljust(cell_width))
            lines.append("".join(parts).rstrip())
        return lines

    def _build_core_cell_text(self, idx: int, pct: float, bar_width: int = 5) -> str:
        meter = self._usage_bar(pct, width=bar_width)
        return f"{idx:>2}[{meter}] {pct:5.1f}%"

    def _format_memory_row(self, row: MemoryUsageSnapshot, content_width: int) -> str:
        meter = self._usage_bar(row.percent, width=self._memory_bar_width(content_width))
        left = f"{row.label}[{meter}]"
        value = row.value_text.rjust(14)
        pad = max(1, content_width - len(left) - len(value))
        return f"{left}{' ' * pad}{value}"

    def _format_bytes(self, num_bytes: int) -> str:
        gib = num_bytes / (1024**3)
        if gib >= 10:
            return f"{gib:,.1f}G"
        return f"{gib:,.2f}G"

    def _get_task_stats(self) -> tuple[int, int, int]:
        now = time.time()
        if self._task_cache is not None and now - self._task_cache_ts < self._task_cache_ttl:
            return self._task_cache

        task_count = 0
        thread_count = 0
        running_count = 0
        try:
            for proc in psutil.process_iter(attrs=["status", "num_threads"]):
                try:
                    info = proc.info
                except Exception:  # noqa: S112
                    continue
                task_count += 1
                thread_count += int(info.get("num_threads") or 0)
                if info.get("status") == psutil.STATUS_RUNNING:
                    running_count += 1
        except Exception:
            if self._task_cache is not None:
                return self._task_cache
            self._task_cache = (0, 0, 0)
            self._task_cache_ts = now
            return self._task_cache

        self._task_cache = (task_count, thread_count, running_count)
        self._task_cache_ts = now
        return self._task_cache


class SystemHealthWidget(Widget):
    """Container for focused system health sections."""

    def __init__(self) -> None:
        super().__init__(classes="system-health")
        self._renderer = SystemHealthRenderer()

    def compose(self) -> ComposeResult:
        yield DateTimeWidget(self._renderer)
        yield CPUUsageWidget(self._renderer)
        yield Horizontal(
            MemorySwapWidget(self._renderer),
            SystemInfoWidget(self._renderer),
            classes="system-health-resource-row",
        )


@dataclass(frozen=True)
class SystemInfoSnapshot:
    """Structured values for the textual system info widget."""

    tasks: int
    threads: int
    running: int
    load_values: tuple[float, float, float] | None
    uptime: str


class DateTimeWidget(Widget):
    """Date/time section for the system health area."""

    def __init__(self, renderer: SystemHealthRenderer) -> None:
        super().__init__(classes="system-health-section system-health-datetime")
        self._renderer = renderer

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Static("Date/Time:", classes="system-health-label system-health-datetime-label"),
            Static(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                classes="system-health-value system-health-datetime-value",
            ),
            classes="system-health-inline-row system-health-datetime-row",
        )


class CPUUsageWidget(Widget):
    """CPU per-core usage section for the system health area."""

    def __init__(self, renderer: SystemHealthRenderer) -> None:
        super().__init__(classes="system-health-section system-health-cpu")
        self._renderer = renderer

    def compose(self) -> ComposeResult:
        rows = self._renderer.cpu_usage_rows(width=self.size.width)
        if not rows:
            yield Static("No CPU data", classes="system-health-muted-value")
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
                self._renderer._usage_bar(core.percent, width=5),
                classes=(
                    f"cpu-core-bar system-health-meter-{self._renderer._usage_color(core.percent)}"
                ),
            ),
            Static(f"{core.percent:5.1f}%", classes=value_class),
            classes="cpu-core-cell",
        )


class MemorySwapWidget(Widget):
    """Memory and swap usage section."""

    def __init__(self, renderer: SystemHealthRenderer) -> None:
        super().__init__(classes="system-health-section system-health-memory-swap")
        self._renderer = renderer

    def compose(self) -> ComposeResult:
        for row in self._renderer.memory_usage_rows(width=self.size.width):
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
            Static("Tasks:", classes="system-health-label system-info-label"),
            Static(f"{snapshot.tasks:>3}", classes="system-info-value system-info-primary-value"),
            Static("Thr:", classes="system-info-label system-info-secondary-label"),
            Static(
                f"{snapshot.threads:>4}",
                classes="system-info-value system-info-primary-value",
            ),
            Static("Run:", classes="system-info-label system-info-running-label"),
            Static(f"{snapshot.running:>2}", classes="system-info-value system-info-primary-value"),
            classes="system-health-inline-row system-info-row",
        )

        if snapshot.load_values is None:
            yield Horizontal(
                Static("Load:", classes="system-health-label system-info-label"),
                Static("n/a", classes="system-info-value system-info-muted-value"),
                classes="system-health-inline-row system-info-row",
            )
        else:
            load_1, load_5, load_15 = snapshot.load_values
            yield Horizontal(
                Static("Load:", classes="system-health-label system-info-label"),
                Static(
                    f"{load_1:.2f}",
                    classes="system-info-value system-info-primary-value",
                ),
                Static(
                    f"{load_5:.2f}",
                    classes="system-info-value system-info-secondary-value",
                ),
                Static(
                    f"{load_15:.2f}",
                    classes="system-info-value system-info-tertiary-value",
                ),
                classes="system-health-inline-row system-info-row",
            )

        yield Horizontal(
            Static("Uptime:", classes="system-health-label system-info-label"),
            Static(snapshot.uptime, classes="system-info-value system-info-primary-value"),
            classes="system-health-inline-row system-info-row",
        )
