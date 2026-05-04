"""System health renderer and widget."""

import time

import psutil
from rich.text import Text
from textual.app import RenderResult
from textual.widget import Widget


class SystemHealthRenderer:
    """Builds CPU, memory, swap, and uptime status text."""

    def __init__(self) -> None:
        self._task_cache: tuple[int, int, int] | None = None
        self._task_cache_ts: float = 0.0
        self._task_cache_ttl: float = 1.5  # 1.5s TTL
        self._cpu_primed = False
        # Prime CPU percent once so the first render call already has data
        _ = psutil.cpu_percent(interval=0.1, percpu=True)
        self._cpu_primed = True

    def render(self) -> Text:
        content_width = 116
        right_block_width = min(44, max(36, content_width // 3))
        left_block_width = max(28, content_width - right_block_width - 3)

        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        uptime_s = int(time.time() - psutil.boot_time())
        tasks, threads, running = self._get_task_stats()

        try:
            load_1, load_5, load_15 = psutil.getloadavg()
        except (AttributeError, OSError):
            load_1 = load_5 = load_15 = -1.0

        text = Text()
        text.append("Date/Time: ", style="bright_cyan")
        text.append(time.strftime("%Y-%m-%d %H:%M:%S"), style="bright_white")
        text.append("\n")

        for line in self._build_core_grid_lines(cpu_per_core, left_width=left_block_width):
            text.append(line)
            text.append("\n")

        mem_bar_width = max(20, left_block_width - 6)
        self._append_memory_line(
            text,
            label="Mem",
            percent=mem.percent,
            used=mem.used,
            total=mem.total,
            right_style="bright_white",
            left_width=left_block_width,
            right_suffix=self._task_summary(tasks, threads, running),
            bar_width=mem_bar_width,
        )
        self._append_memory_line(
            text,
            label="Swp",
            percent=swap.percent,
            used=swap.used,
            total=swap.total,
            right_style="dim",
            left_width=left_block_width,
            right_suffix=self._load_summary(load_1, load_5, load_15),
            bar_width=mem_bar_width,
        )

        uptime_left = Text()
        uptime_right = Text()
        self._append_aligned_value(uptime_right, "", width=14, style="dim")
        uptime_right.append("  Uptime:", style="bright_cyan")
        uptime_right.append(f" {self._format_uptime(uptime_s)}", style="bright_white")
        self._append_two_column_line(text, uptime_left, uptime_right, left_width=left_block_width)

        return text

    def _append_memory_line(
        self,
        text: Text,
        label: str,
        percent: float,
        used: int,
        total: int,
        right_style: str,
        left_width: int,
        right_suffix: Text,
        bar_width: int,
    ) -> None:
        left = Text()
        left.append(label, style="bright_cyan")
        left.append("[", style="bright_cyan")
        self._append_segmented_bar(left, percent, bar_width)
        left.append("]", style="bright_cyan")

        right = Text()
        self._append_aligned_value(
            right,
            f"{self._format_bytes(used)}/{self._format_bytes(total)}",
            width=14,
            style=right_style,
        )
        right.append_text(right_suffix)
        self._append_two_column_line(text, left, right, left_width=left_width)

    def _task_summary(self, tasks: int, threads: int, running: int) -> Text:
        text = Text()
        text.append("  Tasks:", style="bright_cyan")
        text.append(f" {tasks:>3}", style="bright_white")
        text.append("  Thr:", style="cyan")
        text.append(f" {threads:>4}", style="bright_white")
        text.append("  Run:", style="green")
        text.append(f" {running:>2}", style="bright_white")
        return text

    def _load_summary(self, load_1: float, load_5: float, load_15: float) -> Text:
        text = Text()
        if load_1 >= 0:
            text.append("  Load:", style="bright_cyan")
            text.append(f" {load_1:.2f}", style="bright_white")
            text.append(f" {load_5:.2f}", style="cyan")
            text.append(f" {load_15:.2f}", style="bright_blue")
        else:
            text.append("  Load: n/a", style="dim")
        return text

    def _usage_bar(self, percent: float, width: int, style: str = "bright_white") -> Text:
        filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * width))
        bar = Text()
        for i in range(width):
            if i < filled:
                bar.append("█", style=style)
            else:
                bar.append("░", style="dim")
        return bar

    def _usage_color(self, percent: float) -> str:
        if percent >= 85:
            return "red"
        if percent >= 60:
            return "yellow"
        return "green"

    def _append_segmented_bar(self, text: Text, percent: float, width: int) -> None:
        """Append an htop-like multi-color bar segment to ``text``."""
        filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * width))
        if filled <= 0:
            text.append(" " * width, style="dim")
            return

        palette = ["green", "green", "green", "green", "cyan", "magenta", "yellow", "yellow"]
        for i in range(filled):
            text.append("|", style=palette[i % len(palette)])
        if filled < width:
            text.append(" " * (width - filled), style="dim")

    def _format_uptime(self, seconds: int) -> str:
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _build_core_grid_lines(self, cpu_per_core: list[float], left_width: int) -> list[Text]:
        if not cpu_per_core:
            return [Text("No CPU data", style="dim")]

        rows = 3
        cols = (len(cpu_per_core) + rows - 1) // rows
        cell_width = max(14, left_width // max(1, cols))
        bar_width = 5
        lines: list[Text] = []
        for row in range(rows):
            line = Text()
            for col in range(cols):
                idx = col * rows + row
                if idx >= len(cpu_per_core):
                    continue
                pct = cpu_per_core[idx]
                cell = Text()
                cell.append(f"{idx:>2}", style="bright_blue")
                cell.append("[", style="bright_white")
                cell.append(self._usage_bar(pct, width=bar_width, style=self._usage_color(pct)))
                cell.append("]", style="bright_white")
                cell.append(" ")
                cell.append(f"{pct:5.1f}%", style="bright_white" if pct > 0 else "dim")
                line.append_text(cell)
                pad = max(1, cell_width - len(cell.plain))
                line.append(" " * pad)
            lines.append(line)
        return lines

    def _append_two_column_line(
        self,
        text: Text,
        left: Text,
        right: Text,
        left_width: int = 66,
    ) -> None:
        left_plain = left.plain
        pad = max(3, left_width - len(left_plain))
        text.append_text(left)
        text.append(" " * pad)
        text.append_text(right)
        text.append("\n")

    def _append_aligned_value(self, text: Text, value: str, width: int, style: str) -> None:
        text.append(value.rjust(width), style=style)

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
    """CPU, memory, swap, and uptime stats."""

    DEFAULT_CSS = """
    SystemHealthWidget {
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._renderer = SystemHealthRenderer()

    def render(self) -> RenderResult:
        return self._renderer.render()
