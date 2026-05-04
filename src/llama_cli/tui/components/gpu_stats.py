"""Per-slot GPU stats panel."""

from typing import Any

from rich.panel import Panel
from rich.text import Text
from textual.widget import Widget

from llama_manager import GPUStats


class GPUStatsPanel(Widget):
    """Compact htop-style GPU telemetry panel."""

    def __init__(self, gpu: GPUStats | None) -> None:
        super().__init__()
        self._gpu = gpu

    def render(self) -> Panel:  # type: ignore[override]
        if self._gpu is None:
            return Panel(
                Text("GPU stats unavailable", style="dim"), title="GPU", border_style="dim"
            )
        return self._build_panel(self._gpu)

    @staticmethod
    def _build_panel(gpu: GPUStats) -> Panel:
        stats = gpu.get_stats_snapshot()
        gpu_pct = GPUStatsPanel._parse_percent(stats.get("gpu_util"))
        mem_pct = GPUStatsPanel._parse_percent(stats.get("mem_util"))
        cpu_pct = GPUStatsPanel._parse_percent(stats.get("cpu"))
        sys_mem_pct = GPUStatsPanel._parse_percent(stats.get("mem"))

        def _fmt(val: Any) -> str:
            if val is None:
                return "N/A"
            s = str(val).strip()
            return "N/A" if s == "" or s.lower() == "n/a" else s

        text = Text()
        text.append("Device: ", style="bright_white")
        text.append(_fmt(stats.get("device")), style="cyan")
        text.append("\n")

        if gpu_pct is not None or mem_pct is not None:
            GPUStatsPanel._append_usage_line(text, "GPU", gpu_pct, _fmt(stats.get("gpu_util")))
            text.append("  ", style="dim")
            GPUStatsPanel._append_usage_line(text, "VRAM", mem_pct, _fmt(stats.get("mem_util")))
        else:
            GPUStatsPanel._append_usage_line(text, "CPU", cpu_pct, _fmt(stats.get("cpu")))
            text.append("  ", style="dim")
            GPUStatsPanel._append_usage_line(text, "Mem", sys_mem_pct, _fmt(stats.get("mem")))
        text.append("\n")

        temp = _fmt(stats.get("temp"))
        power = _fmt(stats.get("power"))
        if temp != "N/A" or power != "N/A":
            text.append("Temp:", style="bright_cyan")
            text.append(f" {temp}", style="bright_white" if temp != "N/A" else "dim")
            text.append("  ", style="dim")
            text.append("Power:", style="bright_cyan")
            text.append(f" {power}", style="bright_white" if power != "N/A" else "dim")

        return Panel(text, title="GPU", border_style="yellow")

    @staticmethod
    def _append_usage_line(text: Text, label: str, percent: float | None, raw: str) -> None:
        text.append(f"{label}", style="bright_blue")
        text.append("[", style="white")
        if percent is None:
            text.append("?" * 10, style="dim")
        else:
            filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * 10))
            color = GPUStatsPanel._usage_color(percent)
            text.append("|" * filled, style=color)
            if filled < 10:
                text.append(" " * (10 - filled), style="dim")
        text.append("]", style="white")
        text.append(f" {raw}", style="bright_white" if raw != "N/A" else "dim")

    @staticmethod
    def _usage_color(percent: float) -> str:
        if percent >= 85:
            return "red"
        if percent >= 60:
            return "yellow"
        return "green"

    @staticmethod
    def _parse_percent(value: object) -> float | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.upper() == "N/A":
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            parsed = float(text)
        except ValueError:
            return None
        import math

        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed
