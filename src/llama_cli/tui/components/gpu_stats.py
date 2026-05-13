"""Per-slot GPU stats widget."""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widget import Widget
from textual.widgets import Static


class GPUStatsPanel(Widget):
    """Compact GPU telemetry card."""

    def __init__(self, stats: dict[str, Any] | None) -> None:
        super().__init__(classes="gpu-stats")
        self._stats = stats

    def compose(self) -> ComposeResult:
        yield Static("GPU", classes="panel-title gpu-stats-title")

        if self._stats is None:
            yield Static(
                "GPU stats unavailable",
                classes="gpu-stats-unavailable",
            )
            return

        yield from self._build_rows(self._stats)

    def _build_rows(self, stats: dict[str, Any]) -> ComposeResult:
        gpu_pct = self._parse_percent(stats.get("gpu_util"))
        mem_pct = self._parse_percent(stats.get("mem_util"))
        cpu_pct = self._parse_percent(stats.get("cpu"))
        sys_mem_pct = self._parse_percent(stats.get("mem"))

        def _fmt(val: Any) -> str:
            if val is None:
                return "N/A"
            s = str(val).strip()
            return "N/A" if s == "" or s.lower() == "n/a" else s

        yield Horizontal(
            Static("Device:", classes="gpu-stats-label"),
            Static(_fmt(stats.get("device")), classes="gpu-stats-value"),
            classes="gpu-stats-row",
        )

        if gpu_pct is not None or mem_pct is not None:
            yield Horizontal(
                self._usage_item("GPU", gpu_pct, _fmt(stats.get("gpu_util"))),
                self._usage_item("VRAM", mem_pct, _fmt(stats.get("mem_util"))),
                classes="gpu-stats-usage-row",
            )
        else:
            yield Horizontal(
                self._usage_item("CPU", cpu_pct, _fmt(stats.get("cpu"))),
                self._usage_item("Mem", sys_mem_pct, _fmt(stats.get("mem"))),
                classes="gpu-stats-usage-row",
            )

        temp = _fmt(stats.get("temp"))
        power = _fmt(stats.get("power"))
        if temp != "N/A" or power != "N/A":
            row = [
                Static("Temp:", classes="gpu-stats-label"),
                Static(temp, classes=self._value_class(temp)),
                Static("Power:", classes="gpu-stats-label"),
                Static(power, classes=self._value_class(power)),
            ]
            yield Horizontal(*row, classes="gpu-stats-row")

    def _usage_item(self, label: str, percent: float | None, raw: str) -> Container:
        level_class = self._usage_level_class(percent)
        return Container(
            Static(label, classes="gpu-stats-usage-label"),
            Static(
                self._usage_meter(percent),
                classes=f"gpu-stats-usage-bar {level_class}",
            ),
            Static(raw, classes=self._value_class(raw)),
            classes="gpu-stats-usage-item",
        )

    @staticmethod
    def _usage_meter(percent: float | None) -> str:
        if percent is None:
            return "?" * 10
        filled = int(round((max(0.0, min(100.0, percent)) / 100.0) * 10))
        return "|" * filled + " " * (10 - filled)

    @staticmethod
    def _usage_level_class(percent: float | None) -> str:
        if percent is None:
            return "gpu-stats-usage-unknown"
        if percent >= 85:
            return "gpu-stats-usage-high"
        if percent >= 60:
            return "gpu-stats-usage-medium"
        return "gpu-stats-usage-low"

    @staticmethod
    def _value_class(raw: str) -> str:
        return "gpu-stats-muted-value" if raw == "N/A" else "gpu-stats-value"

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
