"""Typed view state shared across the TUI submodule."""

from dataclasses import dataclass
from typing import Any, Literal

from llama_manager.build_pipeline import BuildConfig

MIN_CONTENT_WIDTH = 40
DEFAULT_CONTENT_WIDTH = 116
MAX_CONTENT_WIDTH = 240
CPU_CORE_BAR_WIDTH = 5
CPU_CORE_CELL_WIDTH = 16


@dataclass(frozen=True)
class RiskPromptState:
    """Current risk prompt state."""

    kind: Literal["vram", "hardware"]
    acknowledged: bool


@dataclass(frozen=True)
class CommandMenuState:
    """State needed to render the bottom command menu."""

    risk_prompt: RiskPromptState | None


@dataclass(frozen=True)
class SlotRuntimeStats:
    """Display-ready per-slot runtime counters."""

    tps: str
    pp: str
    tokens_in: str
    tokens_out: str


@dataclass(frozen=True)
class ServerColumnState:
    """State needed to render one server column."""

    alias: str
    profile_name: str
    status: str
    status_label: str
    status_class: str
    backend_label: str
    url: str
    config_summary: str
    runtime_stats: SlotRuntimeStats
    gpu_stats: dict[str, Any] | None
    stale_warning: str | None


@dataclass(frozen=True)
class DashboardSnapshot:
    """Immutable cached telemetry consumed by dashboard render code."""

    cpu_percentages: list[float]
    memory_usage_rows: list[MemoryUsageSnapshot]
    system_info: SystemInfoSnapshot
    gpu_stats_by_alias: dict[str, dict[str, Any]]


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


@dataclass(frozen=True)
class SystemInfoSnapshot:
    """Structured values for the textual system info widget."""

    tasks: int
    threads: int
    running: int
    load_values: tuple[float, float, float] | None
    uptime: str


@dataclass(frozen=True)
class DateTimeSnapshot:
    """Formatted date for the system health datetime row (e.g. Wed 2026-05-20)."""

    date_text: str


@dataclass
class BuildWizardResult:
    """Result returned from the build wizard modal."""

    backends: list[str]
    options: dict[str, BuildConfig | None]
