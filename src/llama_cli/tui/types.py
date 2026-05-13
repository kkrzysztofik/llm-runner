"""Typed view state shared across the TUI submodule."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class RiskPromptState:
    """Current risk prompt state."""

    kind: Literal["vram", "hardware"]
    acknowledged: bool


@dataclass(frozen=True)
class CommandMenuState:
    """State needed to render the bottom command menu."""

    profile_request: str | None
    risk_prompt: RiskPromptState | None
    build_request: bool
    smoke_request: bool


@dataclass(frozen=True)
class ServerColumnState:
    """State needed to render one server column."""

    alias: str
    status: str
    status_class: str
    backend_label: str
    url: str
    config_summary: str
    logs_text: str
    gpu_stats: dict[str, Any] | None
    stale_warning: str | None
    is_unsaved: bool


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
class BuildViewState:
    """State needed to render the build progress panel."""

    visible: bool = False
    build_request: bool = False
    selected_backend: str | None = None
    in_progress: bool = False
    stage: str | None = None
    message: str | None = None
    is_retrying: bool = False
    retries_remaining: int = 0
    last_result_success: bool | None = None
    artifact_path: str | None = None
    error_message: str | None = None
    progress_percent: int = 0
