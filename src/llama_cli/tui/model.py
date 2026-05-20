"""Dashboard model for the Textual TUI.

The model owns mutable runtime state. It intentionally avoids Textual objects
and Rich renderables so it can be inspected independently from the UI layer.
"""

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

import psutil

from llama_manager import (
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    ServerConfig,
    ServerManager,
    collect_nvtop_stats,
)
from llama_manager.build_pipeline import BuildConfig

from .types import DateTimeSnapshot, MemoryUsageSnapshot, RiskPromptState, SystemInfoSnapshot


class DashboardModel:
    """Mutable dashboard state shared by controller and view models."""

    STATUS_MESSAGE_LIFETIME_S: float = 30.0

    def __init__(
        self,
        configs: list[ServerConfig],
        gpu_indices: list[int],
        slots: list[ModelSlot] | None = None,
    ) -> None:
        self.config = Config()
        self.configs = configs
        self.gpu_indices = gpu_indices
        self.slots = slots or []
        self.log_buffers: dict[str, LogBuffer] = {
            cfg.alias: LogBuffer(redact_sensitive=True) for cfg in configs
        }
        self.gpu_stats: list[GPUStats] = [
            GPUStats(idx, collector=self.make_collector(idx)) for idx in gpu_indices
        ]
        self.running = True
        self.launch_result: LaunchResult | None = None
        self.risk_prompt: RiskPromptState | None = None

        self.status_messages: list[tuple[float, str]] = []
        self.status_lock = threading.Lock()
        self.stale_warnings: dict[str, str] = {}
        _ = psutil.cpu_percent(interval=0.1, percpu=True)

        self.build_request = False
        self.build_selected_backends: list[str] | None = None
        self.build_in_progress = False
        self.build_result: Literal["success", "failed"] | None = None
        self.build_error: str | None = None
        self.build_artifact: str | None = None
        self.build_stage: str | None = None
        self.build_progress_percent: float = 0.0
        self.build_is_retrying = False
        self.build_retries_remaining: int = 0
        self.build_cancel_event: threading.Event | None = None
        self.build_selected_backends_options: dict[str, BuildConfig | None] = {}
        self.unsaved_slots: set[str] = set()

        self.server_manager = ServerManager()
        self.slot_states: dict[str, str] = {}
        self.server_processes: dict[str, Any] = {}

    def make_collector(self, device_index: int) -> Callable[[], dict[str, Any]]:
        """Create a GPU collector bound to a device index."""
        return lambda: collect_nvtop_stats(device_index)

    def stop(self) -> None:
        """Stop the dashboard."""
        self.running = False

    def set_risk_prompt(
        self, kind: Literal["vram", "hardware"], acknowledged: bool = False
    ) -> None:
        """Set the active risk prompt."""
        self.risk_prompt = RiskPromptState(kind=kind, acknowledged=acknowledged)

    def clear_risk_prompt(self) -> None:
        """Clear the active risk prompt."""
        self.risk_prompt = None

    def push_status_message(self, message: str) -> None:
        """Push a status message to the bounded TUI message buffer."""
        with self.status_lock:
            self.status_messages.append((time.monotonic(), message))
            if len(self.status_messages) > 5:
                self.status_messages.pop(0)

    def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
        """Return status messages newer than ``since_ts`` and not expired."""
        cutoff = time.monotonic() - self.STATUS_MESSAGE_LIFETIME_S
        with self.status_lock:
            return [(ts, msg) for ts, msg in self.status_messages if ts > since_ts and ts >= cutoff]

    def cpu_percentages(self) -> list[float]:
        """Return current per-core CPU usage percentages."""
        from llama_manager import collect_cpu_percentages

        return collect_cpu_percentages(percpu=True)

    def memory_usage_rows(self) -> list[MemoryUsageSnapshot]:
        """Return memory and swap usage snapshots for the dashboard."""
        from llama_manager import collect_memory_usage

        data = collect_memory_usage()
        mem = data["mem"]
        swp = data["swp"]
        return [
            MemoryUsageSnapshot(
                label=str(mem["label"]),
                percent=float(mem["percent"] if isinstance(mem["percent"], float) else 0.0),
                value_text=str(mem["value_text"]),
            ),
            MemoryUsageSnapshot(
                label=str(swp["label"]),
                percent=float(swp["percent"] if isinstance(swp["percent"], float) else 0.0),
                value_text=str(swp["value_text"]),
            ),
        ]

    def system_info_snapshot(self) -> SystemInfoSnapshot:
        """Return process, load, and uptime state for the dashboard."""
        from llama_manager import collect_system_info

        data = collect_system_info()
        return SystemInfoSnapshot(
            tasks=data["tasks"],  # type: ignore[arg-type]
            threads=data["threads"],  # type: ignore[arg-type]
            running=data["running"],  # type: ignore[arg-type]
            load_values=data["load_values"],  # type: ignore[arg-type]
            uptime=data["uptime"],  # type: ignore[arg-type]
        )

    def current_datetime_snapshot(self) -> DateTimeSnapshot:
        """Return the current local date for display (Wed 2026-05-20)."""
        now = datetime.now()
        return DateTimeSnapshot(date_text=f"{now.strftime('%a')} {now.strftime('%Y-%m-%d')}")
