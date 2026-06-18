"""Dashboard model for the Textual TUI.

The model owns mutable runtime state. It intentionally avoids Textual objects
and Rich renderables so it can be inspected independently from the UI layer.
"""

import threading
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from llama_manager import (
    Config,
    GPUStats,
    GpuTelemetrySelector,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    ServerConfig,
    ServerManager,
    collect_gpu_stats,
    collector_for_config,
    gpu_index_for_config,
    selector_for_config,
)
from llama_manager.build_pipeline import BuildConfig
from llama_manager.slot_stats import SlotStatsSnapshot

from .types import (
    DashboardSnapshot,
    DateTimeSnapshot,
    MemoryUsageSnapshot,
    RiskPromptState,
    SystemInfoSnapshot,
)


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
            GPUStats(
                gpu_index_for_config(cfg),
                collector=collector_for_config(cfg),
                selector=selector_for_config(cfg),
            )
            for cfg in configs
        ]
        self.running = True
        self.launch_result: LaunchResult | None = None
        self.risk_prompt: RiskPromptState | None = None

        self.status_messages: list[tuple[float, str]] = []
        self.status_lock = threading.Lock()
        self.stale_warnings: dict[str, str] = {}
        self.system_health_lock = threading.Lock()
        self.cached_cpu_percentages: list[float] = []
        self.cached_memory_usage_rows: list[MemoryUsageSnapshot] = []
        self.cached_gpu_stats_by_alias: dict[str, dict[str, Any]] = {
            cfg.alias: gpu.get_cached_stats_snapshot()
            for cfg, gpu in zip(configs, self.gpu_stats, strict=False)
        }
        self.cached_slot_stats_by_alias: dict[str, SlotStatsSnapshot] = {}
        self.cached_system_info_snapshot = SystemInfoSnapshot(
            tasks=0,
            threads=0,
            running=0,
            load_values=None,
            uptime="0:00",
        )

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
        """Create a legacy CUDA collector bound to a device ordinal."""
        selector = GpuTelemetrySelector(backend="cuda", ordinal=device_index)
        return lambda: collect_gpu_stats(selector)

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
        """Return cached per-core CPU usage percentages."""
        with self.system_health_lock:
            return list(self.cached_cpu_percentages)

    def memory_usage_rows(self) -> list[MemoryUsageSnapshot]:
        """Return cached memory and swap usage snapshots for the dashboard."""
        with self.system_health_lock:
            return list(self.cached_memory_usage_rows)

    def system_info_snapshot(self) -> SystemInfoSnapshot:
        """Return cached process, load, and uptime state for the dashboard."""
        with self.system_health_lock:
            return self.cached_system_info_snapshot

    def dashboard_snapshot(self) -> DashboardSnapshot:
        """Return immutable cached telemetry for render-only dashboard code."""
        with self.system_health_lock:
            return DashboardSnapshot(
                cpu_percentages=list(self.cached_cpu_percentages),
                memory_usage_rows=list(self.cached_memory_usage_rows),
                system_info=self.cached_system_info_snapshot,
                gpu_stats_by_alias={
                    alias: dict(stats) for alias, stats in self.cached_gpu_stats_by_alias.items()
                },
            )

    def collect_system_health_snapshot(
        self,
    ) -> tuple[list[float], list[MemoryUsageSnapshot], SystemInfoSnapshot]:
        """Collect live system-health state for background cache refresh."""
        from llama_manager import collect_cpu_percentages, collect_memory_usage, collect_system_info

        cpu = collect_cpu_percentages(percpu=True)
        memory_data = collect_memory_usage()
        mem = memory_data["mem"]
        swp = memory_data["swp"]
        memory_rows = [
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
        system_data = collect_system_info()
        system_info = SystemInfoSnapshot(
            tasks=system_data["tasks"],  # type: ignore[arg-type]
            threads=system_data["threads"],  # type: ignore[arg-type]
            running=system_data["running"],  # type: ignore[arg-type]
            load_values=system_data["load_values"],  # type: ignore[arg-type]
            uptime=system_data["uptime"],  # type: ignore[arg-type]
        )
        return cpu, memory_rows, system_info

    def apply_system_health_snapshot(
        self,
        cpu: list[float],
        memory_rows: list[MemoryUsageSnapshot],
        system_info: SystemInfoSnapshot,
    ) -> None:
        """Store system-health state collected off the UI thread."""
        with self.system_health_lock:
            self.cached_cpu_percentages = list(cpu)
            self.cached_memory_usage_rows = list(memory_rows)
            self.cached_system_info_snapshot = system_info

    def apply_gpu_stats_snapshot(self, gpu_stats_by_alias: dict[str, dict[str, Any]]) -> None:
        """Store GPU telemetry collected off the UI thread."""
        with self.system_health_lock:
            self.cached_gpu_stats_by_alias = {
                alias: dict(stats) for alias, stats in gpu_stats_by_alias.items()
            }

    def set_cached_gpu_stats(self, alias: str, stats: dict[str, Any]) -> None:
        """Set one slot's cached GPU telemetry without probing hardware."""
        with self.system_health_lock:
            self.cached_gpu_stats_by_alias[alias] = dict(stats)

    def remove_cached_gpu_stats(self, alias: str) -> None:
        """Remove cached GPU telemetry for a deleted or replaced slot."""
        with self.system_health_lock:
            self.cached_gpu_stats_by_alias.pop(alias, None)

    def apply_slot_stats_snapshot(self, stats_by_alias: dict[str, SlotStatsSnapshot]) -> None:
        """Store slot stats collected off the UI thread."""
        with self.system_health_lock:
            self.cached_slot_stats_by_alias = dict(stats_by_alias)

    def set_cached_slot_stats(self, alias: str, stats: SlotStatsSnapshot) -> None:
        """Set one slot's cached stats without fetching live data."""
        with self.system_health_lock:
            self.cached_slot_stats_by_alias[alias] = stats

    def slot_stats_snapshot(self) -> dict[str, SlotStatsSnapshot]:
        """Return a snapshot of cached slot stats for render-only code."""
        with self.system_health_lock:
            return dict(self.cached_slot_stats_by_alias)

    def collect_memory_usage_rows_now(self) -> list[MemoryUsageSnapshot]:
        """Return live memory and swap usage snapshots for non-refresh callers."""
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

    def collect_system_info_snapshot_now(self) -> SystemInfoSnapshot:
        """Return live process, load, and uptime state for non-refresh callers."""
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
