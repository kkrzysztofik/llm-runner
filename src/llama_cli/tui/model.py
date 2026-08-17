"""Dashboard model for the Textual TUI.

The model owns mutable runtime state. It intentionally avoids Textual objects
and Rich renderables so it can be inspected independently from the UI layer.
"""

import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Literal

from llama_manager import (
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    ServerConfig,
    ServerManager,
    collector_for_config,
    gpu_index_for_config,
    selector_for_config,
)
from llama_manager.build_pipeline import BuildConfig, BuildProgress
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

        self.status_messages: deque[tuple[float, str]] = deque(maxlen=5)
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

        self.build_in_progress = False
        self.build_error: str | None = None
        self.build_artifact: str | None = None
        self.build_progress: BuildProgress | None = None
        self.build_cancel_event: threading.Event | None = None
        self.build_selected_backends_options: dict[str, BuildConfig | None] = {}
        self.unsaved_slots: set[str] = set()

        self.server_manager = ServerManager()
        self.slot_states: dict[str, str] = {}
        self.server_processes: dict[str, Any] = {}

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

    def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
        """Return status messages newer than ``since_ts`` and not expired."""
        cutoff = time.monotonic() - self.STATUS_MESSAGE_LIFETIME_S
        with self.status_lock:
            return [(ts, m) for ts, m in self.status_messages if ts > since_ts and ts >= cutoff]

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

    def snapshot_for_probe(self) -> tuple[tuple[str, GPUStats, ServerConfig], ...]:
        """Frozen (alias, gpu, config) triples for background telemetry workers.

        Must be called on the UI thread (or via ``call_from_thread``) so the
        ``configs`` / ``gpu_stats`` length invariant is observed atomically.
        """
        with self.system_health_lock:
            return tuple(
                (cfg.alias, gpu, cfg) for cfg, gpu in zip(self.configs, self.gpu_stats, strict=True)
            )

    def apply_gpu_stats_snapshot(self, gpu_stats_by_alias: dict[str, dict[str, Any]]) -> None:
        """Merge GPU telemetry collected off the UI thread; prune removed aliases."""
        with self.system_health_lock:
            self.cached_gpu_stats_by_alias.update(
                {alias: dict(stats) for alias, stats in gpu_stats_by_alias.items()}
            )
            live_aliases = {cfg.alias for cfg in self.configs}
            for alias in list(self.cached_gpu_stats_by_alias):
                if alias not in live_aliases:
                    del self.cached_gpu_stats_by_alias[alias]

    def set_cached_gpu_stats(self, alias: str, stats: dict[str, Any]) -> None:
        """Set one slot's cached GPU telemetry without probing hardware."""
        with self.system_health_lock:
            self.cached_gpu_stats_by_alias[alias] = dict(stats)

    def remove_cached_gpu_stats(self, alias: str) -> None:
        """Remove cached GPU telemetry for a deleted or replaced slot."""
        with self.system_health_lock:
            self.cached_gpu_stats_by_alias.pop(alias, None)

    def apply_slot_stats_snapshot(
        self,
        stats_by_alias: dict[str, SlotStatsSnapshot],
        live_aliases: set[str] | None = None,
    ) -> None:
        """Merge slot stats collected off the UI thread; prune removed aliases.

        *live_aliases* should be derived from the UI-thread-taken probe snapshot so
        workers do not read ``self.configs`` (mutated on the UI thread).
        """
        with self.system_health_lock:
            self.cached_slot_stats_by_alias.update(stats_by_alias)
            aliases = (
                live_aliases if live_aliases is not None else {cfg.alias for cfg in self.configs}
            )
            for alias in list(self.cached_slot_stats_by_alias):
                if alias not in aliases:
                    del self.cached_slot_stats_by_alias[alias]

    def slot_stats_snapshot(self) -> dict[str, SlotStatsSnapshot]:
        """Return a snapshot of cached slot stats for render-only code."""
        with self.system_health_lock:
            return dict(self.cached_slot_stats_by_alias)

    def current_datetime_snapshot(self) -> DateTimeSnapshot:
        """Return the current local date for display (Wed 2026-05-20)."""
        now = datetime.now()
        return DateTimeSnapshot(date_text=f"{now.strftime('%a')} {now.strftime('%Y-%m-%d')}")
