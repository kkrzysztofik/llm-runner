"""Dashboard model for the Textual TUI.

The model owns mutable runtime state. It intentionally avoids Textual objects
and Rich renderables so it can be inspected independently from the UI layer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Literal

import psutil

from llama_cli.gpu_collectors import collect_nvtop_stats
from llama_manager import (
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    ServerConfig,
    ServerManager,
)

from .types import MemoryUsageSnapshot, RiskPromptState, SystemInfoSnapshot


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

        self.profile_status: dict[str, str] = {}
        self.profile_flavor: dict[str, str] = {}
        self.profile_cancel_events: dict[str, threading.Event] = {}
        self.profile_lock = threading.Lock()

        self.status_messages: list[tuple[float, str]] = []
        self.status_lock = threading.Lock()
        self.stale_warnings: dict[str, str] = {}
        self._task_cache: tuple[int, int, int] | None = None
        self._task_cache_ts: float = 0.0
        self._task_cache_ttl: float = 1.5
        _ = psutil.cpu_percent(interval=0.1, percpu=True)

        self.profile_request: str | None = None
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
        self.smoke_request = False
        self.unsaved_slots: set[str] = set()

        self.server_manager = ServerManager()
        self.slot_states: dict[str, str] = {}
        self.server_processes: dict[str, Any] = {}

    def make_collector(self, device_index: int) -> Callable[[], dict[str, Any]]:
        """Create a GPU collector bound to a device index."""

        def collector() -> dict[str, Any]:
            return collect_nvtop_stats(device_index)

        return collector

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
        samples = psutil.cpu_percent(interval=None, percpu=True)
        return [float(sample) for sample in samples]

    def memory_usage_rows(self) -> list[MemoryUsageSnapshot]:
        """Return memory and swap usage snapshots for the dashboard."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return [
            MemoryUsageSnapshot(
                label="Mem",
                percent=float(mem.percent),
                value_text=f"{self._format_bytes(int(mem.used))}/{self._format_bytes(int(mem.total))}",
            ),
            MemoryUsageSnapshot(
                label="Swp",
                percent=float(swap.percent),
                value_text=f"{self._format_bytes(int(swap.used))}/{self._format_bytes(int(swap.total))}",
            ),
        ]

    def system_info_snapshot(self) -> SystemInfoSnapshot:
        """Return process, load, and uptime state for the dashboard."""
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

    def current_datetime_text(self) -> str:
        """Return the current local date/time string for display."""
        return time.strftime("%Y-%m-%d %H:%M:%S")

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

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        gib = num_bytes / (1024**3)
        if gib >= 10:
            return f"{gib:,.1f}G"
        return f"{gib:,.2f}G"

    @staticmethod
    def _format_uptime(seconds: int) -> str:
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
