"""View models for the Textual dashboard."""

from __future__ import annotations

import psutil

from llama_manager import (
    GPUStats,
    ServerConfig,
    SlotState,
)
from llama_manager.build_pipeline import BuildConfig

from .model import DashboardModel
from .types import (
    CommandMenuState,
    CPUCoreSnapshot,
    MemoryUsageSnapshot,
    ServerColumnState,
    SystemInfoSnapshot,
)

BACKEND_LABELS: dict[str, str] = {
    "sycl": "SYCL",
    "cuda": "CUDA",
    "llama_cpp": "CPU",
}


class DashboardViewModel:
    """Derives immutable display state from ``DashboardModel``."""

    def __init__(self, model: DashboardModel) -> None:
        self.model = model

    def command_menu(self) -> CommandMenuState:
        return CommandMenuState(
            profile_request=self.model.profile_request,
            risk_prompt=self.model.risk_prompt,
            build_request=self.model.build_request,
            smoke_request=self.model.smoke_request,
        )

    def gpu_telemetry_lines(self) -> list[str]:
        lines: list[str] = []
        for gpu in self.model.gpu_stats:
            lines.append(gpu.format_stats_text())
        return lines

    def server_column_count(self) -> int:
        return max(1, len(self.model.configs))

    def can_select_build_target(self) -> bool:
        return self.model.build_request and self.model.build_selected_backends is None

    @property
    def build_selected_backends(self) -> list[str] | None:
        return self.model.build_selected_backends

    @property
    def build_in_progress(self) -> bool:
        return self.model.build_in_progress

    @property
    def build_result(self) -> str | None:
        return self.model.build_result

    @property
    def build_error(self) -> str | None:
        return self.model.build_error

    @property
    def build_selected_backends_options(self) -> dict[str, BuildConfig | None]:
        return self.model.build_selected_backends_options

    @property
    def build_stage(self) -> str | None:
        return self.model.build_stage

    @property
    def build_progress_percent(self) -> float:
        return self.model.build_progress_percent

    def cpu_usage_rows(self, width: int | None = None) -> list[list[CPUCoreSnapshot]]:
        content_width = self._content_width(width)
        cpu_per_core = self.model.cpu_percentages()
        if not cpu_per_core:
            return []

        max_cols = max(1, content_width // 16)
        rows = max(1, (len(cpu_per_core) + max_cols - 1) // max_cols)
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

    def memory_usage_rows(self) -> list[MemoryUsageSnapshot]:
        return self.model.memory_usage_rows()

    def system_info_snapshot(self) -> SystemInfoSnapshot:
        return self.model.system_info_snapshot()

    def current_datetime_text(self) -> str:
        return self.model.current_datetime_text()

    def system_notices(self) -> list[str]:
        notices: list[str] = []
        launch_result = self.model.launch_result
        if launch_result is not None:
            if launch_result.is_blocked():
                notices.append("Launch blocked: no slots could be launched")
            elif launch_result.is_degraded():
                notices.append("Launch degraded: some slots blocked")

        risk_prompt = self.model.risk_prompt
        if risk_prompt is not None:
            if risk_prompt.kind == "vram":
                notices.append("VRAM risk acknowledgement required [y/n]")
            elif risk_prompt.acknowledged:
                notices.append("Risky operation acknowledged")
            else:
                notices.append("Hardware risk acknowledgement required [y/n]")

        with self.model.profile_lock:
            running_profiles = [
                alias for alias, status in self.model.profile_status.items() if status == "running"
            ]
        if running_profiles:
            notices.append(f"Profiles running: {', '.join(running_profiles)}")

        return notices

    def active_profile_status(self) -> dict[str, str]:
        with self.model.profile_lock:
            return {
                alias: status
                for alias, status in self.model.profile_status.items()
                if status != "idle"
            }

    def column(self, slot_index: int) -> ServerColumnState | None:
        configs = self.model.configs
        if slot_index >= len(configs):
            return None

        cfg = configs[slot_index]
        gpu: GPUStats | None = (
            self.model.gpu_stats[slot_index] if slot_index < len(self.model.gpu_stats) else None
        )
        status = self._resolve_slot_status(cfg.alias)
        return ServerColumnState(
            alias=cfg.alias,
            status=status,
            status_class=f"server-column-status-{status.replace('_', '-')}",
            backend_label=BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"]),
            url=f"http://{self.model.config.host}:{cfg.port}",
            config_summary=f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}",
            logs_text=self.model.log_buffers[cfg.alias].get_text(
                empty_message="Waiting for output..."
            ),
            gpu_stats=gpu.get_stats_snapshot() if gpu is not None else None,
            stale_warning=self.stale_warning(cfg),
            is_unsaved=cfg.alias in self.model.unsaved_slots,
        )

    def stale_warning(self, cfg: ServerConfig) -> str | None:
        """Return the cached stale-profile warning for a config."""
        return self.model.stale_warnings.get(cfg.alias)

    def _resolve_slot_status(self, alias: str) -> str:
        state = self.model.slot_states.get(alias, SlotState.OFFLINE.value)
        status = state
        if state == SlotState.RUNNING.value:
            proc = self.model.server_processes.get(alias)
            if not proc:
                status = SlotState.CRASHED.value
            elif hasattr(proc, "poll"):
                if proc.poll() is not None:
                    status = SlotState.CRASHED.value
            else:
                pid = getattr(proc, "pid", None)
                if not (pid and psutil.pid_exists(pid)):
                    status = SlotState.CRASHED.value
        return status

    @staticmethod
    def _content_width(width: int | None) -> int:
        if width is None or width <= 0:
            return 116
        return min(240, max(40, width))
