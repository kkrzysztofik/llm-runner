"""View models for the Textual dashboard."""

import logging
import time

from llama_manager import (
    GPUStats,
    ServerConfig,
    SlotState,
    resolve_slot_runtime_status,
)
from llama_manager.build_pipeline import BuildConfig
from llama_manager.config import Config
from llama_manager.config.builder import create_tui_profile_registry

from .model import DashboardModel
from .types import (
    CommandMenuState,
    CPUCoreSnapshot,
    DateTimeSnapshot,
    MemoryUsageSnapshot,
    ServerColumnState,
    SystemInfoSnapshot,
)

logger = logging.getLogger(__name__)

BACKEND_LABELS: dict[str, str] = {
    "sycl": "SYCL",
    "cuda": "CUDA",
    "llama_cpp": "CPU",
}


class DashboardViewModel:
    """Derives immutable display state from ``DashboardModel``."""

    def __init__(self, model: DashboardModel) -> None:
        self.model = model
        self._deferred_resolve: set[str] = set()

    def command_menu(self) -> CommandMenuState:
        return CommandMenuState(
            risk_prompt=self.model.risk_prompt,
            build_request=self.model.build_request,
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

    def current_datetime_snapshot(self) -> DateTimeSnapshot:
        return self.model.current_datetime_snapshot()

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

        return notices

    def column(self, slot_index: int) -> ServerColumnState | None:
        start = time.perf_counter()
        configs = self.model.configs
        if slot_index >= len(configs):
            logger.debug(
                "DashboardViewModel.column: empty slot_index=%d configs=%d",
                slot_index,
                len(configs),
            )
            return None

        cfg = configs[slot_index]
        gpu: GPUStats | None = (
            self.model.gpu_stats[slot_index] if slot_index < len(self.model.gpu_stats) else None
        )
        status = self._resolve_slot_status(cfg.alias)
        gpu_stats = gpu.get_stats_snapshot() if gpu is not None else None
        state = ServerColumnState(
            alias=cfg.alias,
            status=status,
            status_class=f"server-column-status-{status.replace('_', '-')}",
            backend_label=BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"]),
            url=f"http://{self.model.config.deployment.host}:{cfg.port}",
            config_summary=f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}",
            logs_text=self.model.log_buffers[cfg.alias].get_text(
                empty_message="Waiting for output..."
            ),
            gpu_stats=gpu_stats,
            stale_warning=self.stale_warning(cfg),
            is_unsaved=cfg.alias in self.model.unsaved_slots,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "DashboardViewModel.column: built slot_index=%d alias=%s status=%s "
            "gpu_cached=%s logs_chars=%d duration_ms=%.1f",
            slot_index,
            cfg.alias,
            status,
            gpu_stats is not None,
            len(state.logs_text),
            duration_ms,
        )
        return state

    def stale_warning(self, cfg: ServerConfig) -> str | None:
        """Return the cached stale-profile warning for a config."""
        return self.model.stale_warnings.get(cfg.alias)

    def profile_options(self, config: Config | None = None) -> list[tuple[str, str]]:
        """Return display label/value pairs for the profile dropdown."""
        cfg = config or Config()
        registry = create_tui_profile_registry(cfg)
        return [
            (
                f"{profile.profile_id} - {profile.description or profile.profile_id}",
                profile.profile_id,
            )
            for profile in registry.profiles
        ]

    def mark_slot_launching(self, alias: str) -> None:
        """Hold LAUNCHING display for one refresh cycle after process starts."""
        self._deferred_resolve.add(alias)

    def _resolve_slot_status(self, alias: str) -> str:
        state = self.model.slot_states.get(alias, SlotState.OFFLINE.value)
        proc = self.model.server_processes.get(alias)
        if state == SlotState.RUNNING.value and alias in self._deferred_resolve:
            self._deferred_resolve.discard(alias)
            return SlotState.LAUNCHING.value
        return resolve_slot_runtime_status(state, proc)

    @staticmethod
    def _content_width(width: int | None) -> int:
        if width is None or width <= 0:
            return 116
        return min(240, max(40, width))
