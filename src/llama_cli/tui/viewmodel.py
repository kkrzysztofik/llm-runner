"""View models for the Textual dashboard."""

import logging
import time

from llama_manager import (
    ServerConfig,
    SlotState,
    resolve_slot_runtime_status,
)
from llama_manager.config import Config
from llama_manager.config.builder import create_tui_profile_registry
from llama_manager.slot_stats import SlotStatsSnapshot

from .model import DashboardModel
from .types import (
    CommandMenuState,
    CPUCoreSnapshot,
    DateTimeSnapshot,
    MemoryUsageSnapshot,
    ServerColumnState,
    SlotRuntimeStats,
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

    def command_menu(self) -> CommandMenuState:
        return CommandMenuState(
            risk_prompt=self.model.risk_prompt,
        )

    def server_column_count(self) -> int:
        return max(1, len(self.model.configs))

    def cpu_usage_rows(self, width: int | None = None) -> list[list[CPUCoreSnapshot]]:
        content_width = 116 if width is None or width <= 0 else min(240, max(40, width))
        cpu_per_core = self.model.dashboard_snapshot().cpu_percentages
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
        return self.model.dashboard_snapshot().memory_usage_rows

    def system_info_snapshot(self) -> SystemInfoSnapshot:
        return self.model.dashboard_snapshot().system_info

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
        snapshot = self.model.dashboard_snapshot()
        status = self._resolve_slot_status(cfg.alias)
        gpu_stats = snapshot.gpu_stats_by_alias.get(cfg.alias)

        # Load cached slot stats for this server alias
        cached_stats: SlotStatsSnapshot | None = None
        stats_by_alias = self.model.slot_stats_snapshot()
        if cfg.alias in stats_by_alias:
            cached_stats = stats_by_alias[cfg.alias]

        if cached_stats is not None:
            display = cached_stats.to_display()
            runtime_stats = SlotRuntimeStats(
                tps=display["tps"],
                pp=display["pp"],
                tokens_in=display["tokens_in"],
                tokens_out=display["tokens_out"],
            )
        else:
            runtime_stats = SlotRuntimeStats(tps="--", pp="--", tokens_in="0", tokens_out="0")

        state = ServerColumnState(
            alias=cfg.alias,
            profile_name=cfg.alias,
            status=status,
            status_label=status.replace("_", " ").title(),
            status_class=f"server-column-status-{status.replace('_', '-')}",
            backend_label=BACKEND_LABELS.get(cfg.backend, BACKEND_LABELS["llama_cpp"]),
            url=f"http://{self.model.config.deployment.host}:{cfg.port}",
            config_summary=f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}",
            runtime_stats=runtime_stats,
            gpu_stats=gpu_stats,
            stale_warning=self.stale_warning(cfg),
        )
        duration_ms = (time.perf_counter() - start) * 1000
        # log_count, not a sum over line lengths: this runs per panel per refresh and
        # argument expressions are evaluated even when DEBUG is disabled.
        logger.debug(
            "DashboardViewModel.column: built slot_index=%d alias=%s status=%s "
            "gpu_cached=%s duration_ms=%.1f",
            slot_index,
            cfg.alias,
            status,
            gpu_stats is not None,
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

    def _resolve_slot_status(self, alias: str) -> str:
        state = self.model.slot_states.get(alias, SlotState.OFFLINE.value)
        proc = self.model.server_processes.get(alias)
        return resolve_slot_runtime_status(state, proc)
