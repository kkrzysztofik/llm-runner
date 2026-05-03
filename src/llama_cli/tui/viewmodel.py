"""View models for the Textual dashboard."""

from __future__ import annotations

from llama_manager import (
    GPUStats,
    ProfileFlavor,
    ServerConfig,
    get_gpu_identifier,
    load_profile_with_staleness,
)

from .model import DashboardModel
from .types import CommandMenuState, ServerColumnState, SlotStatusState, SystemStatusState


class DashboardViewModel:
    """Derives immutable display state from ``DashboardModel``."""

    def __init__(self, model: DashboardModel) -> None:
        self.model = model

    def command_menu(self) -> CommandMenuState:
        return CommandMenuState(
            profile_request=self.model.profile_request,
            risk_prompt=self.model.risk_prompt,
        )

    def system_status(self) -> SystemStatusState:
        return SystemStatusState(
            gpu_lines=self.gpu_telemetry_lines(),
            notices=self.system_notices(),
        )

    def gpu_telemetry_lines(self) -> list[str]:
        lines: list[str] = []
        for gpu in self.model.gpu_stats:
            lines.append(gpu.format_stats_text())
        return lines

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

    def status_messages(self) -> list[str]:
        return self.model.recent_status_messages()

    def column(self, slot_index: int, stale_warning: str | None = None) -> ServerColumnState | None:
        configs = self.model.configs
        if slot_index >= len(configs):
            return None

        cfg = configs[slot_index]
        gpu: GPUStats | None = (
            self.model.gpu_stats[slot_index] if slot_index < len(self.model.gpu_stats) else None
        )
        return ServerColumnState(
            config=cfg,
            buffer=self.model.log_buffers[cfg.alias],
            gpu=gpu,
            host=self.model.config.host,
            stale_warning=stale_warning,
            slot_states=self.model.slot_states,
            server_processes=self.model.server_processes,
            is_unsaved=cfg.alias in self.model.unsaved_slots,
        )

    def slot_status(self, configs: list[ServerConfig] | None = None) -> SlotStatusState:
        return SlotStatusState(
            configs=self.model.configs if configs is None else configs,
            slot_states=self.model.slot_states,
            server_processes=self.model.server_processes,
            log_buffers=self.model.log_buffers,
            host=self.model.config.host,
            unsaved_slots=self.model.unsaved_slots,
        )

    def stale_warning(self, cfg: ServerConfig) -> str | None:
        """Check whether the cached profile for a config is stale."""
        try:
            from llama_cli.commands.profile import get_driver_version

            _record, staleness = load_profile_with_staleness(
                profiles_dir=self.model.config.profiles_dir,
                gpu_identifier=get_gpu_identifier(cfg.backend),
                backend=cfg.backend,
                flavor=ProfileFlavor.BALANCED,
                current_driver_version=get_driver_version(cfg.backend),
                current_binary_version=self.model.config.server_binary_version or "unknown",
                staleness_days=self.model.config.profile_staleness_days,
            )
        except Exception:
            return None

        if staleness is None or not staleness.is_stale:
            return None

        reasons = "; ".join(reason.value.replace("_", " ").title() for reason in staleness.reasons)
        return f"\u26a0 profile stale \u2014 {reasons}"
