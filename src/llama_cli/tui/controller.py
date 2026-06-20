"""Dashboard controller for the Textual TUI."""

import contextlib
import json
import logging
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType
from typing import Any, Literal

from llama_manager import (
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelIndexEntry,
    ModelSlot,
    ProfileFlavor,
    RiskAckResult,
    ServerConfig,
    ServerManager,
    SlotState,
    collector_for_config,
    compute_slot_transition,
    get_gpu_identifier,
    gpu_index_for_config,
    launch_orchestrate,
    load_model_index,
    load_profile_with_staleness,
    model_index_path,
    refresh_model_index,
    resolve_risk_action,
    selector_for_config,
)
from llama_manager.build_pipeline import (
    BuildConfig,
    BuildPipeline,
    BuildProgress,
    run_build_for_backend,
)
from llama_manager.config.profiles import resolve_profile_id
from llama_manager.logging_setup import (
    suppress_build_pipeline_stderr_for_tui,
    update_file_level,
    update_stderr_level,
)
from llama_manager.slot_stats import (
    collect_slot_stats,
    load_profile_stats,
    load_slot_stats,
    save_profile_stats,
    save_slot_stats,
    update_profile_stats,
)

from .components.config_modal import ConfigPayload
from .components.slot_profile_modal import SlotProfilePayload
from .constants import MSG_BUILD_CANCELLED, MSG_BUILD_FAILED
from .model import DashboardModel
from .textual_app import DashboardApp
from .viewmodel import DashboardViewModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncSlotPlan:
    """UI-thread plan for a background slot launch."""

    success: bool
    messages: list[str]
    alias: str
    profile_id: str
    old_alias: str | None


@dataclass(frozen=True)
class AsyncSlotStageResult:
    """State staged on the UI thread before a worker starts a slot process."""

    success: bool
    messages: list[str]
    alias: str
    log_buffer: LogBuffer | None


class DashboardController:
    """Controller for the Textual TUI dashboard — commands, lifecycle, and background work.

    The controller is the central hub that coordinates between the model
    (:class:`~.model.DashboardModel`), the view model
    (:class:`~.viewmodel.DashboardViewModel`), and the Textual app
    (:class:`~.textual_app.DashboardApp`). It owns all user-facing command
    handlers (launch, build, slot management, config editing, risk prompts)
    and manages background threads for builds.

    Responsibilities:

    * **Lifecycle** — register signal handlers (SIGINT/SIGTERM), start/stop the
      TUI loop, and perform graceful shutdown of server processes.
    * **Launch orchestration** — delegate to :func:`~llama_manager.launch_orchestrate`
      and map results to UI state (risk prompts, slot states, status messages).
    * **Build pipeline** — run :func:`~llama_manager.build_pipeline.run_build_for_backend`
      in a daemon thread, expose progress via :attr:`build_progress`, and
      coordinate with the build wizard modal.
    * **Slot management** — create/replace slots from the add-slot modal, track
      slot state transitions, and detect duplicate slot IDs.
    * **Config editing** — persist edited values from the config modal and
      optionally trigger a server restart.
    * **Risk prompts** — surface VRAM and hardware mismatch warnings, resolve
      user acknowledgment (proceed / abort / quit).

    Args:
        configs: List of :class:`~llama_manager.ServerConfig` instances to manage.
        gpu_indices: GPU device indices to monitor (one per backend).
        slots: Optional list of :class:`~llama_manager.ModelSlot` instances.
        register_signals: If ``True``, register SIGINT/SIGTERM handlers on init.
    """

    def __init__(
        self,
        configs: list[ServerConfig],
        gpu_indices: list[int],
        slots: list[ModelSlot] | None = None,
        register_signals: bool = True,
    ) -> None:
        self.model = DashboardModel(configs=configs, gpu_indices=gpu_indices, slots=slots)
        self.view_model = DashboardViewModel(self.model)

        # Load persisted slot stats so the TUI shows last-known values immediately
        self._load_persisted_slot_stats()

        # Build pipeline state
        self._build_pipeline: BuildPipeline | None = None
        self.build_in_progress = False
        self.build_progress: BuildProgress | None = None
        self._original_sigint_handler: Callable[[int, FrameType | None], Any] | int | None = None
        self._build_wizard: Any = None  # BuildModalScreen | None
        self._model_index_cache: list[ModelIndexEntry] | None = None
        self._model_index_lock = threading.Lock()
        self._model_index_refreshing = False

        if register_signals:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    @property
    def config(self) -> Config:
        return self.model.config

    @property
    def configs(self) -> list[ServerConfig]:
        return self.model.configs

    @configs.setter
    def configs(self, value: list[ServerConfig]) -> None:
        self.model.configs = value

    @property
    def gpu_indices(self) -> list[int]:
        return self.model.gpu_indices

    @property
    def slots(self) -> list[ModelSlot]:
        return self.model.slots

    def _load_persisted_slot_stats(self) -> None:
        """Load persisted slot stats from disk into the model cache."""
        try:
            self.model.apply_slot_stats_snapshot(load_slot_stats())
        except Exception:
            logger.debug("failed to load persisted slot stats", exc_info=True)

    def refresh_slot_stats(self) -> None:
        """Collect live slot stats for all running configs and persist changes."""
        current = self.model.slot_stats_snapshot()
        updated = dict(current)
        profile_stats = load_profile_stats()
        profile_stats_changed = False
        changed = False
        for cfg in self.model.configs:
            try:
                stats = collect_slot_stats(cfg.alias, self.model.config.deployment.host, cfg.port)
                if stats is None:
                    continue
                updated[cfg.alias] = stats
                changed = True
                profile_id = self.resolve_profile_id_for_config(cfg)
                if profile_id is not None:
                    profile_stats = update_profile_stats(
                        profile_stats,
                        profile_id,
                        self._profile_stats_session_id(cfg.alias),
                        stats,
                    )
                    profile_stats_changed = True
            except Exception:
                logger.exception("refresh_slot_stats: failed to collect for %s", cfg.alias)
        if changed:
            try:
                self.model.apply_slot_stats_snapshot(updated)
                save_slot_stats(updated)
            except Exception:
                logger.exception("refresh_slot_stats: failed to persist slot stats")
        if profile_stats_changed:
            try:
                save_profile_stats(profile_stats)
            except Exception:
                logger.exception("refresh_slot_stats: failed to persist profile stats")

    def resolve_profile_id_for_config(self, cfg: ServerConfig) -> str | None:
        """Resolve a live server config alias to a registered profile ID."""
        registry = self._build_tui_registry()
        resolved = resolve_profile_id(registry, cfg.alias)
        if resolved is not None:
            return resolved
        if cfg.alias.endswith("-coding"):
            return resolve_profile_id(registry, cfg.alias.removesuffix("-coding"))
        return None

    def _profile_stats_session_id(self, alias: str) -> str:
        """Return a stable ID for the current server process behind *alias*."""
        process = self.server_processes.get(alias)
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            return f"{alias}:{pid}"
        return alias

    @property
    def log_buffers(self) -> dict[str, LogBuffer]:
        return self.model.log_buffers

    @property
    def gpu_stats(self) -> list[GPUStats]:
        return self.model.gpu_stats

    @property
    def running(self) -> bool:
        return self.model.running

    @running.setter
    def running(self, value: bool) -> None:
        self.model.running = value

    @property
    def launch_result(self) -> LaunchResult | None:
        return self.model.launch_result

    @launch_result.setter
    def launch_result(self, value: LaunchResult | None) -> None:
        self.model.launch_result = value

    @property
    def risks_acknowledged(self) -> bool:
        return bool(self.model.risk_prompt and self.model.risk_prompt.acknowledged)

    @property
    def active_risk_kind(self) -> Literal["vram", "hardware"] | None:
        return self.model.risk_prompt.kind if self.model.risk_prompt is not None else None

    @active_risk_kind.setter
    def active_risk_kind(self, value: Literal["vram", "hardware"] | None) -> None:
        if value is None:
            self.model.clear_risk_prompt()
        else:
            self.model.set_risk_prompt(value, acknowledged=self.risks_acknowledged)

    @property
    def _build_request(self) -> bool:
        return self.model.build_request

    @_build_request.setter
    def _build_request(self, value: bool) -> None:
        self.model.build_request = value

    @property
    def unsaved_slots(self) -> set[str]:
        return self.model.unsaved_slots

    @property
    def server_manager(self) -> ServerManager:
        return self.model.server_manager

    @property
    def slot_states(self) -> dict[str, str]:
        return self.model.slot_states

    @slot_states.setter
    def slot_states(self, value: dict[str, str]) -> None:
        self.model.slot_states = value

    @property
    def server_processes(self) -> dict[str, Any]:
        return self.model.server_processes

    @server_processes.setter
    def server_processes(self, value: dict[str, Any]) -> None:
        self.model.server_processes = value

    @property
    def _status_messages(self) -> list[tuple[float, str]]:
        return self.model.status_messages

    @property
    def _status_lock(self) -> threading.Lock:
        return self.model.status_lock

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals by stopping the TUI loop.

        If a build is in progress, release the build lock before stopping.
        """
        # Release build lock if in progress
        if self.build_in_progress and self._build_pipeline is not None:
            self._build_pipeline.release_lock()
            self.build_in_progress = False

        self.stop()

    def _signal_handler_build(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler specifically for build process.

        Releases build lock and stops the build gracefully.
        """
        if self.build_in_progress and self._build_pipeline is not None:
            self._push_status_message("Build interrupted by user, releasing lock...")
            self._build_pipeline.release_lock()
            self.build_in_progress = False
            # Set cancel event for the build thread
            cancel_event = getattr(self.model, "build_cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()

    def _make_collector(self, device_index: int) -> Callable[[], dict[str, Any]]:
        """Create a GPU collector bound to a specific device index."""
        return self.model.make_collector(device_index)

    def can_select_build_target(self) -> bool:
        return self.view_model.can_select_build_target()

    def stop(self) -> None:
        """Stop the TUI loop gracefully."""
        self.running = False

    def _build_risk_panel_required(self, kind: Literal["vram", "hardware"] = "hardware") -> None:
        self.model.set_risk_prompt(kind=kind, acknowledged=False)

    def _build_risk_panel_acknowledged(
        self, kind: Literal["vram", "hardware"] = "hardware"
    ) -> None:
        self.model.set_risk_prompt(kind=kind, acknowledged=True)

    # How long (seconds) a status message remains visible across renders.
    _STATUS_MESSAGE_LIFETIME_S: float = DashboardModel.STATUS_MESSAGE_LIFETIME_S

    def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
        """Return status messages newer than ``since_ts``."""
        return self.model.get_status_messages_since(since_ts)

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        self.server_manager.cleanup_servers()

    def request_quit(self) -> None:
        """Request a graceful shutdown from the UI."""
        if self.model.risk_prompt is not None:
            if self.active_risk_kind == "hardware":
                self.handle_hardware_warning("q")
            return
        self._graceful_shutdown()

    def interrupt(self) -> None:
        """Request an interrupt from the UI (graceful shutdown when no risk prompt)."""
        if self.model.risk_prompt is not None:
            return

        self._graceful_shutdown()

    def refresh_display(self) -> None:
        """Request a refresh message from the UI."""
        if self.model.risk_prompt is not None:
            return
        self._push_status_message("Display refreshed.")

    def request_build(self) -> None:
        """Start the build selection flow from the UI."""
        if self.model.risk_prompt is not None:
            return
        self._build_request = True
        self._push_status_message("Select build target: [1] SYCL  [2] CUDA  [3] Both")

    def cancel_pending_prompt(self) -> bool:
        """Cancel any pending build prompt."""
        if self._build_request:
            self._build_request = False
            self._push_status_message("Build selection cancelled.")
            return True

        return False

    def acknowledge_risk(self) -> None:
        """Acknowledge the active risk prompt."""
        if self.model.risk_prompt is None:
            return
        if self.active_risk_kind == "vram":
            self.handle_vram_risk("y")
        else:
            self.handle_hardware_warning("y")

    def reject_risk(self) -> None:
        """Reject the active risk prompt."""
        if self.model.risk_prompt is None:
            return
        if self.active_risk_kind == "vram":
            self.handle_vram_risk("n")
        else:
            self.handle_hardware_warning("n")

    def _push_status_message(self, message: str) -> None:
        """Push a message to the TUI-safe status buffer and trigger a refresh.

        This method is safe to call from TUI handlers — it only mutates the
        status buffer and leaves rendering to Textual.
        """
        logger.debug("status: %s", message)
        self.model.push_status_message(message)

    def refresh_stale_warnings(self, get_driver_version: Callable[[str], str]) -> None:
        """Refresh cached stale-profile warnings for all configured slots."""
        warnings: dict[str, str] = {}
        for cfg in self.configs:
            try:
                _record, staleness = load_profile_with_staleness(
                    profiles_dir=self.config.paths.profiles_dir,
                    gpu_identifier=get_gpu_identifier(cfg.backend),
                    backend=cfg.backend,
                    flavor=ProfileFlavor.BALANCED,
                    current_driver_version=get_driver_version(cfg.backend),
                    current_binary_version=self.config.server_binary_version or "unknown",
                    staleness_days=self.config.profile_staleness_days,
                )
            except OSError, ValueError, KeyError:
                continue

            if staleness is None or not staleness.is_stale:
                continue

            reasons = "; ".join(
                reason.value.replace("_", " ").title() for reason in staleness.reasons
            )
            warnings[cfg.alias] = f"profile stale - {reasons}"

        self.model.stale_warnings = warnings

    def _build_tui_registry(self) -> Any:
        """Build the TUI profile registry (built-in + custom profiles from disk)."""
        from llama_manager.config.builder import create_tui_profile_registry

        return create_tui_profile_registry(self.config)

    def compute_add_slot_from_form(
        self,
        values: dict[str, str],
    ) -> tuple[bool, list[str], str, ServerConfig | None]:
        """Validate form values and resolve profile config without mutating TUI state."""
        from llama_manager.slot_manager import compute_add_slot_from_form

        registry = self._build_tui_registry()
        return compute_add_slot_from_form(values, self.config, registry=registry)

    def apply_add_slot_from_form(
        self,
        new_cfg: ServerConfig,
        profile_id: str,
        startup_callback: Callable[[], None] | None = None,
    ) -> tuple[bool, list[str]]:
        """Apply a resolved profile config to dashboard runtime state."""
        from llama_manager.slot_manager import upsert_profile_slot

        state = {
            "log_buffers": self.log_buffers,
            "server_processes": self.server_processes,
            "slot_states": self.slot_states,
            "unsaved_slots": self.unsaved_slots,
            "slots": self.slots,
        }
        success, messages, _updated_state = upsert_profile_slot(
            new_cfg,
            profile_id,
            self.configs,
            self.gpu_indices,
            self.gpu_stats,
            self.server_manager,
            state,
            startup_callback=startup_callback,
        )
        for msg in messages:
            self._push_status_message(msg)
        active_aliases = {cfg.alias for cfg in self.configs}
        self.model.apply_gpu_stats_snapshot(
            {
                cfg.alias: gpu.get_cached_stats_snapshot()
                for cfg, gpu in zip(self.configs, self.gpu_stats, strict=False)
            }
        )
        self.model.stale_warnings = {
            alias: warning
            for alias, warning in self.model.stale_warnings.items()
            if alias in active_aliases
        }
        return success, messages

    def remove_live_slot(self, alias: str) -> bool:
        """Stop and remove one live slot from dashboard runtime state."""
        from llama_manager.slot_manager import remove_profile_slot

        state = {
            "log_buffers": self.log_buffers,
            "server_processes": self.server_processes,
            "slot_states": self.slot_states,
            "unsaved_slots": self.unsaved_slots,
            "slots": self.slots,
        }
        success, messages, _updated_state = remove_profile_slot(
            alias,
            self.configs,
            self.gpu_indices,
            self.gpu_stats,
            self.log_buffers,
            self.server_manager,
            state,
        )
        for msg in messages:
            self._push_status_message(msg)
        active_aliases = {cfg.alias for cfg in self.configs}
        self.model.apply_gpu_stats_snapshot(
            {
                cfg.alias: gpu.get_cached_stats_snapshot()
                for cfg, gpu in zip(self.configs, self.gpu_stats, strict=False)
            }
        )
        self.model.stale_warnings = {
            stale_alias: warning
            for stale_alias, warning in self.model.stale_warnings.items()
            if stale_alias in active_aliases
        }
        return success

    def prepare_async_slot_launch(
        self,
        new_cfg: ServerConfig,
        profile_id: str,
    ) -> AsyncSlotPlan:
        """Prepare a background slot launch without mutating dashboard state."""
        from llama_manager.slot_manager import device_class_for_config

        target_device = device_class_for_config(new_cfg)
        old_alias = next(
            (
                existing_cfg.alias
                for existing_cfg in self.configs
                if device_class_for_config(existing_cfg) == target_device
            ),
            None,
        )
        return AsyncSlotPlan(
            success=True,
            messages=[],
            alias=new_cfg.alias,
            profile_id=profile_id,
            old_alias=old_alias,
        )

    def stage_async_slot_launch(
        self,
        new_cfg: ServerConfig,
        old_alias: str | None,
    ) -> AsyncSlotStageResult:
        """Commit launching slot state on the UI thread before process start."""
        from llama_manager.slot_manager import (
            device_class_for_config,
            remove_slot_runtime_state,
        )

        alias = new_cfg.alias
        target_device = device_class_for_config(new_cfg)
        state = {
            "log_buffers": self.log_buffers,
            "server_processes": self.server_processes,
            "slot_states": self.slot_states,
            "unsaved_slots": self.unsaved_slots,
            "slots": self.slots,
        }
        messages: list[str] = []

        if old_alias is None:
            self.configs.append(new_cfg)
            self.gpu_indices.append(gpu_index_for_config(new_cfg))
            new_gpu = GPUStats(
                gpu_index_for_config(new_cfg),
                collector=collector_for_config(new_cfg),
                selector=selector_for_config(new_cfg),
            )
            self.gpu_stats.append(new_gpu)
        else:
            existing_index = next(
                (
                    idx
                    for idx, existing_cfg in enumerate(self.configs)
                    if existing_cfg.alias == old_alias
                ),
                None,
            )
            if existing_index is None:
                messages.append(
                    f"Unable to replace '{old_alias}' on {target_device}: slot not found"
                )
                for msg in messages:
                    self._push_status_message(msg)
                return AsyncSlotStageResult(False, messages, alias, None)

            remove_slot_runtime_state(old_alias, state)
            self.model.remove_cached_gpu_stats(old_alias)
            self.configs[existing_index] = new_cfg
            gpu_idx = gpu_index_for_config(new_cfg)
            self.gpu_indices[existing_index] = gpu_idx
            new_gpu = GPUStats(
                gpu_idx,
                collector=collector_for_config(new_cfg),
                selector=selector_for_config(new_cfg),
            )
            self.gpu_stats[existing_index] = new_gpu

        log_buffer = LogBuffer(redact_sensitive=True)
        self.log_buffers[alias] = log_buffer
        self.model.set_cached_gpu_stats(alias, new_gpu.get_cached_stats_snapshot())
        self.unsaved_slots.add(alias)
        self.slots.append(ModelSlot(slot_id=alias, model_path=new_cfg.model, port=new_cfg.port))
        self.slot_states[alias] = SlotState.LAUNCHING.value
        self.model.stale_warnings = {
            stale_alias: warning
            for stale_alias, warning in self.model.stale_warnings.items()
            if stale_alias in {cfg.alias for cfg in self.configs}
        }

        messages.append(f"Slot '{alias}' launching...")
        for msg in messages:
            self._push_status_message(msg)
        return AsyncSlotStageResult(True, messages, alias, log_buffer)

    def complete_async_slot_launch(
        self,
        alias: str,
        profile_id: str,
        old_alias: str | None,
        process: Any | None,
    ) -> tuple[bool, list[str]]:
        """Commit final slot launch state on the UI thread."""
        from llama_manager.slot_manager import device_class_for_config

        cfg = next((item for item in self.configs if item.alias == alias), None)
        target_device = device_class_for_config(cfg) if cfg is not None else "unknown"
        messages: list[str] = []

        if process is None:
            self.slot_states[alias] = SlotState.CRASHED.value
            messages.append(f"Slot '{alias}' failed to start: no process returned")
            for msg in messages:
                self._push_status_message(msg)
            return False, messages

        self.server_processes[alias] = process
        old_state = self.slot_states.get(alias)
        self.slot_states[alias] = SlotState.RUNNING.value
        result = compute_slot_transition(alias, old_state, SlotState.RUNNING)
        if result is not None:
            message, _color = result
            messages.append(message)
            logger.info("slot %s: %s", alias, message)

        port = cfg.port if cfg is not None else "unknown"
        if old_alias is None:
            messages.append(f"Added profile '{profile_id}' as '{alias}' on {target_device}:{port}")
        else:
            messages.append(
                f"Replaced '{old_alias}' with profile '{profile_id}' as "
                f"'{alias}' on {target_device}:{port}"
            )

        for msg in messages:
            self._push_status_message(msg)
        return True, messages

    def prepare_async_slot_remove(self, alias: str) -> tuple[bool, list[str]]:
        """Validate slot removal on the UI thread before worker shutdown."""
        if not any(existing_cfg.alias == alias for existing_cfg in self.configs):
            return False, [f"Unable to remove '{alias}': slot not found"]
        return True, []

    def commit_async_slot_remove(self, alias: str) -> tuple[bool, list[str]]:
        """Commit slot removal state on the UI thread after worker shutdown."""
        from llama_manager.slot_manager import remove_slot_runtime_state

        existing_index = next(
            (idx for idx, existing_cfg in enumerate(self.configs) if existing_cfg.alias == alias),
            None,
        )
        if existing_index is None:
            messages = [f"Unable to remove '{alias}': slot not found"]
            for msg in messages:
                self._push_status_message(msg)
            return False, messages

        if len(self.configs) != len(self.gpu_indices) or len(self.configs) != len(self.gpu_stats):
            raise RuntimeError("slot runtime lists must remain length-synchronized")

        del self.configs[existing_index]
        del self.gpu_indices[existing_index]
        del self.gpu_stats[existing_index]
        state = {
            "log_buffers": self.log_buffers,
            "server_processes": self.server_processes,
            "slot_states": self.slot_states,
            "unsaved_slots": self.unsaved_slots,
            "slots": self.slots,
        }
        remove_slot_runtime_state(alias, state)
        self.model.remove_cached_gpu_stats(alias)
        self.model.stale_warnings = {
            stale_alias: warning
            for stale_alias, warning in self.model.stale_warnings.items()
            if stale_alias in {cfg.alias for cfg in self.configs}
        }
        messages = [f"Removed slot '{alias}'"]
        for msg in messages:
            self._push_status_message(msg)
        return True, messages

    def add_slot_from_form(self, values: dict[str, str]) -> bool:
        """Create or replace a slot from modal profile selection."""
        logger.debug(
            "add_slot_from_form: enter values=%r configs_before=%r",
            values,
            [c.alias for c in self.configs],
        )
        success, messages, profile_id, new_cfg = self.compute_add_slot_from_form(values)
        for msg in messages:
            self._push_status_message(msg)
        if not success or new_cfg is None:
            logger.debug(
                "add_slot_from_form: compute failed success=%s messages=%r",
                success,
                messages,
            )
            return False

        apply_success, apply_messages = self.apply_add_slot_from_form(new_cfg, profile_id)
        logger.debug(
            "add_slot_from_form: result success=%s messages=%r configs_after=%r",
            apply_success,
            messages + apply_messages,
            [c.alias for c in self.configs],
        )
        return apply_success

    def cancel_add_slot_form(self) -> None:
        """Emit a status message when the add-slot modal is cancelled."""
        self._push_status_message("Slot configuration cancelled")

    def get_stale_warning(self, cfg: ServerConfig) -> str | None:
        """Return a warning string when the cached profile is stale."""
        return self.view_model.stale_warning(cfg)

    def _update_risk_panel_state(self, result: RiskAckResult | None) -> None:
        if result is None:
            self.model.clear_risk_prompt()
            return
        if result.has_risks:
            if result.risks_acknowledged:
                self._build_risk_panel_acknowledged()
            else:
                self._build_risk_panel_required()
            return
        self.model.clear_risk_prompt()

    def _apply_risk_action(self, action: str) -> None:
        if action in ("acknowledge", "proceed"):
            self.active_risk_kind = None
        elif action == "abort":
            self.running = False
            self.active_risk_kind = None
        elif action == "quit":
            self._graceful_shutdown()
            self.active_risk_kind = None

    def handle_hardware_warning(self, key: str) -> str:
        """Handle hardware mismatch warning key press.

        Args:
            key: The key pressed by the user.

        Returns:
            'acknowledge' if user acknowledged, 'abort' if rejected,
            'quit' if user pressed q, or 'ignore' for unknown keys.

        """
        action = resolve_risk_action(key, "hardware")
        self._apply_risk_action(action)
        return action

    def handle_vram_risk(self, key: str) -> str:
        """Handle VRAM risk confirmation key press.

        Args:
            key: The key pressed by the user.

        Returns:
            'proceed' if user confirmed, 'abort' if rejected,
            or 'ignore' for unknown keys.

        """
        action = resolve_risk_action(key, "vram")
        self._apply_risk_action(action)
        return action

    def handle_slot_transition(self, slot_id: str, new_state: SlotState) -> None:
        """Handle a slot state transition and update the UI.

        Args:
            slot_id: The slot identifier.
            new_state: The new state for the slot.
        """
        old_state = self.slot_states.get(slot_id)
        self.slot_states[slot_id] = new_state.value

        result = compute_slot_transition(slot_id, old_state, new_state)
        if result is None:
            return

        message, _color = result
        self._push_status_message(message)

    def _graceful_shutdown(self) -> None:
        """Initiate graceful shutdown of all server processes."""
        if not self.running:
            return

        self._push_status_message("Shutting down...")
        self.server_manager.cleanup_servers()
        self.running = False

    def save_config(self, payload: ConfigPayload) -> None:
        """Persist edited config values and optionally restart all servers.

        Args:
            payload: Typed config values and restart flag from the modal.
        """
        from llama_manager import apply_config_updates

        result = apply_config_updates(self.model.config, payload.to_config_updates())

        if result.errors:
            for error in result.errors:
                self._push_status_message(error)
            return

        if result.updated_fields:
            self._push_status_message("Config saved to disk.")
            # Live-update logging levels if they changed
            if "log_file_level" in result.updated_fields:
                with contextlib.suppress(Exception):
                    update_file_level(self.config.log_file_level)
            if "log_stderr_level" in result.updated_fields:
                with contextlib.suppress(Exception):
                    update_stderr_level(self.config.log_stderr_level)

        if payload.restart:
            self._push_status_message("Restarting servers with new config…")
            self.server_manager.cleanup_servers()
            self._push_status_message(
                "Servers stopped. Use 'uv run llm-runner' to relaunch with updated config."
            )
            self.running = False

    def clean_model_cache(self) -> tuple[bool, str]:
        """Delete the model index cache file and clear in-memory cache.

        Returns:
            A tuple of (success, message).
        """
        idx_path = model_index_path(self.config)

        if not idx_path.exists():
            return (False, "No model cache to clean")

        try:
            idx_path.unlink()
            with self._model_index_lock:
                self._model_index_cache = None
            return (True, "Model cache cleaned")
        except OSError as exc:
            return (False, f"Failed to clean model cache: {exc}")

    def list_slot_profiles(self) -> list[tuple[Any, str]]:
        """Return list of ``(SlotProfileSpec, source)`` tuples for all profiles.

        Source is ``'builtin'`` or ``'custom'``.
        """
        from llama_manager.slot_profile_store import custom_slot_profile_exists

        registry = self._build_tui_registry()
        result: list[tuple[Any, str]] = []
        for p in registry.profiles:
            is_custom = custom_slot_profile_exists(p.profile_id)
            source = "custom" if is_custom else "builtin"
            result.append((p, source))
        return result

    def _builtin_profile_ids(self) -> set[str]:
        """Return the set of known built-in profile IDs."""
        return {"summary-balanced", "summary-fast", "qwen35"}

    def is_profile_in_use(self, profile_id: str) -> bool:
        """Check if *profile_id* is used by any currently running server config."""
        for cfg in self.configs:
            if cfg.alias == profile_id or cfg.alias == f"{profile_id}-coding":
                return True
        return False

    def update_slot_profile(self, original_profile_id: str, payload: SlotProfilePayload) -> bool:
        """Update an existing slot profile.

        Handles both built-in override and custom profile update.

        Args:
            original_profile_id: The profile_id that was used when the profile
                was loaded.
            payload: Typed form values from the edit modal.

        Returns:
            True if the profile was updated successfully, False otherwise.
        """
        ok, profile_id = self._validate_slot_profile_payload(payload, require_device=True)
        if not ok:
            return False

        from llama_manager.slot_profile_store import upsert_custom_slot_profile

        spec = self._payload_to_spec(profile_id, payload)
        if spec is None:
            return False

        return self._save_profile_with_status(
            lambda: upsert_custom_slot_profile(original_profile_id, spec),
            success_message=f"Profile '{profile_id}' updated",
        )

    def delete_slot_profile(self, profile_id: str) -> bool:
        """Delete/hide a slot profile. Returns True if successful.

        Args:
            profile_id: The profile identifier to delete/hide.

        Returns:
            True if the profile was found and acted on, False otherwise.
        """
        from llama_manager.slot_profile_store import delete_custom_slot_profile

        if self.is_profile_in_use(profile_id):
            self._push_status_message(f"Cannot delete '{profile_id}': in use by running slot")
            return False

        builtin_ids = self._builtin_profile_ids()
        try:
            result = delete_custom_slot_profile(profile_id, builtin_ids)
        except Exception as exc:
            self._push_status_message(f"Error deleting profile: {exc}")
            return False

        if result:
            self._push_status_message(f"Profile '{profile_id}' deleted")
        else:
            self._push_status_message(f"Profile '{profile_id}' not found")
        return result

    def save_slot_profile_from_form(self, payload: SlotProfilePayload) -> bool:
        """Save a custom slot profile from the modal form payload.

        Args:
            payload: Typed form values from the slot profile modal.

        Returns:
            True if the profile was saved successfully, False otherwise.
        """
        ok, profile_id = self._validate_slot_profile_payload(payload, require_device=False)
        if not ok:
            return False

        from llama_manager.slot_profile_store import save_custom_slot_profile

        if self._profile_id_exists(profile_id):
            self._push_status_message(f"Profile ID '{profile_id}' already exists")
            return False

        spec = self._payload_to_spec(profile_id, payload)
        if spec is None:
            return False

        return self._save_profile_with_status(
            lambda: save_custom_slot_profile(spec),
            success_message=None,
        )

    def _validate_slot_profile_payload(
        self, payload: SlotProfilePayload, *, require_device: bool
    ) -> tuple[bool, str]:
        """Validate the common shape of a slot-profile modal payload.

        Returns (ok, normalized_profile_id). The id is normalised
        (lowercase, spaces→dashes) and returned even on failure so callers
        can reuse it in error messages.
        """
        profile_id = payload.profile_id.strip().lower().replace(" ", "-")
        if not profile_id:
            self._push_status_message("Profile ID is required")
            return False, profile_id

        if require_device and not payload.device:
            self._push_status_message("Device is required")
            return False, profile_id

        if not payload.model:
            self._push_status_message("Model path is required")
            return False, profile_id

        if not (1024 <= payload.port <= 65535):
            self._push_status_message("Port must be between 1024 and 65535")
            return False, profile_id

        if payload.ctx_size <= 0 or payload.ubatch_size <= 0 or payload.threads <= 0:
            self._push_status_message("ctx_size, ubatch_size, and threads must be positive")
            return False, profile_id

        if not self._validate_n_gpu_layers(payload.n_gpu_layers):
            return False, profile_id

        if not self._validate_chat_template_kwargs(payload.chat_template_kwargs):
            return False, profile_id

        return True, profile_id

    @staticmethod
    def _validate_n_gpu_layers(ngl: int | str) -> bool:
        """Return True when *ngl* is 'all' or a non-negative integer."""
        if ngl == "all":
            return True
        try:
            ngl_int = int(ngl)
        except TypeError, ValueError:
            return False
        return ngl_int >= 0

    @staticmethod
    def _validate_chat_template_kwargs(ctk: str) -> bool:
        """Return True when *ctk* is empty, a non-string, or valid JSON."""
        if not ctk or not isinstance(ctk, str):
            return True
        try:
            json.loads(ctk)
        except TypeError, ValueError:
            return False
        return True

    def _profile_id_exists(self, profile_id: str) -> bool:
        from llama_manager.config.builder import create_tui_profile_registry

        registry = create_tui_profile_registry(self.config)
        return any(p.profile_id == profile_id for p in registry.profiles)

    def _payload_to_spec(self, profile_id: str, payload: SlotProfilePayload):
        from .components.slot_profile_modal import payload_to_slot_profile_spec

        try:
            return payload_to_slot_profile_spec(profile_id, payload)
        except ValueError as exc:
            self._push_status_message(str(exc))
            return None

    def _save_profile_with_status(self, save_fn, *, success_message: str | None) -> bool:
        try:
            save_fn()
        except ValueError as exc:
            self._push_status_message(str(exc))
            return False
        if success_message:
            self._push_status_message(success_message)
        return True

    def _handle_build_progress(self, progress: BuildProgress) -> None:
        """Handle build progress updates from pipeline.

        Args:
            progress: BuildProgress from the pipeline
        """
        self.build_progress = progress
        self.model.build_stage = progress.stage
        self.model.build_progress_percent = progress.progress_percent
        self.model.build_is_retrying = progress.is_retrying
        if progress.retries_remaining is not None:
            self.model.build_retries_remaining = progress.retries_remaining

        if self.build_in_progress:
            if progress.is_retrying:
                retry_message = f"Build retrying: {progress.message}"
                if progress.retries_remaining is not None:
                    retry_message += f" (retries remaining: {progress.retries_remaining})"
                self._push_status_message(retry_message)
            elif progress.status == "failed":
                self._push_status_message(f"Build failed: {progress.message}")
            elif progress.status == "success":
                self._push_status_message("Build completed successfully.")

        # Push to wizard modal if active
        wizard = self._build_wizard
        if wizard is not None:
            wizard.update_progress(progress)

    # -- Build lifecycle --------------------------------------------------

    def handle_build_selection(
        self, backends: list[str], options: dict[str, BuildConfig | None] | None = None
    ) -> None:
        """Initiate build for the given backends.

        Called from BuildModalScreen when the user presses 1/2/3.
        Starts a background thread so the TUI stays responsive.

        Args:
            backends: List of backend names to build.
            options: Optional build configuration options from the wizard.
        """
        if options is not None:
            self.model.build_selected_backends_options = options
        else:
            self.model.build_selected_backends_options = {}
        self._run_build_background(backends)

    def handle_build_with_wizard(
        self, backends: list[str], wizard: Any
    ) -> None:  # BuildModalScreen
        """Initiate build for the given backends, keeping the wizard modal open.

        The wizard modal stays open showing live progress. When the build
        completes (success or failure), the controller calls back to the
        wizard to display the result and dismiss.
        """
        self._run_build_background(backends, wizard=wizard)

    def cancel_build(self) -> None:
        """Signal cancellation and terminate any in-flight compile/configure subprocess."""
        cancel_event = getattr(self.model, "build_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()
        pipeline = getattr(self, "_build_pipeline", None)
        if pipeline is not None:
            pipeline.kill_active_subprocess()

    def _build_cancel_event_is_set(self) -> bool:
        cancel_evt = self.model.build_cancel_event
        return cancel_evt is not None and cancel_evt.is_set()

    def _abort_build_wizard(self, wizard: Any, message: str) -> None:
        self.model.build_error = message
        self._push_status_message(message)
        if wizard is not None:
            wizard.set_build_result(False, error_message=message)

    def _run_wizard_backend(self, backend: str, wizard: Any) -> bool:
        """Run one backend in the wizard thread. Returns False when cancelled or failed."""
        if self._build_cancel_event_is_set():
            self._abort_build_wizard(wizard, MSG_BUILD_CANCELLED)
            return False
        if wizard is not None:
            wizard.set_building_backend(backend)
        if self._build_single_backend(backend):
            return True
        self.model.build_result = "failed"
        if wizard is not None:
            wizard.set_build_result(
                False,
                error_message=self.model.build_error or MSG_BUILD_FAILED,
            )
        return False

    def _run_build_background(
        self, backends: list[str], wizard: Any = None
    ) -> None:  # BuildModalScreen | None
        """Run the build pipeline in a daemon thread.

        Args:
            backends: List of backends to build (e.g. ["sycl"] or ["sycl", "cuda"]).
            wizard: Optional wizard modal that stays open during the build.
        """
        self.model.build_in_progress = True
        self.build_in_progress = True
        self.model.build_selected_backends = backends
        self.model.build_cancel_event = threading.Event()
        self._build_wizard = wizard

        def _do_build() -> None:
            with suppress_build_pipeline_stderr_for_tui():
                self._execute_build_loop(backends, wizard)

        threading.Thread(target=_do_build, name="build-worker", daemon=True).start()

    def _execute_build_loop(self, backends: list[str], wizard: Any) -> None:
        """Execute the build loop for given backends. Handles success/failure states."""
        try:
            for backend in backends:
                if not self._build_all_targets_for_backend(backend, wizard):
                    return
            self.model.build_result = "success"
            self._push_status_message("Build completed successfully!")
            if wizard is not None:
                artifact_path = self.model.build_artifact
                wizard.set_build_result(True, artifact_path=artifact_path)
        except Exception as exc:
            self.model.build_result = "failed"
            self.model.build_error = str(exc)
            self._push_status_message(f"Build failed: {exc}")
            if wizard is not None:
                wizard.set_build_result(False, error_message=str(exc))
        finally:
            self.model.build_in_progress = False
            self.build_in_progress = False
            self._build_wizard = None

    def _build_all_targets_for_backend(self, backend: str, wizard: Any) -> bool:
        """Build all targets for a backend. Returns False to abort build loop."""
        targets = ("sycl", "cuda") if backend == "both" else (backend,)
        return all(self._run_wizard_backend(target, wizard) for target in targets)

    def _build_single_backend(self, backend: str) -> bool:
        """Build for a single backend; returns True on success."""
        try:
            self._push_status_message(f"Building for {backend} backend...")
            config_overrides = self.model.build_selected_backends_options.get(backend)
            result = run_build_for_backend(
                backend=backend,
                dry_run=False,
                config=self.config,
                progress_callback=self._handle_build_progress,
                pipeline_callback=lambda p: setattr(self, "_build_pipeline", p),
                config_overrides=config_overrides,
                cancel_event=self.model.build_cancel_event,
            )
            if not result.success:
                self.model.build_error = result.error_message or MSG_BUILD_FAILED
                self._push_status_message(f"Build failed: {result.error_message}")
                return False
            if result.artifact and result.artifact.binary_path:
                self.model.build_artifact = str(result.artifact.binary_path)
                self._push_status_message(f"Artifact: {result.artifact.binary_path}")
            return True
        except Exception as exc:
            self.model.build_error = str(exc)
            self._push_status_message(f"Build failed: {exc}")
            return False

    def _handle_launch_result(self, launch_result: LaunchResult | None) -> None:
        if launch_result is None:
            return
        if launch_result.is_blocked():
            self._push_status_message("launch blocked - no slots could be launched")
            if launch_result.errors is not None:
                for error_detail in launch_result.errors.errors:
                    self._push_status_message(
                        f"{error_detail.error_code}: {error_detail.why_blocked}"
                    )
            raise SystemExit(1)

        if launch_result.is_degraded():
            self._push_status_message("launch degraded - some slots blocked")
            for warning in launch_result.warnings or []:
                self._push_status_message(warning)

    def build_llama_cpp(self, backend: str = "sycl", dry_run: bool = False) -> bool:
        """Build llama.cpp using BuildPipeline.

        Args:
            backend: Build backend ("sycl" or "cuda")
            dry_run: If True, print commands without executing

        Returns:
            True if build successful, False otherwise
        """

        def _set_pipeline(pipeline: BuildPipeline) -> None:
            self._build_pipeline = pipeline
            self.build_in_progress = True

        # Capture original SIGINT handler before replacing it
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler_build)

        # Create a fresh cancel event for this build
        self.model.build_cancel_event = threading.Event()

        try:
            self._push_status_message(f"Building for {backend} backend...")
            if dry_run:
                self._push_status_message("DRY RUN MODE - commands will not be executed")

            with suppress_build_pipeline_stderr_for_tui():
                result = run_build_for_backend(
                    backend=backend,
                    dry_run=dry_run,
                    config=self.config,
                    progress_callback=self._handle_build_progress,
                    pipeline_callback=_set_pipeline,
                    cancel_event=self.model.build_cancel_event,
                )

            if result.success:
                self._push_status_message("Build completed successfully!")
                if result.artifact:
                    self._push_status_message(f"Artifact: {result.artifact.binary_path}")
                return True
            else:
                self._push_status_message(f"Build failed: {result.error_message}")
                return False
        finally:
            self.build_in_progress = False
            # Restore original SIGINT handler
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)

    def run(self, acknowledged: bool = False) -> None:
        from llama_cli.commands.profile import get_driver_version

        # Delegate launch orchestration to the pure library function
        result = launch_orchestrate(
            self.configs,
            self.config,
            self.server_manager,
            self.log_buffers,
            get_driver_version,
            acknowledged=acknowledged,
        )

        self.configs = result.updated_configs
        self.refresh_stale_warnings(get_driver_version)

        for msg in result.status_messages:
            self._push_status_message(msg)

        if result.empty:
            self._run_tui_loop_without_servers()
            return

        # Map risk result to Textual prompt state
        self._update_risk_panel_state(result.risk_result)
        if not acknowledged and result.risk_result is not None:
            for detail in result.risk_result.risk_details:
                self._push_status_message(
                    f"warning: risky operation in {detail['alias']}: {detail['risk']} — "
                    f"press 'y' to acknowledge, 'n' to abort"
                )

        self.launch_result = result.launch_result

        # CLI boundary: stderr printing and SystemExit for blocked launches
        if result.launch_result is not None:
            self._handle_launch_result(result.launch_result)

        self.server_processes = result.processes
        self.slot_states = result.slot_states

        try:
            DashboardApp(self).run()
        finally:
            self._cleanup()

    def _run_tui_loop_without_servers(self) -> None:
        """Run the TUI loop without any server processes.

        Used when no slots are configured - allows user to add slots interactively.
        """
        try:
            DashboardApp(self).run()
        finally:
            self._cleanup()

    def load_model_index(self) -> list[ModelIndexEntry]:
        """Load cached model index from disk."""
        with self._model_index_lock:
            if self._model_index_cache is not None:
                return list(self._model_index_cache)

        entries = load_model_index(self.config)
        with self._model_index_lock:
            self._model_index_cache = entries
        return list(entries)

    def refresh_model_index(
        self,
        progress_callback: Callable[[list[ModelIndexEntry], int, int, int], None] | None = None,
        *,
        progressive: bool = False,
    ) -> tuple[list[ModelIndexEntry], int, int]:
        """Refresh the model index by scanning config.models_dir."""
        entries, total, errors = refresh_model_index(
            self.config,
            progress_callback=progress_callback,
            progressive=progressive,
        )
        with self._model_index_lock:
            self._model_index_cache = entries
        return entries, total, errors

    def refresh_model_index_async(
        self,
        progress_callback: Callable[[list[ModelIndexEntry], int, int, int], None] | None = None,
        complete_callback: Callable[[list[ModelIndexEntry], int, int], None] | None = None,
    ) -> bool:
        """Refresh the model index in a background thread.

        Returns ``False`` when an index refresh is already running.
        """
        with self._model_index_lock:
            if self._model_index_refreshing:
                return False
            self._model_index_refreshing = True

        def _progress(
            entries: list[ModelIndexEntry],
            scanned: int,
            total: int,
            errors: int,
        ) -> None:
            with self._model_index_lock:
                self._model_index_cache = entries
            if progress_callback is not None:
                progress_callback(entries, scanned, total, errors)

        def _run_refresh() -> None:
            try:
                entries, total, errors = self.refresh_model_index(
                    progress_callback=_progress,
                    progressive=True,
                )
                if complete_callback is not None:
                    complete_callback(entries, total, errors)
            except Exception as exc:
                self._push_status_message(f"Model indexing failed: {exc}")
            finally:
                with self._model_index_lock:
                    self._model_index_refreshing = False

        threading.Thread(target=_run_refresh, name="model-index-worker", daemon=True).start()
        return True

    def is_model_index_refreshing(self) -> bool:
        """Return True while the background model index refresh is running."""
        with self._model_index_lock:
            return self._model_index_refreshing

    def model_index_path(self) -> str:
        """Return the path string where the model index cache is stored."""
        return str(model_index_path(self.config))
