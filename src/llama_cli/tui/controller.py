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
from llama_manager.config.reasoning_effort import (
    REASONING_EFFORT_JSON_CONFLICT,
    chat_template_kwargs_has_reasoning_effort,
)
from llama_manager.logging_setup import (
    suppress_build_pipeline_stderr_for_tui,
    update_file_level,
    update_stderr_level,
)
from llama_manager.slot_stats import (
    ProfileStatsAggregate,
    SlotStatsSnapshot,
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
    handlers (launch, build, slot management, config editing, risk prompts).

    Responsibilities:

    * **Lifecycle** — register signal handlers (SIGINT/SIGTERM), start/stop the
      TUI loop, and perform graceful shutdown of server processes.
    * **Launch orchestration** — delegate to :func:`~llama_manager.launch_orchestrate`
      and map results to UI state (risk prompts, slot states, status messages).
    * **Build pipeline** — run :func:`~llama_manager.build_pipeline.run_build_for_backend`
      via :meth:`begin_build` / :meth:`run_build_loop`, expose progress via
      :attr:`build_progress`, and coordinate with the build wizard modal. The
      thread is owned by ``DashboardApp``, not here.
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

        # Build pipeline state (build_in_progress lives on the model — single source of truth)
        self._build_pipeline: BuildPipeline | None = None
        self.build_progress: BuildProgress | None = None
        self._build_wizard: Any = None  # BuildModalScreen | None
        self._model_index_cache: list[ModelIndexEntry] | None = None
        self._model_index_lock = threading.Lock()
        self._model_index_refreshing = False

        if register_signals:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    @property
    def build_in_progress(self) -> bool:
        """Whether a build is currently running (delegates to the model)."""
        return self.model.build_in_progress

    @build_in_progress.setter
    def build_in_progress(self, value: bool) -> None:
        self.model.build_in_progress = value

    @property
    def config(self) -> Config:
        return self.model.config

    @property
    def configs(self) -> list[ServerConfig]:
        return self.model.configs

    @configs.setter
    def configs(self, value: list[ServerConfig]) -> None:
        self.model.configs = value

    def _load_persisted_slot_stats(self) -> None:
        """Load persisted slot stats from disk into the model cache."""
        try:
            self.model.apply_slot_stats_snapshot(load_slot_stats())
        except Exception:
            logger.debug("failed to load persisted slot stats", exc_info=True)

    def refresh_slot_stats(
        self,
        targets: tuple[tuple[str, Any, ServerConfig], ...] | None = None,
    ) -> None:
        """Collect live slot stats for all running configs and persist changes.

        *targets* is a frozen ``(alias, gpu, config)`` snapshot taken on the UI
        thread. Callers on a worker thread must pass one — iterating
        ``model.configs`` live races slot add/remove and can skip or double-visit
        slots mid-collection. Defaults to a fresh snapshot for UI-thread callers.
        """
        probe_targets = self.model.snapshot_for_probe() if targets is None else targets
        configs = [cfg for _alias, _gpu, cfg in probe_targets]
        current = self.model.slot_stats_snapshot()
        updated = dict(current)
        live_aliases = {cfg.alias for cfg in configs}
        changed = False
        for alias in list(updated):
            if alias not in live_aliases:
                del updated[alias]
                changed = True

        registry = self._build_tui_registry()
        profile_id_by_alias = {
            cfg.alias: self.resolve_profile_id_for_config(cfg, registry=registry) for cfg in configs
        }
        profile_stats = load_profile_stats()
        profile_stats_changed = False
        for cfg in configs:
            try:
                stats = collect_slot_stats(cfg.alias, self.model.config.deployment.host, cfg.port)
                if stats is None:
                    continue
                if updated.get(cfg.alias) != stats:
                    updated[cfg.alias] = stats
                    changed = True
                    profile_stats, stats_changed = self._record_profile_stats(
                        profile_stats, profile_id_by_alias, cfg.alias, stats
                    )
                    profile_stats_changed |= stats_changed
            except Exception:
                logger.exception("refresh_slot_stats: failed to collect for %s", cfg.alias)
        self._persist_slot_stats_if_changed(updated, live_aliases, changed)
        self._persist_profile_stats_if_changed(profile_stats, profile_stats_changed)

    def _record_profile_stats(
        self,
        profile_stats: dict[str, ProfileStatsAggregate],
        profile_id_by_alias: dict[str, str | None],
        alias: str,
        stats: SlotStatsSnapshot,
    ) -> tuple[dict[str, ProfileStatsAggregate], bool]:
        """Fold a slot's live stats into the profile aggregate.

        Returns ``(aggregate, changed)`` where *aggregate* is the (possibly new)
        profile-stats dict and *changed* is True when a new aggregate was produced.
        """
        profile_id = profile_id_by_alias.get(alias)
        if profile_id is None:
            return profile_stats, False
        updated = update_profile_stats(
            profile_stats, profile_id, self._profile_stats_session_id(alias), stats
        )
        return updated, True

    def _persist_slot_stats_if_changed(
        self,
        updated: dict[str, SlotStatsSnapshot],
        live_aliases: set[str],
        changed: bool,
    ) -> None:
        """Persist the slot-stats snapshot when it changed."""
        if not changed:
            return
        try:
            self.model.apply_slot_stats_snapshot(updated, live_aliases=live_aliases)
            save_slot_stats(updated)
        except Exception:
            logger.exception("refresh_slot_stats: failed to persist slot stats")

    def _persist_profile_stats_if_changed(
        self, profile_stats: dict[str, ProfileStatsAggregate], changed: bool
    ) -> None:
        """Persist the profile-stats aggregate when it changed."""
        if not changed:
            return
        try:
            save_profile_stats(profile_stats)
        except Exception:
            logger.exception("refresh_slot_stats: failed to persist profile stats")

    def resolve_profile_id_for_config(
        self, cfg: ServerConfig, *, registry: Any | None = None
    ) -> str | None:
        """Resolve a live server config alias to a registered profile ID."""
        resolved_registry = self._build_tui_registry() if registry is None else registry
        resolved = resolve_profile_id(resolved_registry, cfg.alias)
        if resolved is not None:
            return resolved
        if cfg.alias.endswith("-coding"):
            return resolve_profile_id(resolved_registry, cfg.alias.removesuffix("-coding"))
        return None

    def _profile_stats_session_id(self, alias: str) -> str:
        """Return a stable ID for the current server process behind *alias*."""
        process = self.model.server_processes.get(alias)
        pid = getattr(process, "pid", None)
        if isinstance(pid, int) and pid > 0:
            return f"{alias}:{pid}"
        return alias

    @property
    def running(self) -> bool:
        return self.model.running

    @running.setter
    def running(self, value: bool) -> None:
        self.model.running = value

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
    def server_manager(self) -> ServerManager:
        return self.model.server_manager

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals by stopping the TUI loop.

        If a build is in progress, release the build lock before stopping.
        """
        # Release build lock if in progress
        if self.build_in_progress and self._build_pipeline is not None:
            self._build_pipeline.release_lock()
            self.build_in_progress = False

        self.stop()

    def stop(self) -> None:
        """Stop the TUI loop gracefully."""
        self.running = False

    def _build_risk_panel_required(self, kind: Literal["vram", "hardware"] = "hardware") -> None:
        self.model.set_risk_prompt(kind=kind, acknowledged=False)

    def _build_risk_panel_acknowledged(
        self, kind: Literal["vram", "hardware"] = "hardware"
    ) -> None:
        self.model.set_risk_prompt(kind=kind, acknowledged=True)

    def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
        """Return status messages newer than ``since_ts``."""
        return self.model.get_status_messages_since(since_ts)

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        self.server_manager.cleanup_servers()

    def request_quit(self) -> bool:
        """Handle risk short-circuit on the UI thread; return True to dispatch shutdown.

        Does not run ``cleanup_servers`` — the app's ``_run_shutdown`` worker owns that.
        """
        if self.model.risk_prompt is not None:
            if self.active_risk_kind == "hardware":
                return self.handle_hardware_warning("q") == "quit"
            return False
        return True

    def interrupt(self) -> bool:
        """Request interrupt; return True when the app should dispatch shutdown.

        Does not run ``cleanup_servers`` — the app's ``_run_shutdown`` worker owns that.
        """
        return self.model.risk_prompt is None

    def refresh_display(self) -> None:
        """Request a refresh message from the UI."""
        if self.model.risk_prompt is not None:
            return
        self._push_status_message("Display refreshed.")

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

    def prepare_async_slot_launch(self, new_cfg: ServerConfig) -> AsyncSlotPlan:
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
            "log_buffers": self.model.log_buffers,
            "server_processes": self.model.server_processes,
            "slot_states": self.model.slot_states,
            "unsaved_slots": self.model.unsaved_slots,
            "slots": self.model.slots,
        }
        messages: list[str] = []

        if old_alias is None:
            self.configs.append(new_cfg)
            self.model.gpu_indices.append(gpu_index_for_config(new_cfg))
            new_gpu = GPUStats(
                gpu_index_for_config(new_cfg),
                collector=collector_for_config(new_cfg),
                selector=selector_for_config(new_cfg),
            )
            self.model.gpu_stats.append(new_gpu)
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
            self.model.gpu_indices[existing_index] = gpu_idx
            new_gpu = GPUStats(
                gpu_idx,
                collector=collector_for_config(new_cfg),
                selector=selector_for_config(new_cfg),
            )
            self.model.gpu_stats[existing_index] = new_gpu

        log_buffer = LogBuffer(redact_sensitive=True)
        self.model.log_buffers[alias] = log_buffer
        self.model.set_cached_gpu_stats(alias, new_gpu.get_cached_stats_snapshot())
        self.model.unsaved_slots.add(alias)
        self.model.slots.append(
            ModelSlot(slot_id=alias, model_path=new_cfg.model, port=new_cfg.port)
        )
        self.model.slot_states[alias] = SlotState.LAUNCHING.value
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
            self.model.slot_states[alias] = SlotState.CRASHED.value
            messages.append(f"Slot '{alias}' failed to start: no process returned")
            for msg in messages:
                self._push_status_message(msg)
            return False, messages

        self.model.server_processes[alias] = process
        old_state = self.model.slot_states.get(alias)
        self.model.slot_states[alias] = SlotState.RUNNING.value
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

        if len(self.configs) != len(self.model.gpu_indices) or len(self.configs) != len(
            self.model.gpu_stats
        ):
            raise RuntimeError("slot runtime lists must remain length-synchronized")

        del self.configs[existing_index]
        del self.model.gpu_indices[existing_index]
        del self.model.gpu_stats[existing_index]
        state = {
            "log_buffers": self.model.log_buffers,
            "server_processes": self.model.server_processes,
            "slot_states": self.model.slot_states,
            "unsaved_slots": self.model.unsaved_slots,
            "slots": self.model.slots,
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

    def cancel_add_slot_form(self) -> None:
        """Emit a status message when the add-slot modal is cancelled."""
        self._push_status_message("Slot configuration cancelled")

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
            # Clear risk only; blocking cleanup runs on the slot-ops shutdown worker.
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

    def _graceful_shutdown(self) -> None:
        """Initiate graceful shutdown of all server processes."""
        if not self.running:
            return

        self._push_status_message("Shutting down...")
        self.server_manager.cleanup_servers()
        self.running = False

    def save_config(self, payload: ConfigPayload) -> bool:
        """Persist edited config values; return True if servers should stop off-thread.

        Args:
            payload: Typed config values and restart flag from the modal.

        Returns:
            True when ``payload.restart`` is set and the app should dispatch the
            slot-ops shutdown worker for ``cleanup_servers`` + ``running = False``.
        """
        from llama_manager import apply_config_updates

        result = apply_config_updates(self.model.config, payload.to_config_updates())

        if result.errors:
            for error in result.errors:
                self._push_status_message(error)
            return False

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
            return True
        return False

    def clean_model_cache(self) -> tuple[bool, str]:
        """Delete the model index cache file and clear in-memory cache.

        Returns:
            A tuple of (success, message).
        """
        idx_path = model_index_path()

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
        from llama_manager.slot_profile_store import load_custom_slot_profiles

        registry = self._build_tui_registry()
        custom_ids = {p.profile_id for p in load_custom_slot_profiles()}
        result: list[tuple[Any, str]] = []
        for p in registry.profiles:
            source = "custom" if p.profile_id in custom_ids else "builtin"
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
            if chat_template_kwargs_has_reasoning_effort(payload.chat_template_kwargs):
                self._push_status_message(REASONING_EFFORT_JSON_CONFLICT)
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
        """Return True when *ctk* is empty, a non-string, or valid JSON without reasoning_effort."""
        if not ctk or not isinstance(ctk, str):
            return True
        try:
            json.loads(ctk)
        except TypeError, ValueError:
            return False
        return not chat_template_kwargs_has_reasoning_effort(ctk)

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
        # Single immutable snapshot — derived fields stay coherent for readers.
        self.build_progress = progress
        self.model.build_progress = progress

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

    def begin_build(
        self,
        options: dict[str, BuildConfig | None] | None = None,
        wizard: Any = None,  # BuildModalScreen | None
    ) -> None:
        """Reserve build state on the UI thread before a worker runs the pipeline.

        Pairs with :meth:`run_build_loop`. Split so the caller owns the thread —
        see ``DashboardApp.start_build``.

        Args:
            options: Optional build configuration options from the wizard.
            wizard: Optional wizard modal that stays open during the build.
        """
        self.model.build_selected_backends_options = options if options is not None else {}
        self.model.build_in_progress = True
        self.model.build_cancel_event = threading.Event()
        self._build_wizard = wizard

    def cancel_build(self) -> None:
        """Signal cancellation; the cancel watcher terminates the process tree off-thread."""
        cancel_event = getattr(self.model, "build_cancel_event", None)
        if cancel_event is not None:
            cancel_event.set()

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
        if wizard is not None:
            wizard.set_build_result(
                False,
                error_message=self.model.build_error or MSG_BUILD_FAILED,
            )
        return False

    def run_build_loop(self, backends: list[str], wizard: Any = None) -> None:
        """Run the build pipeline. Blocks — call from a worker thread only.

        Pairs with :meth:`begin_build`, which must have run on the UI thread first.

        Args:
            backends: List of backends to build (e.g. ["sycl"] or ["sycl", "cuda"]).
            wizard: Optional wizard modal that stays open during the build.
        """
        with suppress_build_pipeline_stderr_for_tui():
            self._execute_build_loop(backends, wizard)

    def _execute_build_loop(self, backends: list[str], wizard: Any) -> None:
        """Execute the build loop for given backends. Handles success/failure states."""
        try:
            for backend in backends:
                if not self._build_all_targets_for_backend(backend, wizard):
                    return
            self._push_status_message("Build completed successfully!")
            if wizard is not None:
                artifact_path = self.model.build_artifact
                wizard.set_build_result(True, artifact_path=artifact_path)
        except Exception as exc:
            self.model.build_error = str(exc)
            self._push_status_message(f"Build failed: {exc}")
            if wizard is not None:
                wizard.set_build_result(False, error_message=str(exc))
        finally:
            self.model.build_in_progress = False
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

    def run(self, acknowledged: bool = False) -> None:
        from llama_cli.commands.profile import get_driver_version

        # Delegate launch orchestration to the pure library function
        result = launch_orchestrate(
            self.configs,
            self.config,
            self.server_manager,
            self.model.log_buffers,
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

        self.model.launch_result = result.launch_result

        # CLI boundary: stderr printing and SystemExit for blocked launches
        if result.launch_result is not None:
            self._handle_launch_result(result.launch_result)

        self.model.server_processes = result.processes
        self.model.slot_states = result.slot_states

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
                return self._model_index_cache

        entries = load_model_index()
        with self._model_index_lock:
            self._model_index_cache = entries
        return entries

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

        Deliberately a daemon thread rather than a Textual worker, unlike the build
        (see ``DashboardApp.start_build``): shutdown joins workers, so a full rescan
        of a large or network-mounted models dir would stall quit for as long as the
        scan takes. Killing a scan mid-flight costs nothing — the cache is written
        atomically and the scan is idempotent, so there is no ``finally`` worth
        waiting for.

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
