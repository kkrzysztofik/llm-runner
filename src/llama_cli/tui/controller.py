"""TUIApp controller — manages server lifecycle, state, and rendering."""

import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from llama_cli.gpu_collectors import collect_nvtop_stats
from llama_manager import (
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    ProfileFlavor,
    ServerConfig,
    ServerManager,
    SlotState,
    get_gpu_identifier,
    load_profile_with_staleness,
    profile_to_override_dict,
)
from llama_manager.build_pipeline import (
    BuildBackend,
    BuildConfig,
    BuildPipeline,
    BuildProgress,
)
from llama_manager.config import (
    create_default_profile_registry,
    merge_config_overrides,
    resolve_profile_config,
)
from llama_manager.server import detect_risky_operations

from .components.alerts import (
    build_profile_status_panel,
    build_risk_panel_acknowledged,
    build_risk_panel_required,
    build_status_messages_panel,
    build_status_panel,
    build_system_status_panel,
)
from .components.menu import build_command_menu
from .components.panels import (
    build_column_panel,
    build_placeholder_panel,
)
from .constants import RISK_ACK_LABEL
from .textual_app import TextualDashboardApp
from .types import DashboardSnapshot


class TUIApp:
    """Main TUI application with 2-column layout."""

    def __init__(
        self,
        configs: list[ServerConfig],
        gpu_indices: list[int],
        slots: list[ModelSlot] | None = None,
    ):
        self.config = Config()
        self.configs = configs
        self.gpu_indices = gpu_indices
        self.slots = slots or []
        self.log_buffers: dict[str, LogBuffer] = {}
        self.gpu_stats: list[GPUStats] = []
        self.running = True
        self.launch_result: LaunchResult | None = None
        self.status_panel: Panel | None = None
        self.risk_panel: Panel | None = None
        self.risks_acknowledged: bool = False
        self.active_risk_kind: str | None = None

        # Profile state
        self._profile_status: dict[str, str] = {}  # alias -> "idle" | "running" | "done" | "failed"
        self._profile_flavor: dict[str, str] = {}  # alias -> flavor string
        self._profile_cancel_events: dict[str, threading.Event] = {}
        self._profile_lock = threading.Lock()

        # TUI-safe status message buffer. Each entry is (timestamp, message).
        self._status_messages: list[tuple[float, str]] = []
        self._status_lock = threading.Lock()

        # Non-blocking profile flavor request state
        self.profile_request: str | None = None
        self._build_request: bool = False
        self._smoke_request: bool = False
        self.unsaved_slots: set[str] = set()

        self.server_manager = ServerManager()

        # Slot state tracking for TUI dashboard
        self.slot_states: dict[str, str] = {}  # alias -> SlotState value
        self.server_processes: dict[str, Any] = {}  # alias -> subprocess.Popen

        for cfg in configs:
            self.log_buffers[cfg.alias] = LogBuffer(redact_sensitive=True)
        for idx in gpu_indices:
            # Pass a bound collector callable with the device index
            self.gpu_stats.append(GPUStats(idx, collector=self._make_collector(idx)))

        # Build pipeline state
        self._build_pipeline: BuildPipeline | None = None
        self.build_in_progress = False
        self.build_progress: BuildProgress | None = None
        self._original_sigint_handler: Callable[[int, FrameType | None], Any] | int | None = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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
            print("\nBuild interrupted by user, releasing lock...", file=sys.stderr)
            self._build_pipeline.release_lock()
            self.build_in_progress = False
            sys.exit(130)  # Standard exit code for Ctrl+C

    def _make_collector(self, device_index: int) -> Callable[[], dict[str, Any]]:
        """Create a GPU collector bound to a specific device index."""

        def collector() -> dict[str, Any]:
            return collect_nvtop_stats(device_index)

        return collector

    def stop(self) -> None:
        """Stop the TUI loop gracefully."""
        self.running = False

    def render(self) -> DashboardSnapshot:
        gpu_lines: list[str] = []
        for gpu in self.gpu_stats:
            gpu.update()
            gpu_lines.append(gpu.format_stats_text())

        alerts_panel = build_system_status_panel(
            gpu_lines=gpu_lines,
            notices=self.build_system_notices(),
        )

        left_panel: Panel | None = None
        if self.configs:
            cfg1 = self.configs[0]
            buffer1 = self.log_buffers[cfg1.alias]
            gpu1 = self.gpu_stats[0] if self.gpu_stats else None
            left_panel = self._build_column_panel(cfg1, buffer1, gpu1)

        if len(self.configs) > 1:
            cfg2 = self.configs[1]
            buffer2 = self.log_buffers[cfg2.alias]
            gpu2 = self.gpu_stats[1] if len(self.gpu_stats) > 1 else None
            right_panel = self._build_column_panel(cfg2, buffer2, gpu2)
        else:
            right_panel = self._build_placeholder_panel()

        return DashboardSnapshot(
            alerts=alerts_panel,
            left=left_panel,
            right=right_panel,
            menu=self._build_command_menu(),
        )

    # ------------------------------------------------------------------
    # Rendering delegation helpers
    # ------------------------------------------------------------------

    def _build_status_panel(self, launch_result: LaunchResult) -> None:
        self.status_panel = build_status_panel(launch_result)

    def _build_risk_panel_required(self, kind: str = "hardware") -> None:
        self.risk_panel = build_risk_panel_required(kind=kind)
        self.risks_acknowledged = False
        self.active_risk_kind = kind

    def _build_risk_panel_acknowledged(self, kind: str = "hardware") -> None:
        self.risk_panel = build_risk_panel_acknowledged()
        self.risks_acknowledged = True
        self.active_risk_kind = kind

    def _build_column_panel(
        self, cfg: ServerConfig, buffer: LogBuffer, gpu: GPUStats | None
    ) -> Panel:
        stale_warning = self.get_stale_warning(cfg)
        return build_column_panel(
            cfg,
            buffer,
            gpu,
            self.config.host,
            stale_warning,
            slot_states=self.slot_states,
            server_processes=self.server_processes,
            is_unsaved=cfg.alias in self.unsaved_slots,
        )

    def _build_placeholder_panel(self) -> Panel:
        return build_placeholder_panel()

    def _build_profile_status_panel(self) -> Panel | None:
        with self._profile_lock:
            active = {a: s for a, s in self._profile_status.items() if s != "idle"}
        return build_profile_status_panel(active, self._profile_flavor)

    def build_system_notices(self) -> list[str]:
        """Build concise status notices shown in the top system panel."""
        notices: list[str] = []

        if self.status_panel is not None and self.launch_result is not None:
            if self.launch_result.is_blocked():
                notices.append("Launch blocked: no slots could be launched")
            elif self.launch_result.is_degraded():
                notices.append("Launch degraded: some slots blocked")

        if self.build_in_progress and self.build_progress is not None:
            notices.append(
                f"Build {self.build_progress.stage}: {self.build_progress.status} "
                f"({self.build_progress.progress_percent}%)"
            )

        if self.risk_panel is not None:
            if self.active_risk_kind == "vram":
                notices.append("VRAM risk acknowledgement required [y/n]")
            elif self.risks_acknowledged:
                notices.append("Risky operation acknowledged")
            else:
                notices.append("Hardware risk acknowledgement required [y/n]")

        with self._profile_lock:
            running_profiles = [a for a, s in self._profile_status.items() if s == "running"]
        if running_profiles:
            notices.append(f"Profiling running: {', '.join(running_profiles)}")

        return notices

    # How long (seconds) a status message remains visible across renders.
    _STATUS_MESSAGE_LIFETIME_S: float = 30.0

    def _build_status_messages_panel(self) -> RenderableType | None:
        with self._status_lock:
            if not self._status_messages:
                return None
            now = time.monotonic()
            messages = [
                msg
                for ts, msg in self._status_messages
                if now - ts < self._STATUS_MESSAGE_LIFETIME_S
            ]
        if not messages:
            return None
        return build_status_messages_panel(messages)

    def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
        """Return status messages newer than ``since_ts``."""
        with self._status_lock:
            return [(ts, msg) for ts, msg in self._status_messages if ts > since_ts]

    def prune_expired_status_messages(self) -> None:
        """Remove status messages older than ``_STATUS_MESSAGE_LIFETIME_S``."""
        cutoff = time.monotonic() - self._STATUS_MESSAGE_LIFETIME_S
        with self._status_lock:
            self._status_messages = [(ts, msg) for ts, msg in self._status_messages if ts >= cutoff]

    def _build_command_menu(self) -> Text:
        return build_command_menu(
            self.profile_request,
            self.risk_panel,
            self.active_risk_kind,
        )

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def _print_acknowledgement_required_and_exit(self) -> None:
        print("error: acknowledgement_required", file=sys.stderr)
        print("  failed_check: acknowledgement_required", file=sys.stderr)
        print(
            "  why_blocked: risky operation detected and not acknowledged",
            file=sys.stderr,
        )
        print(
            "  how_to_fix: use --acknowledge-risky flag or confirm with 'y'",
            file=sys.stderr,
        )
        raise SystemExit(1)

    def _cleanup(self) -> None:
        self.server_manager.cleanup_servers()

    def request_quit(self) -> None:
        """Request a graceful shutdown from the UI."""
        if self.risk_panel is not None:
            if self.active_risk_kind == "hardware":
                self.handle_hardware_warning("q")
            return
        self._graceful_shutdown()

    def interrupt(self) -> None:
        """Request an interrupt from the UI.

        Matches the legacy Ctrl+C path: abort a running profile if one exists,
        otherwise shut the app down.
        """
        if self.risk_panel is not None:
            return

        with self._profile_lock:
            for cfg in self.configs:
                if self._profile_status.get(cfg.alias) == "running":
                    cancel_event = self._profile_cancel_events.get(cfg.alias)
                    if cancel_event is not None:
                        cancel_event.set()
                    self._profile_status[cfg.alias] = "failed"
                    self._push_status_message(f"Profile '{cfg.alias}' aborted.")
                    return

        self._graceful_shutdown()

    def refresh_display(self) -> None:
        """Request a refresh message from the UI."""
        if self.risk_panel is not None:
            return
        self._push_status_message("Display refreshed.")

    def request_profile(self) -> None:
        """Start the profile selection flow from the UI."""
        if self.risk_panel is not None or not self.configs:
            return
        alias = self.configs[0].alias
        with self._profile_lock:
            self._profile_status[alias] = "idle"
        self.profile_request = alias

    def request_build(self) -> None:
        """Start the build selection flow from the UI."""
        if self.risk_panel is not None:
            return
        self._build_request = True
        self._push_status_message("Select build target: [1] SYCL  [2] CUDA  [3] Both")

    def request_smoke(self) -> None:
        """Start the smoke selection flow from the UI."""
        if self.risk_panel is not None:
            return
        self._smoke_request = True
        self._push_status_message("Select smoke scope: [1] Both  [2] Active Slot")

    def cancel_pending_prompt(self) -> bool:
        """Cancel any pending profile, build, or smoke prompt."""
        if self.profile_request is not None:
            self.profile_request = None
            self._push_status_message("Profile selection cancelled.")
            return True

        if self._build_request:
            self._build_request = False
            self._push_status_message("Build selection cancelled.")
            return True

        if self._smoke_request:
            self._smoke_request = False
            self._push_status_message("Smoke selection cancelled.")
            return True

        return False

    def select_pending_option(self, key: str) -> bool:
        """Apply a numeric choice to the current pending prompt."""
        if self.risk_panel is not None:
            return False

        if self.profile_request is not None and key in {"1", "2", "3"}:
            alias = self.profile_request
            self.profile_request = None
            flavor_map = {"1": "balanced", "2": "fast", "3": "quality"}
            self._run_profile_background(alias, flavor_map[key])
            return True

        if self._build_request and key in {"1", "2", "3"}:
            self._build_request = False
            target = {"1": "sycl", "2": "cuda", "3": "both"}.get(key, "both")
            self._push_status_message(f"Feature not fully hooked up in TUI yet: Build {target}")
            return True

        if self._smoke_request and key in {"1", "2"}:
            self._smoke_request = False
            target = "both" if key == "1" else "slot"
            self._push_status_message(f"Feature not fully hooked up in TUI yet: Smoke {target}")
            return True

        return False

    def acknowledge_risk(self) -> None:
        """Acknowledge the active risk prompt."""
        if self.risk_panel is None:
            return
        if self.active_risk_kind == "vram":
            self.handle_vram_risk("y")
        else:
            self.handle_hardware_warning("y")

    def reject_risk(self) -> None:
        """Reject the active risk prompt."""
        if self.risk_panel is None:
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
        with self._status_lock:
            self._status_messages.append((time.monotonic(), message))
            # Keep at most 5 messages
            if len(self._status_messages) > 5:
                self._status_messages.pop(0)

    def add_slot_from_form(self, values: dict[str, str]) -> bool:
        """Create or replace a slot from modal profile selection."""
        profile_id = values.get("profile", "").strip()
        if not profile_id:
            self._push_status_message("Profile is required")
            return False

        registry = create_default_profile_registry(self.config)

        override_config: dict[str, int] | None = None
        port_value = values.get("port", "").strip()
        if port_value:
            normalized_port = {"port": port_value}
            self._normalize_slot_port(normalized_port)
            override_config = {"port": int(normalized_port["port"])}

        try:
            new_cfg = resolve_profile_config(registry, profile_id, override_config=override_config)
        except ValueError:
            allowed = ", ".join(registry.profile_ids)
            self._push_status_message(f"Unknown profile '{profile_id}'. Choose one of: {allowed}")
            return False

        return self._upsert_profile_slot(new_cfg, profile_id)

    def cancel_add_slot_form(self) -> None:
        """Emit a status message when the add-slot modal is cancelled."""
        self._push_status_message("Slot configuration cancelled")

    def _device_class_for_config(self, cfg: ServerConfig) -> str:
        """Return normalized device class name used for replacement logic."""
        return "sycl" if cfg.device.upper().startswith("SYCL") else "cuda"

    def _gpu_index_for_config(self, cfg: ServerConfig) -> int:
        """Return dashboard GPU index for config."""
        return 1 if self._device_class_for_config(cfg) == "sycl" else 0

    def _remove_slot_runtime_state(self, alias: str) -> None:
        """Remove runtime state for one slot alias."""
        self.log_buffers.pop(alias, None)
        self.server_processes.pop(alias, None)
        self.slot_states.pop(alias, None)
        self.unsaved_slots.discard(alias)
        self.slots = [slot for slot in self.slots if slot.slot_id != alias]

    def _register_and_start_slot(self, cfg: ServerConfig) -> None:
        """Register and start one slot using the current config."""
        alias = cfg.alias
        self.log_buffers[alias] = LogBuffer(redact_sensitive=True)
        self.unsaved_slots.add(alias)
        self.slots.append(ModelSlot(slot_id=alias, model_path=cfg.model, port=cfg.port))

        log_handler = lambda line, buf=self.log_buffers[alias]: buf.add_line(line)  # noqa: E731
        procs = self.server_manager.start_servers([cfg], {alias: log_handler})
        if procs:
            self.server_processes[alias] = procs[0]
        self.handle_slot_transition(alias, SlotState.RUNNING)

    def _upsert_profile_slot(self, cfg: ServerConfig, profile_id: str) -> bool:
        """Add profile slot or replace existing slot on same device."""
        target_device = self._device_class_for_config(cfg)
        existing_index = next(
            (
                idx
                for idx, existing_cfg in enumerate(self.configs)
                if self._device_class_for_config(existing_cfg) == target_device
            ),
            None,
        )

        if existing_index is None:
            self.configs.append(cfg)
            gpu_idx = self._gpu_index_for_config(cfg)
            self.gpu_indices.append(gpu_idx)
            self.gpu_stats.append(GPUStats(gpu_idx, collector=self._make_collector(gpu_idx)))
            self._register_and_start_slot(cfg)
            self._push_status_message(
                f"Added profile '{profile_id}' as '{cfg.alias}' on {target_device}:{cfg.port}"
            )
            return True

        old_cfg = self.configs[existing_index]
        old_alias = old_cfg.alias
        if not self.server_manager.shutdown_slot(old_alias):
            self._push_status_message(
                f"Unable to replace '{old_alias}' on {target_device}: shutdown verification failed"
            )
            return False

        self._remove_slot_runtime_state(old_alias)
        self.configs[existing_index] = cfg
        gpu_idx = self._gpu_index_for_config(cfg)
        self.gpu_indices[existing_index] = gpu_idx
        self.gpu_stats[existing_index] = GPUStats(gpu_idx, collector=self._make_collector(gpu_idx))

        self._register_and_start_slot(cfg)
        self._push_status_message(
            f"Replaced '{old_alias}' with profile '{profile_id}' as '{cfg.alias}' on {target_device}:{cfg.port}"
        )
        return True

    def _normalize_slot_port(self, values: dict[str, str]) -> None:
        """Validate and normalise the port field, falling back to 8080."""
        try:
            port = int(values.get("port", ""))
            if port < 1024 or port > 65535:
                self._push_status_message(f"Invalid port {port}, using 8080")
                values["port"] = "8080"
        except ValueError:
            self._push_status_message("Invalid port, using 8080")
            values["port"] = "8080"

    def _run_profile_background(self, alias: str, flavor: str) -> None:
        """Run profiling in a background thread so the TUI stays responsive."""
        with self._profile_lock:
            self._profile_status[alias] = "running"
            self._profile_flavor[alias] = flavor
            self._profile_cancel_events[alias] = threading.Event()

        def _do_profile() -> None:
            try:
                exit_code = self._execute_profile(alias, flavor)
                if exit_code == 0:
                    with self._profile_lock:
                        self._profile_status[alias] = "done"
                else:
                    with self._profile_lock:
                        self._profile_status[alias] = "failed"
            except Exception:
                with self._profile_lock:
                    self._profile_status[alias] = "failed"
            finally:
                with self._profile_lock:
                    self._profile_cancel_events.pop(alias, None)

        threading.Thread(
            target=_do_profile,
            name=f"profile-{alias}",
            daemon=True,
        ).start()

    def _execute_profile(self, alias: str, flavor: str) -> int:
        """Execute the profile benchmark for a given slot.

        Uses ``profile_cli.cmd_profile`` directly (no subprocess).
        Returns 0 on success, 1 on failure.
        """
        from llama_cli.commands.profile import cmd_profile

        with self._profile_lock:
            cancel_event = self._profile_cancel_events.get(alias)

        if cancel_event is None:
            return 1

        return cmd_profile(
            slot_id=alias,
            flavor=flavor,
            quiet=True,
            progress_callback=self._push_status_message,
            cancel_event=cancel_event,
        )

    def _abort_profile(self) -> None:
        """Abort the currently running profile for any slot."""
        for cfg in self.configs:
            with self._profile_lock:
                if self._profile_status.get(cfg.alias) == "running":
                    cancel_event = self._profile_cancel_events.get(cfg.alias)
                    if cancel_event is not None:
                        cancel_event.set()
                    self._profile_status[cfg.alias] = "failed"
                    self._push_status_message(f"Profile '{cfg.alias}' aborted.")

    # ------------------------------------------------------------------
    # Profile staleness helpers
    # ------------------------------------------------------------------

    def get_stale_warning(self, cfg: ServerConfig) -> str | None:
        """Check if the cached profile for a config is stale.

        Returns a warning string or None if the profile is fresh / nonexistent.
        """
        try:
            from llama_cli.commands.profile import _get_driver_version

            _record, staleness = load_profile_with_staleness(
                profiles_dir=self.config.profiles_dir,
                gpu_identifier=get_gpu_identifier(cfg.backend),
                backend=cfg.backend,
                flavor=ProfileFlavor.BALANCED,
                current_driver_version=_get_driver_version(cfg.backend),
                current_binary_version=self.config.server_binary_version or "unknown",
                staleness_days=self.config.profile_staleness_days,
            )
        except Exception:
            return None

        if staleness is None or not staleness.is_stale:
            return None

        reasons = "; ".join(r.value.replace("_", " ").title() for r in staleness.reasons)
        return f"\u26a0 profile stale \u2014 {reasons}"

    def _acknowledge_risks(
        self, launch_attempt_id: str, ack_token: str, acknowledged: bool
    ) -> bool:
        has_risks = False
        for cfg in self.configs:
            if acknowledged and RISK_ACK_LABEL not in cfg.risky_acknowledged:
                cfg.risky_acknowledged.append(RISK_ACK_LABEL)
            for risk in detect_risky_operations(cfg):
                has_risks = True
                self._acknowledge_single_risk(
                    cfg,
                    risk,
                    launch_attempt_id,
                    ack_token,
                    acknowledged,
                )
        if has_risks and acknowledged:
            self.risks_acknowledged = True
        return has_risks

    def _acknowledge_single_risk(
        self,
        cfg: ServerConfig,
        risk: str,
        launch_attempt_id: str,
        ack_token: str,
        acknowledged: bool,
    ) -> None:
        if self.server_manager.is_risk_acknowledged(cfg.alias, risk, launch_attempt_id):
            return

        risk_kind = "vram" if "vram" in risk.lower() else "hardware"

        if not acknowledged:
            self._build_risk_panel_required(risk_kind)
            self._push_status_message(
                f"warning: risky operation in {cfg.alias}: {risk} — "
                f"press 'y' to acknowledge, 'n' to abort"
            )

        self.server_manager.acknowledge_risk(
            cfg.alias,
            risk,
            launch_attempt_id=launch_attempt_id,
            ack_token=ack_token,
        )

    def _update_risk_panel_state(self, has_risks: bool) -> None:
        if has_risks:
            if self.risks_acknowledged:
                self._build_risk_panel_acknowledged()
            else:
                self._build_risk_panel_required()
            return
        self.risk_panel = None
        self.risks_acknowledged = False

    def handle_hardware_warning(self, key: str) -> str:
        """Handle hardware mismatch warning key press.

        Args:
            key: The key pressed by the user.

        Returns:
            'acknowledge' if user acknowledged, 'abort' if rejected,
            'quit' if user pressed q, or 'ignore' for unknown keys.

        """
        if key == "y":
            # User acknowledged the hardware warning
            self.risk_panel = None
            self.active_risk_kind = None
            return "acknowledge"
        if key == "n":
            # User rejected — abort
            self.running = False
            self.active_risk_kind = None
            return "abort"
        if key == "q":
            # User wants to quit
            self._graceful_shutdown()
            self.active_risk_kind = None
            return "quit"
        return "ignore"

    def handle_vram_risk(self, key: str) -> str:
        """Handle VRAM risk confirmation key press.

        Args:
            key: The key pressed by the user.

        Returns:
            'proceed' if user confirmed, 'abort' if rejected,
            or 'ignore' for unknown keys.

        """
        if key == "y":
            # User confirmed the VRAM risk
            self.risk_panel = None
            self.active_risk_kind = None
            return "proceed"
        if key == "n":
            # User rejected — abort
            self.running = False
            self.active_risk_kind = None
            return "abort"
        return "ignore"

    def handle_slot_transition(self, slot_id: str, new_state: SlotState) -> None:
        """Handle a slot state transition and update the UI.

        Args:
            slot_id: The slot identifier.
            new_state: The new state for the slot.
        """
        old_state = self.slot_states.get(slot_id)
        self.slot_states[slot_id] = new_state.value

        # Handle specific transitions
        if old_state is None and new_state == SlotState.RUNNING:
            # First launch - clear any previous status panels
            self.status_panel = None
            self._push_status_message(f"Slot '{slot_id}' launched successfully.")
            return

        transition_messages: dict[tuple[str, str], tuple[str, str]] = {
            (SlotState.LAUNCHING.value, SlotState.RUNNING.value): (
                "Launched",
                "green",
            ),
            (SlotState.RUNNING.value, SlotState.DEGRADED.value): (
                "Degraded",
                "yellow",
            ),
            (SlotState.RUNNING.value, SlotState.CRASHED.value): (
                "Crashed",
                "red",
            ),
            (SlotState.DEGRADED.value, SlotState.OFFLINE.value): (
                "Offline",
                "yellow",
            ),
            (SlotState.CRASHED.value, SlotState.OFFLINE.value): (
                "Offline",
                "red",
            ),
            (SlotState.OFFLINE.value, SlotState.IDLE.value): (
                "Idle",
                "dim",
            ),
        }

        if old_state is not None:
            key = (old_state, new_state.value)
            if key in transition_messages:
                label, color = transition_messages[key]
                msg = f"Slot '{slot_id}': {label} ({color})"
                self._push_status_message(msg)

    def _graceful_shutdown(self) -> None:
        """Initiate graceful shutdown of all server processes."""
        if not self.running:
            return

        self._push_status_message("Shutting down...")
        self.server_manager.cleanup_servers()
        self.running = False

    def _handle_build_progress(self, progress: BuildProgress) -> None:
        """Handle build progress updates from pipeline.

        Args:
            progress: BuildProgress from the pipeline
        """
        self.build_progress = progress

        # Update status panel if build is in progress
        if self.build_in_progress:
            if progress.is_retrying:
                status_text = Text()
                status_text.append("STATUS: ", style="bold yellow")
                status_text.append("RETRYING", style="bold yellow")
                status_text.append(f" - {progress.message}\n", style="dim")
                if progress.retries_remaining is not None:
                    status_text.append(
                        f"Retries remaining: {progress.retries_remaining}",
                        style="dim",
                    )
                self.status_panel = Panel(
                    status_text,
                    title="[yellow]Build In Progress[/yellow]",
                    border_style="yellow",
                )
            elif progress.status == "failed":
                status_text = Text()
                status_text.append("STATUS: ", style="bold red")
                status_text.append("FAILED", style="bold red")
                status_text.append(f" - {progress.message}\n", style="dim")
                self.status_panel = Panel(
                    status_text,
                    title="[red]Build Failed[/red]",
                    border_style="red",
                )
            elif progress.status == "success":
                # Clear status panel on success
                self.status_panel = None

    def _handle_launch_result(self, launch_result: LaunchResult) -> None:
        if launch_result.is_blocked():
            print("error: launch blocked - no slots could be launched", file=sys.stderr)
            if launch_result.errors is not None:
                for error_detail in launch_result.errors.errors:
                    print(f"  {error_detail.error_code}", file=sys.stderr)
                    print(f"    failed_check: {error_detail.failed_check}", file=sys.stderr)
                    print(f"    why_blocked: {error_detail.why_blocked}", file=sys.stderr)
                    print(f"    how_to_fix: {error_detail.how_to_fix}", file=sys.stderr)
            raise SystemExit(1)

        if launch_result.is_degraded():
            print("warning: launch degraded - some slots blocked", file=sys.stderr)
            for warning in launch_result.warnings or []:
                print(f"  warning: {warning}", file=sys.stderr)

    def _apply_profile_overrides(self) -> list[ServerConfig]:
        """Apply cached profile overrides to configs at startup.

        For each config, attempts to load a cached profile. If found and fresh,
        applies the whitelisted profile parameters (threads, ctx_size, ubatch_size,
        cache types) via merge_config_overrides. Reports status via TUI messages.

        Returns:
            Updated configs list with profile overrides applied where applicable.
        """
        from llama_cli.commands.profile import _get_driver_version

        updated_configs: list[ServerConfig] = []

        for cfg in self.configs:
            try:
                gpu_identifier = get_gpu_identifier(cfg.backend)
                driver_version = _get_driver_version(cfg.backend)
                binary_version = self.config.server_binary_version or "unknown"

                record, staleness = load_profile_with_staleness(
                    profiles_dir=self.config.profiles_dir,
                    gpu_identifier=gpu_identifier,
                    backend=cfg.backend,
                    flavor=ProfileFlavor.BALANCED,
                    current_driver_version=driver_version,
                    current_binary_version=binary_version,
                    staleness_days=self.config.profile_staleness_days,
                )
            except Exception:
                self._push_status_message(f"No profile found for {cfg.alias}; using defaults")
                updated_configs.append(cfg)
                continue

            if record is None:
                self._push_status_message(f"No profile found for {cfg.alias}; using defaults")
                updated_configs.append(cfg)
                continue

            if staleness is not None and staleness.is_stale:
                reasons = "; ".join(r.value.replace("_", " ").title() for r in staleness.reasons)
                self._push_status_message(
                    f"Profile stale for {cfg.alias}: {reasons}; using defaults"
                )
                updated_configs.append(cfg)
                continue

            # Profile exists and is fresh - apply overrides
            profile_overrides = profile_to_override_dict(record)

            if not profile_overrides:
                self._push_status_message(f"Profile empty for {cfg.alias}; using defaults")
                updated_configs.append(cfg)
                continue

            # Apply profile overrides via merge_config_overrides.
            # Pass an empty override_dict so profile_overrides take full effect;
            # identity fields (model, alias, device, port, backend, etc.) are
            # restored explicitly below.
            override_dict: dict[str, object] = {}

            merged = merge_config_overrides(
                defaults=self.config,
                slot_config=None,
                workstation_config=None,
                profile_config=profile_overrides,
                override_config=override_dict,
            )

            # Preserve identity fields that shouldn't come from profile
            merged.model = cfg.model
            merged.alias = cfg.alias
            merged.device = cfg.device
            merged.port = cfg.port
            merged.bind_address = cfg.bind_address
            merged.server_bin = cfg.server_bin
            merged.backend = cfg.backend
            merged.tensor_split = cfg.tensor_split
            merged.reasoning_mode = cfg.reasoning_mode
            merged.reasoning_format = cfg.reasoning_format
            merged.chat_template_kwargs = cfg.chat_template_kwargs
            merged.reasoning_budget = cfg.reasoning_budget
            merged.use_jinja = cfg.use_jinja
            merged.n_gpu_layers = cfg.n_gpu_layers
            merged.risky_acknowledged = cfg.risky_acknowledged

            self._push_status_message(
                f"Applied profile: {cfg.alias} (balanced) "
                f"[threads={merged.threads}, ctx={merged.ctx_size}]"
            )
            updated_configs.append(merged)

        return updated_configs

    def build_llama_cpp(self, backend: str = "sycl", dry_run: bool = False) -> bool:
        """Build llama.cpp using BuildPipeline.

        Args:
            backend: Build backend ("sycl" or "cuda")
            dry_run: If True, print commands without executing

        Returns:
            True if build successful, False otherwise
        """
        config = Config()

        # Determine paths
        source_dir = Path(config.llama_cpp_root)
        build_dir = source_dir / ("build_cuda" if backend == "cuda" else "build")
        output_dir = config.builds_dir

        # Create build config
        build_backend = BuildBackend.SYCL if backend == "sycl" else BuildBackend.CUDA
        build_config = BuildConfig(
            backend=build_backend,
            source_dir=source_dir,
            build_dir=build_dir,
            output_dir=output_dir,
            git_remote_url=config.build_git_remote,
            git_branch=config.build_git_branch,
            shallow_clone=True,
            retry_attempts=config.build_retry_attempts,
            retry_delay=config.build_retry_delay,
        )

        # Create and configure pipeline with progress callback
        self._build_pipeline = BuildPipeline(
            build_config, progress_callback=self._handle_build_progress
        )
        self._build_pipeline.dry_run = dry_run
        self.build_in_progress = True

        # Capture original SIGINT handler before replacing it
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler_build)

        try:
            # Run pipeline
            print(f"Building for {backend} backend...", file=sys.stderr)
            if dry_run:
                print("DRY RUN MODE - commands will not be executed", file=sys.stderr)

            result = self._build_pipeline.run()

            if result.success:
                print("Build completed successfully!", file=sys.stderr)
                if result.artifact:
                    print(f"Artifact: {result.artifact.binary_path}", file=sys.stderr)
                return True
            else:
                print(f"Build failed: {result.error_message}", file=sys.stderr)
                return False
        finally:
            self.build_in_progress = False
            # Restore original SIGINT handler
            if self._original_sigint_handler is not None:
                signal.signal(signal.SIGINT, self._original_sigint_handler)

    def run(self, acknowledged: bool = False) -> None:
        # Apply profile overrides at startup (before slot launch validation)
        self.configs = self._apply_profile_overrides()

        # If no slots configured, skip launch and show empty state
        if not self.configs:
            self._push_status_message("No slots configured. Press 'a' to add a slot.")
            self._run_tui_loop_without_servers()
            return

        slots = [
            ModelSlot(slot_id=cfg.alias, model_path=cfg.model, port=cfg.port)
            for cfg in self.configs
        ]

        launch_attempt_id = self.server_manager.begin_launch_attempt()
        ack_token = self.server_manager.issue_ack_token(launch_attempt_id)
        has_risks = self._acknowledge_risks(launch_attempt_id, ack_token, acknowledged)
        self._update_risk_panel_state(has_risks)

        launch_result = self.server_manager.launch_all_slots(slots)
        self._handle_launch_result(launch_result)

        self.launch_result = launch_result
        self._build_status_panel(launch_result)

        launched_slots = launch_result.launched or []
        launched_set = set(launched_slots)

        launched_configs = [cfg for cfg in self.configs if cfg.alias in launched_set]
        launched_log_buffers = {
            alias: buf for alias, buf in self.log_buffers.items() if alias in launched_set
        }

        log_handlers = {
            cfg.alias: lambda line, buf=launched_log_buffers[cfg.alias]: buf.add_line(line)
            for cfg in launched_configs
        }
        processes = self.server_manager.start_servers(launched_configs, log_handlers)

        self.server_processes = {
            cfg.alias: proc for cfg, proc in zip(launched_configs, processes, strict=True)
        }

        # Initialize slot states for launched servers
        for cfg in launched_configs:
            self.handle_slot_transition(cfg.alias, SlotState.RUNNING)

        try:
            TextualDashboardApp(self).run()
        finally:
            self._cleanup()

    def _run_tui_loop_without_servers(self) -> None:
        """Run the TUI loop without any server processes.

        Used when no slots are configured - allows user to add slots interactively.
        """
        try:
            TextualDashboardApp(self).run()
        finally:
            self._cleanup()
