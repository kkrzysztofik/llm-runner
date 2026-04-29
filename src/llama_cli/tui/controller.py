"""TUIApp controller — manages server lifecycle, state, and rendering."""

import queue
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType
from typing import Any

from rich.console import Group
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
    build_gpu_telemetry_panel,
    build_profile_status_panel,
    build_risk_panel_acknowledged,
    build_risk_panel_required,
    build_status_messages_panel,
    build_status_panel,
)
from .components.menu import build_command_menu
from .components.panels import (
    build_column_panel,
    build_placeholder_panel,
    build_slot_status_panel,
)
from .constants import RISK_ACK_LABEL
from .textual_app import TextualDashboardApp
from .types import DashboardSnapshot, TextualLayoutSpec


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
        self.width = 80
        self.height = 24
        self.launch_result: LaunchResult | None = None
        self.status_panel: Panel | None = None
        self.telemetry_panel: Panel | None = None
        self.risk_panel: Panel | None = None
        self.risks_acknowledged: bool = False
        self.active_risk_kind: str | None = None

        # Textual keypress dispatch queue used by unit tests and app actions.
        self._keypress_queue: queue.Queue[str] = queue.Queue()

        # Profile state
        self._profile_status: dict[str, str] = {}  # alias -> "idle" | "running" | "done" | "failed"
        self._profile_flavor: dict[str, str] = {}  # alias -> flavor string
        self._profile_cancel_events: dict[str, threading.Event] = {}
        self._profile_lock = threading.Lock()

        # TUI-safe status message buffer. Each entry is (timestamp, message).
        self._status_messages: list[tuple[float, str]] = []
        self._status_lock = threading.Lock()

        # Non-blocking profile flavor request queue
        self._profile_request: str | None = None
        self._build_request: bool = False
        self._smoke_request: bool = False

        # Slot configuration mode (for adding new slots)
        self._slot_config_state: dict[str, str] = {}  # slot_id -> current prompt field
        self._slot_config_values: dict[str, dict[str, str]] = {}  # slot_id -> {field: value}
        self._unsaved_slots: set[str] = set()

        self.server_manager = ServerManager()

        # Slot state tracking for TUI dashboard
        self._slot_states: dict[str, str] = {}  # alias -> SlotState value
        self._server_processes: dict[str, Any] = {}  # alias -> subprocess.Popen

        for cfg in configs:
            self.log_buffers[cfg.alias] = LogBuffer(redact_sensitive=True)
        for idx in gpu_indices:
            # Pass a bound collector callable with the device index
            self.gpu_stats.append(GPUStats(idx, collector=self._make_collector(idx)))

        # Build pipeline state
        self._build_pipeline: BuildPipeline | None = None
        self._build_in_progress = False
        self._build_progress: BuildProgress | None = None
        self._original_sigint_handler: Callable[[int, FrameType | None], Any] | int | None = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals by stopping the TUI loop.

        If a build is in progress, release the build lock before stopping.
        """
        # Release build lock if in progress
        if self._build_in_progress and self._build_pipeline is not None:
            self._build_pipeline.release_lock()
            self._build_in_progress = False

        self.stop()

    def _signal_handler_build(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler specifically for build process.

        Releases build lock and stops the build gracefully.
        """
        if self._build_in_progress and self._build_pipeline is not None:
            print("\nBuild interrupted by user, releasing lock...", file=sys.stderr)
            self._build_pipeline.release_lock()
            self._build_in_progress = False
            sys.exit(130)  # Standard exit code for Ctrl+C

    def _make_collector(self, device_index: int) -> Callable[[], dict[str, Any]]:
        """Create a GPU collector bound to a specific device index."""

        def collector() -> dict[str, Any]:
            return collect_nvtop_stats(device_index)

        return collector

    def on_resize(self, event: Any) -> None:
        self.width = event.width
        self.height = event.height

    def stop(self) -> None:
        """Stop the TUI loop gracefully."""
        self.running = False

    def build_layout(self) -> TextualLayoutSpec:
        orientation = "horizontal" if self.width >= 80 else "vertical"
        return TextualLayoutSpec(content_orientation=orientation)

    def render(self) -> DashboardSnapshot:
        self._update_gpu_telemetry()

        alerts: list[Panel] = []
        if self.risk_panel is not None:
            alerts.append(self.risk_panel)
        if self.status_panel is not None:
            alerts.append(self.status_panel)

        # Add GPU telemetry panel
        if self.telemetry_panel is not None:
            alerts.append(self.telemetry_panel)

        # Add slot status panel
        slot_status_panel = self._build_slot_status_panel()
        if slot_status_panel is not None:
            alerts.append(slot_status_panel)

        # Add profile status panel
        profile_panel = self._build_profile_status_panel()
        if profile_panel is not None:
            alerts.append(profile_panel)

        # Add status messages panel
        status_msgs_panel = self._build_status_messages_panel()
        if status_msgs_panel is not None:
            alerts.append(status_msgs_panel)

        if alerts:
            alerts_panel = Panel(Group(*alerts), title="System Alerts", border_style="yellow")
        else:
            alerts_panel = Panel(Text("No active alerts", style="dim"), border_style="dim")

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
        stale_warning = self._get_stale_warning(cfg)
        return build_column_panel(cfg, buffer, gpu, self.config.host, stale_warning)

    def _build_placeholder_panel(self) -> Panel:
        return build_placeholder_panel()

    def _build_slot_status_panel(self) -> Panel:
        return build_slot_status_panel(
            self.configs,
            self._slot_states,
            self._server_processes,
            self.log_buffers,
            self.config.host,
            unsaved_slots=self._unsaved_slots,
        )

    def _build_profile_status_panel(self) -> Panel | None:
        with self._profile_lock:
            active = {a: s for a, s in self._profile_status.items() if s != "idle"}
        return build_profile_status_panel(active, self._profile_flavor)

    # How long (seconds) a status message remains visible across renders.
    _STATUS_MESSAGE_LIFETIME_S: float = 30.0

    def _build_status_messages_panel(self) -> Panel | None:
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

    def prune_expired_status_messages(self) -> None:
        """Remove status messages older than ``_STATUS_MESSAGE_LIFETIME_S``."""
        cutoff = time.monotonic() - self._STATUS_MESSAGE_LIFETIME_S
        with self._status_lock:
            self._status_messages = [(ts, msg) for ts, msg in self._status_messages if ts >= cutoff]

    def _build_command_menu(self) -> Text:
        return build_command_menu(
            self._profile_request,
            self._slot_config_state,
            self.risk_panel,
            self.active_risk_kind,
        )

    def _update_gpu_telemetry(self) -> None:
        """Update GPU telemetry panel with latest stats."""
        if not self.gpu_stats:
            self.telemetry_panel = None
            return

        lines: list[str] = []
        for gpu in self.gpu_stats:
            gpu.update()  # Refresh stats from collector
            lines.append(gpu.format_stats_text())

        self.telemetry_panel = build_gpu_telemetry_panel(lines)

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

    def handle_keypress(self, key: str) -> None:
        """Handle a Textual key event through the existing state machine."""
        _KEY_MAP = {
            "return": "\n",
            "escape": "\x1b",
            "backspace": "\x7f",
            "ctrl+c": "^C",
            "ctrl+C": "^C",
            "ctrl_c": "^C",
            "tab": "\t",
        }
        self._keypress_queue.put(_KEY_MAP.get(key, key))
        self._process_keypresses()

    def _handle_profile_key(self, key: str) -> bool:
        """Handle profile-related keys. Returns True if key was consumed."""
        if self._profile_request is not None and key in {"1", "2", "3"}:
            alias = self._profile_request
            self._profile_request = None
            self._wait_for_flavor_selection(alias, preselected_key=key)
            return True

        if key == "^C":
            if self._profile_request is not None:
                self._profile_request = None
                self._push_status_message("Profile selection cancelled.")
                return True
            # Let _on_key handle running profiles / shutdown
            return False

        if key.upper() == "P" and self.configs:
            cfg = self.configs[0]
            self._prompt_profile_flavor(cfg.alias)
            return True

        return False

    def _handle_build_key(self, key: str) -> bool:
        """Handle build-related keys. Returns True if consumed."""
        if self._build_request and key in {"1", "2", "3"}:
            self._build_request = False
            # Map choice to target (1=sycl, 2=cuda, 3=both)
            target = {"1": "sycl", "2": "cuda", "3": "both"}.get(key, "both")
            self._push_status_message(f"Feature not fully hooked up in TUI yet: Build {target}")
            # Placeholder: trigger actual build logic in background thread
            return True

        if self._build_request and key in {"^C", "\x1b"}:
            self._build_request = False
            self._push_status_message("Build selection cancelled.")
            return True

        if key == "b":
            self._build_request = True
            self._push_status_message("Select build target: [1] SYCL  [2] CUDA  [3] Both")
            return True

        return False

    def _handle_smoke_key(self, key: str) -> bool:
        """Handle smoke test-related keys. Returns True if consumed."""
        if self._smoke_request and key in {"1", "2"}:
            self._smoke_request = False
            target = "both" if key == "1" else "slot"
            self._push_status_message(f"Feature not fully hooked up in TUI yet: Smoke {target}")
            # Placeholder: trigger smoke logic
            return True

        if self._smoke_request and key in {"^C", "\x1b"}:
            self._smoke_request = False
            self._push_status_message("Smoke selection cancelled.")
            return True

        if key == "s":
            self._smoke_request = True
            self._push_status_message("Select smoke scope: [1] Both  [2] Active Slot")
            return True

        return False

    def _handle_risk_key(self, key: str) -> None:
        """Handle hardware warning and VRAM risk keys."""
        if key in ("y", "n"):
            if self.active_risk_kind == "vram":
                result = self.handle_vram_risk(key)
            else:
                result = self.handle_hardware_warning(key)
            if result in ("proceed", "abort"):
                self.active_risk_kind = None
        elif key == "q" and self.active_risk_kind != "vram":
            result = self.handle_hardware_warning(key)
            if result in ("acknowledge", "abort", "quit"):
                self.active_risk_kind = None

    def _process_keypresses(self) -> None:
        """Drain the keypress queue and handle relevant keys."""
        while not self._keypress_queue.empty():
            try:
                key = self._keypress_queue.get_nowait()
            except queue.Empty:
                break

            # Handle slot configuration mode first
            if self._slot_config_state:
                self._process_slot_config_input(key)
                continue

            if self._handle_profile_key(key):
                continue

            if self._handle_build_key(key):
                continue

            if self._handle_smoke_key(key):
                continue

            if self.risk_panel is not None:
                self._handle_risk_key(key)
                continue

            self._on_key(key)

        # Handle queued profile flavor request non-blockingly
        if self._profile_request is not None:
            alias = self._profile_request
            self._profile_request = None
            self._wait_for_flavor_selection(alias)

    def _prompt_profile_flavor(self, alias: str) -> None:
        """Queue a non-blocking profile flavor request for the given alias.

        The actual prompt is processed by _process_keypresses via the keypress
        queue so the Textual event loop is never blocked.
        """
        # Check if already profiling
        with self._profile_lock:
            if self._profile_status.get(alias) == "running":
                return

        # Clear any previous status
        with self._profile_lock:
            self._profile_status[alias] = "idle"

        # Queue the alias so _process_keypresses can handle it non-blockingly
        self._profile_request = alias

    def _wait_for_flavor_selection(self, alias: str, preselected_key: str | None = None) -> None:
        """Non-blocking flavor selection — drain keypress queue once per cycle.

        Called from _process_keypresses; does NOT sleep or block.
        If the user hasn't pressed a valid key yet, the request stays queued
        and _process_keypresses will try again on the next dashboard refresh.
        """
        flavor_map = {"1": "balanced", "2": "fast", "3": "quality"}

        if preselected_key in flavor_map:
            self._profile_request = None
            self._run_profile_background(alias, flavor_map[preselected_key])
            return

        while not self._keypress_queue.empty():
            try:
                key = self._keypress_queue.get_nowait()
            except queue.Empty:
                break
            if key in flavor_map:
                self._profile_request = None
                self._run_profile_background(alias, flavor_map[key])
                return
            if key == "^C":
                self._profile_request = None
                self._push_status_message(f"Profile '{alias}' cancelled.")
                return
            # Ignore other keys silently

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

    # ------------------------------------------------------------------
    # Slot configuration (add new slots)
    # ------------------------------------------------------------------

    SLOT_CONFIG_FIELDS = ["model", "port", "backend", "threads", "ctx_size"]
    SLOT_CONFIG_PROMPTS = {
        "model": "Model path: ",
        "port": "Port: ",
        "backend": "Backend (cuda/sycl): ",
        "threads": "Threads: ",
        "ctx_size": "Context size: ",
    }

    def _prompt_add_slot(self) -> None:
        """Start the slot configuration flow."""
        slot_id = f"slot{len(self.configs) + 1}"
        self._slot_config_state[slot_id] = "model"
        self._slot_config_values[slot_id] = {}
        self._push_status_message(f"Adding {slot_id}: {self.SLOT_CONFIG_PROMPTS['model']}")

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
        self._server_processes.pop(alias, None)
        self._slot_states.pop(alias, None)
        self._unsaved_slots.discard(alias)
        self.slots = [slot for slot in self.slots if slot.slot_id != alias]

    def _register_and_start_slot(self, cfg: ServerConfig) -> None:
        """Register and start one slot using the current config."""
        alias = cfg.alias
        self.log_buffers[alias] = LogBuffer(redact_sensitive=True)
        self._unsaved_slots.add(alias)
        self.slots.append(ModelSlot(slot_id=alias, model_path=cfg.model, port=cfg.port))

        log_handler = lambda line, buf=self.log_buffers[alias]: buf.add_line(line)  # noqa: E731
        procs = self.server_manager.start_servers([cfg], {alias: log_handler})
        if procs:
            self._server_processes[alias] = procs[0]
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

    def _process_slot_config_input(self, key: str) -> None:
        """Process input during slot configuration mode.

        Non-blocking - called from _process_keypresses.
        """
        # Find any slots in config mode
        slots_to_process = list(self._slot_config_state.keys())
        for slot_id in slots_to_process:
            current_field = self._slot_config_state.get(slot_id)
            if current_field is None:
                continue

            # Handle Enter key - advance to next field or complete
            if key == "\n":
                self._advance_slot_config_field(slot_id)
                continue

            # Handle Escape key - cancel
            if key == "\x1b":
                self._cancel_slot_config(slot_id)
                continue

            # Handle backspace
            if key in {"\x7f", "\b"}:
                current_value = self._slot_config_values.get(slot_id, {}).get(current_field, "")
                if current_value:
                    self._slot_config_values[slot_id][current_field] = current_value[:-1]
                continue

            # Handle regular character input (only for value fields)
            if len(key) == 1 and current_field in self.SLOT_CONFIG_FIELDS:
                current_value = self._slot_config_values.get(slot_id, {}).get(current_field, "")
                self._slot_config_values[slot_id][current_field] = current_value + key

    def _advance_slot_config_field(self, slot_id: str) -> None:
        """Advance to the next field in slot configuration."""
        values = self._slot_config_values.get(slot_id, {})
        current_field = self._slot_config_state.get(slot_id)

        if current_field is None:
            return

        if not self._validate_slot_field(current_field, values):
            return

        # Find next field
        fields = self.SLOT_CONFIG_FIELDS
        try:
            current_idx = fields.index(current_field)
            if current_idx + 1 < len(fields):
                next_field = fields[current_idx + 1]
                self._slot_config_state[slot_id] = next_field
                self._push_status_message(
                    f"Adding {slot_id}: {self.SLOT_CONFIG_PROMPTS[next_field]}"
                )
            else:
                # All fields complete - create the slot
                self._finalize_slot_config(slot_id)
        except ValueError:
            self._cancel_slot_config(slot_id)

    def _validate_slot_field(self, current_field: str, values: dict[str, str]) -> bool:
        """Validate and normalise the current slot config field.

        Returns ``False`` if the caller should abort advancing (e.g. model
        path is empty).
        """
        if current_field == "port":
            self._normalize_slot_port(values)
        elif current_field == "threads":
            self._normalize_slot_threads(values)
        elif current_field == "ctx_size":
            self._normalize_slot_ctx_size(values)
        elif current_field == "backend":
            self._normalize_slot_backend(values)
        elif current_field == "model" and not values.get("model", "").strip():
            self._push_status_message("Model path required")
            return False
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

    def _normalize_slot_threads(self, values: dict[str, str]) -> None:
        """Validate and normalise the threads field, falling back to 4."""
        try:
            threads = int(values.get("threads", ""))
            if threads < 1:
                self._push_status_message("Invalid threads, using 4")
                values["threads"] = "4"
        except ValueError:
            self._push_status_message("Invalid threads, using 4")
            values["threads"] = "4"

    def _normalize_slot_ctx_size(self, values: dict[str, str]) -> None:
        """Validate and normalise the ctx_size field, falling back to 2048."""
        try:
            ctx_size = int(values.get("ctx_size", ""))
            if ctx_size < 512:
                self._push_status_message("Invalid ctx_size, using 2048")
                values["ctx_size"] = "2048"
        except ValueError:
            self._push_status_message("Invalid ctx_size, using 2048")
            values["ctx_size"] = "2048"

    def _normalize_slot_backend(self, values: dict[str, str]) -> None:
        """Validate and normalise the backend field, falling back to sycl."""
        backend = values.get("backend", "").lower()
        if backend not in ("cuda", "sycl"):
            self._push_status_message("Invalid backend, using sycl")
            values["backend"] = "sycl"
        else:
            values["backend"] = backend

    def _finalize_slot_config(self, slot_id: str) -> None:
        """Create the slot from collected configuration values."""
        values = self._slot_config_values.get(slot_id, {})
        if not values.get("model"):
            self._push_status_message("Slot configuration cancelled")
            self._slot_config_state.pop(slot_id, None)
            self._slot_config_values.pop(slot_id, None)
            return

        self._add_slot_from_values(slot_id, values)

        # Clean up config state
        self._slot_config_state.pop(slot_id, None)
        self._slot_config_values.pop(slot_id, None)

    def _add_slot_from_values(self, slot_id: str, values: dict[str, str]) -> None:
        """Add a configured slot and start its server process."""
        # Create ServerConfig for the new slot
        backend = values.get("backend", "sycl")
        port = int(values.get("port", "8080"))
        threads = int(values.get("threads", "4"))
        ctx_size = int(values.get("ctx_size", "2048"))

        # Determine device string from backend
        device = "SYCL" if backend == "sycl" else "CUDA"
        new_cfg = ServerConfig(
            alias=slot_id,
            model=values["model"],
            device=device,
            port=port,
            ctx_size=ctx_size,
            ubatch_size=512,
            threads=threads,
            backend=backend,
        )

        self.configs.append(new_cfg)
        self.log_buffers[slot_id] = LogBuffer(redact_sensitive=True)
        self._unsaved_slots.add(slot_id)

        # Determine GPU index: SYCL runs on Intel Arc (slot 1 by convention),
        # CUDA runs on NVIDIA (slot 0 by convention). Adjust via gpu_indices if needed.
        gpu_idx = 1 if backend == "sycl" else 0
        self.gpu_indices.append(gpu_idx)
        self.gpu_stats.append(GPUStats(gpu_idx, collector=self._make_collector(gpu_idx)))

        # Register the slot and start the server process so the new slot becomes live.
        new_slot = ModelSlot(slot_id=slot_id, model_path=values["model"], port=port)
        self.slots.append(new_slot)
        log_handler = lambda line, buf=self.log_buffers[slot_id]: buf.add_line(line)  # noqa: E731
        procs = self.server_manager.start_servers([new_cfg], {slot_id: log_handler})
        if procs:
            self._server_processes[slot_id] = procs[0]
        self.handle_slot_transition(slot_id, SlotState.RUNNING)

        self._push_status_message(f"Added {slot_id}: {values['model']} on {backend}:{port}")

    def _cancel_slot_config(self, slot_id: str) -> None:
        """Cancel slot configuration."""
        self._push_status_message("Slot configuration cancelled")
        self._slot_config_state.pop(slot_id, None)
        self._slot_config_values.pop(slot_id, None)

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

    def _get_stale_warning(self, cfg: ServerConfig) -> str | None:
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

    def _on_key(self, key: str) -> None:
        """Handle key presses from the keypress queue."""
        # Handle Ctrl+C — abort profile or graceful shutdown
        if key == "^C":
            # Check if a profile is running
            with self._profile_lock:
                for cfg in self.configs:
                    if self._profile_status.get(cfg.alias) == "running":
                        cancel_event = self._profile_cancel_events.get(cfg.alias)
                        if cancel_event is not None:
                            cancel_event.set()
                        self._profile_status[cfg.alias] = "failed"
                        self._push_status_message(f"Profile '{cfg.alias}' aborted.")
                        return

            # No profile running — graceful shutdown
            self._graceful_shutdown()
            return

        # Handle 'q' key — quit
        if key == "q":
            self._graceful_shutdown()
            return

        # Handle 'r' key — refresh display
        if key == "r":
            self._push_status_message("Display refreshed.")
            return

        # Handle 'a' key — add new slot
        if key == "a":
            self._prompt_add_slot()
            return

        # Handle 'P' key — trigger profiling on the focused slot
        if key.upper() == "P" and self.configs:
            cfg = self.configs[0]
            self._prompt_profile_flavor(cfg.alias)

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
        old_state = self._slot_states.get(slot_id)
        self._slot_states[slot_id] = new_state.value

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
        self._build_progress = progress

        # Update status panel if build is in progress
        if self._build_in_progress:
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
        self._build_in_progress = True

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
            self._build_in_progress = False
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

        self._server_processes = {
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
