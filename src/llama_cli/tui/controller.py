"""Dashboard controller for the Textual TUI."""

import signal
import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any

from rich.panel import Panel
from rich.text import Text

from llama_manager import (
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    RiskAckResult,
    ServerConfig,
    ServerManager,
    SlotState,
    add_slot_from_form,
    compute_slot_transition,
    launch_orchestrate,
    resolve_risk_action,
)
from llama_manager.build_pipeline import (
    BuildPipeline,
    BuildProgress,
    run_build_for_backend,
)

from .components.launch_status import LaunchStatusPanelRenderer
from .components.profile_status import ProfileStatusPanelRenderer
from .components.risk import RiskPanelRenderer
from .components.server_column import ServerColumnPanel
from .components.slot_status import SlotStatusPanel
from .components.status_messages import StatusMessagesRenderer
from .components.system_status import SystemStatusPanelRenderer
from .model import DashboardModel
from .textual_app import DashboardApp
from .types import ServerColumnState
from .viewmodel import DashboardViewModel


class DashboardController:
    """Controller for TUI commands, lifecycle, and background work."""

    def __init__(
        self,
        configs: list[ServerConfig],
        gpu_indices: list[int],
        slots: list[ModelSlot] | None = None,
    ) -> None:
        self.model = DashboardModel(configs=configs, gpu_indices=gpu_indices, slots=slots)
        self.view_model = DashboardViewModel(self.model)
        self._status_panel_renderer = LaunchStatusPanelRenderer()
        self._risk_panel_renderer = RiskPanelRenderer()
        self._profile_status_renderer = ProfileStatusPanelRenderer()
        self._status_messages_renderer = StatusMessagesRenderer()
        self._system_status_renderer = SystemStatusPanelRenderer()
        self.status_panel: Panel | None = None

        # Build pipeline state
        self._build_pipeline: BuildPipeline | None = None
        self.build_in_progress = False
        self.build_progress: BuildProgress | None = None
        self._original_sigint_handler: Callable[[int, FrameType | None], Any] | int | None = None

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
    def active_risk_kind(self) -> str | None:
        return self.model.risk_prompt.kind if self.model.risk_prompt is not None else None

    @active_risk_kind.setter
    def active_risk_kind(self, value: str | None) -> None:
        if value is None:
            self.model.clear_risk_prompt()
        else:
            self.model.set_risk_prompt(value, acknowledged=self.risks_acknowledged)

    @property
    def risk_panel(self) -> Panel | None:
        prompt = self.model.risk_prompt
        if prompt is None:
            return None
        if prompt.acknowledged:
            return self._risk_panel_renderer.acknowledged()
        return self._risk_panel_renderer.required(prompt.kind)

    @risk_panel.setter
    def risk_panel(self, value: object | None) -> None:
        if value is None:
            self.model.clear_risk_prompt()
        elif self.model.risk_prompt is None:
            self.model.set_risk_prompt("hardware", acknowledged=False)

    @property
    def profile_request(self) -> str | None:
        return self.model.profile_request

    @profile_request.setter
    def profile_request(self, value: str | None) -> None:
        self.model.profile_request = value

    @property
    def _build_request(self) -> bool:
        return self.model.build_request

    @_build_request.setter
    def _build_request(self, value: bool) -> None:
        self.model.build_request = value

    @property
    def _smoke_request(self) -> bool:
        return self.model.smoke_request

    @_smoke_request.setter
    def _smoke_request(self, value: bool) -> None:
        self.model.smoke_request = value

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
    def _profile_status(self) -> dict[str, str]:
        return self.model.profile_status

    @property
    def _profile_flavor(self) -> dict[str, str]:
        return self.model.profile_flavor

    @property
    def _profile_cancel_events(self) -> dict[str, threading.Event]:
        return self.model.profile_cancel_events

    @property
    def _profile_lock(self) -> threading.Lock:
        return self.model.profile_lock

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
            print("\nBuild interrupted by user, releasing lock...", file=sys.stderr)
            self._build_pipeline.release_lock()
            self.build_in_progress = False
            sys.exit(130)  # Standard exit code for Ctrl+C

    def _make_collector(self, device_index: int) -> Callable[[], dict[str, Any]]:
        """Create a GPU collector bound to a specific device index."""
        return self.model.make_collector(device_index)

    def stop(self) -> None:
        """Stop the TUI loop gracefully."""
        self.running = False

    def render_panels(self) -> tuple[Panel, Panel | None, Panel, Text]:
        """Render a snapshot of the dashboard for tests and non-Textual callers."""
        alerts_panel = self._system_status_renderer.render_panel(self.view_model.system_status())
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

        from .components.menu import CommandMenuRenderer

        return (
            alerts_panel,
            left_panel,
            right_panel,
            CommandMenuRenderer().render(self.view_model.command_menu()),
        )

    # ------------------------------------------------------------------
    # Rendering delegation helpers
    # ------------------------------------------------------------------

    def _build_status_panel(self, launch_result: LaunchResult) -> None:
        self.status_panel = self._status_panel_renderer.render(launch_result)

    def _build_risk_panel_required(self, kind: str = "hardware") -> None:
        self.model.set_risk_prompt(kind=kind, acknowledged=False)

    def _build_risk_panel_acknowledged(self, kind: str = "hardware") -> None:
        self.model.set_risk_prompt(kind=kind, acknowledged=True)

    def _build_column_panel(
        self, cfg: ServerConfig, buffer: LogBuffer, gpu: GPUStats | None
    ) -> Panel:
        stale_warning = self.get_stale_warning(cfg)
        return ServerColumnPanel(
            ServerColumnState(
                config=cfg,
                buffer=buffer,
                gpu=gpu,
                host=self.config.host,
                stale_warning=stale_warning,
                slot_states=self.slot_states,
                server_processes=self.server_processes,
                is_unsaved=cfg.alias in self.unsaved_slots,
            )
        ).render()

    def _build_placeholder_panel(self) -> Panel:
        return SlotStatusPanel(self.view_model.slot_status(configs=[])).render()

    def _build_profile_status_panel(self) -> Panel | None:
        return self._profile_status_renderer.render(
            self.view_model.active_profile_status(),
            self._profile_flavor,
        )

    def build_system_notices(self) -> list[str]:
        """Build concise status notices shown in the top system panel."""
        notices = self.view_model.system_notices()

        if self.build_in_progress and self.build_progress is not None:
            notices.append(
                f"Build {self.build_progress.stage}: {self.build_progress.status} "
                f"({self.build_progress.progress_percent}%)"
            )

        return notices

    # How long (seconds) a status message remains visible across renders.
    _STATUS_MESSAGE_LIFETIME_S: float = DashboardModel.STATUS_MESSAGE_LIFETIME_S

    def _build_status_messages_panel(self) -> Text | None:
        return self._status_messages_renderer.render(self.view_model.status_messages())

    def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
        """Return status messages newer than ``since_ts``."""
        return self.model.get_status_messages_since(since_ts)

    def prune_expired_status_messages(self) -> None:
        """Remove status messages older than ``_STATUS_MESSAGE_LIFETIME_S``."""
        self.model.prune_expired_status_messages()

    def _build_command_menu(self) -> Text:
        from .components.menu import CommandMenuRenderer

        return CommandMenuRenderer().render(self.view_model.command_menu())

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

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
        self.model.push_status_message(message)

    def add_slot_from_form(self, values: dict[str, str]) -> bool:
        """Create or replace a slot from modal profile selection."""
        state = {
            "log_buffers": self.log_buffers,
            "server_processes": self.server_processes,
            "slot_states": self.slot_states,
            "unsaved_slots": self.unsaved_slots,
            "slots": self.slots,
        }
        success, messages, _updated_state = add_slot_from_form(
            values,
            self.config,
            self.configs,
            self.gpu_indices,
            self.gpu_stats,
            self.server_manager,
            state,
            self._make_collector,
        )
        for msg in messages:
            self._push_status_message(msg)
        return success

    def cancel_add_slot_form(self) -> None:
        """Emit a status message when the add-slot modal is cancelled."""
        self._push_status_message("Slot configuration cancelled")

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
        """Return a warning string when the cached profile is stale."""
        return self.view_model.stale_warning(cfg)

    def _update_risk_panel_state(self, result: RiskAckResult) -> None:
        if result.has_risks:
            if result.risks_acknowledged:
                self._build_risk_panel_acknowledged()
            else:
                self._build_risk_panel_required()
            return
        self.risk_panel = None

    def _apply_risk_action(self, action: str) -> None:
        if action in ("acknowledge", "proceed"):
            self.risk_panel = None
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
        if old_state is None and new_state == SlotState.RUNNING:
            # First launch - clear any previous status panels
            self.status_panel = None

        self._push_status_message(message)

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

    def build_llama_cpp(self, backend: str = "sycl", dry_run: bool = False) -> bool:
        """Build llama.cpp using BuildPipeline.

        Args:
            backend: Build backend ("sycl" or "cuda")
            dry_run: If True, print commands without executing

        Returns:
            True if build successful, False otherwise
        """
        config = Config()

        def _set_pipeline(pipeline: BuildPipeline) -> None:
            self._build_pipeline = pipeline
            self.build_in_progress = True

        # Capture original SIGINT handler before replacing it
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler_build)

        try:
            print(f"Building for {backend} backend...", file=sys.stderr)
            if dry_run:
                print("DRY RUN MODE - commands will not be executed", file=sys.stderr)

            result = run_build_for_backend(
                backend=backend,
                dry_run=dry_run,
                config=config,
                progress_callback=self._handle_build_progress,
                pipeline_callback=_set_pipeline,
            )

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
        from llama_cli.commands.profile import _get_driver_version

        # Delegate launch orchestration to the pure library function
        result = launch_orchestrate(
            self.configs,
            self.config,
            self.server_manager,
            self.log_buffers,
            _get_driver_version,
            acknowledged=acknowledged,
        )

        self.configs = result.updated_configs

        for msg in result.status_messages:
            self._push_status_message(msg)

        if result.empty:
            self._run_tui_loop_without_servers()
            return

        # Map risk result to TUI panels
        self._update_risk_panel_state(result.risk_result)
        for detail in result.risk_result.risk_details:
            if not acknowledged:
                self._build_risk_panel_required(detail["risk_kind"])
                self._push_status_message(
                    f"warning: risky operation in {detail['alias']}: {detail['risk']} — "
                    f"press 'y' to acknowledge, 'n' to abort"
                )

        self.launch_result = result.launch_result
        self._build_status_panel(result.launch_result)

        # CLI boundary: stderr printing and SystemExit for blocked launches
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
