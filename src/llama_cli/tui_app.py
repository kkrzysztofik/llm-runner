"""TUI application for llm-runner.

This module provides a Rich-based live terminal interface for managing
multiple llama-server instances with real-time log streaming, GPU stats,
and configuration display.
"""

import os
import queue
import select
import signal
import sys
import threading
import time
import tty
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any

import psutil
from rich.console import ConsoleDimensions, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from llama_cli.colors import Colors
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
)
from llama_manager.build_pipeline import (
    BuildBackend,
    BuildConfig,
    BuildPipeline,
    BuildProgress,
)
from llama_manager.server import detect_risky_operations

RISK_ACK_LABEL = "warning_bypass"
RISK_CONFIRM_PROMPT = "Confirm risky operation [y/N]: "
STATUS_PREFIX = "STATUS: "
STYLE_BOLD_RED = "bold red"
STYLE_BOLD_YELLOW = "bold yellow"


@contextmanager
def _cbreak_stdin() -> Any:
    if os.name == "nt" or not sys.stdin.isatty():
        yield
        return

    try:
        import termios

        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (ImportError, OSError, ValueError):
        yield
        return

    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


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

        # Keypress input polling infrastructure
        self._keypress_queue: queue.Queue[str] = queue.Queue()
        self._input_thread: threading.Thread | None = None

        # Profile state
        self._profile_status: dict[str, str] = {}  # alias -> "idle" | "running" | "done" | "failed"
        self._profile_flavor: dict[str, str] = {}  # alias -> flavor string
        self._profile_cancel_events: dict[str, threading.Event] = {}
        self._profile_lock = threading.Lock()

        # TUI-safe status message buffer (avoids print() inside Live context)
        self._status_messages: list[str] = []
        self._status_lock = threading.Lock()

        # Non-blocking profile flavor request queue
        self._profile_request: str | None = None

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

    def on_resize(self, event: ConsoleDimensions) -> None:
        self.width = event.width
        self.height = event.height

    def stop(self) -> None:
        """Stop the TUI loop gracefully."""
        self.running = False

    def build_layout(self) -> Layout:
        layout = Layout(name="main")
        layout.split_column(
            Layout(name="alerts", size=8),
            Layout(name="content", ratio=1),
        )

        if self.width >= 80:
            layout["content"].split_row(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1),
            )
        else:
            layout["content"].split_column(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1),
            )
        return layout

    def render(self) -> Layout:
        self._update_gpu_telemetry()
        layout = self.build_layout()

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
            layout["alerts"].update(
                Panel(Group(*alerts), title="System Alerts", border_style="yellow")
            )
        else:
            layout["alerts"].update(
                Panel(Text("No active alerts", style="dim"), border_style="dim")
            )

        if self.configs:
            cfg1 = self.configs[0]
            buffer1 = self.log_buffers[cfg1.alias]
            gpu1 = self.gpu_stats[0] if self.gpu_stats else None
            layout["left"].update(self._build_column_panel(cfg1, buffer1, gpu1))

        if len(self.configs) > 1:
            cfg2 = self.configs[1]
            buffer2 = self.log_buffers[cfg2.alias]
            gpu2 = self.gpu_stats[1] if len(self.gpu_stats) > 1 else None
            layout["right"].update(self._build_column_panel(cfg2, buffer2, gpu2))
        else:
            layout["right"].update(self._build_placeholder_panel())

        return layout

    def _build_status_panel(self, launch_result: LaunchResult) -> None:
        if launch_result.is_success():
            self.status_panel = None
            return

        status_text = Text()
        if launch_result.is_blocked():
            status_text.append(STATUS_PREFIX, style=STYLE_BOLD_RED)
            status_text.append("BLOCKED", style="bold red reverse")
            status_text.append("\n\n")
            if launch_result.errors is not None:
                status_text.append("FR-005 Error Details:\n", style=STYLE_BOLD_YELLOW)
                for error_detail in launch_result.errors.errors:
                    status_text.append(f"  - {error_detail.error_code}\n", style="red")
                    status_text.append(
                        f"    failed_check: {error_detail.failed_check}\n",
                        style="dim",
                    )
                    status_text.append(
                        f"    why_blocked: {error_detail.why_blocked}\n",
                        style="dim",
                    )
                    status_text.append(
                        f"    how_to_fix: {error_detail.how_to_fix}\n\n",
                        style="dim",
                    )
            self.status_panel = Panel(
                status_text,
                title="[red]Launch Failed[/red]",
                border_style="red",
            )
            return

        status_text.append(STATUS_PREFIX, style=STYLE_BOLD_YELLOW)
        status_text.append("DEGRADED", style=STYLE_BOLD_YELLOW)
        status_text.append(" (partial success)\n\n", style="dim")
        launched = launch_result.launched or []
        if launched:
            status_text.append("Launched slots:\n", style="bold green")
            for slot_id in launched:
                status_text.append(f"  + {slot_id}\n", style="green")
            status_text.append("\n")
        for warning in launch_result.warnings or []:
            status_text.append(f"  ! {warning}\n", style="yellow")
        self.status_panel = Panel(
            status_text,
            title="[yellow]Launch Degraded[/yellow]",
            border_style="yellow",
        )

    def _build_risk_panel_required(self, kind: str = "hardware") -> None:
        text = Text()
        text.append("RISK STATUS: ", style="bold")
        text.append(" ACKNOWLEDGEMENT REQUIRED ", style="bold red reverse")
        text.append("\nLaunch is blocked until you acknowledge risky operations.")
        self.risk_panel = Panel(text, title="Risk Management", border_style="red")
        self.risks_acknowledged = False
        self.active_risk_kind = kind

    def _build_risk_panel_acknowledged(self, kind: str = "hardware") -> None:
        text = Text()
        text.append("RISK STATUS: ", style="bold")
        text.append(" ACKNOWLEDGED ", style="bold green reverse")
        text.append("\nRisky operations (privileged ports, non-loopback bind) were acknowledged.")
        self.risk_panel = Panel(text, title="Risk Management", border_style="green")
        self.risks_acknowledged = True
        self.active_risk_kind = kind

    def _build_column_panel(
        self, cfg: ServerConfig, buffer: LogBuffer, gpu: GPUStats | None
    ) -> Panel:
        color_code = Colors.get_code(cfg.alias)
        color_style = color_code if color_code else "white"

        header = Text()
        header.append(f"{cfg.alias.upper()} ", style=f"bold {color_style}")
        header.append(f"http://{self.config.host}:{cfg.port}/v1", style="dim")
        header.append("\n")
        header.append(
            f"Device: {cfg.device} | Ctx: {cfg.ctx_size} | Threads: {cfg.threads}", style="cyan"
        )

        # Stale profile warning badge
        stale_warning = self._get_stale_warning(cfg)
        if stale_warning:
            header.append("\n")
            header.append(stale_warning, style="yellow")

        header.append("\n\n")

        logs_text = buffer.get_text(empty_message="Waiting for output...")
        logs = Panel(Text(logs_text), title="Logs", border_style="dim")
        gpu_renderable = (
            Panel(Text(gpu.format_stats_text()), title="GPU", border_style="yellow")
            if gpu is not None
            else Panel(Text("GPU stats unavailable", style="dim"), title="GPU", border_style="dim")
        )
        return Panel(Group(header, gpu_renderable, logs), border_style=color_style)

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
        self._stop_input_polling()

    # ------------------------------------------------------------------
    # Input polling infrastructure
    # ------------------------------------------------------------------

    def _start_input_polling(self) -> None:
        """Start a daemon thread that polls stdin for keypresses."""
        if self._input_thread is not None and self._input_thread.is_alive():
            return
        self._input_thread = threading.Thread(
            target=self._input_poller,
            name="tui-input-poller",
            daemon=True,
        )
        self._input_thread.start()

    def _stop_input_polling(self) -> None:
        """Stop the input polling thread."""
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
            self._input_thread = None

    def _poll_windows_keypress(self) -> None:
        """Poll for a single keypress on Windows via msvcrt."""
        import msvcrt  # type: ignore[import-not-found]

        if msvcrt.kbhit():  # type: ignore[attr-defined]
            ch = msvcrt.getch()  # type: ignore[attr-defined]
            # Handle Ctrl+C (0x03)
            if ch == b"\x03":
                self._keypress_queue.put("^C")
            elif ch and len(ch) == 1:
                self._keypress_queue.put(ch.decode("utf-8", errors="replace"))

    def _poll_posix_keypress(self) -> None:
        """Poll for a single keypress on POSIX systems via select."""
        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if ready:
            ch = sys.stdin.read(1)
            if ch:
                self._keypress_queue.put(ch)

    def _input_poller(self) -> None:
        """Daemon thread: poll stdin for single-character keypresses."""
        is_windows = os.name == "nt"
        poll_fn = self._poll_windows_keypress if is_windows else self._poll_posix_keypress
        with _cbreak_stdin():
            while self.running:
                try:
                    poll_fn()
                except Exception:
                    # Don't crash the TUI if stdin polling fails
                    time.sleep(0.1)

    def _process_keypresses(self) -> None:
        """Drain the keypress queue and handle relevant keys."""
        while not self._keypress_queue.empty():
            try:
                key = self._keypress_queue.get_nowait()
            except queue.Empty:
                break

            if self._profile_request is not None and key in {"1", "2", "3"}:
                alias = self._profile_request
                self._profile_request = None
                self._wait_for_flavor_selection(alias, preselected_key=key)
                continue

            # Handle Ctrl+C during profiling — abort the current profile
            if key == "^C":
                self._abort_profile()
            # Handle 'P' key — trigger profiling on the focused slot
            if key.upper() == "P" and self.configs:
                # Use the first config (left column = focused slot)
                cfg = self.configs[0]
                self._prompt_profile_flavor(cfg.alias)
                continue

            # Handle hardware warning keys (y/n/q) and VRAM risk keys (y/n)
            if self.risk_panel is not None:
                if key in ("y", "n"):
                    if self.active_risk_kind == "vram":
                        result = self.handle_vram_risk(key)
                    else:
                        result = self.handle_hardware_warning(key)
                    if result in ("proceed", "abort"):
                        self.active_risk_kind = None
                        continue
                elif key == "q" and self.active_risk_kind != "vram":
                    result = self.handle_hardware_warning(key)
                    if result in ("acknowledge", "abort", "quit"):
                        self.active_risk_kind = None
                        continue

        # Handle queued profile flavor request non-blockingly
        if self._profile_request is not None:
            alias = self._profile_request
            self._profile_request = None
            self._wait_for_flavor_selection(alias)

    def _prompt_profile_flavor(self, alias: str) -> None:
        """Queue a non-blocking profile flavor request for the given alias.

        The actual prompt is processed by _process_keypresses via the keypress
        queue so the TUI Live context is never blocked.
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
        and _process_keypresses will try again on the next TUI render cycle.
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

        This method is safe to call inside the Live context — it does NOT
        call print() or console.print().
        """
        with self._status_lock:
            self._status_messages.append(message)
            # Keep at most 5 messages
            if len(self._status_messages) > 5:
                self._status_messages.pop(0)

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
        from llama_cli.profile_cli import cmd_profile

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
            from llama_cli.profile_cli import _get_driver_version

            record, staleness = load_profile_with_staleness(
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

    def _build_profile_status_panel(self) -> Panel | None:
        """Build a panel showing active profile operations."""
        with self._profile_lock:
            active = {a: s for a, s in self._profile_status.items() if s != "idle"}
        if not active:
            return None

        text = Text()
        for alias, status in active.items():
            flavor = self._profile_flavor.get(alias, "unknown")
            if status == "running":
                text.append("\u25b6 ", style="yellow")
                text.append(f"Profiling {alias}: {flavor} ", style="yellow")
                text.append("[running...]", style="dim")
            elif status == "done":
                text.append("\u2713 ", style="green")
                text.append(f"Profile {alias}: {flavor} ", style="green")
                text.append("[done]", style="dim")
            elif status == "failed":
                text.append("\u2717 ", style="red")
                text.append(f"Profile {alias}: {flavor} ", style="red")
                text.append("[failed]", style="dim")
            text.append("\n")

        return Panel(text, title="Profile Status", border_style="yellow")

    def _build_status_messages_panel(self) -> Panel | None:
        """Build a panel showing TUI-safe status messages."""
        with self._status_lock:
            if not self._status_messages:
                return None
            messages = list(self._status_messages)
            self._status_messages.clear()

        text = Text()
        for msg in messages:
            text.append(msg + "\n", style="green")
        return Panel(text, title="Status", border_style="green")

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
            self._build_risk_panel_acknowledged()
            return
        self.risk_panel = None
        self.risks_acknowledged = False

    def _build_placeholder_panel(self) -> Panel:
        """Build a placeholder panel for the right column when only one config exists."""
        return Panel(
            Text("[dim]No secondary config[/dim]"),
            title="Status",
            border_style="dim",
        )

    def _build_slot_status_panel(self) -> Panel:
        """Build a panel showing per-slot status (health, logs, GPU stats, backend label)."""
        sections: list[Text] = []

        for cfg in self.configs:
            alias = cfg.alias
            state = self._slot_states.get(alias, SlotState.OFFLINE.value)

            status = state
            if state == SlotState.RUNNING.value:
                proc = self._server_processes.get(alias)
                if not proc or not (proc.pid and psutil.pid_exists(proc.pid)):
                    status = SlotState.CRASHED.value

            # Determine backend label
            backend_map: dict[str, str] = {
                "sycl": "SYCL",
                "cuda": "CUDA",
                "llama_cpp": "CPU",
            }
            backend_label = backend_map.get(cfg.backend, backend_map["llama_cpp"])

            # Color for status
            status_colors: dict[str, str] = {
                SlotState.RUNNING.value: "green",
                SlotState.LAUNCHING.value: "yellow",
                SlotState.DEGRADED.value: "yellow",
                SlotState.CRASHED.value: "red",
                SlotState.OFFLINE.value: "dim",
                SlotState.IDLE.value: "dim",
            }
            color = status_colors.get(status, "white")

            # Build section for this slot
            header = Text()
            header.append(f"[{alias}] ", style="bold")
            header.append(f"{status.upper()} ", style=color)
            header.append(f"| {backend_label} ", style="cyan")
            header.append(f"| http://{self.config.host}:{cfg.port}", style="dim")
            header.append("\n")

            # Log buffer preview (last 3 lines)
            buffer = self.log_buffers.get(alias)
            if buffer is not None:
                log_lines = buffer.get_lines()[-3:] if buffer.get_lines() else []
                log_text = "\n".join(log_lines) if log_lines else "  (no logs yet)"
                if log_text:
                    header.append(Text(log_text + "\n", style="dim"))

            sections.append(header)

        group = Group(*sections)
        return Panel(group, title="Slot Status", border_style="blue")

    def _update_gpu_telemetry(self) -> None:
        """Update GPU telemetry panel with latest stats."""
        if not self.gpu_stats:
            self.telemetry_panel = None
            return

        lines: list[str] = []
        for gpu in self.gpu_stats:
            gpu.update()  # Refresh stats from collector
            lines.append(gpu.format_stats_text())

        telemetry_text = Text("\n".join(lines))
        self.telemetry_panel = Panel(
            telemetry_text,
            title="GPU Telemetry",
            border_style="yellow",
        )

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
                status_text.append(STATUS_PREFIX, style=STYLE_BOLD_YELLOW)
                status_text.append("RETRYING", style=STYLE_BOLD_YELLOW)
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
                status_text.append(STATUS_PREFIX, style=STYLE_BOLD_RED)
                status_text.append("FAILED", style=STYLE_BOLD_RED)
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

        # Start input polling for keypresses
        self._start_input_polling()

        try:
            with Live(
                self.render(),
                screen=True,
                refresh_per_second=10,
                auto_refresh=False,
                vertical_overflow="ellipsis",
            ) as live:
                while self.running:
                    # Process any pending keypresses
                    self._process_keypresses()
                    time.sleep(0.1)
                    live.update(self.render(), refresh=True)
        finally:
            self._stop_input_polling()

        self._cleanup()
