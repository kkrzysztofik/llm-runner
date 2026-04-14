"""TUI application for llm-runner.

This module provides a Rich-based live terminal interface for managing
multiple llama-server instances with real-time log streaming, GPU stats,
and configuration display.
"""

import signal
import sys
import time
from collections.abc import Callable
from typing import Any

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
    ServerConfig,
    ServerManager,
)
from llama_manager.server import detect_risky_operations

RISK_ACK_LABEL = "warning_bypass"
RISK_CONFIRM_PROMPT = "Confirm risky operation [y/N]: "
STYLE_BOLD_YELLOW = "bold yellow"


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
        self.risk_panel: Panel | None = None
        self.risks_acknowledged: bool = False

        self.server_manager = ServerManager()

        for cfg in configs:
            self.log_buffers[cfg.alias] = LogBuffer(redact_sensitive=True)
        for idx in gpu_indices:
            # Pass a bound collector callable with the device index
            self.gpu_stats.append(GPUStats(idx, collector=self._make_collector(idx)))

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame: object | None) -> None:
        """Handle shutdown signals by stopping the TUI loop."""
        self.stop()

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
        layout = self.build_layout()

        alerts: list[Panel] = []
        if self.risk_panel is not None:
            alerts.append(self.risk_panel)
        if self.status_panel is not None:
            alerts.append(self.status_panel)

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
            status_text.append("STATUS: ", style="bold red")
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

        status_text.append("STATUS: ", style=STYLE_BOLD_YELLOW)
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

    def _build_risk_panel_required(self) -> None:
        text = Text()
        text.append("RISK STATUS: ", style="bold")
        text.append(" ACKNOWLEDGEMENT REQUIRED ", style="bold red reverse")
        text.append("\nLaunch is blocked until you acknowledge risky operations.")
        self.risk_panel = Panel(text, title="Risk Management", border_style="red")
        self.risks_acknowledged = False

    def _build_risk_panel_acknowledged(self) -> None:
        text = Text()
        text.append("RISK STATUS: ", style="bold")
        text.append(" ACKNOWLEDGED ", style="bold green reverse")
        text.append("\nRisky operations (privileged ports, non-loopback bind) were acknowledged.")
        self.risk_panel = Panel(text, title="Risk Management", border_style="green")
        self.risks_acknowledged = True

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

        if not acknowledged:
            self._build_risk_panel_required()
            print(f"warning: risky operation detected in {cfg.alias}: {risk}")
            try:
                response = input(RISK_CONFIRM_PROMPT).strip().lower()
            except EOFError:
                self._print_acknowledgement_required_and_exit()
            else:
                if response != "y":
                    self._print_acknowledgement_required_and_exit()

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

        log_handlers = {
            cfg.alias: lambda line, buf=buf: buf.add_line(line)
            for cfg, buf in zip(self.configs, self.log_buffers.values(), strict=True)
        }
        self.server_manager.start_servers(self.configs, log_handlers)

        with Live(
            self.render(),
            screen=True,
            refresh_per_second=10,
            auto_refresh=False,
            vertical_overflow="ellipsis",
        ) as live:
            while self.running:
                time.sleep(0.1)
                live.update(self.render(), refresh=True)

        self._cleanup()
