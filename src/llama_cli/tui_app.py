# TUI application


import signal
import sys
import time

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from llama_manager import (
    Color,
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    ServerConfig,
    ServerManager,
)


class TUIApp:
    """Main TUI application with 2-column layout"""

    def __init__(
        self,
        configs: list[ServerConfig],
        gpu_indices: list[int],
        slots: list[ModelSlot] | None = None,
    ):
        self.configs = configs
        self.gpu_indices = gpu_indices
        self.slots = slots or []
        self.log_buffers: dict[str, LogBuffer] = {}
        self.gpu_stats: list[GPUStats] = []
        self.console = Console()
        self.running = True
        self.width = 80
        self.height = 24
        self.launch_result: LaunchResult | None = None
        self.status_panel: Panel | None = None

        # Initialize ServerManager for lifecycle management
        self.server_manager = ServerManager()

        # Initialize buffers and GPU stats
        for cfg in configs:
            self.log_buffers[cfg.alias] = LogBuffer(redact_sensitive=True)
        for idx in gpu_indices:
            self.gpu_stats.append(GPUStats(idx))

        # Setup signal handlers through ServerManager
        signal.signal(signal.SIGINT, self.server_manager.on_interrupt)
        signal.signal(signal.SIGTERM, self.server_manager.on_terminate)

    def start_servers(
        self,
        launch_result: LaunchResult | None = None,
    ) -> None:
        """Start all server processes with log buffering and status handling.

        Args:
            launch_result: Optional LaunchResult from slot-based launch (T020)
                          If provided, renders status panel for degraded/blocked states.
        """
        # Store launch result for status panel rendering (T021)
        self.launch_result = launch_result

        # Create log handlers that wrap LogBuffer.add_line
        log_handlers = {
            cfg.alias: lambda line, buf=buf: buf.add_line(line)
            for cfg, buf in zip(self.configs, self.log_buffers.values(), strict=True)
        }

        # Start servers via ServerManager with log handler delegation
        self.server_manager.start_servers(self.configs, log_handlers)

        # Build status panel for degraded/full-block states (T021)
        # Do NOT print directly while Rich Live is active - update layout instead
        if launch_result is not None:
            self._build_status_panel(launch_result)

    def on_resize(self, event) -> None:
        """Handle terminal resize events"""
        self.width = event.columns
        self.height = event.rows

    def build_layout(self) -> Layout:
        """Build dynamic layout based on terminal width and status panel (T021)"""
        layout = Layout(name="main")

        if self.status_panel is not None:
            # Status panel at top
            layout.split_column(
                Layout(name="status", ratio=1),
                Layout(name="content", ratio=3),
            )
            # Content area splits into columns
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
        else:
            # No status panel - standard layout
            if self.width >= 80:
                layout.split_row(
                    Layout(name="left", ratio=1),
                    Layout(name="right", ratio=1),
                )
            else:
                layout.split_column(
                    Layout(name="left", ratio=1),
                    Layout(name="right", ratio=1),
                )

        return layout

    def render(self) -> Layout:
        """Render the TUI layout with optional status panel (T021)"""
        layout = self.build_layout()

        # Render status panel if degraded/blocked (T021)
        if self.status_panel is not None:
            # Place status panel at top for visibility
            layout["status"].update(self.status_panel)

        # Render left column (first config)
        if self.configs:
            cfg1 = self.configs[0]
            buffer1 = self.log_buffers[cfg1.alias]
            gpu1 = self.gpu_stats[0] if len(self.gpu_stats) > 0 else None

            left_panel = self._build_column_panel(cfg1, buffer1, gpu1)
            layout["left"].update(left_panel)

        # Render right column (second config, if exists)
        if len(self.configs) > 1:
            cfg2 = self.configs[1]
            buffer2 = self.log_buffers[cfg2.alias]
            gpu2 = self.gpu_stats[1] if len(self.gpu_stats) > 1 else None

            right_panel = self._build_column_panel(cfg2, buffer2, gpu2)
            layout["right"].update(right_panel)

        return layout

    def _build_status_panel(self, launch_result: LaunchResult) -> None:
        """Build status panel for degraded/full-block states (T021).

        Renders consistent status panel updates without printing directly
        while Rich Live is active.

        Args:
            launch_result: LaunchResult from launch_all_slots()
        """
        if launch_result.is_success():
            # No status panel needed for success
            self.status_panel = None
            return

        # Build status panel content
        status_text = Text()

        if launch_result.is_blocked():
            # Full block - red border, critical styling
            status_text.append("STATUS: ", style="bold red")
            status_text.append("BLOCKED", style="bold red reverse")
            status_text.append("\n\n")

            if launch_result.errors is not None:
                status_text.append("FR-005 Error Details:\n", style="bold yellow")
                for error_detail in launch_result.errors.errors:
                    status_text.append(f"  • {error_detail.error_code}\n", style="red")
                    status_text.append(
                        f"    failed_check: {error_detail.failed_check}\n", style="dim"
                    )
                    status_text.append(
                        f"    why_blocked: {error_detail.why_blocked}\n", style="dim"
                    )
                    status_text.append(
                        f"    how_to_fix: {error_detail.how_to_fix}\n\n", style="dim"
                    )

            self.status_panel = Panel(
                status_text,
                title="[red]Launch Failed[/red]",
                border_style="red",
            )

        elif launch_result.is_degraded():
            # Degraded - yellow/orange border, warning styling
            status_text.append("STATUS: ", style="bold yellow")
            status_text.append("DEGRADED", style="bold yellow")
            status_text.append(" (partial success)\n\n", style="dim")

            # Launched slots
            launched = launch_result.launched or []
            if launched:
                status_text.append("Launched slots:\n", style="bold green")
                for slot_id in launched:
                    status_text.append(f"  ✓ {slot_id}\n", style="green")
                status_text.append("\n")

            # Blocked slots/warnings
            if launch_result.warnings:
                status_text.append("Blocked/Warning slots:\n", style="bold yellow")
                for warning in launch_result.warnings:
                    status_text.append(f"  ⚠ {warning}\n", style="yellow")

            self.status_panel = Panel(
                status_text,
                title="[yellow]Launch Degraded[/yellow]",
                border_style="yellow",
            )

    def _build_column_panel(
        self, cfg: ServerConfig, buffer: LogBuffer, gpu: GPUStats | None
    ) -> Panel:
        """Build a column panel for a single server"""
        color_code = Color.get_code(cfg.alias)
        color_style = color_code if color_code else "white"

        # Header with model name
        header_text = Text()
        header_text.append(f"  {cfg.alias.upper()}  ", style=f"bold {color_style}")
        header_text.append(f"  http://{Config().host}:{cfg.port}/v1", style="dim")

        # Config summary
        config_text = Text()
        config_text.append("Device: ", style="bold")
        config_text.append(cfg.device or "Auto", style="cyan")
        config_text.append(" | ")
        config_text.append("Ctx: ", style="bold")
        config_text.append(f"{cfg.ctx_size:,}", style="yellow")
        config_text.append(" | ")
        config_text.append("Threads: ", style="bold")
        config_text.append(f"{cfg.threads}", style="yellow")
        config_text.append(" | ")
        config_text.append("UBatch: ", style="bold")
        config_text.append(f"{cfg.ubatch_size}", style="yellow")

        # GPU stats panel
        gpu_panel = (
            gpu.get_rich_renderable()
            if gpu
            else Panel(
                Text("[dim]GPU stats unavailable[/]"),
                title="GPU Stats",
                border_style="dim",
            )
        )

        # Log buffer
        logs_panel = buffer.get_rich_renderable()

        # Combine all into a vertical layout using Columns
        vertical_content = Columns(
            [
                header_text,
                config_text,
                gpu_panel,
                logs_panel,
            ],
            expand=True,
        )

        return Panel(
            vertical_content,
            title="",
            border_style=color_style,
            padding=(1, 2),
        )

    def _cleanup(self) -> None:
        """Clean up all processes and resources via ServerManager"""
        self.server_manager.cleanup_servers()

    def run(self) -> None:
        """Run the TUI with optional slot-based launch (T020/T021)"""
        # Convert ServerConfig to ModelSlot for slot-based launch
        slots: list[ModelSlot] = []
        for cfg in self.configs:
            slot = ModelSlot(
                slot_id=cfg.alias,
                model_path=cfg.model,
                port=cfg.port,
            )
            slots.append(slot)

        # Launch slots with status handling (T020)
        launch_result = self.server_manager.launch_all_slots(slots)

        # Handle blocked status immediately (T020)
        if launch_result.is_blocked():
            # Print FR-005 details to stderr and exit
            print("error: launch blocked - no slots could be launched", file=sys.stderr)
            if launch_result.errors is not None:
                for error_detail in launch_result.errors.errors:
                    print(f"  {error_detail.error_code}", file=sys.stderr)
                    print(f"    failed_check: {error_detail.failed_check}", file=sys.stderr)
                    print(f"    why_blocked: {error_detail.why_blocked}", file=sys.stderr)
                    print(f"    how_to_fix: {error_detail.how_to_fix}", file=sys.stderr)
            sys.exit(1)

        # Handle degraded status with warnings (T020)
        if launch_result.is_degraded():
            print("warning: launch degraded - some slots blocked", file=sys.stderr)
            for warning in launch_result.warnings or []:
                print(f"  warning: {warning}", file=sys.stderr)

        # Store launch result for TUI status panel (T021)
        self.launch_result = launch_result
        self._build_status_panel(launch_result)

        # Start servers with log buffering
        log_handlers = {
            cfg.alias: lambda line, buf=buf: buf.add_line(line)
            for cfg, buf in zip(self.configs, self.log_buffers.values(), strict=True)
        }

        # Start servers via ServerManager with log handler delegation
        self.server_manager.start_servers(self.configs, log_handlers)

        # Run the live display
        with Live(
            self.render(),
            console=self.console,
            screen=True,
            refresh_per_second=10,
            auto_refresh=False,
            vertical_overflow="ellipsis",
        ) as live:
            while self.running:
                time.sleep(0.1)
                live.refresh()

        # Cleanup via ServerManager
        self._cleanup()
