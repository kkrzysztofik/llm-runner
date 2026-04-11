# TUI application


import signal
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
    LogBuffer,
    ServerConfig,
    ServerManager,
)


class TUIApp:
    """Main TUI application with 2-column layout"""

    def __init__(self, configs: list[ServerConfig], gpu_indices: list[int]):
        self.configs = configs
        self.gpu_indices = gpu_indices
        self.log_buffers: dict[str, LogBuffer] = {}
        self.gpu_stats: list[GPUStats] = []
        self.console = Console()
        self.running = True
        self.width = 80
        self.height = 24

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

    def start_servers(self) -> None:
        """Start all server processes with log buffering"""
        # Create log handlers that wrap LogBuffer.add_line
        log_handlers = {
            cfg.alias: lambda line, buf=buf: buf.add_line(line)
            for cfg, buf in zip(self.configs, self.log_buffers.values(), strict=True)
        }

        # Start servers via ServerManager with log handler delegation
        processes = self.server_manager.start_servers(self.configs, log_handlers)

        print(f"Started {len(processes)} server(s)")

    def on_resize(self, event) -> None:
        """Handle terminal resize events"""
        self.width = event.columns
        self.height = event.rows

    def build_layout(self) -> Layout:
        """Build dynamic layout based on terminal width"""
        layout = Layout(name="main")

        if self.width >= 80:
            # 2-column layout
            layout.split_row(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1),
            )
        else:
            # Single column layout
            layout.split_column(
                Layout(name="left", ratio=1),
                Layout(name="right", ratio=1),
            )

        return layout

    def render(self) -> Layout:
        """Render the TUI layout"""
        layout = self.build_layout()

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
        """Run the TUI"""
        # Start servers
        self.start_servers()

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
