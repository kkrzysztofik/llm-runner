#!/usr/bin/env python3
"""
run_models_tui.py - TUI for managing multiple llama-server instances
2-column layout with live logs, config, and GPU stats
"""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import psutil

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.panel import Panel as ScrollablePane


# ============================================================
# CONFIGURATION (copied from original script)
# ============================================================


@dataclass
class Config:
    """Server configuration defaults"""

    llama_cpp_root: str = "/home/kmk/src/llama.cpp"
    llama_server_bin_intel: str = f"{llama_cpp_root}/build/bin/llama-server"
    llama_server_bin_nvidia: str = f"{llama_cpp_root}/build_cuda/bin/llama-server"

    model_summary_balanced: str = (
        "/home/kmk/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf"
    )
    model_summary_fast: str = (
        "/home/kmk/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
    )
    model_qwen35: str = (
        "/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
    )
    model_qwen35_both: str = (
        "/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
    )

    host: str = "127.0.0.1"
    summary_balanced_port: int = 8080
    summary_fast_port: int = 8082
    qwen35_port: int = 8081

    summary_balanced_chat_template_kwargs: str = '{"enable_thinking":false}'

    default_n_gpu_layers: int = 99
    default_ctx_size_summary: int = 16144
    default_ctx_size_qwen35: int = 262144
    default_ctx_size_both_summary: int = 16144
    default_ctx_size_both_qwen35: int = 262144
    default_n_gpu_layers_qwen35: str = "all"
    default_n_gpu_layers_qwen35_both: str = "all"
    default_ubatch_size_summary_balanced: int = 1024
    default_ubatch_size_summary_fast: int = 512
    default_ubatch_size_qwen35: int = 1024
    default_ubatch_size_qwen35_both: int = 1024
    default_threads_summary_balanced: int = 8
    default_threads_summary_fast: int = 8
    default_threads_qwen35: int = 12
    default_threads_qwen35_both: int = 12
    default_cache_type_summary_k: str = "q8_0"
    default_cache_type_summary_v: str = "q8_0"
    default_cache_type_qwen35_k: str = "q8_0"
    default_cache_type_qwen35_v: str = "q8_0"
    default_cache_type_qwen35_both_k: str = "q8_0"
    default_cache_type_qwen35_both_v: str = "q8_0"


@dataclass
class ServerConfig:
    """Individual server configuration"""

    model: str
    alias: str
    device: str
    port: int
    ctx_size: int
    ubatch_size: int
    threads: int
    tensor_split: str = ""
    reasoning_mode: str = "auto"
    reasoning_format: str = "none"
    chat_template_kwargs: str = ""
    reasoning_budget: str = ""
    use_jinja: bool = False
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    n_gpu_layers: Union[int, str] = 99
    server_bin: str = ""


# ============================================================
# COLOR UTILITIES (copied from original script)
# ============================================================


class Color:
    """Rich-compatible color names"""

    COLORS: Dict[str, str] = {
        "summary-balanced": "blue",
        "summary-fast": "yellow",
        "qwen35-coding": "green",
    }

    @staticmethod
    def get_code(server_name: str) -> Optional[str]:
        return Color.COLORS.get(server_name)


def prefix_output(server_name: str, line: str) -> str:
    """Format log line with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    return "[%s][%s] %s" % (timestamp, server_name, line)


# ============================================================
# SERVER COMMAND BUILDER (copied from original script)
# ============================================================


def build_server_cmd(cfg: ServerConfig) -> List[str]:
    """Build llama-server command arguments"""
    cmd = [
        cfg.server_bin or Config().llama_server_bin_intel,
        "--model",
        cfg.model,
        "--alias",
        cfg.alias,
        "--n-gpu-layers",
        str(cfg.n_gpu_layers),
        "--split-mode",
        "layer",
        "--ctx-size",
        str(cfg.ctx_size),
        "--flash-attn",
        "on",
        "--cache-type-k",
        cfg.cache_type_k,
        "--cache-type-v",
        cfg.cache_type_v,
        "--batch-size",
        "2048",
        "--ubatch-size",
        str(cfg.ubatch_size),
        "--threads",
        str(cfg.threads),
        "--poll",
        "50",
        "--mmap",
        "--host",
        Config().host,
        "--port",
        str(cfg.port),
        "--no-webui",
    ]

    if cfg.device:
        cmd.extend(["--device", cfg.device])
    if cfg.reasoning_mode:
        cmd.extend(["--reasoning", cfg.reasoning_mode])
    if cfg.reasoning_format:
        cmd.extend(["--reasoning-format", cfg.reasoning_format])
    if cfg.tensor_split:
        cmd.extend(["--tensor-split", cfg.tensor_split])
    if cfg.chat_template_kwargs:
        cmd.extend(["--chat-template-kwargs", cfg.chat_template_kwargs])
    if cfg.reasoning_budget:
        cmd.extend(["--reasoning-budget", cfg.reasoning_budget])
    if cfg.use_jinja:
        cmd.append("--jinja")

    return cmd


# ============================================================
# VALIDATION (copied from original script)
# ============================================================


def validate_port(port: int, name: str = "port") -> None:
    """Validate port number"""
    if not isinstance(port, int) or port < 1 or port > 65535:
        print(
            f"error: {name} must be between 1 and 65535, got: {port}", file=sys.stderr
        )
        sys.exit(1)


def validate_threads(threads: int, name: str = "threads") -> None:
    """Validate thread count"""
    if not isinstance(threads, int) or threads < 1:
        print(f"error: {name} must be greater than 0, got: {threads}", file=sys.stderr)
        sys.exit(1)


def require_model(model_path: str) -> None:
    """Check if model file exists"""
    if not os.path.isfile(model_path):
        print(f"error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)


def require_executable(bin_path: str, name: str = "binary") -> None:
    """Check if executable exists"""
    if not os.access(bin_path, os.X_OK):
        print(f"error: {name} not found or not executable: {bin_path}", file=sys.stderr)
        sys.exit(1)


def validate_ports(
    port1: int, port2: int, name1: str = "port1", name2: str = "port2"
) -> None:
    """Validate ports are different"""
    if port1 == port2:
        print(
            f"error: {name1} and {name2} must be different, got: {port1}",
            file=sys.stderr,
        )
        sys.exit(1)


# ============================================================
# LOG BUFFER (thread-safe, scrollable)
# ============================================================


class LogBuffer:
    """Thread-safe log buffer with autoscroll support"""

    def __init__(self, max_lines: int = 50):
        self.lines: deque = deque(maxlen=max_lines)
        self.lock = threading.Lock()
        self.running = True
        self.auto_scroll = True

    def add_line(self, line: str) -> None:
        """Add a log line"""
        with self.lock:
            self.lines.append(line)

    def clear(self) -> None:
        """Clear all lines"""
        with self.lock:
            self.lines.clear()

    def stop(self) -> None:
        """Stop the buffer"""
        self.running = False

    def get_rich_renderable(self) -> Panel:
        """Get a Rich renderable for this buffer (Panel auto-scrolls)"""
        with self.lock:
            if not self.lines:
                text = Text("[dim]Waiting for output...[/]")
            else:
                text = Text("\n".join(self.lines))

        return Panel(text, title="Logs", border_style="dim")

    def get_stats(self) -> str:
        """Get buffer stats for display"""
        with self.lock:
            return f"[dim]{len(self.lines)} lines[/]"


# ============================================================
# GPU STATS COLLECTOR (nvtop + psutil)
# ============================================================


class GPUStats:
    """Collect GPU stats from nvtop and psutil"""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.stats: dict = {}
        self.last_update = 0
        self.update_interval = 0.5

    def update(self) -> None:
        """Update GPU stats from nvtop or psutil"""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        self.stats = self._get_nvtop_stats()
        self.last_update = current_time

    def _get_nvtop_stats(self) -> dict:
        """Get stats from nvtop JSON output"""
        try:
            result = subprocess.run(
                ["nvtop", "-s"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            all_gpus = json.loads(result.stdout)
            if self.device_index < len(all_gpus):
                gpu = all_gpus[self.device_index]
                return {
                    "device": gpu.get("device_name", "Unknown"),
                    "gpu_util": gpu.get("gpu_util", "N/A"),
                    "mem_util": gpu.get("mem_util", "N/A"),
                    "temp": gpu.get("temp", "N/A"),
                    "power": gpu.get("power_draw", "N/A"),
                }
        except Exception:
            pass

        # Fallback to psutil
        return {
            "device": f"GPU {self.device_index}",
            "cpu": f"{psutil.cpu_percent():.0f}%",
            "mem": f"{psutil.virtual_memory().percent:.0f}%",
        }

    def get_rich_renderable(self) -> Panel:
        """Get a panel with GPU stats"""
        self.update()

        stats_text = Text()
        stats_text.append("Device: ", style="bold")
        stats_text.append(self.stats.get("device", "N/A"), style="cyan")

        if "gpu_util" in self.stats:
            stats_text.append("GPU: ", style="bold")
            stats_text.append(str(self.stats.get("gpu_util", "N/A")), style="green")
            stats_text.append(" | Mem: ", style="bold")
            stats_text.append(str(self.stats.get("mem_util", "N/A")), style="yellow")

        if "temp" in self.stats:
            stats_text.append("\nTemp: ", style="bold")
            stats_text.append(str(self.stats.get("temp", "N/A")), style="red")

        if "power" in self.stats and self.stats["power"] != "N/A":
            stats_text.append("\nPower: ", style="bold")
            stats_text.append(str(self.stats["power"]), style="magenta")

        return Panel(
            stats_text,
            title="[bold yellow]GPU Stats[/]",
            border_style="yellow",
        )


# ============================================================
# TUI APPLICATION
# ============================================================


class TUIApp:
    """Main TUI application with 2-column layout"""

    def __init__(self, configs: List[ServerConfig], gpu_indices: List[int]):
        self.configs = configs
        self.gpu_indices = gpu_indices
        self.log_buffers: Dict[str, LogBuffer] = {}
        self.gpu_stats: List[GPUStats] = []
        self.console = Console()
        self.running = True
        self.processes: List[subprocess.Popen] = []
        self.threads: List[threading.Thread] = []
        self.pids: List[int] = []
        self.shutting_down = False
        self.width = 80
        self.height = 24

        # Initialize buffers and GPU stats
        for cfg in configs:
            self.log_buffers[cfg.alias] = LogBuffer()
        for idx in gpu_indices:
            self.gpu_stats.append(GPUStats(idx))

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)

    def _signal_handler(self, signum, frame) -> None:
        """Handle signals for graceful shutdown"""
        self._cleanup()
        sys.exit(130)

    def _cleanup(self) -> None:
        """Clean up all processes and resources"""
        if self.shutting_down:
            return
        self.shutting_down = True

        # Stop log buffers
        for buffer in self.log_buffers.values():
            buffer.stop()

        # Kill processes
        for proc in self.processes:
            try:
                proc.terminate()
            except Exception:
                pass

        time.sleep(0.5)

        for proc in self.processes:
            try:
                proc.kill()
            except Exception:
                pass

        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=1)

    def _read_log_output(self, pipe, server_name: str, is_stderr: bool = False) -> None:
        """Read log output from process and add to buffer"""
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                formatted = prefix_output(server_name, line.rstrip("\n"))
                self.log_buffers[server_name].add_line(formatted)
        except Exception:
            pass
        finally:
            pipe.close()

    def start_servers(self) -> None:
        """Start all server processes with log buffering"""
        for i, cfg in enumerate(self.configs):
            cmd = build_server_cmd(cfg)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.processes.append(proc)
            self.pids.append(proc.pid)

            # Start log reading threads
            threading.Thread(
                target=self._read_log_output,
                args=(proc.stdout, cfg.alias, False),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._read_log_output,
                args=(proc.stderr, cfg.alias, True),
                daemon=True,
            ).start()

            self.threads.extend(
                [t for t in threading.enumerate() if "read_log_output" in str(t)]
            )

            print(f"Started {cfg.alias} (PID {proc.pid})")

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
        self, cfg: ServerConfig, buffer: LogBuffer, gpu: Optional[GPUStats]
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
        from rich.columns import Columns

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

        # Cleanup
        self._cleanup()


# ============================================================
# SERVER CONFIG CREATION FUNCTIONS (copied from original script)
# ============================================================


def create_summary_balanced_cfg(
    port: int,
    ctx_size: Optional[int] = None,
    ubatch_size: Optional[int] = None,
    threads: Optional[int] = None,
    cache_k: Optional[str] = None,
    cache_v: Optional[str] = None,
) -> ServerConfig:
    cfg = Config()
    return ServerConfig(
        model=cfg.model_summary_balanced,
        alias="summary-balanced",
        device="SYCL0",
        port=port,
        ctx_size=ctx_size or cfg.default_ctx_size_summary,
        ubatch_size=ubatch_size or cfg.default_ubatch_size_summary_balanced,
        threads=threads or cfg.default_threads_summary_balanced,
        reasoning_mode="off",
        reasoning_format="deepseek",
        chat_template_kwargs=cfg.summary_balanced_chat_template_kwargs,
        use_jinja=True,
        cache_type_k=cache_k or cfg.default_cache_type_summary_k,
        cache_type_v=cache_v or cfg.default_cache_type_summary_v,
    )


def create_summary_fast_cfg(
    port: int,
    ctx_size: Optional[int] = None,
    ubatch_size: Optional[int] = None,
    threads: Optional[int] = None,
    cache_k: Optional[str] = None,
    cache_v: Optional[str] = None,
) -> ServerConfig:
    cfg = Config()
    return ServerConfig(
        model=cfg.model_summary_fast,
        alias="summary-fast",
        device="SYCL0",
        port=port,
        ctx_size=ctx_size or cfg.default_ctx_size_summary,
        ubatch_size=ubatch_size or cfg.default_ubatch_size_summary_fast,
        threads=threads or cfg.default_threads_summary_fast,
        cache_type_k=cache_k or cfg.default_cache_type_summary_k,
        cache_type_v=cache_v or cfg.default_cache_type_summary_v,
    )


def create_qwen35_cfg(
    port: int,
    ctx_size: Optional[int] = None,
    ubatch_size: Optional[int] = None,
    threads: Optional[int] = None,
    cache_k: Optional[str] = None,
    cache_v: Optional[str] = None,
    n_gpu_layers: Union[int, str] = "all",
    model: Optional[str] = None,
    server_bin: str = "",
) -> ServerConfig:
    cfg = Config()
    return ServerConfig(
        model=model or cfg.model_qwen35,
        alias="qwen35-coding",
        device="",
        port=port,
        ctx_size=ctx_size or cfg.default_ctx_size_qwen35,
        ubatch_size=ubatch_size or cfg.default_ubatch_size_qwen35,
        threads=threads or cfg.default_threads_qwen35,
        cache_type_k=cache_k or cfg.default_cache_type_qwen35_k,
        cache_type_v=cache_v or cfg.default_cache_type_qwen35_v,
        n_gpu_layers=n_gpu_layers,
        server_bin=server_bin or cfg.llama_server_bin_nvidia,
    )


# ============================================================
# MAIN
# ============================================================


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="TUI for managing multiple llama-server instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s both                     Run summary-balanced and qwen35 side-by-side
  %(prog)s both --port 8080 --port2 8081
  %(prog)s summary-balanced --port 8080
  %(prog)s qwen35 --port 8081
  %(prog)s summary-fast

GPU Mapping:
  - NVIDIA (CUDA) -> GPU 0 (RTX 3090)
  - Intel (SYCL)  -> GPU 1 (Arc B580)
        """,
    )

    modes = ["both", "summary-balanced", "summary-fast", "qwen35"]
    parser.add_argument(
        "mode",
        choices=modes,
        help=f"Mode to run: {' | '.join(modes)}",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port for primary model",
    )
    parser.add_argument(
        "--port2",
        "-p2",
        type=int,
        help="Port for secondary model",
    )

    return parser.parse_args()


def check_prereqs() -> None:
    """Check prerequisites"""
    cfg = Config()
    require_executable(cfg.llama_server_bin_intel, "Intel llama-server")
    if cfg.llama_server_bin_nvidia:
        require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")


def main():
    args = parse_args()

    # Check prerequisites
    check_prereqs()

    # Create configs based on mode
    configs = []
    gpu_indices = []

    if args.mode == "both":
        configs = [
            create_summary_balanced_cfg(args.port or 8080),
            create_qwen35_cfg(args.port2 or 8081),
        ]
        # GPU mapping: summary-balanced (SYCL) -> GPU 1, qwen35 (CUDA) -> GPU 0
        gpu_indices = [1, 0]
        print("Starting both models...")

    elif args.mode == "summary-balanced":
        configs = [create_summary_balanced_cfg(args.port or 8080)]
        gpu_indices = [1]  # SYCL -> GPU 1
        print("Starting summary-balanced...")

    elif args.mode == "summary-fast":
        configs = [create_summary_fast_cfg(args.port or 8082)]
        gpu_indices = [1]  # SYCL -> GPU 1
        print("Starting summary-fast...")

    elif args.mode == "qwen35":
        configs = [create_qwen35_cfg(args.port or 8081)]
        gpu_indices = [0]  # CUDA -> GPU 0
        print("Starting qwen35-coding...")

    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

    # Validate ports
    for cfg in configs:
        validate_port(cfg.port, cfg.alias)

    # Validate ports are different if multiple configs
    if len(configs) > 1:
        validate_ports(
            configs[0].port,
            configs[1].port,
            configs[0].alias + " port",
            configs[1].alias + " port",
        )

    # Validate models exist
    for cfg in configs:
        require_model(cfg.model)

    # Run TUI
    app = TUIApp(configs, gpu_indices)
    app.run()


if __name__ == "__main__":
    main()
