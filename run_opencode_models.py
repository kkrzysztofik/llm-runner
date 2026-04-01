#!/usr/bin/env python3
"""
run_opencode_models.py - Manage multiple llama-server instances
"""

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class Config:
    """Server configuration defaults"""

    # Paths
    llama_cpp_root: str = "/home/kmk/src/llama.cpp"
    llama_server_bin_intel: str = f"{llama_cpp_root}/build/bin/llama-server"
    llama_server_bin_nvidia: str = f"{llama_cpp_root}/build_cuda/bin/llama-server"

    # Models
    model_summary_balanced: str = (
        "/home/kmk/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf"
    )
    model_summary_fast: str = (
        "/home/kmk/models/unsloth/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"
    )
    model_qwen35: str = (
        "/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
    )
    model_qwen35_both: str = (
        "/home/kmk/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
    )

    # Network
    host: str = "127.0.0.1"
    summary_balanced_port: int = 8080
    summary_fast_port: int = 8082
    qwen35_port: int = 8081

    # Model-specific defaults
    summary_balanced_chat_template_kwargs: str = '{"enable_thinking":false}'

    # Server defaults
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
# COLOR UTILITIES
# ============================================================


class Color:
    """ANSI color codes"""

    RESET = "\033[0m"
    BOLD = "\033[1m"

    COLORS: Dict[str, str] = {
        "summary-balanced": "\033[1;34m",  # Blue
        "summary-fast": "\033[1;33m",  # Yellow
        "qwen35-coding": "\033[1;32m",  # Green
    }

    @staticmethod
    def get_code(server_name: str) -> Optional[str]:
        return Color.COLORS.get(server_name)

    @staticmethod
    def is_enabled() -> bool:
        return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def prefix_output(server_name: str, line: str) -> str:
    """Format log line with timestamp and color"""
    timestamp = time.strftime("%H:%M:%S")
    color_code = Color.get_code(server_name)

    if Color.is_enabled() and color_code:
        return f"{color_code}{Color.BOLD}[%s][%s]\033[0m %s" % (
            timestamp,
            server_name,
            line,
        )
    return "[%s][%s] %s" % (timestamp, server_name, line)


# ============================================================
# SERVER COMMAND BUILDER
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
# VALIDATION
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
# SERVER PROCESS MANAGER
# ============================================================


class ServerManager:
    """Manages server processes"""

    def __init__(self):
        self.pids: List[int] = []
        self.shutting_down: bool = False
        self.servers: List[subprocess.Popen] = []

    def cleanup_servers(self) -> None:
        """Clean up all server processes"""
        if self.shutting_down:
            return
        self.shutting_down = True

        running_pids = []
        for pid in self.pids:
            try:
                os.kill(pid, 0)
                running_pids.append(pid)
            except OSError:
                pass

        if not running_pids:
            return

        print(
            f"warning: Sending TERM to {len(running_pids)} server(s)...",
            file=sys.stderr,
        )

        for pid in running_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        time.sleep(1)

        stubborn_pids = []
        for pid in running_pids:
            try:
                os.kill(pid, 0)
                stubborn_pids.append(pid)
            except OSError:
                pass

        if stubborn_pids:
            print(
                f"warning: Killing {len(stubborn_pids)} stubborn server(s)...",
                file=sys.stderr,
            )
            for pid in stubborn_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

        for proc in self.servers:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass

    def on_interrupt(self, signum, frame) -> None:
        """Handle SIGINT"""
        self.cleanup_servers()
        sys.exit(130)

    def on_terminate(self, signum, frame) -> None:
        """Handle SIGTERM"""
        self.cleanup_servers()
        sys.exit(143)

    def _stream_pipe(self, pipe, server_name: str, is_stderr: bool = False) -> None:
        if pipe is None:
            return
        for line in iter(pipe.readline, ""):
            formatted = prefix_output(server_name, line.rstrip("\n"))
            if is_stderr:
                print(formatted, file=sys.stderr, flush=True)
            else:
                print(formatted, flush=True)
        pipe.close()

    def start_server_background(
        self, server_name: str, cmd: List[str]
    ) -> subprocess.Popen:
        """Start a server in background with output redirection"""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        self.pids.append(proc.pid)
        self.servers.append(proc)

        threading.Thread(
            target=self._stream_pipe,
            args=(proc.stdout, server_name, False),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._stream_pipe,
            args=(proc.stderr, server_name, True),
            daemon=True,
        ).start()

        return proc

    def run_server_foreground(self, server_name: str, cmd: List[str]) -> int:
        proc = self.start_server_background(server_name, cmd)
        return proc.wait()

    def wait_for_any(self) -> int:
        while True:
            for proc in self.servers:
                code = proc.poll()
                if code is not None:
                    return code
            time.sleep(0.2)


# ============================================================
# SERVER STARTUP FUNCTIONS
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
# DRY RUN OUTPUT
# ============================================================


def dry_run(mode: str, primary_port: str = "", secondary_port: str = "") -> None:
    """Print command without executing"""
    cfg = Config()

    summary_balanced_port = (
        int(primary_port) if primary_port else cfg.summary_balanced_port
    )
    summary_fast_port = int(primary_port) if primary_port else cfg.summary_fast_port
    qwen35_port = int(primary_port) if primary_port else cfg.qwen35_port
    qwen35_port_both = int(secondary_port) if secondary_port else cfg.qwen35_port

    print("=== DRY RUN MODE ===")
    print(f"Mode: {mode}")
    print(f"llama-server (Intel): {cfg.llama_server_bin_intel}")
    print(f"llama-server (NVIDIA): {cfg.llama_server_bin_nvidia}")
    print(f"summary-balanced model: {cfg.model_summary_balanced}")
    print(f"summary-fast model: {cfg.model_summary_fast}")
    print(f"qwen35 model: {cfg.model_qwen35}")
    print(f"qwen35 both model: {cfg.model_qwen35_both}")
    print()

    if mode in ("summary-balanced", "llama32"):
        server_cfg = create_summary_balanced_cfg(summary_balanced_port)
        print("summary-balanced:")
        print(f"  Port: {summary_balanced_port}")
        print(f"  Device: SYCL0")
        print(f"  Context: {cfg.default_ctx_size_summary}")
        print(f"  Threads: {cfg.default_threads_summary_balanced}")
        print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
        print(f"  Reasoning: off")
        print(f"  Reasoning Format: deepseek")
        print(f"  Jinja: True")
        print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
        cmd = build_server_cmd(server_cfg)
        print(f"  Command: {' '.join(cmd)}")
        print()

    elif mode == "summary-fast":
        server_cfg = create_summary_fast_cfg(summary_fast_port)
        print("summary-fast:")
        print(f"  Port: {summary_fast_port}")
        print(f"  Device: SYCL0")
        print(f"  Context: {cfg.default_ctx_size_summary}")
        print(f"  Threads: {cfg.default_threads_summary_fast}")
        print(f"  UBatch: {cfg.default_ubatch_size_summary_fast}")
        cmd = build_server_cmd(server_cfg)
        print(f"  Command: {' '.join(cmd)}")
        print()

    elif mode == "qwen35":
        server_cfg = create_qwen35_cfg(
            qwen35_port,
            n_gpu_layers=cfg.default_n_gpu_layers_qwen35,
            model=cfg.model_qwen35,
            server_bin=cfg.llama_server_bin_nvidia,
        )
        print("qwen35:")
        print(f"  Port: {qwen35_port}")
        print(f"  Device: NVIDIA (CUDA)")
        print(f"  Context: {cfg.default_ctx_size_qwen35}")
        print(f"  Threads: {cfg.default_threads_qwen35}")
        print(f"  UBatch: {cfg.default_ubatch_size_qwen35}")
        print(
            f"  KV cache: {cfg.default_cache_type_qwen35_k}/{cfg.default_cache_type_qwen35_v}"
        )
        print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35}")
        cmd = build_server_cmd(server_cfg)
        print(f"  Command: {' '.join(cmd)}")
        print()

    elif mode == "both":
        server_cfg1 = create_summary_balanced_cfg(summary_balanced_port)
        server_cfg2 = create_qwen35_cfg(
            qwen35_port_both,
            ctx_size=cfg.default_ctx_size_both_qwen35,
            ubatch_size=cfg.default_ubatch_size_qwen35_both,
            threads=cfg.default_threads_qwen35_both,
            cache_k=cfg.default_cache_type_qwen35_both_k,
            cache_v=cfg.default_cache_type_qwen35_both_v,
            n_gpu_layers=cfg.default_n_gpu_layers_qwen35_both,
            model=cfg.model_qwen35_both,
            server_bin=cfg.llama_server_bin_nvidia,
        )
        print("summary-balanced:")
        print(f"  Port: {summary_balanced_port}")
        print(f"  Device: SYCL0")
        print(f"  Context: {cfg.default_ctx_size_both_summary}")
        print(f"  Threads: {cfg.default_threads_summary_balanced}")
        print(f"  UBatch: {cfg.default_ubatch_size_summary_balanced}")
        print(
            f"  KV cache: {cfg.default_cache_type_summary_k}/{cfg.default_cache_type_summary_v}"
        )
        print(f"  Reasoning: off")
        print(f"  Reasoning Format: deepseek")
        print(f"  Jinja: True")
        print(f"  Chat Template Kwargs: {cfg.summary_balanced_chat_template_kwargs}")
        cmd1 = build_server_cmd(server_cfg1)
        print(f"  Command: {' '.join(cmd1)}")
        print()
        print("qwen35:")
        print(f"  Port: {qwen35_port_both}")
        print(f"  Device: NVIDIA (CUDA)")
        print(f"  Context: {cfg.default_ctx_size_both_qwen35}")
        print(f"  Threads: {cfg.default_threads_qwen35_both}")
        print(f"  UBatch: {cfg.default_ubatch_size_qwen35_both}")
        print(
            f"  KV cache: {cfg.default_cache_type_qwen35_both_k}/{cfg.default_cache_type_qwen35_both_v}"
        )
        print(f"  n-gpu-layers: {cfg.default_n_gpu_layers_qwen35_both}")
        cmd2 = build_server_cmd(server_cfg2)
        print(f"  Command: {' '.join(cmd2)}")
        print()

    else:
        print(
            "error: dry-run requires a mode argument "
            "(summary-balanced|summary-fast|qwen35|both)",
            file=sys.stderr,
        )
        usage()
        sys.exit(1)


# ============================================================
# MAIN
# ============================================================


def usage() -> None:
    print("""Usage:
  run_opencode_models.py summary-balanced [port]
  run_opencode_models.py summary-fast [port]
  run_opencode_models.py qwen35 [port]
  run_opencode_models.py both [summary_balanced_port qwen35_port]
  run_opencode_models.py dry-run summary-balanced|summary-fast|qwen35|both [ports...]

Examples:
  run_opencode_models.py summary-balanced
  run_opencode_models.py summary-fast 8082
  run_opencode_models.py qwen35 8080
  run_opencode_models.py both 8080 8081
  run_opencode_models.py dry-run both""")


def check_prereqs() -> None:
    cfg = Config()
    require_executable(cfg.llama_server_bin_intel, "Intel llama-server")


def main() -> None:
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    mode = sys.argv[1]

    # Setup signal handlers
    manager = ServerManager()
    signal.signal(signal.SIGINT, manager.on_interrupt)
    signal.signal(signal.SIGTERM, manager.on_terminate)
    atexit.register(manager.cleanup_servers)

    # Initialize
    Color.is_enabled()
    check_prereqs()
    os.environ["ZES_ENABLE_SYSMAN"] = "1"

    # Handle dry-run
    if mode == "dry-run":
        if len(sys.argv) < 3:
            print("error: dry-run requires a mode argument", file=sys.stderr)
            usage()
            sys.exit(1)
        mode = sys.argv[2]
        primary_port = sys.argv[3] if len(sys.argv) > 3 else ""
        secondary_port = sys.argv[4] if len(sys.argv) > 4 else ""
        dry_run(mode, primary_port, secondary_port)
        sys.exit(0)

    # Parse and execute mode
    cfg = Config()

    try:
        if mode in ("summary-balanced", "llama32"):
            port = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.summary_balanced_port
            validate_port(port, "summary-balanced port")
            require_model(cfg.model_summary_balanced)
            print(f"Starting summary-balanced at http://{cfg.host}:{port}/v1")
            server_cfg = create_summary_balanced_cfg(port)
            cmd = build_server_cmd(server_cfg)
            sys.exit(manager.run_server_foreground("summary-balanced", cmd))

        elif mode == "summary-fast":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.summary_fast_port
            validate_port(port, "summary-fast port")
            require_model(cfg.model_summary_fast)
            print(f"Starting summary-fast at http://{cfg.host}:{port}/v1")
            server_cfg = create_summary_fast_cfg(port)
            cmd = build_server_cmd(server_cfg)
            sys.exit(manager.run_server_foreground("summary-fast", cmd))

        elif mode == "qwen35":
            port = int(sys.argv[2]) if len(sys.argv) > 2 else cfg.qwen35_port
            validate_port(port, "qwen35 port")
            require_model(cfg.model_qwen35)
            require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")
            print(
                f"Starting qwen35-coding at http://{cfg.host}:{port}/v1 (NVIDIA CUDA)"
            )
            server_cfg = create_qwen35_cfg(port)
            cmd = build_server_cmd(server_cfg)
            sys.exit(manager.run_server_foreground("qwen35-coding", cmd))

        elif mode == "both":
            port32 = (
                int(sys.argv[2]) if len(sys.argv) > 2 else cfg.summary_balanced_port
            )
            port35 = int(sys.argv[3]) if len(sys.argv) > 3 else cfg.qwen35_port
            validate_port(port32, "summary-balanced port")
            validate_port(port35, "qwen35 port")
            validate_ports(port32, port35, "summary-balanced port", "qwen35 port")
            require_model(cfg.model_summary_balanced)
            require_model(cfg.model_qwen35_both)
            require_executable(cfg.llama_server_bin_nvidia, "NVIDIA llama-server")
            server_cfg1 = create_summary_balanced_cfg(
                port32,
                ctx_size=cfg.default_ctx_size_both_summary,
                ubatch_size=cfg.default_ubatch_size_summary_balanced,
                threads=cfg.default_threads_summary_balanced,
                cache_k=cfg.default_cache_type_summary_k,
                cache_v=cfg.default_cache_type_summary_v,
            )
            server_cfg2 = create_qwen35_cfg(
                port35,
                ctx_size=cfg.default_ctx_size_both_qwen35,
                ubatch_size=cfg.default_ubatch_size_qwen35_both,
                threads=cfg.default_threads_qwen35_both,
                cache_k=cfg.default_cache_type_qwen35_both_k,
                cache_v=cfg.default_cache_type_qwen35_both_v,
                n_gpu_layers=cfg.default_n_gpu_layers_qwen35_both,
                model=cfg.model_qwen35_both,
                server_bin=cfg.llama_server_bin_nvidia,
            )
            cmd1 = build_server_cmd(server_cfg1)
            cmd2 = build_server_cmd(server_cfg2)
            print(f"summary-balanced: http://{cfg.host}:{port32}/v1")
            print(f"qwen35-coding: http://{cfg.host}:{port35}/v1")
            manager.start_server_background("summary-balanced", cmd1)
            manager.start_server_background("qwen35-coding", cmd2)
            code = manager.wait_for_any()
            manager.cleanup_servers()
            sys.exit(code)

        else:
            usage()
            sys.exit(1)

    except (ValueError, IndexError):
        usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
