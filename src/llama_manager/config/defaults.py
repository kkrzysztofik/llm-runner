"""Config and SmokeProbeConfiguration dataclasses."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _default_llama_cpp_root() -> str:
    """Return the default llama.cpp checkout path under XDG cache."""
    if llama_cpp_root := os.environ.get("LLAMA_CPP_ROOT"):
        return llama_cpp_root
    xdg_cache_base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return str(Path(xdg_cache_base) / "llm-runner" / "llama.cpp")


@dataclass
class Config:
    """Server configuration defaults.

    Computed binary defaults are derived from llama_cpp_root:
    - llama_server_bin_intel: SYCL backend path (build/bin/llama-server)
    - llama_server_bin_nvidia: CUDA backend path (build_cuda/bin/llama-server)

    These defaults are used by build_server_cmd() when ServerConfig.server_bin
    is not explicitly provided.
    """

    # Paths
    llama_cpp_root: str = field(default_factory=_default_llama_cpp_root)
    llama_server_bin_intel: str = ""
    llama_server_bin_nvidia: str = ""

    def __post_init__(self) -> None:
        """Compute derived paths from llama_cpp_root after dataclass init."""
        if not self.llama_server_bin_intel:
            self.llama_server_bin_intel = str(
                Path(self.llama_cpp_root) / "build" / "bin" / "llama-server"
            )
        if not self.llama_server_bin_nvidia:
            self.llama_server_bin_nvidia = str(
                Path(self.llama_cpp_root) / "build_cuda" / "bin" / "llama-server"
            )

    # Models
    model_summary_balanced: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_SUMMARY_BALANCED", "models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-IQ4_XS.gguf"
        )
    )
    model_summary_fast: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_SUMMARY_FAST",
            "models/unsloth/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        )
    )
    model_qwen35: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_QWEN35", "models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf"
        )
    )
    model_qwen35_both: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL_QWEN35_BOTH",
            "models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_XS.gguf",
        )
    )

    # General models base directory (not tied to any profile)
    models_dir: str = field(
        default_factory=lambda: os.environ.get("MODELS_DIR", str(Path.home() / "models"))
    )

    # Network
    host: str = "127.0.0.1"
    summary_balanced_port: int = 8080
    summary_fast_port: int = 8082
    qwen35_port: int = 8081

    # Model-specific defaults
    summary_balanced_chat_template_kwargs: str = '{"enable_thinking":false}'
    summary_fast_chat_template_kwargs: str = '{"enable_thinking":false}'

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

    # M2 build setup XDG directories
    xdg_cache_base: str = field(
        default_factory=lambda: os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    )
    xdg_state_base: str = field(
        default_factory=lambda: os.environ.get(
            "XDG_STATE_HOME", str(Path.home() / ".local" / "state")
        )
    )
    xdg_data_base: str = field(
        default_factory=lambda: os.environ.get(
            "XDG_DATA_HOME", str(Path.home() / ".local" / "share")
        )
    )

    # M2 build pipeline configuration
    build_git_remote: str = "https://github.com/ggerganov/llama.cpp.git"
    build_git_branch: str = "master"
    build_retry_attempts: int = 3
    build_retry_delay: int = 5
    build_max_reports: int = 50
    build_output_truncate_bytes: int = 8192
    toolchain_timeout_seconds: int = 30

    # Profile management
    profile_staleness_days: int = 30
    server_binary_version: str = field(
        default_factory=lambda: os.environ.get("SERVER_BINARY_VERSION", "")
    )

    # Smoke probe configuration
    smoke_listen_timeout_s: int = 120
    smoke_http_request_timeout_s: int = 10
    smoke_inter_slot_delay_s: int = 2
    smoke_max_tokens: int = 16
    smoke_prompt: str = "Respond with exactly one word."
    smoke_skip_models_discovery: bool = False
    smoke_api_key: str = ""
    smoke_first_token_timeout_s: int = 1200
    smoke_total_chat_timeout_s: int = 1500

    # GGUF metadata extraction
    gguf_metadata_prefix_cap_bytes: int = 32 * 1024 * 1024  # 32 MiB
    gguf_metadata_parse_timeout_s: float = 5.0

    # TUI
    tui_launch_timeout_s: int = 120
    tui_refresh_interval_ms: int = 1000

    # Probe
    probe_latency_threshold_s: int = 10

    # Lockfile
    lock_stale_threshold_s: int = 300

    # M2 XDG path utilities
    @property
    def venv_path(self) -> Path:
        """Return the virtual environment path.

        Returns:
            Path to $XDG_CACHE_HOME/llm-runner/venv or ~/.cache/llm-runner/venv
        """
        return Path(self.xdg_cache_base) / "llm-runner" / "venv"

    @property
    def builds_dir(self) -> Path:
        """Return the builds directory path.

        Returns:
            Path to $XDG_STATE_HOME/llm-runner/builds
        """
        return Path(self.xdg_state_base) / "llm-runner" / "builds"

    @property
    def reports_dir(self) -> Path:
        """Return the reports directory path.

        Returns:
            Path to $XDG_DATA_BASE/llm-runner/reports
        """
        return Path(self.xdg_data_base) / "llm-runner" / "reports"

    @property
    def build_lock_path(self) -> Path:
        """Return the build lock file path.

        Returns:
            Path to $XDG_CACHE_BASE/llm-runner/.build.lock
        """
        return Path(self.xdg_cache_base) / "llm-runner" / ".build.lock"

    @property
    def profiles_dir(self) -> Path:
        """Return the profiles directory path.

        Returns:
            Path to $XDG_RUNTIME_DIR/llm-runner/profiles when XDG_RUNTIME_DIR
            is set to an absolute path; otherwise falls back to the system
            temporary directory at <tempdir>/llm-runner/profiles.
        """
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if not runtime_dir or not Path(runtime_dir).is_absolute():
            runtime_dir = tempfile.gettempdir()
        return Path(runtime_dir).resolve() / "llm-runner" / "profiles"


@dataclass
class SmokeProbeConfiguration:
    """Configuration for smoke probe phases.

    Translates Config defaults into probe-specific parameters used by
    probe_slot() and the smoke CLI entry point.
    """

    inter_slot_delay_s: int = 2
    listen_timeout_s: int = 120
    http_request_timeout_s: int = 10
    max_tokens: int = 16
    prompt: str = "Respond with exactly one word."
    skip_models_discovery: bool = False
    api_key: str = ""
    model_id_override: str | None = None
    first_token_timeout_s: int = 1200
    total_chat_timeout_s: int = 1500

    def __post_init__(self) -> None:
        """Validate smoke probe configuration parameters."""
        if not (8 <= self.max_tokens <= 32):
            raise ValueError("max_tokens must be between 8 and 32")
        if self.listen_timeout_s < 1:
            raise ValueError("listen_timeout_s must be at least 1")
        if self.http_request_timeout_s < 1:
            raise ValueError("http_request_timeout_s must be at least 1")
        if self.first_token_timeout_s < 1:
            raise ValueError("first_token_timeout_s must be at least 1")
        if self.total_chat_timeout_s < 1:
            raise ValueError("total_chat_timeout_s must be at least 1")
