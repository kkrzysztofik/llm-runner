"""Config dataclasses — domain-focused sub-dataclasses composed by Config."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .spec_decode import SpeculativeDecodingConfig


def _default_llama_cpp_root() -> str:
    """Return the default llama.cpp checkout path under XDG cache."""
    if llama_cpp_root := os.environ.get("LLAMA_CPP_ROOT"):
        return llama_cpp_root
    xdg_cache_base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    return str(Path(xdg_cache_base) / "llm-runner" / "llama.cpp")


# ---------------------------------------------------------------------------
# Sub-dataclasses (domain-focused)
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """Filesystem paths: XDG bases, binary locations, model directories."""

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
    llama_cpp_root: str = field(default_factory=_default_llama_cpp_root)
    llama_server_bin_intel: str = ""
    llama_server_bin_nvidia: str = ""
    models_dir: str = field(
        default_factory=lambda: os.environ.get("MODELS_DIR", str(Path.home() / "models"))
    )

    def __post_init__(self) -> None:
        """Compute derived binary paths from llama_cpp_root."""
        if not self.llama_server_bin_intel:
            self.llama_server_bin_intel = str(
                Path(self.llama_cpp_root) / "build" / "bin" / "llama-server"
            )
        if not self.llama_server_bin_nvidia:
            self.llama_server_bin_nvidia = str(
                Path(self.llama_cpp_root) / "build_cuda" / "bin" / "llama-server"
            )

    @property
    def venv_path(self) -> Path:
        return Path(self.xdg_cache_base) / "llm-runner" / "venv"

    @property
    def builds_dir(self) -> Path:
        return Path(self.xdg_state_base) / "llm-runner" / "builds"

    @property
    def reports_dir(self) -> Path:
        return Path(self.xdg_data_base) / "llm-runner" / "reports"

    @property
    def build_lock_path(self) -> Path:
        return Path(self.xdg_cache_base) / "llm-runner" / ".build.lock"

    @property
    def profiles_dir(self) -> Path:
        data_base = Path(self.xdg_data_base).resolve()
        return data_base / "llm-runner" / "profiles"

    @property
    def logs_dir(self) -> Path:
        state_base = Path(self.xdg_state_base).resolve()
        return state_base / "llm-runner" / "logs"


@dataclass
class BuildPipelineConfig:
    """llama.cpp build pipeline settings."""

    git_remote: str = "https://github.com/ggerganov/llama.cpp.git"
    git_branch: str = "master"
    retry_attempts: int = 3
    retry_delay: int = 5
    max_reports: int = 50
    output_truncate_bytes: int = 8192
    args_default: str = ""
    toolchain_timeout_seconds: int = 30


@dataclass
class SmokeConfig:
    """Smoke probe configuration."""

    listen_timeout_s: int = 120
    http_request_timeout_s: int = 10
    inter_slot_delay_s: int = 2
    max_tokens: int = 16
    prompt: str = "Respond with exactly one word."
    skip_models_discovery: bool = False
    api_key: str = ""
    first_token_timeout_s: int = 1200
    total_chat_timeout_s: int = 1500


@dataclass
class ServerDefaultsConfig:
    """Server defaults: per-profile (builtin) + template (new profiles)."""

    # -- Per-profile defaults (builtin profiles: summary-balanced, qwen35, etc.) --
    n_gpu_layers: int = 99
    ctx_size_summary: int = 16144
    ctx_size_qwen35: int = 262144
    ctx_size_both_summary: int = 16144
    ctx_size_both_qwen35: int = 262144
    n_gpu_layers_qwen35: str = "all"
    n_gpu_layers_qwen35_both: str = "all"
    ubatch_size_summary_balanced: int = 1024
    ubatch_size_summary_fast: int = 512
    ubatch_size_qwen35: int = 1024
    ubatch_size_qwen35_both: int = 1024
    threads_summary_balanced: int = 8
    threads_summary_fast: int = 8
    threads_qwen35: int = 12
    threads_qwen35_both: int = 12
    cache_type_summary_k: str = "q8_0"
    cache_type_summary_v: str = "q8_0"
    cache_type_qwen35_k: str = "q8_0"
    cache_type_qwen35_v: str = "q8_0"
    cache_type_qwen35_both_k: str = "q8_0"
    cache_type_qwen35_both_v: str = "q8_0"

    # -- Template defaults (new custom profiles, Config modal prefill) --
    port: int = 8080
    ctx_size: int = 4096
    ubatch_size: int = 512
    threads: int = 8
    n_gpu_layers_profile: str = "all"
    bind_address: str = "127.0.0.1"
    batch_size: int = 2048
    poll_ms: int = 50
    n_predict: int = 32768
    parallel: int = 4
    threads_batch: int = 0
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    reasoning_mode: str = "auto"
    reasoning_format: str = "none"
    reasoning_budget: str = ""
    use_jinja: bool = False
    chat_template_kwargs: str = "{}"
    mmproj: str = ""
    spec_type: str = ""
    spec_ngram_size_n: int = 24
    draft_min: int = 48
    draft_max: int = 64
    spec_draft_n_max: int = 0
    spec_draft_p_min: float = 0.0
    spec_draft_cache_type_k: str = ""
    spec_draft_cache_type_v: str = ""
    spec_draft_device: str = ""

    @property
    def spec_decode(self) -> SpeculativeDecodingConfig:
        return SpeculativeDecodingConfig(
            spec_type=self.spec_type,
            spec_ngram_size_n=self.spec_ngram_size_n,
            draft_min=self.draft_min,
            draft_max=self.draft_max,
            spec_draft_n_max=self.spec_draft_n_max,
            spec_draft_p_min=self.spec_draft_p_min,
            spec_draft_cache_type_k=self.spec_draft_cache_type_k,
            spec_draft_cache_type_v=self.spec_draft_cache_type_v,
            spec_draft_device=self.spec_draft_device,
            reasoning_mode=self.reasoning_mode,
            reasoning_format=self.reasoning_format,
            reasoning_budget=self.reasoning_budget,
        )

    @spec_decode.setter
    def spec_decode(self, value: SpeculativeDecodingConfig) -> None:
        self.spec_type = value.spec_type
        self.spec_ngram_size_n = value.spec_ngram_size_n
        self.draft_min = value.draft_min
        self.draft_max = value.draft_max
        self.spec_draft_n_max = value.spec_draft_n_max
        self.spec_draft_p_min = value.spec_draft_p_min
        self.spec_draft_cache_type_k = value.spec_draft_cache_type_k
        self.spec_draft_cache_type_v = value.spec_draft_cache_type_v
        self.spec_draft_device = value.spec_draft_device
        self.reasoning_mode = value.reasoning_mode
        self.reasoning_format = value.reasoning_format
        self.reasoning_budget = value.reasoning_budget


# ---------------------------------------------------------------------------
# DeploymentConfig — model paths, network, and per-model chat templates
# ---------------------------------------------------------------------------


@dataclass
class DeploymentConfig:
    """Model paths, network ports, and per-model chat template overrides."""

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
    host: str = "127.0.0.1"
    summary_balanced_port: int = 8080
    summary_fast_port: int = 8082
    qwen35_port: int = 8081
    summary_balanced_chat_template_kwargs: str = '{"enable_thinking":false}'
    summary_fast_chat_template_kwargs: str = '{"enable_thinking":false}'


# ---------------------------------------------------------------------------
# Main Config — composes sub-dataclasses + keeps miscellaneous fields
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Server configuration defaults.

    Composes domain-focused sub-dataclasses:
    - ``paths``: XDG bases, binary paths, model directories
    - ``build``: llama.cpp build pipeline settings
    - ``smoke``: smoke probe parameters
    - ``server_defaults``: template defaults for new profiles
    - ``deployment``: model paths, network ports, and chat template overrides

    Top-level fields are kept for misc settings that don't fit a domain.
    """

    # -- Sub-dataclasses (source of truth) --
    paths: PathsConfig = field(default_factory=PathsConfig)
    build: BuildPipelineConfig = field(default_factory=BuildPipelineConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    server_defaults: ServerDefaultsConfig = field(default_factory=ServerDefaultsConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

    # -- Top-level: misc --
    profile_staleness_days: int = 30
    server_binary_version: str = field(
        default_factory=lambda: os.environ.get("SERVER_BINARY_VERSION", "")
    )
    gguf_metadata_prefix_cap_bytes: int = 32 * 1024 * 1024  # 32 MiB
    gguf_metadata_parse_timeout_s: float = 60.0
    tui_launch_timeout_s: int = 120
    tui_refresh_interval_ms: int = 1000
    probe_latency_threshold_s: int = 10
    lock_stale_threshold_s: int = 300
    log_file_level: str = "DEBUG"
    log_stderr_level: str = "INFO"


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
