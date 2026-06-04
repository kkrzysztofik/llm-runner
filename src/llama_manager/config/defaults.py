"""Config dataclasses — domain-focused sub-dataclasses composed by Config."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# Main Config — composes sub-dataclasses + keeps frequently-accessed fields
# ---------------------------------------------------------------------------


# Legacy flat field name -> (sub-dataclass field on Config, attr on sub-dataclass)
_CONFIG_LEGACY_ROUTE: dict[str, tuple[str, str]] = {
    "llama_cpp_root": ("paths", "llama_cpp_root"),
    "models_dir": ("paths", "models_dir"),
    "llama_server_bin_intel": ("paths", "llama_server_bin_intel"),
    "llama_server_bin_nvidia": ("paths", "llama_server_bin_nvidia"),
    "build_git_remote": ("build", "git_remote"),
    "build_git_branch": ("build", "git_branch"),
    "smoke_listen_timeout_s": ("smoke", "listen_timeout_s"),
    "smoke_http_request_timeout_s": ("smoke", "http_request_timeout_s"),
    "smoke_first_token_timeout_s": ("smoke", "first_token_timeout_s"),
    "smoke_total_chat_timeout_s": ("smoke", "total_chat_timeout_s"),
    "smoke_inter_slot_delay_s": ("smoke", "inter_slot_delay_s"),
    "smoke_max_tokens": ("smoke", "max_tokens"),
    "smoke_prompt": ("smoke", "prompt"),
    "smoke_skip_models_discovery": ("smoke", "skip_models_discovery"),
    "smoke_api_key": ("smoke", "api_key"),
    "default_profile_port": ("server_defaults", "port"),
    "default_profile_ctx_size": ("server_defaults", "ctx_size"),
    "default_profile_ubatch_size": ("server_defaults", "ubatch_size"),
    "default_profile_threads": ("server_defaults", "threads"),
    "default_profile_n_gpu_layers": ("server_defaults", "n_gpu_layers"),
    "default_bind_address": ("server_defaults", "bind_address"),
    "default_batch_size": ("server_defaults", "batch_size"),
    "default_poll_ms": ("server_defaults", "poll_ms"),
    "default_n_predict": ("server_defaults", "n_predict"),
    "default_parallel": ("server_defaults", "parallel"),
    "default_threads_batch": ("server_defaults", "threads_batch"),
    "default_profile_cache_type_k": ("server_defaults", "cache_type_k"),
    "default_profile_cache_type_v": ("server_defaults", "cache_type_v"),
    "default_use_jinja": ("server_defaults", "use_jinja"),
    "default_profile_chat_template_kwargs": ("server_defaults", "chat_template_kwargs"),
    "default_mmproj": ("server_defaults", "mmproj"),
    "default_spec_type": ("server_defaults", "spec_type"),
    "default_spec_ngram_size_n": ("server_defaults", "spec_ngram_size_n"),
    "default_draft_min": ("server_defaults", "draft_min"),
    "default_draft_max": ("server_defaults", "draft_max"),
    "default_spec_draft_n_max": ("server_defaults", "spec_draft_n_max"),
    "default_spec_draft_p_min": ("server_defaults", "spec_draft_p_min"),
    "default_spec_draft_cache_type_k": ("server_defaults", "spec_draft_cache_type_k"),
    "default_spec_draft_cache_type_v": ("server_defaults", "spec_draft_cache_type_v"),
    "default_spec_draft_device": ("server_defaults", "spec_draft_device"),
    "default_reasoning_mode": ("server_defaults", "reasoning_mode"),
    "default_reasoning_format": ("server_defaults", "reasoning_format"),
    "default_reasoning_budget": ("server_defaults", "reasoning_budget"),
}


@dataclass
class Config:
    """Server configuration defaults.

    Composes domain-focused sub-dataclasses:
    - ``paths``: XDG bases, binary paths, model directories
    - ``build``: llama.cpp build pipeline settings
    - ``smoke``: smoke probe parameters
    - ``server_defaults``: template defaults for new profiles

    Top-level fields are kept for model paths, network, and misc settings
    that are frequently accessed across the codebase.

    For backward compatibility, the old flat field names are accepted as
    optional kwargs and routed to the appropriate sub-dataclass.
    """

    # -- Sub-dataclasses (source of truth) --
    paths: PathsConfig = field(default_factory=PathsConfig)
    build: BuildPipelineConfig = field(default_factory=BuildPipelineConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    server_defaults: ServerDefaultsConfig = field(default_factory=ServerDefaultsConfig)

    # -- Top-level: model paths --
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

    # -- Top-level: network --
    host: str = "127.0.0.1"
    summary_balanced_port: int = 8080
    summary_fast_port: int = 8082
    qwen35_port: int = 8081

    # -- Top-level: model-specific defaults --
    summary_balanced_chat_template_kwargs: str = '{"enable_thinking":false}'
    summary_fast_chat_template_kwargs: str = '{"enable_thinking":false}'

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

    def __init__(self, **kwargs: Any) -> None:
        """Initialise Config, routing legacy flat kwargs to sub-dataclasses.

        Accepts both the new sub-dataclass kwargs (``paths``, ``build``, etc.)
        and the old flat field names (``llama_cpp_root``, ``smoke_listen_timeout_s``,
        etc.). Old flat names are routed into the corresponding sub-dataclass.
        """
        import dataclasses

        # Extract legacy flat fields and group by target sub-dataclass
        paths_kw: dict[str, Any] = {}
        build_kw: dict[str, Any] = {}
        smoke_kw: dict[str, Any] = {}
        sd_kw: dict[str, Any] = {}

        for key in list(kwargs):
            route = _CONFIG_LEGACY_ROUTE.get(key)
            if route is None:
                continue
            sub_field, attr = route
            val = kwargs.pop(key)
            if sub_field == "paths":
                paths_kw[attr] = val
            elif sub_field == "build":
                build_kw[attr] = val
            elif sub_field == "smoke":
                smoke_kw[attr] = val
            elif sub_field == "server_defaults":
                sd_kw[attr] = val

        # Build sub-dataclass instances from legacy kwargs
        if "paths" not in kwargs and paths_kw:
            kwargs["paths"] = PathsConfig(**paths_kw)
        if "build" not in kwargs and build_kw:
            kwargs["build"] = BuildPipelineConfig(**build_kw)
        if "smoke" not in kwargs and smoke_kw:
            kwargs["smoke"] = SmokeConfig(**smoke_kw)
        if "server_defaults" not in kwargs and sd_kw:
            kwargs["server_defaults"] = ServerDefaultsConfig(**sd_kw)

        # Handle default_spec_decode (SpeculativeDecodingConfig from persistence)
        if "default_spec_decode" in kwargs:
            from .spec_decode import SpeculativeDecodingConfig

            spec_cfg = kwargs.pop("default_spec_decode")
            if isinstance(spec_cfg, SpeculativeDecodingConfig):
                for fld_name in spec_cfg.__dataclass_fields__:  # type: ignore[attr-defined]
                    sd_kw[fld_name] = getattr(spec_cfg, fld_name)
                if "server_defaults" not in kwargs:
                    kwargs["server_defaults"] = ServerDefaultsConfig(**sd_kw)
                else:
                    for k, v in sd_kw.items():
                        setattr(kwargs["server_defaults"], k, v)

        # Initialize all dataclass fields manually
        for fld in dataclasses.fields(self.__class__):
            if fld.name in kwargs:
                object.__setattr__(self, fld.name, kwargs[fld.name])
            elif fld.default is not dataclasses.MISSING:
                object.__setattr__(self, fld.name, fld.default)
            elif fld.default_factory is not dataclasses.MISSING:
                object.__setattr__(self, fld.name, fld.default_factory())

        # Run __post_init__ on paths to compute derived binary paths
        if hasattr(self.paths, "__post_init__"):
            self.paths.__post_init__()

    # -----------------------------------------------------------------------
    # Backward-compatible properties for fields moved into sub-dataclasses
    # -----------------------------------------------------------------------

    # PathsConfig
    @property
    def xdg_cache_base(self) -> str:
        return self.paths.xdg_cache_base

    @property
    def xdg_state_base(self) -> str:
        return self.paths.xdg_state_base

    @property
    def xdg_data_base(self) -> str:
        return self.paths.xdg_data_base

    @property
    def llama_cpp_root(self) -> str:
        return self.paths.llama_cpp_root

    @llama_cpp_root.setter
    def llama_cpp_root(self, value: str) -> None:
        self.paths.llama_cpp_root = value
        self.paths.__post_init__()

    @property
    def llama_server_bin_intel(self) -> str:
        return self.paths.llama_server_bin_intel

    @llama_server_bin_intel.setter
    def llama_server_bin_intel(self, value: str) -> None:
        self.paths.llama_server_bin_intel = value

    @property
    def llama_server_bin_nvidia(self) -> str:
        return self.paths.llama_server_bin_nvidia

    @llama_server_bin_nvidia.setter
    def llama_server_bin_nvidia(self, value: str) -> None:
        self.paths.llama_server_bin_nvidia = value

    @property
    def models_dir(self) -> str:
        return self.paths.models_dir

    @models_dir.setter
    def models_dir(self, value: str) -> None:
        self.paths.models_dir = value

    # PathsConfig derived properties
    @property
    def venv_path(self) -> Path:
        return self.paths.venv_path

    @property
    def builds_dir(self) -> Path:
        return self.paths.builds_dir

    @property
    def reports_dir(self) -> Path:
        return self.paths.reports_dir

    @property
    def build_lock_path(self) -> Path:
        return self.paths.build_lock_path

    @property
    def profiles_dir(self) -> Path:
        return self.paths.profiles_dir

    @property
    def logs_dir(self) -> Path:
        return self.paths.logs_dir

    # BuildPipelineConfig
    @property
    def build_git_remote(self) -> str:
        return self.build.git_remote

    @build_git_remote.setter
    def build_git_remote(self, value: str) -> None:
        self.build.git_remote = value

    @property
    def build_git_branch(self) -> str:
        return self.build.git_branch

    @build_git_branch.setter
    def build_git_branch(self, value: str) -> None:
        self.build.git_branch = value

    @property
    def build_retry_attempts(self) -> int:
        return self.build.retry_attempts

    @build_retry_attempts.setter
    def build_retry_attempts(self, value: int) -> None:
        self.build.retry_attempts = value

    @property
    def build_retry_delay(self) -> int:
        return self.build.retry_delay

    @build_retry_delay.setter
    def build_retry_delay(self, value: int) -> None:
        self.build.retry_delay = value

    @property
    def build_max_reports(self) -> int:
        return self.build.max_reports

    @build_max_reports.setter
    def build_max_reports(self, value: int) -> None:
        self.build.max_reports = value

    @property
    def build_output_truncate_bytes(self) -> int:
        return self.build.output_truncate_bytes

    @build_output_truncate_bytes.setter
    def build_output_truncate_bytes(self, value: int) -> None:
        self.build.output_truncate_bytes = value

    @property
    def build_args_default(self) -> str:
        return self.build.args_default

    @build_args_default.setter
    def build_args_default(self, value: str) -> None:
        self.build.args_default = value

    @property
    def toolchain_timeout_seconds(self) -> int:
        return self.build.toolchain_timeout_seconds

    @toolchain_timeout_seconds.setter
    def toolchain_timeout_seconds(self, value: int) -> None:
        self.build.toolchain_timeout_seconds = value

    # SmokeConfig
    @property
    def smoke_listen_timeout_s(self) -> int:
        return self.smoke.listen_timeout_s

    @smoke_listen_timeout_s.setter
    def smoke_listen_timeout_s(self, value: int) -> None:
        self.smoke.listen_timeout_s = value

    @property
    def smoke_http_request_timeout_s(self) -> int:
        return self.smoke.http_request_timeout_s

    @smoke_http_request_timeout_s.setter
    def smoke_http_request_timeout_s(self, value: int) -> None:
        self.smoke.http_request_timeout_s = value

    @property
    def smoke_inter_slot_delay_s(self) -> int:
        return self.smoke.inter_slot_delay_s

    @smoke_inter_slot_delay_s.setter
    def smoke_inter_slot_delay_s(self, value: int) -> None:
        self.smoke.inter_slot_delay_s = value

    @property
    def smoke_max_tokens(self) -> int:
        return self.smoke.max_tokens

    @smoke_max_tokens.setter
    def smoke_max_tokens(self, value: int) -> None:
        self.smoke.max_tokens = value

    @property
    def smoke_prompt(self) -> str:
        return self.smoke.prompt

    @smoke_prompt.setter
    def smoke_prompt(self, value: str) -> None:
        self.smoke.prompt = value

    @property
    def smoke_skip_models_discovery(self) -> bool:
        return self.smoke.skip_models_discovery

    @smoke_skip_models_discovery.setter
    def smoke_skip_models_discovery(self, value: bool) -> None:
        self.smoke.skip_models_discovery = value

    @property
    def smoke_api_key(self) -> str:
        return self.smoke.api_key

    @smoke_api_key.setter
    def smoke_api_key(self, value: str) -> None:
        self.smoke.api_key = value

    @property
    def smoke_first_token_timeout_s(self) -> int:
        return self.smoke.first_token_timeout_s

    @smoke_first_token_timeout_s.setter
    def smoke_first_token_timeout_s(self, value: int) -> None:
        self.smoke.first_token_timeout_s = value

    @property
    def smoke_total_chat_timeout_s(self) -> int:
        return self.smoke.total_chat_timeout_s

    @smoke_total_chat_timeout_s.setter
    def smoke_total_chat_timeout_s(self, value: int) -> None:
        self.smoke.total_chat_timeout_s = value

    # ServerDefaultsConfig
    @property
    def default_profile_port(self) -> int:
        return self.server_defaults.port

    @default_profile_port.setter
    def default_profile_port(self, value: int) -> None:
        self.server_defaults.port = value

    @property
    def default_profile_ctx_size(self) -> int:
        return self.server_defaults.ctx_size

    @default_profile_ctx_size.setter
    def default_profile_ctx_size(self, value: int) -> None:
        self.server_defaults.ctx_size = value

    @property
    def default_profile_ubatch_size(self) -> int:
        return self.server_defaults.ubatch_size

    @default_profile_ubatch_size.setter
    def default_profile_ubatch_size(self, value: int) -> None:
        self.server_defaults.ubatch_size = value

    @property
    def default_profile_threads(self) -> int:
        return self.server_defaults.threads

    @default_profile_threads.setter
    def default_profile_threads(self, value: int) -> None:
        self.server_defaults.threads = value

    @property
    def default_profile_n_gpu_layers(self) -> str:
        return self.server_defaults.n_gpu_layers_profile

    @default_profile_n_gpu_layers.setter
    def default_profile_n_gpu_layers(self, value: str) -> None:
        self.server_defaults.n_gpu_layers_profile = value

    @property
    def default_bind_address(self) -> str:
        return self.server_defaults.bind_address

    @default_bind_address.setter
    def default_bind_address(self, value: str) -> None:
        self.server_defaults.bind_address = value

    @property
    def default_batch_size(self) -> int:
        return self.server_defaults.batch_size

    @default_batch_size.setter
    def default_batch_size(self, value: int) -> None:
        self.server_defaults.batch_size = value

    @property
    def default_poll_ms(self) -> int:
        return self.server_defaults.poll_ms

    @default_poll_ms.setter
    def default_poll_ms(self, value: int) -> None:
        self.server_defaults.poll_ms = value

    @property
    def default_n_predict(self) -> int:
        return self.server_defaults.n_predict

    @default_n_predict.setter
    def default_n_predict(self, value: int) -> None:
        self.server_defaults.n_predict = value

    @property
    def default_parallel(self) -> int:
        return self.server_defaults.parallel

    @default_parallel.setter
    def default_parallel(self, value: int) -> None:
        self.server_defaults.parallel = value

    @property
    def default_threads_batch(self) -> int:
        return self.server_defaults.threads_batch

    @default_threads_batch.setter
    def default_threads_batch(self, value: int) -> None:
        self.server_defaults.threads_batch = value

    @property
    def default_profile_cache_type_k(self) -> str:
        return self.server_defaults.cache_type_k

    @default_profile_cache_type_k.setter
    def default_profile_cache_type_k(self, value: str) -> None:
        self.server_defaults.cache_type_k = value

    @property
    def default_profile_cache_type_v(self) -> str:
        return self.server_defaults.cache_type_v

    @default_profile_cache_type_v.setter
    def default_profile_cache_type_v(self, value: str) -> None:
        self.server_defaults.cache_type_v = value

    @property
    def default_spec_decode(self) -> SpeculativeDecodingConfig:
        return SpeculativeDecodingConfig(
            spec_type=self.server_defaults.spec_type,
            spec_ngram_size_n=self.server_defaults.spec_ngram_size_n,
            draft_min=self.server_defaults.draft_min,
            draft_max=self.server_defaults.draft_max,
            spec_draft_n_max=self.server_defaults.spec_draft_n_max,
            spec_draft_p_min=self.server_defaults.spec_draft_p_min,
            spec_draft_cache_type_k=self.server_defaults.spec_draft_cache_type_k,
            spec_draft_cache_type_v=self.server_defaults.spec_draft_cache_type_v,
            spec_draft_device=self.server_defaults.spec_draft_device,
            reasoning_mode=self.server_defaults.reasoning_mode,
            reasoning_format=self.server_defaults.reasoning_format,
            reasoning_budget=self.server_defaults.reasoning_budget,
        )

    @default_spec_decode.setter
    def default_spec_decode(self, value: SpeculativeDecodingConfig) -> None:
        self.server_defaults.spec_type = value.spec_type
        self.server_defaults.spec_ngram_size_n = value.spec_ngram_size_n
        self.server_defaults.draft_min = value.draft_min
        self.server_defaults.draft_max = value.draft_max
        self.server_defaults.spec_draft_n_max = value.spec_draft_n_max
        self.server_defaults.spec_draft_p_min = value.spec_draft_p_min
        self.server_defaults.spec_draft_cache_type_k = value.spec_draft_cache_type_k
        self.server_defaults.spec_draft_cache_type_v = value.spec_draft_cache_type_v
        self.server_defaults.spec_draft_device = value.spec_draft_device
        self.server_defaults.reasoning_mode = value.reasoning_mode
        self.server_defaults.reasoning_format = value.reasoning_format
        self.server_defaults.reasoning_budget = value.reasoning_budget

    @property
    def default_reasoning_mode(self) -> str:
        return self.server_defaults.reasoning_mode

    @default_reasoning_mode.setter
    def default_reasoning_mode(self, value: str) -> None:
        self.server_defaults.reasoning_mode = value

    @property
    def default_reasoning_format(self) -> str:
        return self.server_defaults.reasoning_format

    @default_reasoning_format.setter
    def default_reasoning_format(self, value: str) -> None:
        self.server_defaults.reasoning_format = value

    @property
    def default_reasoning_budget(self) -> str:
        return self.server_defaults.reasoning_budget

    @default_reasoning_budget.setter
    def default_reasoning_budget(self, value: str) -> None:
        self.server_defaults.reasoning_budget = value

    @property
    def default_use_jinja(self) -> bool:
        return self.server_defaults.use_jinja

    @default_use_jinja.setter
    def default_use_jinja(self, value: bool) -> None:
        self.server_defaults.use_jinja = value

    @property
    def default_profile_chat_template_kwargs(self) -> str:
        return self.server_defaults.chat_template_kwargs

    @default_profile_chat_template_kwargs.setter
    def default_profile_chat_template_kwargs(self, value: str) -> None:
        self.server_defaults.chat_template_kwargs = value

    @property
    def default_mmproj(self) -> str:
        return self.server_defaults.mmproj

    @default_mmproj.setter
    def default_mmproj(self, value: str) -> None:
        self.server_defaults.mmproj = value

    @property
    def default_spec_type(self) -> str:
        return self.server_defaults.spec_type

    @default_spec_type.setter
    def default_spec_type(self, value: str) -> None:
        self.server_defaults.spec_type = value

    @property
    def default_spec_ngram_size_n(self) -> int:
        return self.server_defaults.spec_ngram_size_n

    @default_spec_ngram_size_n.setter
    def default_spec_ngram_size_n(self, value: int) -> None:
        self.server_defaults.spec_ngram_size_n = value

    @property
    def default_draft_min(self) -> int:
        return self.server_defaults.draft_min

    @default_draft_min.setter
    def default_draft_min(self, value: int) -> None:
        self.server_defaults.draft_min = value

    @property
    def default_draft_max(self) -> int:
        return self.server_defaults.draft_max

    @default_draft_max.setter
    def default_draft_max(self, value: int) -> None:
        self.server_defaults.draft_max = value

    @property
    def default_spec_draft_n_max(self) -> int:
        return self.server_defaults.spec_draft_n_max

    @default_spec_draft_n_max.setter
    def default_spec_draft_n_max(self, value: int) -> None:
        self.server_defaults.spec_draft_n_max = value

    @property
    def default_spec_draft_p_min(self) -> float:
        return self.server_defaults.spec_draft_p_min

    @default_spec_draft_p_min.setter
    def default_spec_draft_p_min(self, value: float) -> None:
        self.server_defaults.spec_draft_p_min = value

    @property
    def default_spec_draft_cache_type_k(self) -> str:
        return self.server_defaults.spec_draft_cache_type_k

    @default_spec_draft_cache_type_k.setter
    def default_spec_draft_cache_type_k(self, value: str) -> None:
        self.server_defaults.spec_draft_cache_type_k = value

    @property
    def default_spec_draft_cache_type_v(self) -> str:
        return self.server_defaults.spec_draft_cache_type_v

    @default_spec_draft_cache_type_v.setter
    def default_spec_draft_cache_type_v(self, value: str) -> None:
        self.server_defaults.spec_draft_cache_type_v = value

    @property
    def default_spec_draft_device(self) -> str:
        return self.server_defaults.spec_draft_device

    @default_spec_draft_device.setter
    def default_spec_draft_device(self, value: str) -> None:
        self.server_defaults.spec_draft_device = value

    # -- Per-profile default properties (routed to server_defaults) --
    @property
    def default_n_gpu_layers(self) -> int:
        return self.server_defaults.n_gpu_layers

    @default_n_gpu_layers.setter
    def default_n_gpu_layers(self, value: int) -> None:
        self.server_defaults.n_gpu_layers = value

    @property
    def default_ctx_size_summary(self) -> int:
        return self.server_defaults.ctx_size_summary

    @default_ctx_size_summary.setter
    def default_ctx_size_summary(self, value: int) -> None:
        self.server_defaults.ctx_size_summary = value

    @property
    def default_ctx_size_qwen35(self) -> int:
        return self.server_defaults.ctx_size_qwen35

    @default_ctx_size_qwen35.setter
    def default_ctx_size_qwen35(self, value: int) -> None:
        self.server_defaults.ctx_size_qwen35 = value

    @property
    def default_ctx_size_both_summary(self) -> int:
        return self.server_defaults.ctx_size_both_summary

    @default_ctx_size_both_summary.setter
    def default_ctx_size_both_summary(self, value: int) -> None:
        self.server_defaults.ctx_size_both_summary = value

    @property
    def default_ctx_size_both_qwen35(self) -> int:
        return self.server_defaults.ctx_size_both_qwen35

    @default_ctx_size_both_qwen35.setter
    def default_ctx_size_both_qwen35(self, value: int) -> None:
        self.server_defaults.ctx_size_both_qwen35 = value

    @property
    def default_n_gpu_layers_qwen35(self) -> str:
        return self.server_defaults.n_gpu_layers_qwen35

    @default_n_gpu_layers_qwen35.setter
    def default_n_gpu_layers_qwen35(self, value: str) -> None:
        self.server_defaults.n_gpu_layers_qwen35 = value

    @property
    def default_n_gpu_layers_qwen35_both(self) -> str:
        return self.server_defaults.n_gpu_layers_qwen35_both

    @default_n_gpu_layers_qwen35_both.setter
    def default_n_gpu_layers_qwen35_both(self, value: str) -> None:
        self.server_defaults.n_gpu_layers_qwen35_both = value

    @property
    def default_ubatch_size_summary_balanced(self) -> int:
        return self.server_defaults.ubatch_size_summary_balanced

    @default_ubatch_size_summary_balanced.setter
    def default_ubatch_size_summary_balanced(self, value: int) -> None:
        self.server_defaults.ubatch_size_summary_balanced = value

    @property
    def default_ubatch_size_summary_fast(self) -> int:
        return self.server_defaults.ubatch_size_summary_fast

    @default_ubatch_size_summary_fast.setter
    def default_ubatch_size_summary_fast(self, value: int) -> None:
        self.server_defaults.ubatch_size_summary_fast = value

    @property
    def default_ubatch_size_qwen35(self) -> int:
        return self.server_defaults.ubatch_size_qwen35

    @default_ubatch_size_qwen35.setter
    def default_ubatch_size_qwen35(self, value: int) -> None:
        self.server_defaults.ubatch_size_qwen35 = value

    @property
    def default_ubatch_size_qwen35_both(self) -> int:
        return self.server_defaults.ubatch_size_qwen35_both

    @default_ubatch_size_qwen35_both.setter
    def default_ubatch_size_qwen35_both(self, value: int) -> None:
        self.server_defaults.ubatch_size_qwen35_both = value

    @property
    def default_threads_summary_balanced(self) -> int:
        return self.server_defaults.threads_summary_balanced

    @default_threads_summary_balanced.setter
    def default_threads_summary_balanced(self, value: int) -> None:
        self.server_defaults.threads_summary_balanced = value

    @property
    def default_threads_summary_fast(self) -> int:
        return self.server_defaults.threads_summary_fast

    @default_threads_summary_fast.setter
    def default_threads_summary_fast(self, value: int) -> None:
        self.server_defaults.threads_summary_fast = value

    @property
    def default_threads_qwen35(self) -> int:
        return self.server_defaults.threads_qwen35

    @default_threads_qwen35.setter
    def default_threads_qwen35(self, value: int) -> None:
        self.server_defaults.threads_qwen35 = value

    @property
    def default_threads_qwen35_both(self) -> int:
        return self.server_defaults.threads_qwen35_both

    @default_threads_qwen35_both.setter
    def default_threads_qwen35_both(self, value: int) -> None:
        self.server_defaults.threads_qwen35_both = value

    @property
    def default_cache_type_summary_k(self) -> str:
        return self.server_defaults.cache_type_summary_k

    @default_cache_type_summary_k.setter
    def default_cache_type_summary_k(self, value: str) -> None:
        self.server_defaults.cache_type_summary_k = value

    @property
    def default_cache_type_summary_v(self) -> str:
        return self.server_defaults.cache_type_summary_v

    @default_cache_type_summary_v.setter
    def default_cache_type_summary_v(self, value: str) -> None:
        self.server_defaults.cache_type_summary_v = value

    @property
    def default_cache_type_qwen35_k(self) -> str:
        return self.server_defaults.cache_type_qwen35_k

    @default_cache_type_qwen35_k.setter
    def default_cache_type_qwen35_k(self, value: str) -> None:
        self.server_defaults.cache_type_qwen35_k = value

    @property
    def default_cache_type_qwen35_v(self) -> str:
        return self.server_defaults.cache_type_qwen35_v

    @default_cache_type_qwen35_v.setter
    def default_cache_type_qwen35_v(self, value: str) -> None:
        self.server_defaults.cache_type_qwen35_v = value

    @property
    def default_cache_type_qwen35_both_k(self) -> str:
        return self.server_defaults.cache_type_qwen35_both_k

    @default_cache_type_qwen35_both_k.setter
    def default_cache_type_qwen35_both_k(self, value: str) -> None:
        self.server_defaults.cache_type_qwen35_both_k = value

    @property
    def default_cache_type_qwen35_both_v(self) -> str:
        return self.server_defaults.cache_type_qwen35_both_v

    @default_cache_type_qwen35_both_v.setter
    def default_cache_type_qwen35_both_v(self, value: str) -> None:
        self.server_defaults.cache_type_qwen35_both_v = value


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
