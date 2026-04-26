# Config & ServerConfig dataclasses


import os
import re
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
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
        self.llama_server_bin_intel = str(
            Path(self.llama_cpp_root) / "build" / "bin" / "llama-server"
        )
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
    profile_staleness_days: int = field(default_factory=lambda: 30)
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
class ServerConfig:
    """Individual server configuration"""

    model: str
    alias: str
    device: str
    port: int
    ctx_size: int
    ubatch_size: int
    threads: int
    bind_address: str = "127.0.0.1"
    tensor_split: str = ""
    reasoning_mode: str = "auto"
    reasoning_format: str = "none"
    chat_template_kwargs: str = ""
    reasoning_budget: str = ""
    use_jinja: bool = False
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    n_gpu_layers: int | str = 99
    server_bin: str = ""
    backend: str = "llama_cpp"
    risky_acknowledged: list[str] = field(default_factory=list)


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


# M1 scaffolding

# Regex pattern for slot ID normalization: strip, lowercase, allow only a-z0-9_-
_SLOT_ID_PATTERN = re.compile(r"[^a-z0-9_-]")


def normalize_slot_id(slot_id: str) -> str:
    """Normalize slot ID by stripping whitespace, lowercasing ASCII, allowing only a-z0-9_-.

    Args:
        slot_id: Raw slot identifier string

    Returns:
        Normalized slot ID with only allowed characters (lowercase a-z, digits, underscore, hyphen)

    Raises:
        ValueError: If normalized result is empty after applying allowed character filter

    """
    normalized = _SLOT_ID_PATTERN.sub("", slot_id.strip().lower())
    if not normalized:
        raise ValueError("slot_id must contain at least one valid character after normalization")
    return normalized


def detect_duplicate_slots(slots: list["ModelSlot"]) -> list[str]:
    """Detect duplicate slot IDs in a list of ModelSlot entries.

    Args:
        slots: List of ModelSlot objects to check for duplicates

    Returns:
        List of normalized slot_ids that appear more than once

    """
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for slot in slots:
        normalized = normalize_slot_id(slot.slot_id)
        if normalized in seen:
            if normalized not in duplicates:
                duplicates.append(normalized)
        else:
            seen[normalized] = 1
    return duplicates


@dataclass
class ModelSlot:
    """Model slot configuration for multi-slot serving"""

    slot_id: str
    model_path: str
    port: int


def validate_slot_id(slot_id: str) -> "ValidationResult":
    """Validate and normalize a slot ID.

    Args:
        slot_id: Raw slot identifier string

    Returns:
        ValidationResult indicating success or failure with error details

    """
    try:
        normalized = normalize_slot_id(slot_id)
        return ValidationResult(
            slot_id=normalized,
            passed=True,
        )
    except ValueError as e:
        return ValidationResult(
            slot_id=slot_id,
            passed=False,
            failed_check="slot_id_validation",
            error_code=ErrorCode.INVALID_SLOT_ID,
            error_message=str(e),
        )


def validate_slot_port(port: int, slot_id: str) -> "ValidationResult":
    """Validate a slot port number.

    Args:
        port: Port number to validate
        slot_id: Slot identifier for error reporting

    Returns:
        ValidationResult indicating success or failure with error details

    """
    if not isinstance(port, int) or port < 1 or port > 65535:
        return ValidationResult(
            slot_id=slot_id,
            passed=False,
            failed_check="port_range",
            error_code=ErrorCode.PORT_INVALID,
            error_message=f"port must be between 1 and 65535, got: {port}",
        )
    return ValidationResult(
        slot_id=slot_id,
        passed=True,
    )


class ErrorCode(StrEnum):
    """Error code enum for validation with deterministic string ordering"""

    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PATH_INVALID = "PATH_INVALID"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    PORT_CONFLICT = "PORT_CONFLICT"
    PORT_INVALID = "PORT_INVALID"
    THREADS_INVALID = "THREADS_INVALID"
    CONFIG_ERROR = "CONFIG_ERROR"
    INVALID_SLOT_ID = "INVALID_SLOT_ID"
    DUPLICATE_SLOT = "DUPLICATE_SLOT"
    RUNTIME_DIR_UNAVAILABLE = "RUNTIME_DIR_UNAVAILABLE"
    LOCKFILE_INTEGRITY_FAILURE = "LOCKFILE_INTEGRITY_FAILURE"
    ARTIFACT_PERSISTENCE_FAILURE = "ARTIFACT_PERSISTENCE_FAILURE"
    BACKEND_NOT_ELIGIBLE = "BACKEND_NOT_ELIGIBLE"
    # M2 build setup error codes
    TOOLCHAIN_MISSING = "TOOLCHAIN_MISSING"
    BUILD_LOCK_HELD = "BUILD_LOCK_HELD"
    VENV_CORRUPT = "VENV_CORRUPT"
    PYTHON_NOT_FOUND = "PYTHON_NOT_FOUND"
    BUILD_FAILED = "BUILD_FAILED"
    PREFLIGHT_FAILURE = "PREFLIGHT_FAILURE"
    GIT_CLONE_FAILED = "GIT_CLONE_FAILED"
    GIT_CHECKOUT_FAILED = "GIT_CHECKOUT_FAILED"
    REPORT_WRITE_FAILURE = "REPORT_WRITE_FAILURE"
    TOOL_VERSION_MISMATCH = "TOOL_VERSION_MISMATCH"


class SlotState(StrEnum):
    """State of a model slot in the TUI dashboard."""

    IDLE = "idle"
    LAUNCHING = "launching"
    RUNNING = "running"
    DEGRADED = "degraded"
    CRASHED = "crashed"
    OFFLINE = "offline"


class SmokePhase(StrEnum):
    """Phase of a smoke probe for a single slot."""

    LISTEN = "listen"
    MODELS = "models"
    CHAT = "chat"
    COMPLETE = "complete"


class SmokeFailurePhase(StrEnum):
    """Phase at which a smoke probe failed."""

    LISTEN = "listen"
    MODELS = "models"
    CHAT = "chat"


class SmokeProbeStatus(StrEnum):
    """Outcome of a smoke probe for a single slot."""

    PASS = "pass"  # noqa: S105
    FAIL = "fail"  # noqa: S105
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    MODEL_NOT_FOUND = "model_not_found"
    AUTH_FAILURE = "auth_failure"


class VRamRecommendation(StrEnum):
    """VRAM heuristic recommendation for model loading."""

    PROCEED = "proceed"
    WARN = "warn"
    CONFIRM_REQUIRED = "confirm_required"


class DoctorCheckStatus(StrEnum):
    """Status of a doctor check result."""

    PASS = "pass"  # noqa: S105
    WARN = "warn"
    FAIL = "fail"  # noqa: S105


class GgufParseError(StrEnum):
    """Error types for GGUF metadata extraction."""

    CORRUPT_FILE = "CORRUPT_FILE"
    PARSE_TIMEOUT = "PARSE_TIMEOUT"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    READ_ERROR = "READ_ERROR"


@dataclass
class ValidationResult:
    """Result of a validation check with slot identity for T003 deterministic sorting"""

    slot_id: str
    passed: bool
    failed_check: str = ""
    error_code: ErrorCode | None = None
    error_message: str = ""

    @property
    def valid(self) -> bool:
        """Alias for passed to maintain backward compatibility"""
        return self.passed


@dataclass
class ErrorDetail:
    """FR-005 structured actionable error detail"""

    error_code: ErrorCode
    failed_check: str
    why_blocked: str
    how_to_fix: str
    docs_ref: str | None = None


@dataclass
class MultiValidationError:
    """FR-005 container for multiple validation errors with deterministic ordering"""

    errors: list[ErrorDetail]

    @property
    def error_count(self) -> int:
        """Return the number of errors in this multi-error"""
        return len(self.errors)

    def sort_errors(self) -> None:
        """Sort errors in-place by slot configuration sequence, then failed_check ascending.

        This provides stable, deterministic ordering for consistent error output.
        Slots are ordered alphabetically by slot_id; within each slot, failed_check is sorted alphabetically.

        Slot ID extraction:
        - Pattern: failed_check starts with "slot_<slot_id>_<check>"
        - slot_id is the second underscore-separated component (e.g., "slot_slot1_a_check" -> slot1)
        - Normalize by stripping "slot_" prefix (e.g., "slot_slot1" -> "slot1")
        """
        if not self.errors:
            return

        slot_ids = sorted(
            {
                slot_id
                for error in self.errors
                if (slot_id := _extract_slot_id(error.failed_check)) is not None
            }
        )
        slot_order = {slot_id: idx for idx, slot_id in enumerate(slot_ids)}

        self.errors = sorted(
            self.errors,
            key=lambda error: _error_sort_key(error, slot_order, len(slot_ids) + 1),
        )


def _extract_slot_id(failed_check: str) -> str | None:
    if not failed_check.startswith("slot_"):
        return None

    parts = failed_check.split("_", maxsplit=2)
    if len(parts) < 2 or not parts[1]:
        return None
    return parts[1]


def _error_sort_key(
    error: ErrorDetail,
    slot_order: dict[str, int],
    default_index: int,
) -> tuple[int, str]:
    slot_id = _extract_slot_id(error.failed_check)
    if slot_id is None:
        return (default_index, error.failed_check)
    return (slot_order.get(slot_id, default_index), error.failed_check)
