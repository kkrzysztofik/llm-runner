"""llama_manager package - Core library for llm-runner.

This package provides the core business logic for managing multiple
llama-server instances, including configuration, server lifecycle,
GPU statistics, log buffering, and lockfile management. It exports:

- Config & ServerConfig dataclasses for hardware-specific defaults and
  per-instance launch parameters
- Factory functions (create_*_cfg) that translate Config into ServerConfig
- GPU statistics collection via nvtop/psutil
- Thread-safe real-time log streaming via LogBuffer
- Subprocess lifecycle management via ServerManager
- Lockfile and artifact I/O functions (create_lock, read_lock,
  release_lock, update_lock, write_artifact, resolve_runtime_dir)
- Server command building and validation utilities
"""

from .build_pipeline import (
    GGML_CUDA,
    GGML_SYCL,
    BuildArtifact,
    BuildBackend,
    BuildConfig,
    BuildLock,
    BuildProgress,
)
from .config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    detect_duplicate_slots,
    normalize_slot_id,
)
from .config_builder import (
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
)
from .gpu_stats import GPUStats
from .log_buffer import LogBuffer
from .process_manager import (
    ArtifactMetadata,
    DryRunArtifactPayload,
    LaunchResult,
    LockMetadata,
    ServerManager,
    ValidationException,
    create_lock,
    read_lock,
    release_lock,
    resolve_runtime_dir,
    update_lock,
    write_artifact,
)
from .reports import (
    FailureReport,
    MutatingActionLogEntry,
    redact_sensitive,
    rotate_reports,
    write_failure_report,
)
from .server import (
    DryRunSlotPayload,
    ValidationResults,
    VllmEligibility,
    build_dry_run_slot_payload,
    build_server_cmd,
    require_executable,
    require_model,
    validate_backend_eligibility,
    validate_port,
    validate_ports,
    validate_server_config,
    validate_slots,
    validate_threads,
)
from .setup_venv import (
    VenvResult,
    check_venv_integrity,
    create_venv,
    get_venv_path,
)
from .toolchain import (
    CMAKE_HINT,
    CMAKE_MINIMUM_VERSION,
    CUDA_HINT,
    CUDA_REQUIRED_TOOLS,
    GCC_HINT,
    GIT_HINT,
    MAKE_HINT,
    NVTOP_HINT,
    SYCL_HINT,
    SYCL_REQUIRED_TOOLS,
    ToolchainErrorDetail,
    ToolchainHint,
    ToolchainStatus,
    detect_tool,
    get_toolchain_hints,
    parse_version,
    version_at_least,
)

# Re-export redact_sensitive from server for backward compatibility
# This avoids circular import issues

__all__ = [
    # Build pipeline
    "BuildConfig",
    "BuildArtifact",
    "BuildProgress",
    "BuildLock",
    "BuildBackend",
    "GGML_SYCL",
    "GGML_CUDA",
    # Config
    "Config",
    "ServerConfig",
    "ModelSlot",
    "ErrorCode",
    "ErrorDetail",
    "MultiValidationError",
    "normalize_slot_id",
    "detect_duplicate_slots",
    # Dry-run payload types
    "DryRunSlotPayload",
    "VllmEligibility",
    "ValidationResults",
    # Server
    "build_server_cmd",
    "build_dry_run_slot_payload",
    "validate_port",
    "validate_ports",
    "validate_threads",
    "validate_slots",
    "validate_backend_eligibility",
    "validate_server_config",
    "require_model",
    "require_executable",
    "redact_sensitive",
    # Config builders
    "create_summary_balanced_cfg",
    "create_summary_fast_cfg",
    "create_qwen35_cfg",
    # Reports
    "FailureReport",
    "MutatingActionLogEntry",
    "redact_sensitive",
    "write_failure_report",
    "rotate_reports",
    # Virtual environment
    "VenvResult",
    "get_venv_path",
    "create_venv",
    "check_venv_integrity",
    # Toolchain
    "ToolchainStatus",
    "ToolchainHint",
    "ToolchainErrorDetail",
    "SYCL_REQUIRED_TOOLS",
    "CUDA_REQUIRED_TOOLS",
    "CMAKE_MINIMUM_VERSION",
    "GCC_HINT",
    "MAKE_HINT",
    "GIT_HINT",
    "CMAKE_HINT",
    "SYCL_HINT",
    "CUDA_HINT",
    "NVTOP_HINT",
    "detect_tool",
    "get_toolchain_hints",
    "parse_version",
    "version_at_least",
    # Components
    "LogBuffer",
    "GPUStats",
    "ServerManager",
    # Lockfile and artifacts
    "ArtifactMetadata",
    "DryRunArtifactPayload",
    "LaunchResult",
    "LockMetadata",
    "ValidationException",
    "create_lock",
    "read_lock",
    "release_lock",
    "resolve_runtime_dir",
    "update_lock",
    "write_artifact",
]
