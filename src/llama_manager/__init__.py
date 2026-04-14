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
from .server import (
    DryRunSlotPayload,
    ValidationResults,
    VllmEligibility,
    build_dry_run_slot_payload,
    build_server_cmd,
    redact_sensitive,
    require_executable,
    require_model,
    validate_backend_eligibility,
    validate_port,
    validate_ports,
    validate_server_config,
    validate_slots,
    validate_threads,
)

__all__ = [
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
