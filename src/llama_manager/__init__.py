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

from .benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    SubprocessResult,
    build_benchmark_cmd,
    parse_benchmark_output,
    run_benchmark,
)
from .build_pipeline import (
    BuildArtifact,
    BuildBackend,
    BuildConfig,
    BuildLock,
    BuildProgress,
)
from .config import (
    Config,
    DoctorCheckStatus,
    ErrorCode,
    ErrorDetail,
    GgufParseError,
    ModelSlot,
    MultiValidationError,
    ServerConfig,
    SlotState,
    SmokeFailurePhase,
    SmokePhase,
    SmokeProbeConfiguration,
    SmokeProbeStatus,
    VRamRecommendation,
    detect_duplicate_slots,
    normalize_slot_id,
    validate_slot_id,
    validate_slot_port,
)
from .config_builder import (
    create_qwen35_cfg,
    create_smoke_config,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
)
from .gpu_stats import GPUStats, get_gpu_identifier
from .log_buffer import LogBuffer
from .metadata import (
    GGUFMetadataRecord,
    extract_gguf_metadata,
    normalize_filename,
)
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
from .profile_cache import (
    CURRENT_SCHEMA_VERSION,
    PROFILE_OVERRIDE_FIELDS,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    StalenessReason,
    StalenessResult,
    check_staleness,
    compute_driver_version_hash,
    compute_gpu_identifier,
    ensure_profiles_dir,
    get_profile_path,
    load_profile_with_staleness,
    profile_to_override_dict,
    read_profile,
    write_profile,
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
from .smoke import (
    ConsecutiveFailureCounter,
    ProvenanceRecord,
    SmokeCompositeReport,
    SmokeProbeResult,
    compute_overall_exit_code,
    probe_slot,
    resolve_provenance,
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

# Re-export redact_sensitive from reports module
# This avoids circular import issues

__all__ = [
    # Build pipeline
    "BuildConfig",
    "BuildArtifact",
    "BuildProgress",
    "BuildLock",
    "BuildBackend",
    # Config
    "Config",
    "ServerConfig",
    "ModelSlot",
    "ErrorCode",
    "ErrorDetail",
    "MultiValidationError",
    "normalize_slot_id",
    "detect_duplicate_slots",
    # Enums
    "SlotState",
    "SmokePhase",
    "SmokeFailurePhase",
    "SmokeProbeStatus",
    "VRamRecommendation",
    "DoctorCheckStatus",
    "GgufParseError",
    # Smoke probe config
    "SmokeProbeConfiguration",
    "validate_slot_id",
    "validate_slot_port",
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
    "get_gpu_identifier",
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
    # Smoke probe
    "SmokeProbeResult",
    "SmokeCompositeReport",
    "ProvenanceRecord",
    "ConsecutiveFailureCounter",
    "probe_slot",
    "resolve_provenance",
    "compute_overall_exit_code",
    # GGUF metadata
    "GGUFMetadataRecord",
    "extract_gguf_metadata",
    "normalize_filename",
    # Config builders
    "create_smoke_config",
    # Benchmark
    "SubprocessResult",
    "BenchmarkResult",
    "BenchmarkRunner",
    "build_benchmark_cmd",
    "parse_benchmark_output",
    "run_benchmark",
    # Profile cache
    "ProfileFlavor",
    "ProfileMetrics",
    "ProfileRecord",
    "StalenessReason",
    "StalenessResult",
    "PROFILE_OVERRIDE_FIELDS",
    "CURRENT_SCHEMA_VERSION",
    "compute_gpu_identifier",
    "compute_driver_version_hash",
    "ensure_profiles_dir",
    "get_profile_path",
    "read_profile",
    "write_profile",
    "check_staleness",
    "load_profile_with_staleness",
    "profile_to_override_dict",
]
