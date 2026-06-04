"""llama_manager package - Core library for llm-runner.

Public API — consumers should prefer deep imports for new code.
This surface is maintained for backward compatibility.

This package provides the core business logic for managing multiple
llama-server instances, including configuration, server lifecycle,
GPU statistics, log buffering, and lockfile management.
"""

from .benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    SubprocessResult,
    build_benchmark_cmd,
    run_benchmark,
)
from .config import (
    BuildPipelineConfig,
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    PathsConfig,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    ServerConfig,
    ServerDefaultsConfig,
    SlotProfileRegistry,
    SlotProfileSpec,
    SlotState,
    SmokeConfig,
    SmokeProbeConfiguration,
    SpeculativeDecodingConfig,
    ValidationException,
    apply_config_updates,
    build_config,
    create_default_profile_registry,
    load_profile_with_staleness,
    resolve_backend_from_profile,
    write_profile,
)
from .dry_run import DryRunResult, run_dry_run, write_dry_run_artifact
from .gpu_telemetry import (
    GPUStats,
    GpuTelemetrySelector,
    collect_gpu_stats,
    collector_for_config,
    get_gpu_identifier,
    selector_for_config,
)
from .log_buffer import LogBuffer
from .model_index import (
    ModelIndexEntry,
    load_model_index,
    model_index_path,
    refresh_model_index,
)
from .orchestration import (
    AuditLogger,
    LaunchResult,
    RiskAckManager,
    ServerManager,
    launch_orchestrate,
    resolve_runtime_dir,
)
from .probe import SmokeCompositeReport
from .profile_orchestrator import (
    BenchmarkConfig,
    resolve_profile_slot,
)
from .risk_ack import RISK_ACK_LABEL, RiskAckResult, resolve_risk_action
from .slot_manager import gpu_index_for_config
from .slot_state import compute_slot_transition, resolve_slot_runtime_status
from .smoke import SmokeTarget, resolve_smoke_targets, run_smoke_probes
from .system_stats import (
    collect_cpu_percentages,
    collect_memory_usage,
    collect_system_info,
)
from .validation import (
    DryRunSlotPayload,
    build_server_cmd,
    require_executable,
    require_model,
    validate_port,
    validate_ports,
)

__all__ = [
    # Audit
    "AuditLogger",
    # Benchmark
    "BenchmarkResult",
    "BenchmarkRunner",
    "SubprocessResult",
    "build_benchmark_cmd",
    "run_benchmark",
    # Config
    "BuildPipelineConfig",
    "Config",
    "PathsConfig",
    "ServerDefaultsConfig",
    "SmokeConfig",
    "SmokeProbeConfiguration",
    "SpeculativeDecodingConfig",
    # Config types
    "ErrorCode",
    "ErrorDetail",
    "ModelIndexEntry",
    "ModelSlot",
    "MultiValidationError",
    "ProfileFlavor",
    "ProfileMetrics",
    "ProfileRecord",
    "ServerConfig",
    "ServerDefaultsConfig",
    "SlotProfileRegistry",
    "SlotProfileSpec",
    "SlotState",
    "ValidationException",
    # Config functions
    "apply_config_updates",
    "build_config",
    "create_default_profile_registry",
    "load_profile_with_staleness",
    "resolve_backend_from_profile",
    "write_profile",
    # Dry-run
    "DryRunResult",
    "DryRunSlotPayload",
    "run_dry_run",
    "write_dry_run_artifact",
    # GPU telemetry
    "GPUStats",
    "GpuTelemetrySelector",
    "collect_gpu_stats",
    "collector_for_config",
    "get_gpu_identifier",
    "selector_for_config",
    # Log buffer
    "LogBuffer",
    # Model index
    "ModelIndexEntry",
    "load_model_index",
    "model_index_path",
    "refresh_model_index",
    # Orchestration
    "BenchmarkConfig",
    "LaunchResult",
    "RiskAckManager",
    "ServerManager",
    "launch_orchestrate",
    "resolve_profile_slot",
    "resolve_runtime_dir",
    # Risk acknowledgement
    "RISK_ACK_LABEL",
    "RiskAckResult",
    "resolve_risk_action",
    # Slot management
    "gpu_index_for_config",
    # Slot state
    "compute_slot_transition",
    "resolve_slot_runtime_status",
    # Smoke probe
    "SmokeCompositeReport",
    "SmokeTarget",
    "resolve_smoke_targets",
    "run_smoke_probes",
    # System stats
    "collect_cpu_percentages",
    "collect_memory_usage",
    "collect_system_info",
    # Validation
    "build_server_cmd",
    "require_executable",
    "require_model",
    "validate_port",
    "validate_ports",
]
