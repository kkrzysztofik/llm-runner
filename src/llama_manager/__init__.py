"""llama_manager package - Core library for llm-runner."""

from .config import (
    Config,
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    ProfileFlavor,
    ServerConfig,
    SlotState,
    apply_config_updates,
    build_config,
    load_profile_with_staleness,
)
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
    LaunchResult,
    ServerManager,
    launch_orchestrate,
)
from .risk_ack import RiskAckResult, resolve_risk_action
from .slot_manager import gpu_index_for_config
from .slot_state import compute_slot_transition, resolve_slot_runtime_status
from .system_stats import (
    collect_cpu_percentages,
    collect_memory_usage,
    collect_system_info,
)
from .validation import require_executable, require_model, validate_port, validate_ports

__all__ = [
    "Config",
    "ErrorCode",
    "ErrorDetail",
    "ModelIndexEntry",
    "ModelSlot",
    "ProfileFlavor",
    "ServerConfig",
    "SlotState",
    "apply_config_updates",
    "build_config",
    "load_profile_with_staleness",
    "GPUStats",
    "GpuTelemetrySelector",
    "collect_gpu_stats",
    "collector_for_config",
    "get_gpu_identifier",
    "selector_for_config",
    "LogBuffer",
    "load_model_index",
    "model_index_path",
    "refresh_model_index",
    "LaunchResult",
    "ServerManager",
    "launch_orchestrate",
    "RiskAckResult",
    "resolve_risk_action",
    "gpu_index_for_config",
    "compute_slot_transition",
    "resolve_slot_runtime_status",
    "collect_cpu_percentages",
    "collect_memory_usage",
    "collect_system_info",
    "require_executable",
    "require_model",
    "validate_port",
    "validate_ports",
]
