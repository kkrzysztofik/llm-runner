"""Config subpackage — public API re-exported from focused submodules."""

from .builder import (
    apply_profile_overrides,
    create_default_profile_registry,
    create_default_slot_profiles,
    create_qwen35_cfg,
    create_server_config_from_profile,
    create_smoke_config,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
    merge_config_overrides,
    resolve_profile_config,
)
from .defaults import (
    BuildPipelineConfig,
    Config,
    DeploymentConfig,
    PathsConfig,
    ServerDefaultsConfig,
    SmokeConfig,
    SmokeProbeConfiguration,
)
from .enums import (
    DoctorCheckStatus,
    ErrorCode,
    GgufParseError,
    SlotState,
    SmokeFailurePhase,
    SmokePhase,
    SmokeProbeStatus,
    VRamRecommendation,
)
from .errors import ErrorDetail, MultiValidationError, ValidationException
from .persistence import (
    ConfigUpdateResult,
    apply_config_updates,
    build_config,
    config_file_path,
    load_config_overrides_from_file,
    save_config_to_file,
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
from .profiles import (
    SlotProfileError,
    SlotProfileRegistry,
    SlotProfileSpec,
    resolve_backend_from_profile,
    resolve_profile_id,
)
from .server import (
    ModelSlot,
    ServerConfig,
    detect_duplicate_slots,
    normalize_slot_id,
    validate_slot_id,
    validate_slot_port,
)
from .spec_decode import SpeculativeDecodingConfig

__all__ = [
    # enums
    "DoctorCheckStatus",
    "ErrorCode",
    "GgufParseError",
    "SlotState",
    "SmokeFailurePhase",
    "SmokePhase",
    "SmokeProbeStatus",
    "VRamRecommendation",
    # errors
    "ErrorDetail",
    "MultiValidationError",
    "ValidationException",
    # server
    "ModelSlot",
    "ServerConfig",
    "SpeculativeDecodingConfig",
    "detect_duplicate_slots",
    "normalize_slot_id",
    "validate_slot_id",
    "validate_slot_port",
    # profiles
    "SlotProfileError",
    "SlotProfileRegistry",
    "SlotProfileSpec",
    "resolve_backend_from_profile",
    "resolve_profile_id",
    # defaults
    "BuildPipelineConfig",
    "Config",
    "DeploymentConfig",
    "PathsConfig",
    "ServerDefaultsConfig",
    "SmokeConfig",
    "SmokeProbeConfiguration",
    # profile_cache
    "CURRENT_SCHEMA_VERSION",
    "PROFILE_OVERRIDE_FIELDS",
    "ProfileFlavor",
    "ProfileMetrics",
    "ProfileRecord",
    "StalenessReason",
    "StalenessResult",
    "check_staleness",
    "compute_driver_version_hash",
    "compute_gpu_identifier",
    "ensure_profiles_dir",
    "get_profile_path",
    "load_profile_with_staleness",
    "profile_to_override_dict",
    "read_profile",
    "write_profile",
    # builder
    "apply_profile_overrides",
    "create_default_profile_registry",
    "create_default_slot_profiles",
    "create_qwen35_cfg",
    "create_server_config_from_profile",
    "create_smoke_config",
    "create_summary_balanced_cfg",
    "create_summary_fast_cfg",
    "merge_config_overrides",
    "resolve_profile_config",
    # persistence
    "build_config",
    "config_file_path",
    "load_config_overrides_from_file",
    "save_config_to_file",
    "ConfigUpdateResult",
    "apply_config_updates",
]
