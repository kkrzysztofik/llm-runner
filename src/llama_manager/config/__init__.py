"""Config subpackage — public API re-exported from focused submodules."""

from .builder import (
    apply_profile_overrides,
    create_default_profile_registry,
    create_default_slot_profiles,
    create_server_config_from_profile,
    create_smoke_config,
    create_tui_profile_registry,
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
    SlotState,
    SmokeFailurePhase,
    SmokePhase,
    SmokeProbeStatus,
)
from .errors import ErrorDetail, MultiValidationError, ValidationException
from .load_mode import LOAD_MODE_VALUES, resolve_load_mode
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
from .spec_decode import SpeculativeDecodingConfig, spec_type_members
from .tri_state import TRI_STATE_VALUES, resolve_fit, resolve_reasoning_preserve, resolve_tri_state

__all__ = [
    # enums
    "DoctorCheckStatus",
    "ErrorCode",
    "SlotState",
    "SmokeFailurePhase",
    "SmokePhase",
    "SmokeProbeStatus",
    # errors
    "ErrorDetail",
    "MultiValidationError",
    "ValidationException",
    # server
    "ModelSlot",
    "ServerConfig",
    "SpeculativeDecodingConfig",
    "spec_type_members",
    "LOAD_MODE_VALUES",
    "resolve_load_mode",
    "TRI_STATE_VALUES",
    "resolve_tri_state",
    "resolve_reasoning_preserve",
    "resolve_fit",
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
    "create_tui_profile_registry",
    "create_server_config_from_profile",
    "create_smoke_config",
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
