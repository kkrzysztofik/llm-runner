# StrEnum classes shared across the config subpackage

from enum import StrEnum


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
