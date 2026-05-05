"""orchestration package — process management, lockfiles, and artifacts."""

from .artifact import (
    ArtifactMetadata,
    DryRunArtifactPayload,
    _redact_sensitive_in_dict,  # noqa: F401 — re-exported for tests
    write_artifact,
)
from .launcher import (
    DefaultProcessLauncher,
    ProcessHandle,
    ProcessLauncher,
    ProcessTimeoutError,
)
from .lockfile import (
    LockMetadata,
    check_lockfile_integrity,
    create_lock,
    read_lock,
    release_lock,
    resolve_runtime_dir,
    update_lock,
)
from .manager import (
    REDACTED_VALUE,
    LaunchOrchestrationResult,
    LaunchResult,
    ProcessMetadata,
    ServerManager,
    SlotRuntime,
    ValidationException,
    _append_audit_log,
    _redact_sensitive,
    _rotate_audit_log,
    _verify_shutdown_ownership,
    launch_orchestrate,
)

__all__ = [
    # Lockfile operations
    "LockMetadata",
    "check_lockfile_integrity",
    "create_lock",
    "read_lock",
    "release_lock",
    "resolve_runtime_dir",
    "update_lock",
    # Artifact operations
    "write_artifact",
    "DryRunArtifactPayload",
    "ArtifactMetadata",
    # Process launcher
    "ProcessHandle",
    "ProcessLauncher",
    "ProcessTimeoutError",
    "DefaultProcessLauncher",
    # Server lifecycle
    "ServerManager",
    "SlotRuntime",
    "ProcessMetadata",
    "LaunchResult",
    "LaunchOrchestrationResult",
    "launch_orchestrate",
    # Internal (exported for tests)
    "REDACTED_VALUE",
    "_append_audit_log",
    "_redact_sensitive",
    "_rotate_audit_log",
    "_verify_shutdown_ownership",
    # Exceptions
    "ValidationException",
]
