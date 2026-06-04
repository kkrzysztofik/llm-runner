"""orchestration package — process management, lockfiles, and artifacts."""

from ..common.security import REDACTED_VALUE, redact_dict, redact_text
from .artifact import (
    ArtifactMetadata,
    DryRunArtifactPayload,
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
    LaunchOrchestrationResult,
    LaunchResult,
    ProcessMetadata,
    ServerManager,
    SlotRuntime,
    ValidationException,
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
    # Redaction (re-exported from common.security)
    "REDACTED_VALUE",
    "redact_dict",
    "redact_text",
    # Exceptions
    "ValidationException",
]
