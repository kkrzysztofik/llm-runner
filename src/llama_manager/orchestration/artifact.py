"""Artifact I/O — write artifacts with JSON serialization and permission enforcement."""

import os
import stat
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TypedDict

from ..common.constants import DIR_MODE_OWNER_ONLY, FILE_MODE_OWNER_ONLY
from ..common.file_ops import atomic_exclusive_create_json
from ..common.security import REDACTED_VALUE, is_sensitive_key
from ..config import ErrorCode

if TYPE_CHECKING:
    from ..orchestration.lockfile import ValidationException

# Module-local string constants (artifact-specific).
ARTIFACT_CHECK_NAME: Final[str] = "artifact_persistence"
OWNER_ONLY_PERMISSIONS_FAILURE: Final[str] = (
    "artifact persistence failed to enforce required owner-only permissions"
)
PERMISSION_SUPPORT_HINT: Final[str] = (
    "verify runtime path and permission support/chmod limitations before retry"
)
PERMISSION_WRITABILITY_HINT: Final[str] = (
    "verify runtime path writability and filesystem permission support/chmod limitations"
)
MAX_COLLISION_RETRIES: Final[int] = 10


class DryRunArtifactPayload(TypedDict):
    """Dry-run artifact payload contract used by backend persistence."""

    timestamp: str
    slot_scope: list[str]
    resolved_command: dict[str, Any]
    validation_results: dict[str, Any]
    warnings: list[Any]
    environment_redacted: dict[str, Any]


class ArtifactMetadata:
    """FR-007: Artifact metadata for T012 persistence tracking."""

    def __init__(
        self,
        artifact_type: str,
        created_at: float,
        slot_id: str | None = None,
        additional_fields: dict | None = None,
    ) -> None:
        self.artifact_type = artifact_type
        self.created_at = created_at
        self.slot_id = slot_id
        self.additional_fields: dict = additional_fields if additional_fields is not None else {}


def write_artifact(runtime_dir: Path, _slot_id: str, data: DryRunArtifactPayload | dict) -> Path:
    """T012: Write artifact with JSON serialization and 0700/0600 permission enforcement."""
    _validate_artifact_fields(data)
    artifact_dir = runtime_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True, mode=DIR_MODE_OWNER_ONLY)

    dir_mode = stat.S_IMODE(os.stat(artifact_dir).st_mode)
    if dir_mode != DIR_MODE_OWNER_ONLY:
        raise _artifact_error(OWNER_ONLY_PERMISSIONS_FAILURE, PERMISSION_WRITABILITY_HINT)

    timestamp_filename = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    from typing import cast as _cast

    try:
        redacted_data = _redact_sensitive_in_dict(_cast(dict, data))
    except TypeError as e:
        raise _artifact_error(
            f"artifact serialization failed: {e}", "ensure artifact data is JSON-serializable"
        ) from e

    for attempt in range(MAX_COLLISION_RETRIES):
        suffix = f"-{attempt}" if attempt > 0 else ""
        artifact_path = artifact_dir / f"artifact-{timestamp_filename}{suffix}.json"
        try:
            atomic_exclusive_create_json(artifact_path, redacted_data, FILE_MODE_OWNER_ONLY)
            return artifact_path
        except FileExistsError:
            continue
        except TypeError as e:
            raise _artifact_error(
                f"artifact serialization failed: {e}", "ensure artifact data is JSON-serializable"
            ) from e
        except PermissionError as e:
            raise _artifact_error(
                "artifact persistence failed due to permission denied", PERMISSION_SUPPORT_HINT
            ) from e
        except OSError as e:
            raise _artifact_error(
                f"artifact persistence failed: {e}", PERMISSION_SUPPORT_HINT
            ) from e

    raise _artifact_error(
        f"artifact: failed to create unique file after {MAX_COLLISION_RETRIES} attempts",
        PERMISSION_SUPPORT_HINT,
    )


def _validate_artifact_fields(data: DryRunArtifactPayload | dict) -> None:
    """FR-007: Validate required top-level artifact fields."""
    required_fields = [
        "timestamp",
        "slot_scope",
        "resolved_command",
        "validation_results",
        "warnings",
        "environment_redacted",
    ]

    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        raise _artifact_error(
            f"artifact missing required fields: {', '.join(missing_fields)}",
            "ensure artifact data contains all required FR-007 fields",
            check="artifact_validation",
        )

    slot_scope = data.get("slot_scope")
    if not isinstance(slot_scope, list) or not all(isinstance(item, str) for item in slot_scope):
        raise _artifact_error(
            "slot_scope must be list[str]",
            "provide slot_scope as a list of slot IDs",
            check="artifact_validation",
        )

    resolved_command = data.get("resolved_command")
    if not isinstance(resolved_command, dict):
        raise _artifact_error(
            "resolved_command must be an object mapping",
            "provide resolved_command as a mapping keyed by slot ID",
            check="artifact_validation",
        )


def _redact_sensitive_in_dict(data: dict, env_key_prefix: str = "") -> dict:
    """Recursively redact sensitive environment variable values in a nested dict."""
    result = {}
    for key, value in data.items():
        full_key = f"{env_key_prefix}_{key}" if env_key_prefix else key
        if isinstance(value, dict):
            result[key] = _redact_sensitive_in_dict(value, full_key)
        elif isinstance(value, str) and is_sensitive_key(full_key):
            result[key] = REDACTED_VALUE
        else:
            result[key] = value
    return result


def _artifact_error(
    why_blocked: str, how_to_fix: str, check: str = ARTIFACT_CHECK_NAME
) -> "ValidationException":
    """Build artifact ValidationException."""
    from ..config import ErrorDetail, MultiValidationError
    from ..orchestration.lockfile import ValidationException

    detail = ErrorDetail(
        error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
        failed_check=check,
        why_blocked=why_blocked,
        how_to_fix=how_to_fix,
    )
    return ValidationException(MultiValidationError(errors=[detail]))
