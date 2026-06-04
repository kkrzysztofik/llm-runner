"""Artifact I/O — write artifacts with JSON serialization and permission enforcement."""

import os
import stat
import time
from pathlib import Path
from typing import Any, Final, TypedDict

from ..common.constants import DIR_MODE_OWNER_ONLY, FILE_MODE_OWNER_ONLY
from ..common.file_ops import atomic_exclusive_create_json
from ..common.security import redact_dict
from ..config import ErrorCode, ValidationException

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
    warnings: list[str]
    environment_redacted: dict[str, Any]


class ArtifactMetadata:
    """FR-007: Artifact metadata for T012 persistence tracking."""

    def __init__(
        self,
        artifact_type: str,
        created_at: float,
        slot_id: str | None = None,
        additional_fields: dict[str, Any] | None = None,
    ) -> None:
        self.artifact_type = artifact_type
        self.created_at = created_at
        self.slot_id = slot_id
        self.additional_fields: dict[str, Any] = (
            additional_fields if additional_fields is not None else {}
        )


def write_artifact(
    runtime_dir: Path, _slot_id: str, data: DryRunArtifactPayload | dict[str, Any]
) -> Path:
    """T012: Write artifact with JSON serialization and 0700/0600 permission enforcement."""
    _validate_artifact_fields(data)
    artifact_dir = runtime_dir / "artifacts"
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True, mode=DIR_MODE_OWNER_ONLY)
    except OSError as e:
        raise _artifact_error(
            f"failed to create artifact directory: {e}",
            "verify runtime path writability and filesystem permission support",
        ) from e

    try:
        dir_mode = stat.S_IMODE(os.stat(artifact_dir).st_mode)
    except OSError as e:
        raise _artifact_error(
            f"failed to stat artifact directory: {e}",
            "verify runtime path and permission support/chmod limitations",
        ) from e

    if dir_mode != DIR_MODE_OWNER_ONLY:
        raise _artifact_error(OWNER_ONLY_PERMISSIONS_FAILURE, PERMISSION_WRITABILITY_HINT)

    timestamp_filename = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    from typing import cast as _cast

    try:
        redacted_data = redact_dict(_cast(dict, data))
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
    _validate_required_fields(
        data,
        [
            "timestamp",
            "slot_scope",
            "resolved_command",
            "validation_results",
            "warnings",
            "environment_redacted",
        ],
    )
    _validate_isinstance_str(
        data,
        "timestamp",
        "timestamp must be an ISO 8601 string",
        "provide timestamp as an ISO 8601 formatted string (e.g. 2026-04-12T00:00:00Z)",
    )
    _validate_string_list(
        data,
        "slot_scope",
        "slot_scope must be list[str]",
        "provide slot_scope as a list of slot IDs",
    )
    _validate_resolved_command(data)
    _validate_isinstance_dict(
        data,
        "validation_results",
        "validation_results must be an object mapping",
        "provide validation_results as a mapping of check names to results",
    )
    _validate_string_list(
        data,
        "warnings",
        "warnings must be list[str]",
        "provide warnings as a list of string messages",
    )
    _validate_isinstance_dict(
        data,
        "environment_redacted",
        "environment_redacted must be an object mapping",
        "provide environment_redacted as a mapping of environment variable names to redacted values",
    )


def _validate_required_fields(data: DryRunArtifactPayload | dict, fields: list[str]) -> None:
    """Check that all required fields exist in data."""
    missing = [f for f in fields if f not in data]
    if missing:
        raise _artifact_error(
            f"artifact missing required fields: {', '.join(missing)}",
            "ensure artifact data contains all required FR-007 fields",
            check="artifact_validation",
        )


def _validate_isinstance_str(
    data: DryRunArtifactPayload | dict, field: str, why: str, how: str
) -> None:
    """Validate that a field value is a string."""
    value = data.get(field)
    if not isinstance(value, str):
        raise _artifact_error(why, how, check="artifact_validation")


def _validate_string_list(
    data: DryRunArtifactPayload | dict, field: str, why: str, how: str
) -> None:
    """Validate that a field is a list of strings."""
    value = data.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _artifact_error(why, how, check="artifact_validation")


def _validate_isinstance_dict(
    data: DryRunArtifactPayload | dict, field: str, why: str, how: str
) -> None:
    """Validate that a field value is a dict."""
    value = data.get(field)
    if not isinstance(value, dict):
        raise _artifact_error(why, how, check="artifact_validation")


def _validate_resolved_command(data: DryRunArtifactPayload | dict) -> None:
    """Validate the resolved_command dict structure."""
    resolved = data.get("resolved_command")
    if not isinstance(resolved, dict):
        raise _artifact_error(
            "resolved_command must be an object mapping",
            "provide resolved_command as a mapping keyed by slot ID",
            check="artifact_validation",
        )
    for cmd_key, cmd_val in resolved.items():
        if not isinstance(cmd_key, str):
            raise _artifact_error(
                "resolved_command keys must be strings (slot IDs)",
                "provide resolved_command with string keys for each slot ID",
                check="artifact_validation",
            )
        if isinstance(cmd_val, dict):
            for slot_id in cmd_val:
                if not isinstance(slot_id, str):
                    raise _artifact_error(
                        "resolved_command slot IDs must be strings",
                        "ensure all slot identifiers in resolved_command are strings",
                        check="artifact_validation",
                    )


def _artifact_error(
    why_blocked: str, how_to_fix: str, check: str = ARTIFACT_CHECK_NAME
) -> ValidationException:
    """Build artifact ValidationException."""
    from ..config import ErrorDetail, MultiValidationError

    detail = ErrorDetail(
        error_code=ErrorCode.ARTIFACT_PERSISTENCE_FAILURE,
        failed_check=check,
        why_blocked=why_blocked,
        how_to_fix=how_to_fix,
    )
    return ValidationException(MultiValidationError(errors=[detail]))
