# Profile cache schema types, metrics, staleness, flavor enums, and I/O helpers.

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_CURRENT_SCHEMA_VERSION: str = "1.0"

CURRENT_SCHEMA_VERSION: str = _CURRENT_SCHEMA_VERSION

PROFILE_OVERRIDE_FIELDS: frozenset[str] = frozenset(
    ["threads", "ctx_size", "ubatch_size", "cache_type_k", "cache_type_v"],
)

DIR_MODE_OWNER_ONLY: int = 0o700

FILE_MODE_OWNER_ONLY: int = 0o600


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProfileFlavor(StrEnum):
    """Performance profile flavor for GPU benchmarking runs."""

    BALANCED = "balanced"
    FAST = "fast"
    QUALITY = "quality"


class StalenessReason(StrEnum):
    """Reasons a cached profile may be considered stale."""

    DRIVER_CHANGED = "driver_changed"
    BINARY_CHANGED = "binary_changed"
    AGE_EXCEEDED = "age_exceeded"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileMetrics:
    """Benchmark performance metrics for a single profile run.

    Attributes:
        tokens_per_second: Average throughput in tokens per second.
        avg_latency_ms: Average per-token latency in milliseconds.
        peak_vram_mb: Peak VRAM usage in megabytes, or ``None`` if unavailable.
    """

    tokens_per_second: float
    avg_latency_ms: float
    peak_vram_mb: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileMetrics:
        """Deserialize ProfileMetrics from a dict.

        Args:
            data: Dictionary containing metric values.

        Returns:
            A new ProfileMetrics instance.
        """
        return cls(
            tokens_per_second=float(data["tokens_per_second"]),
            avg_latency_ms=float(data["avg_latency_ms"]),
            peak_vram_mb=float(data["peak_vram_mb"])
            if data.get("peak_vram_mb") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ProfileMetrics to a plain dict.

        Returns:
            A dictionary with metric values suitable for JSON serialization.
        """
        return {
            "tokens_per_second": self.tokens_per_second,
            "avg_latency_ms": self.avg_latency_ms,
            "peak_vram_mb": self.peak_vram_mb,
        }


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    """A complete profiling record for a single GPU/backend combination.

    This dataclass is immutable (frozen=True, slots=True) and serves as the
    primary serialisable unit for cached benchmark profiles.

    Attributes:
        schema_version: Schema version string (defaults to module constant).
        gpu_identifier: Human-readable GPU identifier (e.g. "Intel Arc B580").
        backend: Backend string, either ``"cuda"`` or ``"sycl"``.
        flavor: Performance profile flavor.
        driver_version: Driver version string.
        driver_version_hash: SHA-256 of driver version, truncated to 16 hex chars.
        server_binary_version: llama-server binary version string.
        profiled_at: ISO 8601 UTC timestamp of when profiling occurred.
        metrics: Benchmark performance metrics.
        parameters: Arbitrary profiling parameters dict.
    """

    gpu_identifier: str
    backend: str
    flavor: ProfileFlavor
    driver_version: str
    driver_version_hash: str
    server_binary_version: str
    profiled_at: str
    metrics: ProfileMetrics
    schema_version: str = _CURRENT_SCHEMA_VERSION
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileRecord:
        """Deserialize a ProfileRecord from a dict.

        Handles conversion of ``ProfileFlavor`` from its string value and
        delegates to ``ProfileMetrics.from_dict`` for the metrics field.

        Args:
            data: Dictionary containing all profile record fields.

        Returns:
            A new ProfileRecord instance.
        """
        return cls(
            schema_version=data.get("schema_version", _CURRENT_SCHEMA_VERSION),
            gpu_identifier=data["gpu_identifier"],
            backend=data["backend"],
            flavor=ProfileFlavor(data["flavor"]),
            driver_version=data["driver_version"],
            driver_version_hash=data["driver_version_hash"],
            server_binary_version=data["server_binary_version"],
            profiled_at=data["profiled_at"],
            metrics=ProfileMetrics.from_dict(data["metrics"]),
            parameters=data.get("parameters", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a plain dict for JSON persistence.

        Converts ``ProfileFlavor`` to its string value and delegates
        metrics serialization to ``ProfileMetrics.to_dict``.

        Returns:
            A dictionary suitable for JSON serialization.
        """
        return {
            "schema_version": self.schema_version,
            "gpu_identifier": self.gpu_identifier,
            "backend": self.backend,
            "flavor": self.flavor.value,
            "driver_version": self.driver_version,
            "driver_version_hash": self.driver_version_hash,
            "server_binary_version": self.server_binary_version,
            "profiled_at": self.profiled_at,
            "metrics": self.metrics.to_dict(),
            "parameters": self.parameters,
        }


@dataclass(frozen=True, slots=True)
class StalenessResult:
    """Result of a profile staleness check.

    Attributes:
        is_stale: Whether the profile is considered stale.
        reasons: List of reasons contributing to staleness (empty when ``is_stale=False``).
        driver_version_display: Human-readable driver version string.
        age_days: Age of the profile in days (fractional).
    """

    is_stale: bool
    reasons: list[StalenessReason] = field(default_factory=list)
    driver_version_display: str = "unknown"
    age_days: float = 0.0

    @property
    def warning_message(self) -> str:
        """Build a human-readable warning message from staleness reasons.

        Returns:
            A descriptive message listing each reason, or an empty string
            when the profile is not stale.
        """
        if not self.reasons:
            return ""
        parts = [reason.value.replace("_", " ").title() for reason in self.reasons]
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Filename & identifier helpers
# ---------------------------------------------------------------------------

# Regex pattern for filename sanitization: allow a-z, 0-9, dash, underscore, dot
_FILENAME_SANITIZE_PATTERN = re.compile(r"[^a-z0-9_\-\.]")


def _sanitize_filename_component(component: str) -> str:
    """Sanitize a string for safe use as a filesystem path component.

    Matches the ``normalize_slot_id()`` pattern from config.py: strips
    whitespace, lowercases ASCII, and replaces any character that is not
    a lowercase letter, digit, dash, underscore, or dot with an underscore.

    Args:
        component: Raw string to sanitize.

    Returns:
        Sanitized string with only allowed characters.

    Raises:
        ValueError: If the resulting string is empty after sanitization.

    """
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a non-empty string")

    sanitized = _FILENAME_SANITIZE_PATTERN.sub("_", component.strip().lower())

    if not sanitized:
        raise ValueError(
            f"component must contain at least one valid character after "
            f"sanitization, got: {component!r}",
        )
    return sanitized


def compute_gpu_identifier(
    backend: str,
    gpu_name: str,
    device_index: int,
) -> str:
    """Compute a filesystem-safe GPU identifier string.

    Formats:
        CUDA  -> ``nvidia-{sanitized_gpu_name}-{device_index:02d}``
        SYCL  -> ``intel-{sanitized_gpu_name}-{device_index:02d}``

    Args:
        backend: Either ``"cuda"`` or ``"sycl"``.
        gpu_name: Human-readable GPU name (e.g. ``"GeForce RTX 3090"``).
        device_index: Zero-based GPU device index.

    Returns:
        Lowercase, filesystem-safe GPU identifier.

    Raises:
        ValueError: If ``backend`` is not recognized or ``gpu_name`` is invalid.

    """
    prefix_map: dict[str, str] = {
        "cuda": "nvidia",
        "sycl": "intel",
    }

    prefix = prefix_map.get(backend)
    if prefix is None:
        raise ValueError(
            f"unsupported backend: {backend!r}; expected one of: {', '.join(sorted(prefix_map))}",
        )

    sanitized_name = _sanitize_filename_component(gpu_name)
    return f"{prefix}-{sanitized_name}-{device_index:02d}"


def compute_driver_version_hash(driver_version: str) -> str:
    """Compute a truncated SHA-256 hash of a driver version string.

    Args:
        driver_version: The driver version string to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest.

    Raises:
        ValueError: If ``driver_version`` is empty.

    """
    if not isinstance(driver_version, str) or not driver_version:
        raise ValueError("driver_version must be a non-empty string")

    digest = hashlib.sha256(driver_version.encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def ensure_profiles_dir(profiles_dir: Path) -> None:
    """Create the profiles directory if it does not exist.

    The directory is created with owner-only permissions (0o700).

    Args:
        profiles_dir: Path to the profiles directory to create.

    Raises:
        OSError: If the directory cannot be created or permissions cannot be set.

    """
    profiles_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(profiles_dir, DIR_MODE_OWNER_ONLY)


def get_profile_path(
    profiles_dir: Path,
    gpu_identifier: str,
    backend: str,
    flavor: ProfileFlavor,
) -> Path:
    """Build a filesystem-safe path for a cached profile JSON file.

    The filename is constructed as::

        {sanitized_gpu_id}-{sanitized_backend}-{sanitized_flavor}.json

    Path traversal protection is applied: the resolved path is verified to
    remain within ``profiles_dir``.

    Args:
        profiles_dir: The root profiles directory.
        gpu_identifier: GPU identifier (e.g. ``"nvidia-geforce_rtx_3090-00"``).
        backend: Backend string (e.g. ``"cuda"`` or ``"sycl"``).
        flavor: Performance profile flavor enum.

    Returns:
        A ``Path`` object pointing to the profile JSON file.

    Raises:
        ValueError: If any component is empty after sanitization or if the
            resolved path escapes ``profiles_dir`` (path traversal attempt).

    """
    sanitized_gpu = _sanitize_filename_component(gpu_identifier)
    sanitized_backend = _sanitize_filename_component(backend)
    sanitized_flavor = _sanitize_filename_component(flavor.value)

    filename = f"{sanitized_gpu}-{sanitized_backend}-{sanitized_flavor}.json"
    candidate = (profiles_dir / filename).resolve()

    # Path traversal protection: ensure the resolved path is under profiles_dir
    profiles_dir_resolved = profiles_dir.resolve()
    if (
        not str(candidate).startswith(str(profiles_dir_resolved) + os.sep)
        and candidate != profiles_dir_resolved
    ):
        raise ValueError(
            f"profile path escapes profiles_dir: {filename} (resolved to {candidate})",
        )

    return candidate


def _atomic_write_json(file_path: Path, data: dict[str, Any]) -> None:
    """Atomically write a dict as JSON to *file_path*.

    Writes to a temporary file in the same directory, syncs to disk,
    then renames to the target path. Final permissions are set to
    owner-only (0o600).

    Args:
        file_path: Destination file path (must be on the same filesystem
            as the directory containing ``file_path``).
        data: Dictionary to serialize as JSON.

    Raises:
        OSError: If the write, sync, rename, or permission operation fails.

    """
    target_dir = file_path.parent
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_profile_",
        suffix=".json",
        dir=str(target_dir),
    )
    try:
        # Write JSON content
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        # Set permissions before rename so the final file has correct mode
        os.chmod(tmp_path, FILE_MODE_OWNER_ONLY)

        # Atomic rename
        os.replace(tmp_path, str(file_path))

        # Verify permissions after rename
        current_mode = os.stat(file_path).st_mode & 0o777
        if current_mode != FILE_MODE_OWNER_ONLY:
            raise OSError(
                f"profile file permissions mismatch after write: "
                f"expected {FILE_MODE_OWNER_ONLY:o}, got {current_mode:o}",
            )
    except BaseException:
        # Clean up temp file on any failure
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def write_profile(profiles_dir: Path, record: ProfileRecord) -> Path:
    """Write a ``ProfileRecord`` as an atomically-stored JSON file.

    Creates the profiles directory if needed, builds the target path using
    ``get_profile_path``, serialises the record, and performs an atomic write.

    Args:
        profiles_dir: Root directory for profile cache files.
        record: The profile record to persist.

    Returns:
        The path where the profile was written.

    Raises:
        ValueError: If path construction fails (invalid components or path
            traversal).
        OSError: If directory creation or file write fails.

    """
    ensure_profiles_dir(profiles_dir)

    profile_path = get_profile_path(
        profiles_dir,
        gpu_identifier=record.gpu_identifier,
        backend=record.backend,
        flavor=record.flavor,
    )

    _atomic_write_json(profile_path, record.to_dict())
    return profile_path


# ---------------------------------------------------------------------------
# Profile read / staleness helpers
# ---------------------------------------------------------------------------

_REQUIRED_PROFILE_FIELDS: frozenset[str] = frozenset(
    [
        "schema_version",
        "gpu_identifier",
        "backend",
        "flavor",
        "driver_version",
        "driver_version_hash",
        "server_binary_version",
        "profiled_at",
        "metrics",
    ],
)


def read_profile(
    profiles_dir: Path,
    gpu_identifier: str,
    backend: str,
    flavor: ProfileFlavor,
) -> ProfileRecord | None:
    """Read and validate a cached profile JSON file.

    Args:
        profiles_dir: Root directory for profile cache files.
        gpu_identifier: GPU identifier (e.g. ``"nvidia-geforce_rtx_3090-00"``).
        backend: Backend string (e.g. ``"cuda"`` or ``"sycl"``).
        flavor: Performance profile flavor enum.

    Returns:
        A deserialised ``ProfileRecord``, or ``None`` when the file does not
        exist, is corrupted, has missing required fields, or carries an
        unsupported schema version.

    """
    profile_path = get_profile_path(profiles_dir, gpu_identifier, backend, flavor)

    if not profile_path.is_file():
        return None

    try:
        raw = profile_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    # Check for missing required fields
    for field_name in _REQUIRED_PROFILE_FIELDS:
        if field_name not in data:
            return None

    # Reject unsupported schema versions
    schema_version = data.get("schema_version")
    if schema_version != _CURRENT_SCHEMA_VERSION:
        return None

    try:
        return ProfileRecord.from_dict(data)
    except (KeyError, TypeError, ValueError):
        return None


def check_staleness(
    record: ProfileRecord,
    current_driver_version: str,
    current_binary_version: str,
    staleness_days: int,
) -> StalenessResult:
    """Check whether a cached profile is considered stale.

    A profile is stale when **any** of the following conditions hold:

    1. Driver version hash changed since profiling.
    2. Server binary version changed since profiling.
    3. The profile age in days exceeds *staleness_days*.

    Args:
        record: The cached profile record to check.
        current_driver_version: Current GPU driver version string.
        current_binary_version: Current llama-server binary version string.
        staleness_days: Maximum acceptable age in days.

    Returns:
        A ``StalenessResult`` with ``is_stale`` set and a list of reasons
        when the profile is stale, or a non-stale result otherwise.

    """
    reasons: list[StalenessReason] = []

    # Condition 1: Driver changed
    current_driver_hash = compute_driver_version_hash(current_driver_version)
    if current_driver_hash != record.driver_version_hash:
        reasons.append(StalenessReason.DRIVER_CHANGED)

    # Condition 2: Binary changed
    if current_binary_version != record.server_binary_version:
        reasons.append(StalenessReason.BINARY_CHANGED)

    # Condition 3: Age exceeded
    try:
        profiled_dt = datetime.fromisoformat(record.profiled_at)
        # Treat naive datetimes as UTC
        if profiled_dt.tzinfo is None:
            profiled_dt = profiled_dt.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        age_days = (now - profiled_dt).total_seconds() / 86400.0
    except (ValueError, TypeError):
        # If we can't parse the timestamp, treat as stale
        age_days = float("inf")
        reasons.append(StalenessReason.AGE_EXCEEDED)
    else:
        if staleness_days > 0 and age_days > staleness_days:
            reasons.append(StalenessReason.AGE_EXCEEDED)

    return StalenessResult(
        is_stale=len(reasons) > 0,
        reasons=reasons,
        driver_version_display=record.driver_version,
        age_days=round(age_days, 2),
    )


def load_profile_with_staleness(
    profiles_dir: Path,
    gpu_identifier: str,
    backend: str,
    flavor: ProfileFlavor,
    current_driver_version: str,
    current_binary_version: str,
    staleness_days: int,
) -> tuple[ProfileRecord | None, StalenessResult | None]:
    """Read a cached profile and immediately check its staleness.

    Convenience wrapper around :func:`read_profile` and
    :func:`check_staleness`.

    Args:
        profiles_dir: Root directory for profile cache files.
        gpu_identifier: GPU identifier.
        backend: Backend string.
        flavor: Performance profile flavor.
        current_driver_version: Current GPU driver version string.
        current_binary_version: Current llama-server binary version.
        staleness_days: Maximum acceptable age in days.

    Returns:
        A tuple of ``(record, staleness_result)``.  When the profile
        file does not exist, both elements are ``None``.

    """
    record = read_profile(profiles_dir, gpu_identifier, backend, flavor)

    if record is None:
        return None, None

    staleness = check_staleness(
        record,
        current_driver_version=current_driver_version,
        current_binary_version=current_binary_version,
        staleness_days=staleness_days,
    )

    return record, staleness


def profile_to_override_dict(record: ProfileRecord) -> dict[str, Any]:
    """Extract profile parameters that are safe to override.

    Filters ``record.parameters`` through :data:`PROFILE_OVERRIDE_FIELDS`,
    returning only keys that are whitelisted for runtime override.

    Args:
        record: The profile record to extract from.

    Returns:
        A dict containing only the whitelisted parameter key-value pairs.

    """
    return {
        key: value for key, value in record.parameters.items() if key in PROFILE_OVERRIDE_FIELDS
    }
