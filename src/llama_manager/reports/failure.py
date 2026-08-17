"""Failure report — structured build failure reporting."""

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..common.security import redact_sensitive


@dataclass
class FailureReport:
    """Structured report of a failed build attempt.

    This dataclass captures comprehensive information about a build failure,
    including the build artifact state, full build output log, and structured
    error details for actionable debugging.
    """

    report_dir: Path
    timestamp: datetime
    build_artifact_json: str
    build_output_log: str
    error_details_json: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def report_path(self) -> Path:
        """Get the path to the report file.

        Returns:
            Path to the JSON report file in the report directory.
        """
        return self.report_dir / f"failure_{self.timestamp.isoformat()}.json"

    def save_to_file(self) -> Path:
        """Save the failure report to disk as JSON.

        Returns:
            Path to the saved report file.

        Raises:
            IOError: If the report cannot be written to disk.
        """
        report_data: dict[str, Any] = {
            "report_dir": str(self.report_dir),
            "timestamp": self.timestamp.isoformat(),
            "build_artifact": self.build_artifact_json,
            "build_output_log": self.build_output_log,
            "error_details": self.error_details_json,
            "metadata": self.metadata,
        }

        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_path

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Set restrictive permissions (owner read/write only)
        os.chmod(report_path, 0o600)

        return report_path


def write_failure_report(
    report_dir: Path | None = None,
    build_artifact_json: str = "",
    build_output: str = "",
    error_details: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> FailureReport:
    """Write a build failure report to disk.

    Creates a timestamped report directory with:
    - build-artifact.json
    - build-output.log (truncated and redacted)
    - error-details.json

    Args:
        report_dir: Optional custom report directory. If not provided,
            uses Config().reports_dir
        build_artifact_json: JSON string of the build artifact
        build_output: Raw build output to log (will be truncated and redacted)
        error_details: List of error details to serialize
        metadata: Optional metadata dictionary

    Returns:
        FailureReport instance with the created report information

    Raises:
        IOError: If the report cannot be written
    """
    from ..config import Config

    config = Config()
    if report_dir is None:
        report_dir = config.paths.reports_dir

    # Create timestamp-only directory name (no backend suffix)
    timestamp = datetime.now(UTC)
    timestamp_dir = report_dir / timestamp.strftime("%Y%m%d_%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Set directory permissions to 0700
    timestamp_dir.chmod(0o700)

    # Truncate and redact build output
    max_output_len = config.build.output_truncate_bytes
    truncated_output = build_output[:max_output_len]
    redacted_output = redact_sensitive(truncated_output)

    # Serialize error details to JSON
    if error_details is None:
        error_details = []
    error_json = json.dumps(error_details, indent=2, default=str, ensure_ascii=False)

    # Write build-artifact.json
    artifact_path = timestamp_dir / "build-artifact.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write(build_artifact_json)
    artifact_path.chmod(0o600)

    # Write build-output.log
    output_path = timestamp_dir / "build-output.log"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(redacted_output)
    output_path.chmod(0o600)

    # Write error-details.json
    errors_path = timestamp_dir / "error-details.json"
    with open(errors_path, "w", encoding="utf-8") as f:
        f.write(error_json)
    errors_path.chmod(0o600)

    # Serialize build artifact to JSON (assume it's already JSON)
    artifact_data = build_artifact_json

    return FailureReport(
        report_dir=timestamp_dir,
        timestamp=timestamp,
        build_artifact_json=artifact_data,
        build_output_log=redacted_output,
        error_details_json=error_json,
        metadata=metadata or {},
    )
