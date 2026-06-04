"""Failure report — structured build failure reporting."""

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .redaction import redact_sensitive
from .rotation import _rotate_mutating_log


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


@dataclass
class MutatingActionLogEntry:
    """Log entry for mutating actions (git clone, checkout, file operations).

    This dataclass records all state-changing operations performed during
    the build process, including exit codes, truncated output for large
    outputs, and whether redaction was applied for security.
    """

    command: list[str]
    timestamp: datetime
    exit_code: int
    truncated_output: str
    redaction_applied: bool = False
    duration_seconds: float | None = None
    working_dir: Path | None = None
    output_truncated: bool = False

    @property
    def is_success(self) -> bool:
        """Check if the mutating action succeeded."""
        return self.exit_code == 0

    @property
    def was_truncated(self) -> bool:
        """Check if the output was truncated.

        Returns True when truncation was explicitly applied during logging.
        """
        return self.output_truncated

    def format_summary(self) -> str:
        """Generate a human-readable summary of this log entry.

        Returns:
            Formatted summary string suitable for logging or display.
        """
        status = "SUCCESS" if self.is_success else "FAILED"
        duration = f" in {self.duration_seconds:.2f}s" if self.duration_seconds else ""
        truncation = " [TRUNCATED]" if self.was_truncated else ""
        redaction = " [REDACTED]" if self.redaction_applied else ""

        return (
            f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"{status}: {' '.join(self.command)}{duration}{truncation}{redaction}"
        )

    def get_output_with_markers(self, max_length: int = 1000) -> str:
        """Get the output with markers indicating truncation.

        Args:
            max_length: Maximum length of output to return.

        Returns:
            Output string with truncation markers if applicable.
        """
        if len(self.truncated_output) <= max_length:
            return self.truncated_output

        truncated = self.truncated_output[:max_length]
        return f"{truncated}\n... [output truncated] ..."


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


def log_mutating_action(
    command: list[str],
    exit_code: int,
    output: str,
    working_dir: Path | None = None,
) -> MutatingActionLogEntry:
    """Log a mutating action (git clone, venv creation, file operations).

    Records all state-changing operations performed during the build process,
    including exit codes, truncated output for large outputs, and whether
    redaction was applied for security.

    Args:
        command: Command that was executed (as list for subprocess safety)
        exit_code: Exit code from the command
        output: stdout/stderr output from the command
        working_dir: Working directory where the command was executed

    Returns:
        MutatingActionLogEntry with the logged action details

    Examples:
        >>> entry = log_mutating_action(
        ...     command=["git", "clone", "https://github.com/example/repo.git"],
        ...     exit_code=0,
        ...     output="Cloning into 'repo'...",
        ... )
        >>> entry.is_success
        True
    """
    from ..config import Config

    config = Config()
    timestamp = datetime.now(UTC)

    # Redact sensitive information
    redaction_applied = False
    if any(kw in output.upper() for kw in ["KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH"]):
        output = redact_sensitive(output)
        redaction_applied = True

    # Truncate output if too large
    max_output_len = config.build.output_truncate_bytes
    output_truncated = len(output) > max_output_len
    truncated_output = output[:max_output_len]

    entry = MutatingActionLogEntry(
        command=command,
        timestamp=timestamp,
        exit_code=exit_code,
        truncated_output=truncated_output,
        redaction_applied=redaction_applied,
        duration_seconds=None,  # Would need timing info from caller
        working_dir=working_dir,
        output_truncated=output_truncated,
    )

    # Write to XDG state home
    log_path = Path(config.paths.xdg_state_base) / "llm-runner" / "mutating_actions.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Append entry to log file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry.format_summary() + "\n")

    # Rotate if exceeds max entries
    _rotate_mutating_log(log_path, max_entries=1000)

    return entry
