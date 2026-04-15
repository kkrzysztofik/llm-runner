# Build failure reporting and logging for M2

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def redact_sensitive(text: str) -> str:
    """Redact sensitive information from text.

    Replaces patterns matching KEY|TOKEN|SECRET|PASSWORD|AUTH with [REDACTED].
    Uses case-insensitive regex matching.

    Args:
        text: Input text to redact

    Returns:
        Text with sensitive patterns replaced by [REDACTED]

    Examples:
        >>> redact_sensitive("API_KEY=abc123")
        'API_[REDACTED]=abc123'
        >>> redact_sensitive("password: secret123")
        'password: [REDACTED]123'
    """
    # Pattern matches KEY, TOKEN, SECRET, PASSWORD, AUTH (case-insensitive)
    # followed by optional characters, then replaces the sensitive value
    pattern = r"(?:KEY|TOKEN|SECRET|PASSWORD|AUTH)\w*[:\s]*[^,\s\n]+|(?<![a-zA-Z])(?:KEY|TOKEN|SECRET|PASSWORD|AUTH)(?![a-zA-Z])"

    # First pass: replace key=value patterns
    def replace_key_value(match: re.Match) -> str:
        matched = match.group(0)
        # Check if this looks like a key=value pattern
        if ":" in matched or "=" in matched or " " in matched:
            # Replace only the value part
            parts = re.split(r"[:=]", matched, maxsplit=1)
            if len(parts) == 2:
                return f"{parts[0]}: [REDACTED]"
        return "[REDACTED]"

    result = re.sub(pattern, replace_key_value, text, flags=re.IGNORECASE)

    # Second pass: replace standalone sensitive words
    result = re.sub(
        r"\b(KEY|TOKEN|SECRET|PASSWORD|AUTH)\b", "[REDACTED]", result, flags=re.IGNORECASE
    )

    return result


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
        import json

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

    @property
    def is_success(self) -> bool:
        """Check if the mutating action succeeded."""
        return self.exit_code == 0

    @property
    def was_truncated(self) -> bool:
        """Check if the output was truncated.

        Returns True when the output length exceeds the expected max length
        (command length * 1000), indicating truncation occurred.
        """
        return len(self.truncated_output) >= len(self.command) * 1000

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
    from .config import Config

    config = Config()
    if report_dir is None:
        report_dir = config.reports_dir

    # Create timestamp-only directory name (no backend suffix)
    timestamp = datetime.now()
    timestamp_dir = report_dir / timestamp.strftime("%Y%m%d_%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Set directory permissions to 0700
    timestamp_dir.chmod(0o700)

    # Truncate and redact build output
    max_output_len = config.build_output_truncate_bytes
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

    # Serialize build artifact if not already JSON
    artifact_data = build_artifact_json
    if not artifact_data.strip().startswith("{") and not artifact_data.strip().startswith("["):
        try:
            artifact_data = json.dumps(json.loads(artifact_data), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            artifact_data = json.dumps({"raw": artifact_data}, indent=2, ensure_ascii=False)

    return FailureReport(
        report_dir=timestamp_dir,
        timestamp=timestamp,
        build_artifact_json=artifact_data,
        build_output_log=redacted_output,
        error_details_json=error_json,
        metadata=metadata or {},
    )


def rotate_reports(config: Any = None) -> None:
    """Rotate old report directories.

    Scans the reports directory and deletes oldest report directories
    when count exceeds Config.build_max_reports. Uses FIFO rotation
    (oldest first).

    Args:
        config: Optional Config instance. If not provided, creates default.
    """
    from .config import Config

    config = Config() if config is None else config  # type: ignore[assignment]

    reports_path = config.reports_dir  # type: ignore[union-attr]

    if not reports_path.exists():
        return

    # Get all report directories (directories starting with timestamp pattern)
    report_dirs: list[Path] = []
    for entry in reports_path.iterdir():
        if entry.is_dir() and entry.name.startswith("20"):
            # Check if it looks like a timestamp directory (YYYYMMDD_HHMMSS)
            try:
                datetime.strptime(entry.name, "%Y%m%d_%H%M%S")
                report_dirs.append(entry)
            except ValueError:
                continue

    # Sort by modification time (oldest first)
    report_dirs.sort(key=lambda p: p.stat().st_mtime)

    # Delete oldest directories if count exceeds max
    max_reports = config.build_max_reports  # type: ignore[union-attr]
    if len(report_dirs) > max_reports:
        to_delete = report_dirs[: len(report_dirs) - max_reports]
        for report_dir in to_delete:
            try:
                import shutil

                shutil.rmtree(report_dir)
            except OSError:
                # Log but don't fail on deletion errors
                pass
