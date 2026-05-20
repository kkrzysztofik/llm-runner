"""Shared mutable build context passed across pipeline stages within a single run."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .models import BuildArtifact, BuildBackend, BuildConfig, BuildProgress
from .utils import _format_command, _redact_build_text


@dataclass
class _BuildContext:
    """Mutable state shared across all pipeline stages for a single build run."""

    config: BuildConfig
    dry_run: bool
    build_start_time: float = 0.0
    build_output: str = ""
    last_build_command: list[str] = field(default_factory=list)
    last_exit_code: int = 1
    progress_callback: Callable[[BuildProgress], None] | None = None
    cancel_event: threading.Event | None = None
    active_proc: subprocess.Popen[str] | None = None

    def append_command_output(
        self,
        *,
        stage: str,
        command: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Append structured command output to the build report payload."""
        self.last_build_command = command
        self.last_exit_code = returncode
        entry = (
            f"## {stage}\n"
            f"COMMAND: {_format_command(command)}\n"
            f"EXIT_CODE: {returncode}\n\n"
            f"STDOUT:\n{_redact_build_text(stdout)}\n\n"
            f"STDERR:\n{_redact_build_text(stderr)}\n"
        )
        self.build_output = f"{self.build_output}\n\n{entry}" if self.build_output else entry

    def build_reports_dir(self) -> Path:
        """Return the directory used for build logs and failure reports."""
        return Path(self.config.output_dir).parent / "reports"

    def write_build_log(self) -> Path | None:
        """Persist captured command output, returning the path only when written."""
        if not self.build_output:
            return None
        backend_name = (
            self.config.backend.value
            if isinstance(self.config.backend, BuildBackend)
            else str(self.config.backend)
        )
        reports_dir = self.build_reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        build_log_path = (
            reports_dir / f"{int(time.time())}-{time.monotonic_ns()}-{backend_name}.log"
        )
        build_log_path.write_text(_redact_build_text(self.build_output))
        return build_log_path

    def create_artifact(
        self,
        *,
        exit_code: int,
        build_log_path: Path | None,
        failure_report_path: Path | None,
        binary_path: Path | None = None,
        binary_size_bytes: int | None = None,
        git_commit_sha: str = "unknown",
        build_command: list[str] | None = None,
    ) -> BuildArtifact:
        """Create build provenance for success or failed command stages."""
        cmd = (
            build_command
            or self.last_build_command
            or ["cmake", "--build", str(self.config.build_dir)]
        )
        return BuildArtifact(
            artifact_type="llama-server",
            backend=self.config.backend,
            created_at=time.time(),
            git_remote_url=_redact_build_text(self.config.git_remote_url),
            git_commit_sha=git_commit_sha,
            git_branch=self.config.git_branch,
            build_command=cmd,
            build_duration_seconds=time.time() - self.build_start_time,
            exit_code=exit_code,
            binary_path=binary_path,
            binary_size_bytes=binary_size_bytes,
            build_log_path=build_log_path,
            failure_report_path=failure_report_path,
        )

    def write_failure_artifact(self, progress: BuildProgress) -> BuildArtifact:
        """Write failure diagnostics and return an artifact pointing to them."""
        from ..reports import write_failure_report

        logger.info("[failure] writing failure diagnostics for stage=%s", progress.stage)
        build_log_path = self.write_build_log()
        artifact = self.create_artifact(
            exit_code=self.last_exit_code,
            build_log_path=build_log_path,
            failure_report_path=None,
            build_command=self.last_build_command,
        )
        report = write_failure_report(
            report_dir=self.build_reports_dir(),
            build_artifact_json=json.dumps(artifact.to_dict(), indent=2),
            build_output=self.build_output,
            error_details=[
                {
                    "stage": progress.stage,
                    "status": progress.status,
                    "message": _redact_build_text(progress.message),
                }
            ],
            metadata={"backend": self.config.backend.value},
        )
        artifact.failure_report_path = report.report_dir
        report.build_artifact_json = json.dumps(artifact.to_dict(), indent=2)
        report.save_to_file()
        logger.info("[failure] report written to %s", report.report_dir)
        return artifact
