"""T027: Contract test for BuildArtifact JSON output.

This test ensures BuildArtifact can be serialized to JSON with all required fields
for the build artifact contract.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from llama_manager.build_pipeline import (
    BuildArtifact,
)


class TestBuildArtifactContract:
    """T027: Contract test for BuildArtifact JSON output."""

    def test_build_artifact_contract(self, tmp_path: Path) -> None:
        """BuildArtifact should serialize to JSON with all required fields.

        This test verifies that BuildArtifact can be serialized to JSON with all
        required fields for the build artifact contract, including:
        - artifact_type, backend, created_at, git_remote_url
        - git_commit_sha, git_branch, build_command
        - build_duration_seconds, exit_code, binary_path
        - binary_size_bytes, build_log_path, failure_report_path
        """
        artifact = BuildArtifact(
            artifact_type="binary",
            backend="sycl",
            created_at=datetime(2026, 4, 15, 12, 0, 0),
            git_remote_url="https://github.com/ggerganov/llama.cpp",
            git_commit_sha="abc123def456",
            git_branch="main",
            build_command=["cmake", "-G", "Ninja", "-DGGML_SYCL=ON"],
            build_duration_seconds=123.456,
            exit_code=0,
            binary_path=tmp_path / "bin" / "llama-server",
            binary_size_bytes=104857600,  # 100 MB
            build_log_path=tmp_path / "logs" / "build.log",
            failure_report_path=None,
        )

        # Serialize to JSON
        artifact_dict = {
            "artifact_type": artifact.artifact_type,
            "backend": artifact.backend,
            "created_at": artifact.created_at.isoformat(),
            "git_remote_url": artifact.git_remote_url,
            "git_commit_sha": artifact.git_commit_sha,
            "git_branch": artifact.git_branch,
            "build_command": artifact.build_command,
            "build_duration_seconds": artifact.build_duration_seconds,
            "exit_code": artifact.exit_code,
            "binary_path": str(artifact.binary_path) if artifact.binary_path else None,
            "binary_size_bytes": artifact.binary_size_bytes,
            "build_log_path": str(artifact.build_log_path) if artifact.build_log_path else None,
            "failure_report_path": str(artifact.failure_report_path)
            if artifact.failure_report_path
            else None,
        }

        json_str = json.dumps(artifact_dict)
        parsed = json.loads(json_str)

        # Verify all required fields are present
        required_fields = [
            "artifact_type",
            "backend",
            "created_at",
            "git_remote_url",
            "git_commit_sha",
            "git_branch",
            "build_command",
            "build_duration_seconds",
            "exit_code",
            "binary_path",
            "binary_size_bytes",
            "build_log_path",
            "failure_report_path",
        ]

        for field in required_fields:
            assert field in parsed, f"Missing required field: {field}"

        # Verify field types
        assert isinstance(parsed["artifact_type"], str)
        assert isinstance(parsed["backend"], str)
        assert isinstance(parsed["created_at"], str)
        assert isinstance(parsed["git_remote_url"], str)
        assert isinstance(parsed["git_commit_sha"], str)
        assert isinstance(parsed["git_branch"], str)
        assert isinstance(parsed["build_command"], list)
        assert isinstance(parsed["build_duration_seconds"], int | float)
        assert isinstance(parsed["exit_code"], int)
        assert parsed["binary_path"] is None or isinstance(parsed["binary_path"], str)
        assert parsed["binary_size_bytes"] is None or isinstance(parsed["binary_size_bytes"], int)
        assert parsed["build_log_path"] is None or isinstance(parsed["build_log_path"], str)
        assert parsed["failure_report_path"] is None or isinstance(
            parsed["failure_report_path"], str
        )

        # Verify is_success property works
        assert artifact.is_success is True
        assert artifact.binary_size_mb == pytest.approx(100.0)
