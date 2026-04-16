"""T027: Contract test for BuildArtifact JSON output.

This test ensures BuildArtifact can be serialized to JSON with all required fields
for the build artifact contract.
"""

import json
import time
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
            artifact_type="llama-server",
            backend="sycl",
            created_at=time.time(),
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

        # Serialize to JSON using to_dict() method
        artifact_dict = artifact.to_dict()
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
        assert isinstance(parsed["created_at"], float)
        assert isinstance(parsed["git_remote_url"], str)
        assert isinstance(parsed["git_commit_sha"], str)
        assert isinstance(parsed["git_branch"], str)
        assert isinstance(parsed["build_command"], list)
        assert isinstance(parsed["build_duration_seconds"], (int, float))
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
