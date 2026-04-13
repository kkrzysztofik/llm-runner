"""Test fixtures for llm-runner Phase 1 QA.

Provides shared test fixtures for:
- T004: tmp_runtime_dir, sample_lockfile, artifact_writer
- Test helpers for runtime directory resolution and artifact management
"""

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def tmp_runtime_dir(tmp_path: Path) -> Path:
    """Create a temporary runtime directory for testing.

    This fixture provides an isolated runtime directory for testing
    resolve_runtime_dir() and related functionality.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture.

    Returns:
        Path to a temporary directory that exists and is writable.

    """
    return tmp_path


@pytest.fixture
def sample_lockfile(tmp_runtime_dir: Path) -> Path:
    """Create a sample lockfile with valid structure.

    Args:
        tmp_runtime_dir: The temporary runtime directory from fixture.

    Returns:
        Path to the created lockfile.

    """
    lockfile = tmp_runtime_dir / "llm-runner.lock"
    lockfile.write_text(
        json.dumps({"version": "1", "pids": [12345], "started_at": "2026-04-10T00:00:00Z"})
    )
    return lockfile


@pytest.fixture
def artifact_writer(tmp_runtime_dir: Path) -> Any:
    """Create an artifact writer utility for testing.

    Provides a simple utility to write test artifacts to the runtime directory
    and verify their existence.

    Args:
        tmp_runtime_dir: The temporary runtime directory from fixture.

    Returns:
        A function that takes (filename, content, expected_type) and writes
        the artifact, returning the Path to the created file.

    """

    def write_artifact(filename: str, content: str | bytes, expected_type: str = "text") -> Path:
        """Write an artifact to the runtime directory.

        Args:
            filename: Name of the artifact file.
            content: Content to write (string or bytes).
            expected_type: Expected file type ("text" or "binary").

        Returns:
            Path to the created artifact file.

        """
        artifact_path = tmp_runtime_dir / filename
        if isinstance(content, str):
            artifact_path.write_text(content)
        else:
            artifact_path.write_bytes(content)

        assert artifact_path.exists(), f"Artifact {filename} not created"
        return artifact_path

    return write_artifact
