"""Build status helper — read-only inspection of build artifacts and source state."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..config import Config
from .models import BuildArtifact, BuildBackend


@dataclass
class BuildStatus:
    """Snapshot of build state for a single backend.

    All fields are read-only observations — constructing a BuildStatus
    has no side effects beyond debug logging and filesystem reads.
    """

    backend: BuildBackend

    # Artifact info
    artifact_exists: bool
    artifact: BuildArtifact | None  # Parsed from build-artifact.json, or None
    binary_version_output: str | None  # llama-server --version stdout, or None

    # Local source
    source_exists: bool
    source_is_repo: bool
    source_branch: str | None
    source_head_sha: str | None
    source_remote_url: str | None

    # Remote
    configured_branch: str
    remote_branch_sha: str | None  # None if unreachable


def _run_git(args: list[str], *, cwd: Path | None = None, timeout: int = 5) -> str | None:
    """Run a git command and return stripped stdout, or None on any failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            cwd=cwd,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("[status] git command failed: %s — %s", " ".join(args), exc)
        return None


def _parse_artifact_json(json_path: Path) -> BuildArtifact | None:
    """Deserialize build-artifact.json into a BuildArtifact instance."""
    raw = json_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    return BuildArtifact(
        artifact_type=data.get("artifact_type", "llama-server"),
        backend=BuildBackend(data.get("backend", "sycl")),
        created_at=float(data.get("created_at", 0)),
        git_remote_url=data.get("git_remote_url", ""),
        git_commit_sha=data.get("git_commit_sha", "unknown"),
        git_branch=data.get("git_branch", ""),
        build_command=data.get("build_command", []),
        build_duration_seconds=float(data.get("build_duration_seconds", 0)),
        exit_code=int(data.get("exit_code", -1)),
        binary_path=Path(data["binary_path"]) if data.get("binary_path") else None,
        binary_size_bytes=int(data["binary_size_bytes"]) if data.get("binary_size_bytes") else None,
        build_log_path=Path(data["build_log_path"]) if data.get("build_log_path") else None,
        failure_report_path=(
            Path(data["failure_report_path"]) if data.get("failure_report_path") else None
        ),
    )


def get_build_status(backend: BuildBackend, config: Config) -> BuildStatus:
    """Gather read-only build status for a single backend.

    Inspects the build artifact JSON, probes the binary version, and
    collects local/remote git information.  All subprocess calls are
    wrapped with short timeouts and graceful error handling.

    Args:
        backend: The backend to inspect (SYCL or CUDA).
        config: Application-level Config providing paths and defaults.

    Returns:
        BuildStatus snapshot with no side effects.
    """
    builds_dir = config.builds_dir
    artifact_json = builds_dir / backend.value / "build-artifact.json"

    # 1. Built artifact info
    artifact: BuildArtifact | None = None
    artifact_exists = artifact_json.is_file()
    if artifact_exists:
        try:
            artifact = _parse_artifact_json(artifact_json)
            logger.debug("[status] %s artifact loaded from %s", backend.value, artifact_json)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("[status] %s artifact JSON parse error: %s", backend.value, exc)
            artifact = None

    # 2. Binary version probe
    binary_version_output: str | None = None
    if artifact is not None and artifact.binary_path is not None:
        binary_path = artifact.binary_path
        if binary_path.is_file():
            logger.debug("[status] probing binary version: %s", binary_path)
            try:
                r = subprocess.run(
                    [str(binary_path), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if r.returncode == 0 and r.stdout.strip():
                    binary_version_output = r.stdout.strip()
                    logger.debug(
                        "[status] %s binary version: %s", backend.value, binary_version_output
                    )
                else:
                    logger.debug(
                        "[status] %s binary --version returned non-zero or empty", backend.value
                    )
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
                logger.debug("[status] %s binary version probe failed: %s", backend.value, exc)
        else:
            logger.debug("[status] %s binary path does not exist: %s", backend.value, binary_path)

    # 3. Local source clone info
    source_dir = Path(config.llama_cpp_root)
    source_exists = source_dir.is_dir()
    source_is_repo = False
    source_branch: str | None = None
    source_head_sha: str | None = None
    source_remote_url: str | None = None

    if source_exists:
        logger.debug("[status] checking source at %s", source_dir)
        # Check if it's a git repo
        is_repo_output = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=source_dir)
        source_is_repo = is_repo_output == "true"

        if source_is_repo:
            source_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=source_dir)
            source_head_sha = _run_git(["rev-parse", "HEAD"], cwd=source_dir)
            source_remote_url = _run_git(["remote", "get-url", "origin"], cwd=source_dir)

    # 4. Remote branch HEAD
    remote_branch_sha: str | None = None
    remote_url = config.build_git_remote
    branch = config.build_git_branch
    logger.debug("[status] querying remote %s for refs/heads/%s", remote_url, branch)
    ls_remote_output = _run_git(
        ["ls-remote", remote_url, f"refs/heads/{branch}"],
        timeout=10,
    )
    if ls_remote_output:
        # Format: <sha>\trefs/heads/<branch>
        parts = ls_remote_output.split("\t")
        if parts:
            remote_branch_sha = parts[0]
            logger.debug(
                "[status] remote HEAD for %s/%s = %s",
                remote_url,
                branch,
                remote_branch_sha,
            )

    return BuildStatus(
        backend=backend,
        artifact_exists=artifact_exists,
        artifact=artifact,
        binary_version_output=binary_version_output,
        source_exists=source_exists,
        source_is_repo=source_is_repo,
        source_branch=source_branch,
        source_head_sha=source_head_sha,
        source_remote_url=source_remote_url,
        configured_branch=branch,
        remote_branch_sha=remote_branch_sha,
    )
