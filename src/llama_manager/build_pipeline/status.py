"""Build status helper — read-only inspection of build artifacts and source state."""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..config import Config
from .models import BuildArtifact, BuildBackend
from .utils import get_build_env_cmd


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
    binary_exists_untracked: bool  # True when Config default binary exists without provenance
    untracked_binary_path: Path | None  # Path to binary found outside build-artifact.json

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
        logger.debug("[status] git command failed: {} — {}", " ".join(args), exc)
        return None


def _default_binary_path(backend: BuildBackend, config: Config) -> Path:
    """Return the Config default llama-server path for a backend."""
    if backend == BuildBackend.SYCL:
        return Path(config.paths.llama_server_bin_intel)
    return Path(config.paths.llama_server_bin_nvidia)


def _extract_llama_server_version(stdout: str, stderr: str) -> str | None:
    """Parse ``version: …`` from llama-server --version output (stdout or stderr)."""
    for line in f"{stdout}\n{stderr}".splitlines():
        stripped = line.strip()
        if stripped.startswith("version:"):
            return stripped
    return None


def _probe_binary_version(binary_path: Path, backend: BuildBackend) -> str | None:
    """Run ``llama-server --version`` on the executable and return the version line."""
    if not binary_path.is_file():
        logger.debug("[status] %s binary path does not exist: %s", backend.value, binary_path)
        return None
    logger.debug("[status] probing binary version: %s", binary_path)
    cmd = get_build_env_cmd([str(binary_path), "--version"], backend)
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy(),
            cwd=binary_path.parent,
        )
        version_line = _extract_llama_server_version(r.stdout, r.stderr)
        if version_line:
            logger.debug("[status] %s binary version: %s", backend.value, version_line)
            return version_line
        if r.returncode == 0:
            combined = f"{r.stdout}\n{r.stderr}".strip()
            if combined:
                first_line = combined.splitlines()[0].strip()
                if first_line and not re.match(r"^(⚠️|warning:|\[Thread)", first_line):
                    logger.debug(
                        "[status] %s binary version (fallback): %s", backend.value, first_line
                    )
                    return first_line
        logger.debug(
            "[status] %s binary --version produced no parseable version (exit %s)",
            backend.value,
            r.returncode,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("[status] %s binary version probe failed: %s", backend.value, exc)
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


def _load_artifact_from_json(
    backend: BuildBackend, artifact_json: Path
) -> tuple[BuildArtifact | None, bool]:
    artifact_exists = artifact_json.is_file()
    if not artifact_exists:
        return None, False
    try:
        artifact = _parse_artifact_json(artifact_json)
        logger.debug("[status] {} artifact loaded from {}", backend.value, artifact_json)
        return artifact, True
    except (KeyError, TypeError, ValueError, OSError) as exc:
        logger.warning("[status] %s artifact JSON parse error: {}", backend.value, exc)
        return None, True


def _resolve_binary_version(
    backend: BuildBackend,
    config: Config,
    artifact: BuildArtifact | None,
    artifact_exists: bool,
) -> tuple[str | None, Path | None, bool]:
    binary_version_output: str | None = None
    if artifact is not None and artifact.binary_path is not None:
        binary_version_output = _probe_binary_version(artifact.binary_path, backend)

    untracked_binary_path: Path | None = None
    binary_exists_untracked = False
    if artifact_exists:
        return binary_version_output, untracked_binary_path, binary_exists_untracked

    fallback_path = _default_binary_path(backend, config)
    if not fallback_path.is_file():
        return binary_version_output, untracked_binary_path, binary_exists_untracked

    binary_exists_untracked = True
    untracked_binary_path = fallback_path
    logger.debug(
        "[status] %s untracked binary at %s (no build-artifact.json)",
        backend.value,
        fallback_path,
    )
    if binary_version_output is None:
        binary_version_output = _probe_binary_version(fallback_path, backend)
    return binary_version_output, untracked_binary_path, binary_exists_untracked


def _source_git_info(
    source_dir: Path,
) -> tuple[bool, bool, str | None, str | None, str | None]:
    source_exists = source_dir.is_dir()
    if not source_exists:
        return False, False, None, None, None

    logger.debug("[status] checking source at %s", source_dir)
    is_repo_output = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=source_dir)
    source_is_repo = is_repo_output == "true"
    if not source_is_repo:
        return source_exists, False, None, None, None

    source_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=source_dir)
    source_head_sha = _run_git(["rev-parse", "HEAD"], cwd=source_dir)
    source_remote_url = _run_git(["remote", "get-url", "origin"], cwd=source_dir)
    return source_exists, source_is_repo, source_branch, source_head_sha, source_remote_url


def _fetch_remote_branch_sha(config: Config) -> str | None:
    remote_url = config.build.git_remote
    branch = config.build.git_branch
    logger.debug("[status] querying remote %s for refs/heads/%s", remote_url, branch)
    ls_remote_output = _run_git(
        ["ls-remote", remote_url, f"refs/heads/{branch}"],
        timeout=10,
    )
    if not ls_remote_output:
        return None
    parts = ls_remote_output.split("\t")
    if not parts:
        return None
    remote_branch_sha = parts[0]
    logger.debug(
        "[status] remote HEAD for %s/%s = %s",
        remote_url,
        branch,
        remote_branch_sha,
    )
    return remote_branch_sha


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
    artifact_json = config.paths.builds_dir / backend.value / "build-artifact.json"
    artifact, artifact_exists = _load_artifact_from_json(backend, artifact_json)
    binary_version_output, untracked_binary_path, binary_exists_untracked = _resolve_binary_version(
        backend, config, artifact, artifact_exists
    )
    source_dir = Path(config.paths.llama_cpp_root)
    source_exists, source_is_repo, source_branch, source_head_sha, source_remote_url = (
        _source_git_info(source_dir)
    )
    branch = config.build.git_branch
    remote_branch_sha = _fetch_remote_branch_sha(config)

    return BuildStatus(
        backend=backend,
        artifact_exists=artifact_exists,
        artifact=artifact,
        binary_version_output=binary_version_output,
        binary_exists_untracked=binary_exists_untracked,
        untracked_binary_path=untracked_binary_path,
        source_exists=source_exists,
        source_is_repo=source_is_repo,
        source_branch=source_branch,
        source_head_sha=source_head_sha,
        source_remote_url=source_remote_url,
        configured_branch=branch,
        remote_branch_sha=remote_branch_sha,
    )
