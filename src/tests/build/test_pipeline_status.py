"""Tests for BuildStatus and get_build_status().

Tests the read-only build status inspection:
- Artifact existence and JSON parsing
- Binary version probing
- Local source git information
- Remote branch HEAD
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path
from unittest.mock import Mock, patch

from llama_manager.build_pipeline import BuildBackend, BuildStatus
from llama_manager.build_pipeline.status import (
    _extract_llama_server_version,
    _probe_binary_version,
    get_build_status,
)
from llama_manager.config import Config
from llama_manager.config.defaults import BuildPipelineConfig, PathsConfig

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_config(tmp_path: Path) -> Config:
    """Create an isolated Config with temp dirs.

    config.paths.builds_dir returns Path(xdg_state_base) / "llm-runner" / "builds".
    We set xdg_state_base so that builds_dir lives under tmp_path.
    """
    state_root = tmp_path / "state_root"
    builds_dir = state_root / "llm-runner" / "builds"
    builds_dir.mkdir(parents=True)

    source_dir = tmp_path / "llama.cpp"
    source_dir.mkdir(parents=True)

    return Config(
        paths=PathsConfig(
            llama_cpp_root=str(source_dir),
            xdg_state_base=str(state_root),
            xdg_cache_base=str(tmp_path),
            xdg_data_base=str(tmp_path),
        ),
        build=BuildPipelineConfig(),
    )


def _write_artifact_json(config: Config, backend: str, tmp_path: Path, **overrides: object) -> Path:
    """Write a build-artifact.json under config.paths.builds_dir/<backend>/."""
    backend_dir = config.paths.builds_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = backend_dir / "build-artifact.json"

    defaults: dict[str, object] = {
        "artifact_type": "llama-server",
        "backend": backend,
        "created_at": time.time(),
        "git_remote_url": "https://github.com/ggerganov/llama.cpp.git",
        "git_commit_sha": "abcdef1234567890abcdef1234567890abcdef12",
        "git_branch": "master",
        "build_command": ["cmake", "--build"],
        "build_duration_seconds": 60.0,
        "exit_code": 0,
        "binary_path": str(tmp_path / "bin" / "llama-server"),
        "binary_size_bytes": 104857600,
        "build_log_path": str(tmp_path / "build.log"),
        "failure_report_path": None,
    }
    defaults.update(overrides)

    artifact_path.write_text(json.dumps(defaults))
    return artifact_path


def _init_git_repo(repo_dir: Path, branch: str = "main") -> str:
    """Initialize a git repo with one commit and return the commit SHA."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    orig_cwd = os.getcwd()
    try:
        os.chdir(repo_dir)
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], check=True, capture_output=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], check=True, capture_output=True)
        (repo_dir / "README.md").write_text("# test repo")
        subprocess.run(["git", "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], check=True, capture_output=True)
        if branch != "main":
            subprocess.run(["git", "branch", "-M", branch], check=True, capture_output=True)
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/ggerganov/llama.cpp.git"],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    finally:
        os.chdir(orig_cwd)


# ── Artifact tests ───────────────────────────────────────────────────────────


class TestArtifactExists:
    """Tests for artifact existence and JSON parsing."""

    def test_artifact_exists_false_when_no_file(self, tmp_path: Path) -> None:
        """get_build_status should return artifact_exists=False when no artifact file."""
        config = _make_config(tmp_path)

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("llama_manager.build_pipeline.status._probe_binary_version", return_value=None),
        ):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.artifact_exists is False
        assert status.artifact is None
        assert status.binary_version_output is None
        assert status.binary_exists_untracked is False
        assert status.untracked_binary_path is None

    def test_artifact_exists_true_parses_json(self, tmp_path: Path) -> None:
        """get_build_status should parse artifact JSON and set artifact fields."""
        config = _make_config(tmp_path)
        _write_artifact_json(
            config,
            "sycl",
            tmp_path,
            git_commit_sha="deadbeef1234567890deadbeef1234567890deadbeef",
            binary_path=str(tmp_path / "bin" / "llama-server"),
        )

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("llama_manager.build_pipeline.status._probe_binary_version", return_value=None),
        ):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.artifact_exists is True
        assert status.artifact is not None
        assert status.artifact.git_commit_sha == "deadbeef1234567890deadbeef1234567890deadbeef"
        assert status.artifact.binary_path == tmp_path / "bin" / "llama-server"
        assert status.binary_exists_untracked is False

    def test_artifact_json_parse_error_sets_artifact_none(self, tmp_path: Path) -> None:
        """get_build_status should handle invalid JSON gracefully (artifact=None)."""
        config = _make_config(tmp_path)
        backend_dir = config.paths.builds_dir / "sycl"
        backend_dir.mkdir(parents=True, exist_ok=True)
        (backend_dir / "build-artifact.json").write_text("not valid json {{{")

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("llama_manager.build_pipeline.status._probe_binary_version", return_value=None),
        ):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.artifact_exists is True
        assert status.artifact is None


class TestUntrackedBinary:
    """Tests for detecting Config default binaries without provenance JSON."""

    def test_untracked_binary_detected_sycl(self, tmp_path: Path) -> None:
        """get_build_status should detect llama-server at Config intel path without JSON."""
        config = _make_config(tmp_path)
        binary = Path(config.paths.llama_server_bin_intel)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\necho ok")
        binary.chmod(0o755)

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("llama_manager.build_pipeline.status._probe_binary_version", return_value=None),
        ):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.artifact_exists is False
        assert status.binary_exists_untracked is True
        assert status.untracked_binary_path == binary

    def test_untracked_binary_detected_cuda(self, tmp_path: Path) -> None:
        """get_build_status should detect llama-server at Config nvidia path without JSON."""
        config = _make_config(tmp_path)
        binary = Path(config.paths.llama_server_bin_nvidia)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\necho ok")
        binary.chmod(0o755)

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("llama_manager.build_pipeline.status._probe_binary_version", return_value=None),
        ):
            status = get_build_status(BuildBackend.CUDA, config)

        assert status.artifact_exists is False
        assert status.binary_exists_untracked is True
        assert status.untracked_binary_path == binary

    def test_untracked_binary_probes_version(self, tmp_path: Path) -> None:
        """get_build_status should probe --version on untracked binary."""
        config = _make_config(tmp_path)
        binary = Path(config.paths.llama_server_bin_intel)
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\necho 'llama-server v9.9.9'")
        binary.chmod(0o755)

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = Mock(returncode=0, stdout="llama-server v9.9.9\n")
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.binary_exists_untracked is True
        assert status.binary_version_output == "llama-server v9.9.9"

    def test_provenance_takes_precedence_over_untracked_binary(self, tmp_path: Path) -> None:
        """When build-artifact.json exists, untracked fallback is not used."""
        config = _make_config(tmp_path)
        default_bin = Path(config.paths.llama_server_bin_intel)
        default_bin.parent.mkdir(parents=True, exist_ok=True)
        default_bin.write_text("#!/bin/sh\necho default")
        default_bin.chmod(0o755)

        provenance_bin = tmp_path / "provenance" / "llama-server"
        provenance_bin.parent.mkdir(parents=True, exist_ok=True)
        provenance_bin.write_text("#!/bin/sh\necho provenance")
        provenance_bin.chmod(0o755)
        _write_artifact_json(config, "sycl", tmp_path, binary_path=str(provenance_bin))

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("llama_manager.build_pipeline.status._probe_binary_version", return_value=None),
        ):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.artifact_exists is True
        assert status.binary_exists_untracked is False
        assert status.untracked_binary_path is None


# ── Binary version probe tests ──────────────────────────────────────────────


class TestBinaryVersionProbe:
    """Tests for binary version probing."""

    def test_binary_version_probe_runs_when_binary_exists(self, tmp_path: Path) -> None:
        """get_build_status should probe binary version when binary exists."""
        config = _make_config(tmp_path)
        fake_bin = tmp_path / "bin" / "llama-server"
        fake_bin.parent.mkdir(parents=True, exist_ok=True)
        fake_bin.write_text("#!/bin/sh\necho 'llama-server v1.2.3'")
        fake_bin.chmod(0o755)

        _write_artifact_json(config, "sycl", tmp_path, binary_path=str(fake_bin))

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = Mock(returncode=0, stdout="llama-server v1.2.3\n")
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.binary_version_output == "llama-server v1.2.3"

    def test_binary_version_probe_skips_when_binary_missing(self, tmp_path: Path) -> None:
        """get_build_status should skip version probe when binary doesn't exist."""
        config = _make_config(tmp_path)
        _write_artifact_json(
            config,
            "sycl",
            tmp_path,
            binary_path=str(tmp_path / "nonexistent" / "llama-server"),
        )

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            status = get_build_status(BuildBackend.SYCL, config)

        mock_run.assert_not_called()
        assert status.binary_version_output is None

    def test_binary_version_probe_skips_when_exit_nonzero(self, tmp_path: Path) -> None:
        """get_build_status should skip version when binary returns non-zero with no version line."""
        config = _make_config(tmp_path)
        fake_bin = tmp_path / "bin" / "llama-server"
        fake_bin.parent.mkdir(parents=True, exist_ok=True)
        fake_bin.write_text("#!/bin/sh\necho 'error'")
        fake_bin.chmod(0o755)

        _write_artifact_json(config, "sycl", tmp_path, binary_path=str(fake_bin))

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = Mock(returncode=1, stdout="", stderr="fatal error\n")
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.binary_version_output is None

    def test_binary_version_probe_parses_version_line_on_nonzero_exit(self, tmp_path: Path) -> None:
        """get_build_status should parse version: line even when exit code is non-zero."""
        config = _make_config(tmp_path)
        fake_bin = tmp_path / "bin" / "llama-server"
        fake_bin.parent.mkdir(parents=True, exist_ok=True)
        fake_bin.write_text("#!/bin/sh\necho 'version: 9219 (45b455e66)'")
        fake_bin.chmod(0o755)

        _write_artifact_json(config, "sycl", tmp_path, binary_path=str(fake_bin))

        with (
            patch("llama_manager.build_pipeline.status._run_git", return_value=None),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = Mock(
                returncode=139,
                stdout="version: 9219 (45b455e66)\n",
                stderr="",
            )
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.binary_version_output == "version: 9219 (45b455e66)"


class TestExtractLlamaServerVersion:
    """Tests for parsing llama-server --version output."""

    def test_extract_from_stdout(self) -> None:
        assert (
            _extract_llama_server_version(
                "version: 9219 (45b455e66)\nbuilt with IntelLLVM 2026.0.0\n",
                "",
            )
            == "version: 9219 (45b455e66)"
        )

    def test_extract_from_stderr(self) -> None:
        assert (
            _extract_llama_server_version("", "version: 1 (abc12345)\n") == "version: 1 (abc12345)"
        )

    def test_extract_returns_none_when_missing(self) -> None:
        assert _extract_llama_server_version("no version here\n", "") is None


class TestProbeBinaryVersionEnv:
    """Tests for executable version probing command construction."""

    def test_probe_uses_build_env_cmd_for_sycl(self, tmp_path: Path) -> None:
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\necho ok")
        binary.chmod(0o755)

        with (
            patch(
                "llama_manager.build_pipeline.status.get_build_env_cmd",
                return_value=["bash", "-c", "wrapped"],
            ) as mock_env,
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = Mock(
                returncode=0,
                stdout="version: 1 (abc)\n",
                stderr="",
            )
            result = _probe_binary_version(binary, BuildBackend.SYCL)

        mock_env.assert_called_once_with([str(binary), "--version"], BuildBackend.SYCL)
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["bash", "-c", "wrapped"]
        assert result == "version: 1 (abc)"


# ── Local source tests ──────────────────────────────────────────────────────


class TestLocalSource:
    """Tests for local source git information."""

    def test_source_not_a_repo(self, tmp_path: Path) -> None:
        """get_build_status should detect source dir that is not a git repo."""
        config = _make_config(tmp_path)

        with patch("llama_manager.build_pipeline.status._run_git", return_value=None):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.source_exists is True
        assert status.source_is_repo is False
        assert status.source_branch is None
        assert status.source_head_sha is None
        assert status.source_remote_url is None

    def test_source_is_repo_parses_git_info(self, tmp_path: Path) -> None:
        """get_build_status should parse git info from a real git repo (no mocks).

        Note: remote_branch_sha may be a 40-char SHA if network is available,
        or None if not. Both outcomes are valid.
        """
        source_dir = tmp_path / "llama.cpp"
        commit_sha = _init_git_repo(source_dir, branch="feature-branch")

        state_root = tmp_path / "state_root"
        (state_root / "llm-runner" / "builds").mkdir(parents=True)

        config = Config(
            paths=PathsConfig(
                llama_cpp_root=str(source_dir),
                xdg_state_base=str(state_root),
                xdg_cache_base=str(tmp_path),
                xdg_data_base=str(tmp_path),
            ),
            build=BuildPipelineConfig(),
        )

        status = get_build_status(BuildBackend.SYCL, config)

        assert status.source_exists is True
        assert status.source_is_repo is True
        assert status.source_branch == "feature-branch"
        assert status.source_head_sha == commit_sha
        assert status.source_remote_url is not None
        assert "llama.cpp" in status.source_remote_url
        # remote_branch_sha: either 40-char hex SHA (network OK) or None (no network)
        if status.remote_branch_sha is not None:
            assert re.fullmatch(r"[0-9a-f]{40}", status.remote_branch_sha)

    def test_source_clone_missing(self, tmp_path: Path) -> None:
        """get_build_status should detect missing source directory."""
        state_root = tmp_path / "state_root"
        (state_root / "llm-runner" / "builds").mkdir(parents=True)

        config = Config(
            paths=PathsConfig(
                llama_cpp_root=str(tmp_path / "nonexistent" / "llama.cpp"),
                xdg_state_base=str(state_root),
                xdg_cache_base=str(tmp_path),
                xdg_data_base=str(tmp_path),
            ),
            build=BuildPipelineConfig(),
        )

        with patch("llama_manager.build_pipeline.status._run_git", return_value=None):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.source_exists is False
        assert status.source_is_repo is False
        assert status.source_branch is None
        assert status.source_head_sha is None
        assert status.source_remote_url is None


# ── Remote branch tests ─────────────────────────────────────────────────────


class TestRemoteBranch:
    """Tests for remote branch HEAD lookup."""

    def test_remote_branch_reachable(self, tmp_path: Path) -> None:
        """get_build_status should parse remote branch SHA from ls-remote output."""
        config = _make_config(tmp_path)

        with patch("llama_manager.build_pipeline.status._run_git") as mock_git:
            mock_git.return_value = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2\trefs/heads/master"
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.configured_branch == "master"
        assert status.remote_branch_sha == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_remote_branch_unreachable_offline(self, tmp_path: Path) -> None:
        """get_build_status should handle offline/unreachable remote gracefully."""
        config = _make_config(tmp_path)

        with patch("llama_manager.build_pipeline.status._run_git", return_value=None):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.remote_branch_sha is None
        assert status.configured_branch == "master"

    def test_remote_branch_none_for_unknown_branch(self, tmp_path: Path) -> None:
        """get_build_status should return None for a branch that doesn't exist."""
        state_root = tmp_path / "state_root"
        (state_root / "llm-runner" / "builds").mkdir(parents=True)
        (tmp_path / "llama.cpp").mkdir(exist_ok=True)

        config = Config(
            paths=PathsConfig(
                llama_cpp_root=str(tmp_path / "llama.cpp"),
                xdg_state_base=str(state_root),
                xdg_cache_base=str(tmp_path),
                xdg_data_base=str(tmp_path),
            ),
            build=BuildPipelineConfig(git_branch="nonexistent_branch_xyz"),
        )

        with patch("llama_manager.build_pipeline.status._run_git", return_value=None):
            status = get_build_status(BuildBackend.SYCL, config)

        assert status.configured_branch == "nonexistent_branch_xyz"
        assert status.remote_branch_sha is None


# ── BuildStatus dataclass tests ─────────────────────────────────────────────


class TestBuildStatusDataclass:
    """Tests for BuildStatus dataclass construction."""

    def test_build_status_all_fields_settable(self) -> None:
        """BuildStatus should have all fields settable and retrievable."""
        status = BuildStatus(
            backend=BuildBackend.SYCL,
            artifact_exists=True,
            artifact=None,
            binary_version_output="v1.0.0",
            binary_exists_untracked=False,
            untracked_binary_path=None,
            source_exists=True,
            source_is_repo=True,
            source_branch="main",
            source_head_sha="abc123" * 8,
            source_remote_url="https://github.com/ggerganov/llama.cpp.git",
            configured_branch="main",
            remote_branch_sha="def456" * 8,
        )

        assert status.backend == BuildBackend.SYCL
        assert status.artifact_exists is True
        assert status.binary_version_output == "v1.0.0"
        assert status.source_exists is True
        assert status.source_is_repo is True
        assert status.source_branch == "main"
        assert status.source_head_sha == "abc123" * 8
        assert status.source_remote_url == "https://github.com/ggerganov/llama.cpp.git"
        assert status.configured_branch == "main"
        assert status.remote_branch_sha == "def456" * 8

    def test_build_status_cuda_backend(self) -> None:
        """BuildStatus should work with CUDA backend."""
        status = BuildStatus(
            backend=BuildBackend.CUDA,
            artifact_exists=False,
            artifact=None,
            binary_version_output=None,
            binary_exists_untracked=False,
            untracked_binary_path=None,
            source_exists=True,
            source_is_repo=False,
            source_branch=None,
            source_head_sha=None,
            source_remote_url=None,
            configured_branch="master",
            remote_branch_sha=None,
        )

        assert status.backend == BuildBackend.CUDA
        assert status.artifact_exists is False
        assert status.source_is_repo is False
