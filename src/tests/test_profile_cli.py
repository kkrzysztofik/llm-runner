"""Tests for profile_cli — GPU benchmark subcommand.

Test coverage targets:
- require_executable: file existence + permission checks
- _detect_backend: CUDA vs SYCL auto-detection
- _get_driver_version: nvidia-smi / sycl-ls parsing
- cmd_profile: success path, benchmark failure, JSON output
- main: argument validation (missing args, empty slot_id, path traversal,
  invalid flavor, --json flag)
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from unittest.mock import MagicMock, _patch, patch

import pytest

from llama_cli.profile_cli import (
    _default_subprocess_runner,
    _detect_backend,
    _get_driver_version,
    cmd_profile,
    main,
    require_executable,
)
from llama_manager import BenchmarkResult

# ---------------------------------------------------------------------------
# TestRequireExecutable
# ---------------------------------------------------------------------------


class TestRequireExecutable:
    """Tests for require_executable."""

    def test_existing_executable(self, tmp_path: Path) -> None:
        """require_executable should not raise for an existing executable file."""
        exe = tmp_path / "my_binary"
        exe.write_text("#!/bin/sh\necho hello")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        # Should not raise
        require_executable(str(exe))

    def test_nonexistent_raises(self, tmp_path: Path) -> None:
        """require_executable should raise FileNotFoundError for missing files."""
        fake = tmp_path / "does_not_exist"

        with pytest.raises(FileNotFoundError, match="file not found"):
            require_executable(str(fake))

    def test_not_executable_raises(self, tmp_path: Path) -> None:
        """require_executable should raise PermissionError for non-executable files."""
        regular = tmp_path / "not_executable"
        regular.write_text("just text")
        # Explicitly ensure it's NOT executable
        regular.chmod(0o644)

        with pytest.raises(PermissionError, match="not executable"):
            require_executable(str(regular))


# ---------------------------------------------------------------------------
# TestDetectBackend
# ---------------------------------------------------------------------------


class TestDetectBackend:
    """Tests for _detect_backend."""

    def test_cuda_binary_returns_cuda(self, tmp_path: Path) -> None:
        """_detect_backend should return 'cuda' when NVIDIA binary exists."""
        cuda_bin = tmp_path / "llama-server"
        cuda_bin.touch()

        with patch("llama_manager.Config") as mock_cfg:
            cfg = mock_cfg.return_value
            cfg.llama_server_bin_nvidia = str(cuda_bin)
            cfg.llama_server_bin_intel = str(tmp_path / "intel_server")

            result = _detect_backend(cfg)

            assert result == "cuda"

    def test_sycl_fallback(self, tmp_path: Path) -> None:
        """_detect_backend should return 'sycl' when no NVIDIA binary exists."""
        with patch("llama_manager.Config") as mock_cfg:
            cfg = mock_cfg.return_value
            cfg.llama_server_bin_nvidia = str(tmp_path / "nonexistent_nvidia")
            cfg.llama_server_bin_intel = str(tmp_path / "intel_server")

            result = _detect_backend(cfg)

            assert result == "sycl"

    def test_empty_cuda_path_fallback(self, tmp_path: Path) -> None:
        """_detect_backend should return 'sycl' when nvidia path is empty."""
        with patch("llama_manager.Config") as mock_cfg:
            cfg = mock_cfg.return_value
            cfg.llama_server_bin_nvidia = ""
            cfg.llama_server_bin_intel = str(tmp_path / "intel_server")

            result = _detect_backend(cfg)

            assert result == "sycl"


# ---------------------------------------------------------------------------
# TestGetDriverVersion
# ---------------------------------------------------------------------------


class TestGetDriverVersion:
    """Tests for _get_driver_version."""

    def test_nvidia_smi_parsed(self) -> None:
        """_get_driver_version should parse nvidia-smi output for CUDA backend."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "  535.104.05\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = _get_driver_version("cuda")

            assert result == "535.104.05"
            mock_run.assert_called_once_with(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                shell=False,
                timeout=10,
            )

    def test_nvidia_smi_failure_returns_unknown(self) -> None:
        """_get_driver_version should return 'unknown' when nvidia-smi fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = _get_driver_version("cuda")

            assert result == "unknown"

    def test_nvidia_smi_empty_output_returns_unknown(self) -> None:
        """_get_driver_version should return 'unknown' for empty nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = _get_driver_version("cuda")

            assert result == "unknown"

    def test_nvidia_smi_file_not_found(self) -> None:
        """_get_driver_version should return 'unknown' when nvidia-smi is missing."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _get_driver_version("cuda")

            assert result == "unknown"

    def test_sycl_ls_parsed(self) -> None:
        """_get_driver_version should parse sycl-ls output for SYCL backend."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "gpu:0, intel:arc, ir:cu12, 00:15:00:00.000000\nother line without gpu/device\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = _get_driver_version("sycl")

            assert "gpu" in result.lower() or "device" in result.lower()

    def test_sycl_ls_no_matching_line_returns_unknown(self) -> None:
        """_get_driver_version should return 'unknown' when sycl-ls has no gpu/device lines."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "just some random output\ncompletely unrelated line\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = _get_driver_version("sycl")

            assert result == "unknown"

    def test_sycl_ls_file_not_found(self) -> None:
        """_get_driver_version should return 'unknown' when sycl-ls is missing."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _get_driver_version("sycl")

            assert result == "unknown"


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should return 1 and print error when fewer than 2 args given."""
        exit_code = main(["only_one_arg"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "slot_id and a flavor" in captured.err

    def test_no_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should return 1 when no arguments provided at all."""
        exit_code = main([])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "slot_id and a flavor" in captured.err

    def test_empty_slot_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should reject empty slot_id."""
        exit_code = main(["", "balanced"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "slot_id must not be empty" in captured.err

    def test_whitespace_only_slot_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should reject whitespace-only slot_id."""
        exit_code = main(["   ", "balanced"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "slot_id must not be empty" in captured.err

    def test_path_traversal_rejected(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should reject slot_id containing '..'."""
        exit_code = main(["../etc/passwd", "balanced"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "path traversal" in captured.err

    def test_path_traversal_deep(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should reject slot_id with nested path traversal."""
        exit_code = main(["foo/../../bar", "fast"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "path traversal" in captured.err

    def test_invalid_flavor(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should reject invalid flavor values."""
        exit_code = main(["slot1", "invalid_flavor"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "invalid flavor" in captured.err
        assert "balanced, fast, quality" in captured.err

    def test_json_flag_parsed(self) -> None:
        """main should pass --json flag to cmd_profile."""
        with patch("llama_cli.profile_cli.cmd_profile") as mock_cmd:
            mock_cmd.return_value = 0
            exit_code = main(["slot1", "balanced", "--json"])

            assert exit_code == 0
            mock_cmd.assert_called_once_with(
                "slot1",
                "balanced",
                json_output=True,
            )

    def test_json_flag_at_end(self) -> None:
        """main should detect --json flag at the end of args."""
        with patch("llama_cli.profile_cli.cmd_profile") as mock_cmd:
            mock_cmd.return_value = 0
            exit_code = main(["slot1", "quality", "--json"])

            assert exit_code == 0
            mock_cmd.assert_called_once_with(
                "slot1",
                "quality",
                json_output=True,
            )

    def test_valid_call_success(self) -> None:
        """main should forward valid args and return cmd_profile exit code."""
        with patch("llama_cli.profile_cli.cmd_profile") as mock_cmd:
            mock_cmd.return_value = 0
            exit_code = main(["my-slot", "balanced"])

            assert exit_code == 0
            mock_cmd.assert_called_once_with(
                "my-slot",
                "balanced",
                json_output=False,
            )

    def test_valid_call_returns_nonzero(self) -> None:
        """main should return non-zero exit code from cmd_profile on failure."""
        with patch("llama_cli.profile_cli.cmd_profile") as mock_cmd:
            mock_cmd.return_value = 1
            exit_code = main(["my-slot", "fast"])

            assert exit_code == 1


# ---------------------------------------------------------------------------
# TestCmdProfile
# ---------------------------------------------------------------------------


def _make_benchmark_result(
    tps: float = 120.5,
    latency: float = 8.3,
    vram: float | None = 10240.0,
) -> BenchmarkResult:
    """Create a successful BenchmarkResult."""
    return BenchmarkResult(
        tokens_per_second=tps,
        avg_latency_ms=latency,
        peak_vram_mb=vram,
    )


def _build_mock_config(
    tmp_path: Path, cuda_exists: bool = False
) -> tuple[MagicMock, str, Path, _patch]:
    """Build a mocked Config that makes cmd_profile succeed.

    Args:
        tmp_path: pytest tmp_path fixture.
        cuda_exists: Whether the CUDA binary should appear to exist.

    Returns:
        A tuple of (mock config, benchmark binary path, profiles directory,
        patcher). The patcher must be stopped by the caller after cmd_profile
        finishes to keep the Config substitution alive during execution.
    """
    patcher = patch("llama_cli.profile_cli.Config")
    mock_cfg_cls = patcher.start()

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Create the benchmark binary
    bench_bin = tmp_path / "llama-bench"
    bench_bin.write_text("#!/bin/sh")
    bench_bin.chmod(0o755)

    # Create the server binary (used for bench_bin derivation)
    server_bin = tmp_path / "llama-server"
    server_bin.write_text("#!/bin/sh")
    server_bin.chmod(0o755)

    # Create CUDA binary if requested
    if cuda_exists:
        cuda_bin = tmp_path / "cuda-server"
        cuda_bin.write_text("#!/bin/sh")
        cuda_bin.chmod(0o755)

    cfg = mock_cfg_cls.return_value
    cfg.model_summary_balanced = str(tmp_path / "model.gguf")
    cfg.summary_balanced_port = 8080
    cfg.default_threads_summary_balanced = 8
    cfg.default_ctx_size_summary = 16144
    cfg.default_ubatch_size_summary_balanced = 1024
    cfg.default_cache_type_summary_k = "q8_0"
    cfg.default_cache_type_summary_v = "q8_0"
    cfg.default_n_gpu_layers_qwen35 = "all"
    cfg.default_n_gpu_layers = 99
    cfg.server_binary_version = "1.18.0"
    cfg.profiles_dir = profiles_dir
    cfg.llama_server_bin_intel = str(server_bin)
    cfg.llama_server_bin_nvidia = str(tmp_path / "cuda-server") if cuda_exists else ""

    return mock_cfg_cls.return_value, str(bench_bin), profiles_dir, patcher


class TestCmdProfile:
    """Tests for cmd_profile — the core profiling logic."""

    def test_success_path(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should succeed with a valid benchmark result."""
        mock_cfg, bench_bin, profiles_dir, patcher = _build_mock_config(tmp_path)

        benchmark_result = _make_benchmark_result(tps=100.0, latency=10.0, vram=5120.0)

        try:
            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                patch("llama_cli.profile_cli.require_executable"),
                patch("llama_cli.profile_cli.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.profile_cli._get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.profile_cli.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch(
                    "llama_cli.profile_cli.build_benchmark_cmd",
                    return_value=["bench", "-m", "model"],
                ),
                patch("llama_cli.profile_cli.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.profile_cli.ProfileFlavor"),
                patch("llama_cli.profile_cli.ProfileRecord") as mock_record_cls,
                patch("llama_cli.profile_cli.write_profile") as mock_write,
            ):
                mock_record = MagicMock()
                mock_record.to_dict.return_value = {
                    "gpu_identifier": "test-gpu",
                    "backend": "sycl",
                    "flavor": "balanced",
                    "driver_version": "unknown",
                    "driver_version_hash": "abc123",
                    "server_binary_version": "1.18.0",
                    "profiled_at": "2025-01-01T00:00:00Z",
                    "metrics": {
                        "tokens_per_second": 100.0,
                        "avg_latency_ms": 10.0,
                        "peak_vram_mb": 5120.0,
                    },
                    "parameters": {},
                }
                mock_record_cls.return_value = mock_record
                mock_write.return_value = profiles_dir / "profile.json"

                exit_code = cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=False,
                )

                assert exit_code == 0
                captured = capsys.readouterr()
                assert "Profile recorded for slot 'test-slot'" in captured.out
                assert "test-gpu" in captured.out
                assert "sycl" in captured.out
                assert "balanced" in captured.out
        finally:
            patcher.stop()

    def test_benchmark_failure_returns_1(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should return 1 when benchmark returns None."""
        _, _, _, patcher = _build_mock_config(tmp_path)

        try:
            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                patch("llama_cli.profile_cli.require_executable"),
                patch("llama_cli.profile_cli.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.profile_cli._get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.profile_cli.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.profile_cli.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.profile_cli.run_benchmark", return_value=None),
                patch("llama_cli.profile_cli.ProfileFlavor"),
                patch("llama_cli.profile_cli.ProfileRecord"),
            ):
                exit_code = cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=False,
                )

                assert exit_code == 1
                captured = capsys.readouterr()
                assert "benchmark failed" in captured.err
        finally:
            patcher.stop()

    def test_benchmark_exit_code_nonzero(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should return 1 when benchmark subprocess exits non-zero."""
        _, _, _, patcher = _build_mock_config(tmp_path)

        try:
            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                patch("llama_cli.profile_cli.require_executable"),
                patch("llama_cli.profile_cli.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.profile_cli._get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.profile_cli.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.profile_cli.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.profile_cli.run_benchmark", return_value=None),
                patch("llama_cli.profile_cli.ProfileFlavor"),
                patch("llama_cli.profile_cli.ProfileRecord"),
            ):
                exit_code = cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=False,
                )

                assert exit_code == 1
        finally:
            patcher.stop()

    def test_json_output_mode(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile with json_output=True should print valid JSON."""
        _, _, _, patcher = _build_mock_config(tmp_path)

        try:
            benchmark_result = _make_benchmark_result(tps=200.0, latency=5.0, vram=8192.0)
            mock_record = MagicMock()
            mock_record.to_dict.return_value = {
                "gpu_identifier": "test-gpu",
                "backend": "sycl",
                "flavor": "balanced",
                "driver_version": "unknown",
                "driver_version_hash": "abc123",
                "server_binary_version": "1.18.0",
                "profiled_at": "2025-01-01T00:00:00Z",
                "metrics": {
                    "tokens_per_second": 200.0,
                    "avg_latency_ms": 5.0,
                    "peak_vram_mb": 8192.0,
                },
                "parameters": {},
            }

            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                patch("llama_cli.profile_cli.require_executable"),
                patch("llama_cli.profile_cli.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.profile_cli._get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.profile_cli.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.profile_cli.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.profile_cli.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.profile_cli.ProfileFlavor"),
                patch("llama_cli.profile_cli.ProfileRecord") as mock_record_cls,
                patch("llama_cli.profile_cli.write_profile") as mock_write,
            ):
                mock_record_cls.return_value = mock_record
                mock_write.return_value = tmp_path / "profiles" / "profile.json"

                exit_code = cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=True,
                )

                assert exit_code == 0
                captured = capsys.readouterr()
                parsed = json.loads(captured.out)
                assert parsed["gpu_identifier"] == "test-gpu"
                assert parsed["backend"] == "sycl"
                assert parsed["flavor"] == "balanced"
                assert parsed["metrics"]["tokens_per_second"] == 200.0
        finally:
            patcher.stop()

    def test_benchmark_binary_unavailable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should return 1 when the benchmark binary doesn't exist."""
        with patch("llama_cli.profile_cli.Config") as mock_cfg_cls:
            cfg = mock_cfg_cls.return_value
            cfg.llama_server_bin_intel = ""
            cfg.model_summary_balanced = str(tmp_path / "model.gguf")
            cfg.summary_balanced_port = 8080
            cfg.default_threads_summary_balanced = 8
            cfg.default_ctx_size_summary = 16144
            cfg.default_ubatch_size_summary_balanced = 1024
            cfg.default_cache_type_summary_k = "q8_0"
            cfg.default_cache_type_summary_v = "q8_0"
            cfg.default_n_gpu_layers_qwen35 = "all"
            cfg.default_n_gpu_layers = 99
            cfg.server_binary_version = "1.18.0"
            cfg.profiles_dir = tmp_path / "profiles"
            cfg.profiles_dir.mkdir(parents=True, exist_ok=True)

            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                patch(
                    "llama_cli.profile_cli.require_executable",
                    side_effect=FileNotFoundError("file not found: /no/such/binary"),
                ),
            ):
                exit_code = cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=False,
                )

                assert exit_code == 1
                captured = capsys.readouterr()
                assert "benchmark binary unavailable" in captured.err

    def test_lockfile_warning_printed(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should print a warning when a lockfile exists."""
        with patch("llama_cli.profile_cli.Config") as mock_cfg_cls:
            mock_cfg = mock_cfg_cls.return_value
            profiles_dir = tmp_path / "profiles"
            profiles_dir.mkdir(parents=True, exist_ok=True)
            mock_cfg.profiles_dir = profiles_dir

            # Create lockfile in the parent of profiles_dir
            lockfile = tmp_path / "test-slot.lock"
            lockfile.touch()

            benchmark_result = _make_benchmark_result()

            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                patch("llama_cli.profile_cli.require_executable"),
                patch("llama_cli.profile_cli.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.profile_cli._get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.profile_cli.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.profile_cli.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.profile_cli.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.profile_cli.ProfileFlavor"),
                patch("llama_cli.profile_cli.ProfileRecord") as mock_record_cls,
                patch("llama_cli.profile_cli.write_profile") as mock_write,
            ):
                mock_record = MagicMock()
                mock_record.to_dict.return_value = {"test": "data"}
                mock_record_cls.return_value = mock_record
                mock_write.return_value = profiles_dir / "profile.json"

                exit_code = cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=False,
                )

                assert exit_code == 0
                captured = capsys.readouterr()
                # The lockfile check happens before other operations, so the warning
                # should appear in stderr when the lockfile exists.
                assert "lockfile" in captured.err.lower() or "appears to be running" in captured.err

    def test_default_subprocess_runner(self) -> None:
        """_default_subprocess_runner should execute via subprocess.run."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "tokens/s: 100.0"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = _default_subprocess_runner(["echo", "hello"])

            assert result.exit_code == 0
            assert result.stdout == "tokens/s: 100.0"
            assert result.stderr == ""
            mock_run.assert_called_once_with(
                ["echo", "hello"],
                capture_output=True,
                text=True,
                shell=False,
                timeout=600,
            )

    def test_cuda_backend_detection_in_profile(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should pass 'cuda' as backend when detected."""
        _, _, _, patcher = _build_mock_config(tmp_path, cuda_exists=True)

        try:
            benchmark_result = _make_benchmark_result()
            mock_record = MagicMock()
            mock_record.to_dict.return_value = {"test": "data"}

            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="cuda"),
                patch("llama_cli.profile_cli.require_executable"),
                patch("llama_cli.profile_cli.get_gpu_identifier", return_value="nvidia-rtx-3090"),
                patch("llama_cli.profile_cli._get_driver_version", return_value="535.104.05"),
                patch(
                    "llama_cli.profile_cli.compute_driver_version_hash",
                    return_value="def456",
                ),
                patch("llama_cli.profile_cli.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.profile_cli.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.profile_cli.ProfileFlavor"),
                patch("llama_cli.profile_cli.ProfileRecord") as mock_record_cls,
                patch("llama_cli.profile_cli.write_profile") as mock_write,
            ):
                mock_record_cls.return_value = mock_record
                mock_write.return_value = tmp_path / "profiles" / "profile.json"

                exit_code = cmd_profile(
                    slot_id="cuda-slot",
                    flavor="quality",
                    json_output=False,
                )

                assert exit_code == 0
                captured = capsys.readouterr()
                assert "cuda" in captured.out
        finally:
            patcher.stop()

    def test_all_flavors_accepted(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should accept all valid flavors: balanced, fast, quality."""
        for flavor in ["balanced", "fast", "quality"]:
            _, _, _, patcher = _build_mock_config(tmp_path)

            try:
                benchmark_result = _make_benchmark_result()
                mock_record = MagicMock()
                mock_record.to_dict.return_value = {"test": "data"}

                with (
                    patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                    patch("llama_cli.profile_cli.require_executable"),
                    patch("llama_cli.profile_cli.get_gpu_identifier", return_value="test-gpu"),
                    patch("llama_cli.profile_cli._get_driver_version", return_value="unknown"),
                    patch(
                        "llama_cli.profile_cli.compute_driver_version_hash",
                        return_value="abc123",
                    ),
                    patch("llama_cli.profile_cli.build_benchmark_cmd", return_value=["bench"]),
                    patch("llama_cli.profile_cli.run_benchmark", return_value=benchmark_result),
                    patch("llama_cli.profile_cli.ProfileFlavor"),
                    patch("llama_cli.profile_cli.ProfileRecord") as mock_record_cls,
                    patch("llama_cli.profile_cli.write_profile") as mock_write,
                ):
                    mock_record_cls.return_value = mock_record
                    mock_write.return_value = tmp_path / "profiles" / "profile.json"

                    exit_code = cmd_profile(
                        slot_id=f"slot-{flavor}",
                        flavor=flavor,
                        json_output=False,
                    )

                    assert exit_code == 0, f"Failed for flavor: {flavor}"
            finally:
                patcher.stop()

    def test_write_profile_called_with_correct_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """cmd_profile should call write_profile with the profiles_dir from config."""
        with patch("llama_cli.profile_cli.Config") as mock_cfg_cls:
            mock_cfg = mock_cfg_cls.return_value
            profiles_dir = tmp_path / "profiles"
            profiles_dir.mkdir(parents=True, exist_ok=True)
            mock_cfg.profiles_dir = profiles_dir

            benchmark_result = _make_benchmark_result()
            mock_record = MagicMock()
            mock_record.to_dict.return_value = {"test": "data"}

            with (
                patch("llama_cli.profile_cli._detect_backend", return_value="sycl"),
                patch("llama_cli.profile_cli.require_executable"),
                patch("llama_cli.profile_cli.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.profile_cli._get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.profile_cli.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.profile_cli.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.profile_cli.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.profile_cli.ProfileFlavor"),
                patch("llama_cli.profile_cli.ProfileRecord") as mock_record_cls,
                patch("llama_cli.profile_cli.write_profile") as mock_write,
            ):
                mock_record_cls.return_value = mock_record
                mock_write.return_value = profiles_dir / "profile.json"

                cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=False,
                )

                mock_write.assert_called_once()
                call_args = mock_write.call_args
                assert call_args[0][0] == profiles_dir
