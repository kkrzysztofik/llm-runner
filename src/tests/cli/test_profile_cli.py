import json
import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_cli.commands.profile import (
    _default_subprocess_runner,
    _detect_backend,
    cmd_profile,
    get_driver_version,
    main,
    require_executable,
)
from llama_manager import BenchmarkResult, ServerConfig

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

    def test_cuda_backend_returns_cuda(self) -> None:
        """_detect_backend should return 'cuda' for profiles with empty device."""
        cfg = ServerConfig(
            model="/model.gguf",
            alias="qwen35-coding",
            device="",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        result = _detect_backend(cfg)
        assert result == "cuda"

    def test_sycl_backend_returns_sycl(self) -> None:
        """_detect_backend should return 'sycl' for profiles with non-empty device."""
        cfg = ServerConfig(
            model="/model.gguf",
            alias="summary-balanced",
            device="SYCL0",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        result = _detect_backend(cfg)
        assert result == "sycl"

    def test_sycl_fallback_for_summary_slot(self) -> None:
        """_detect_backend should return 'sycl' for summary-balanced slot."""
        cfg = ServerConfig(
            model="/model.gguf",
            alias="slot0",
            device="SYCL0",
            port=8080,
            ctx_size=4096,
            ubatch_size=512,
            threads=4,
        )
        result = _detect_backend(cfg)
        assert result == "sycl"


# ---------------------------------------------------------------------------
# TestGetDriverVersion
# ---------------------------------------------------------------------------


class TestGetDriverVersion:
    """Tests for get_driver_version."""

    def test_nvidia_smi_parsed(self) -> None:
        """get_driver_version should parse nvidia-smi output for CUDA backend."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "  535.104.05\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = get_driver_version("cuda")

            assert result == "535.104.05"
            mock_run.assert_called_once_with(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                shell=False,
                timeout=10,
                check=False,
            )

    def test_nvidia_smi_failure_returns_unknown(self) -> None:
        """get_driver_version should return 'unknown' when nvidia-smi fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = get_driver_version("cuda")

            assert result == "unknown"

    def test_nvidia_smi_empty_output_returns_unknown(self) -> None:
        """get_driver_version should return 'unknown' for empty nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = get_driver_version("cuda")

            assert result == "unknown"

    def test_nvidia_smi_file_not_found(self) -> None:
        """get_driver_version should return 'unknown' when nvidia-smi is missing."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_driver_version("cuda")

            assert result == "unknown"

    def test_sycl_ls_parsed(self) -> None:
        """get_driver_version should parse sycl-ls output for SYCL backend."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "gpu:0, intel:arc, ir:cu12, 00:15:00:00.000000\nother line without gpu/device\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = get_driver_version("sycl")

            assert "gpu" in result.lower() or "device" in result.lower()

    def test_sycl_ls_no_matching_line_returns_unknown(self) -> None:
        """get_driver_version should return 'unknown' when sycl-ls has no gpu/device lines."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "just some random output\ncompletely unrelated line\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = mock_result
            result = get_driver_version("sycl")

            assert result == "unknown"

    def test_sycl_ls_file_not_found(self) -> None:
        """get_driver_version should return 'unknown' when sycl-ls is missing."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_driver_version("sycl")

            assert result == "unknown"


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() CLI entry point."""

    def test_missing_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should return 1 and print error when fewer than 2 args given."""
        with pytest.raises(SystemExit) as exc_info:
            main(["only_one_arg"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "the following arguments are required: flavor" in captured.err

    def test_no_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        """main should return 1 when no arguments provided at all."""
        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "the following arguments are required: slot_id" in captured.err

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
        with pytest.raises(SystemExit) as exc_info:
            main(["slot1", "invalid_flavor"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err
        assert "balanced" in captured.err
        assert "quality" in captured.err

    def test_json_flag_parsed(self) -> None:
        """main should pass --json flag to cmd_profile."""
        with patch("llama_cli.commands.profile.cmd_profile") as mock_cmd:
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
        with patch("llama_cli.commands.profile.cmd_profile") as mock_cmd:
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
        with patch("llama_cli.commands.profile.cmd_profile") as mock_cmd:
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
        with patch("llama_cli.commands.profile.cmd_profile") as mock_cmd:
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


def _build_test_registry(
    tmp_path: Path,
    summary_port: int = 8080,
    fast_port: int = 8082,
    qwen35_port: int = 8081,
) -> tuple[Any, Any]:
    """Build a real profile registry with test-configured ports.

    Returns a tuple of (mock config, registry) where the mock Config
    has all the attributes needed by cmd_profile.
    """
    from llama_manager import (
        Config,
        RunGroupSpec,
        RunProfileRegistry,
        RunProfileSpec,
    )

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Create a real Config with test values
    cfg = Config()
    cfg.model_summary_balanced = str(tmp_path / "model_summary.gguf")
    cfg.model_summary_fast = str(tmp_path / "model_fast.gguf")
    cfg.model_qwen35 = str(tmp_path / "model_qwen35.gguf")
    cfg.summary_balanced_port = summary_port
    cfg.summary_fast_port = fast_port
    cfg.qwen35_port = qwen35_port
    cfg.default_threads_summary_balanced = 8
    cfg.default_threads_summary_fast = 4
    cfg.default_threads_qwen35 = 16
    cfg.default_ctx_size_summary = 32768
    cfg.default_ubatch_size_summary_balanced = 1024
    cfg.default_ubatch_size_summary_fast = 512
    cfg.default_ubatch_size_qwen35 = 2048
    cfg.default_cache_type_summary_k = "q8_0"
    cfg.default_cache_type_summary_v = "q8_0"
    cfg.default_cache_type_qwen35_k = "q8_0"
    cfg.default_cache_type_qwen35_v = "q8_0"
    cfg.default_n_gpu_layers = 99
    cfg.default_n_gpu_layers_qwen35 = "all"
    cfg.server_binary_version = "1.18.0"
    cfg.llama_server_bin_intel = str(tmp_path / "llama-server")
    cfg.llama_server_bin_nvidia = str(tmp_path / "llama-server-cuda")
    cfg.summary_balanced_chat_template_kwargs = ""
    cfg.summary_fast_chat_template_kwargs = ""

    registry = RunProfileRegistry(
        profiles=(
            RunProfileSpec(
                profile_id="summary-balanced",
                description="Run summary-balanced model on Intel SYCL.",
                model=cfg.model_summary_balanced,
                alias="summary-balanced",
                device="SYCL0",
                port=summary_port,
                ctx_size=cfg.default_ctx_size_summary,
                ubatch_size=cfg.default_ubatch_size_summary_balanced,
                threads=cfg.default_threads_summary_balanced,
                reasoning_mode="off",
                reasoning_format="deepseek",
                chat_template_kwargs=cfg.summary_balanced_chat_template_kwargs,
                use_jinja=True,
                cache_type_k=cfg.default_cache_type_summary_k,
                cache_type_v=cfg.default_cache_type_summary_v,
                backend="llama_cpp",
            ),
            RunProfileSpec(
                profile_id="summary-fast",
                description="Run summary-fast model on Intel SYCL.",
                model=cfg.model_summary_fast,
                alias="summary-fast",
                device="SYCL0",
                port=fast_port,
                ctx_size=cfg.default_ctx_size_summary,
                ubatch_size=cfg.default_ubatch_size_summary_fast,
                threads=cfg.default_threads_summary_fast,
                reasoning_mode="off",
                reasoning_format="deepseek",
                chat_template_kwargs=cfg.summary_fast_chat_template_kwargs,
                use_jinja=True,
                cache_type_k=cfg.default_cache_type_summary_k,
                cache_type_v=cfg.default_cache_type_summary_v,
                backend="llama_cpp",
            ),
            RunProfileSpec(
                profile_id="qwen35",
                description="Run qwen35-coding model on NVIDIA CUDA.",
                model=cfg.model_qwen35,
                alias="qwen35-coding",
                device="",
                port=qwen35_port,
                ctx_size=cfg.default_ctx_size_qwen35,
                ubatch_size=cfg.default_ubatch_size_qwen35,
                threads=cfg.default_threads_qwen35,
                cache_type_k=cfg.default_cache_type_qwen35_k,
                cache_type_v=cfg.default_cache_type_qwen35_v,
                n_gpu_layers=cfg.default_n_gpu_layers_qwen35,
                server_bin=cfg.llama_server_bin_nvidia,
                backend="llama_cpp",
            ),
        ),
        run_groups=(
            RunGroupSpec(
                group_id="summary-balanced",
                profile_ids=("summary-balanced",),
                description="Launch the summary-balanced profile.",
            ),
            RunGroupSpec(
                group_id="summary-fast",
                profile_ids=("summary-fast",),
                description="Launch the summary-fast profile.",
            ),
            RunGroupSpec(
                group_id="qwen35",
                profile_ids=("qwen35",),
                description="Launch the qwen35 profile.",
            ),
            RunGroupSpec(
                group_id="both",
                profile_ids=("summary-balanced", "qwen35"),
                description="Launch summary-balanced and qwen35 profiles together.",
            ),
        ),
    )

    return cfg, registry


def _populate_profile_mock_defaults(mock_cfg: Any, tmp_path: Path) -> None:
    """Populate a mock Config with the standard profile test defaults."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    mock_cfg.profiles_dir = profiles_dir
    mock_cfg.summary_balanced_port = 8080
    mock_cfg.summary_fast_port = 8082
    mock_cfg.qwen35_port = 8081
    mock_cfg.default_threads_summary_balanced = 8
    mock_cfg.default_ctx_size_summary = 16144
    mock_cfg.default_ubatch_size_summary_balanced = 1024
    mock_cfg.default_cache_type_summary_k = "q8_0"
    mock_cfg.default_cache_type_summary_v = "q8_0"
    mock_cfg.default_n_gpu_layers = 99
    mock_cfg.default_n_gpu_layers_qwen35 = "all"
    mock_cfg.server_binary_version = "1.18.0"
    mock_cfg.model_summary_balanced = str(tmp_path / "model.gguf")
    mock_cfg.llama_server_bin_intel = str(tmp_path / "llama-server")
    mock_cfg.llama_server_bin_nvidia = ""


@contextmanager
def _build_mock_config(
    tmp_path: Path, cuda_exists: bool = False
) -> Generator[tuple[MagicMock, str, Path, Any]]:
    """Build a mocked Config and registry that makes cmd_profile succeed.

    Args:
        tmp_path: pytest tmp_path fixture.
        cuda_exists: Whether the CUDA binary should appear to exist.

    Yields:
        A tuple of (mock config, benchmark binary path, profiles directory,
        registry). Config is only mocked for the duration of the context.
    """
    patcher = patch("llama_cli.commands.profile.Config")
    mock_cfg_cls = patcher.start()
    try:
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

        # Build a real registry for _resolve_slot_server_config
        _, registry = _build_test_registry(
            tmp_path,
            summary_port=cfg.summary_balanced_port,
            fast_port=cfg.summary_balanced_port + 2,
            qwen35_port=cfg.summary_balanced_port + 1,
        )

        yield mock_cfg_cls.return_value, str(bench_bin), profiles_dir, registry
    finally:
        patcher.stop()


class TestCmdProfile:
    """Tests for cmd_profile — the core profiling logic."""

    def test_success_path(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should succeed with a valid benchmark result."""
        with _build_mock_config(tmp_path) as (mock_cfg, bench_bin, profiles_dir, registry):
            benchmark_result = _make_benchmark_result(tps=100.0, latency=10.0, vram=5120.0)

            with (
                patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
                patch("llama_cli.commands.profile.require_executable"),
                patch("llama_cli.commands.profile.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.commands.profile.get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.commands.profile.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch(
                    "llama_cli.commands.profile.build_benchmark_cmd",
                    return_value=["bench", "-m", "model"],
                ),
                patch("llama_cli.commands.profile.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.commands.profile.ProfileFlavor"),
                patch("llama_cli.commands.profile.ProfileRecord") as mock_record_cls,
                patch("llama_cli.commands.profile.write_profile") as mock_write,
                patch(
                    "llama_manager.profile_orchestrator.create_default_profile_registry",
                    return_value=registry,
                ),
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

    def test_benchmark_failure_returns_1(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should return 1 when benchmark returns None."""
        with (
            _build_mock_config(tmp_path) as (_, _, _, registry),
            patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
            patch("llama_cli.commands.profile.require_executable"),
            patch("llama_cli.commands.profile.get_gpu_identifier", return_value="test-gpu"),
            patch("llama_cli.commands.profile.get_driver_version", return_value="unknown"),
            patch(
                "llama_cli.commands.profile.compute_driver_version_hash",
                return_value="abc123",
            ),
            patch("llama_cli.commands.profile.build_benchmark_cmd", return_value=["bench"]),
            patch("llama_cli.commands.profile.run_benchmark", return_value=None),
            patch("llama_cli.commands.profile.ProfileFlavor"),
            patch("llama_cli.commands.profile.ProfileRecord"),
            patch("llama_manager.profile_orchestrator.create_default_profile_registry"),
        ):
            exit_code = cmd_profile(
                slot_id="test-slot",
                flavor="balanced",
                json_output=False,
            )

            assert exit_code == 1
            captured = capsys.readouterr()
            assert "benchmark failed" in captured.err

    def test_benchmark_exit_code_nonzero(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should return 1 when benchmark subprocess exits non-zero."""
        with (
            _build_mock_config(tmp_path) as (_, _, _, registry),
            patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
            patch("llama_cli.commands.profile.require_executable"),
            patch("llama_cli.commands.profile.get_gpu_identifier", return_value="test-gpu"),
            patch("llama_cli.commands.profile.get_driver_version", return_value="unknown"),
            patch(
                "llama_cli.commands.profile.compute_driver_version_hash",
                return_value="abc123",
            ),
            patch("llama_cli.commands.profile.build_benchmark_cmd", return_value=["bench"]),
            patch(
                "llama_cli.commands.profile.run_benchmark",
                return_value=None,
            ),
            patch("llama_cli.commands.profile.ProfileFlavor"),
            patch("llama_cli.commands.profile.ProfileRecord"),
            patch("llama_manager.profile_orchestrator.create_default_profile_registry"),
        ):
            exit_code = cmd_profile(
                slot_id="test-slot",
                flavor="balanced",
                json_output=False,
            )

            assert exit_code == 1

    def test_json_output_mode(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile with json_output=True should print valid JSON."""
        with _build_mock_config(tmp_path) as (_, _, _, registry):
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
                patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
                patch("llama_cli.commands.profile.require_executable"),
                patch("llama_cli.commands.profile.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.commands.profile.get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.commands.profile.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.commands.profile.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.commands.profile.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.commands.profile.ProfileFlavor"),
                patch("llama_cli.commands.profile.ProfileRecord") as mock_record_cls,
                patch("llama_cli.commands.profile.write_profile") as mock_write,
                patch(
                    "llama_manager.profile_orchestrator.create_default_profile_registry",
                    return_value=registry,
                ),
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

    def test_benchmark_binary_unavailable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should return 1 when the benchmark binary doesn't exist."""
        with (
            _build_mock_config(tmp_path) as (_, _, _, registry),
            patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
            patch(
                "llama_cli.commands.profile.require_executable",
                side_effect=FileNotFoundError("file not found: /no/such/binary"),
            ),
            patch(
                "llama_manager.profile_orchestrator.create_default_profile_registry",
                return_value=registry,
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
        with _build_mock_config(tmp_path) as (_, _, _, registry):
            # Create lockfile in the parent of profiles_dir
            lockfile = tmp_path / "test-slot.lock"
            lockfile.touch()

            benchmark_result = _make_benchmark_result()

            with (
                patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
                patch(
                    "llama_manager.profile_orchestrator.resolve_benchmark_binary",
                    return_value="/fake/llama-bench",
                ),
                patch("llama_cli.commands.profile.require_executable"),
                patch("llama_cli.commands.profile.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.commands.profile.get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.commands.profile.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.commands.profile.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.commands.profile.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.commands.profile.ProfileFlavor"),
                patch("llama_cli.commands.profile.ProfileRecord") as mock_record_cls,
                patch("llama_cli.commands.profile.write_profile") as mock_write,
                patch(
                    "llama_manager.profile_orchestrator.create_default_profile_registry",
                    return_value=registry,
                ),
            ):
                mock_record = MagicMock()
                mock_record.to_dict.return_value = {"test": "data"}
                mock_record_cls.return_value = mock_record
                mock_write.return_value = tmp_path / "profiles" / "profile.json"

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
        """_default_subprocess_runner should execute via subprocess.Popen with cancel support."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.read.return_value = "tokens/s: 100.0"
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""

        with patch("subprocess.Popen", return_value=mock_proc):
            result = _default_subprocess_runner(["echo", "hello"])

            assert result.exit_code == 0
            assert result.stdout == "tokens/s: 100.0"
            assert result.stderr == ""

    def test_cuda_backend_detection_in_profile(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should pass 'cuda' as backend when detected."""
        with _build_mock_config(tmp_path, cuda_exists=True) as (_, _, _, registry):
            benchmark_result = _make_benchmark_result()
            mock_record = MagicMock()
            mock_record.to_dict.return_value = {"test": "data"}

            with (
                patch("llama_cli.commands.profile._detect_backend", return_value="cuda"),
                patch("llama_cli.commands.profile.require_executable"),
                patch(
                    "llama_cli.commands.profile.get_gpu_identifier", return_value="nvidia-rtx-3090"
                ),
                patch("llama_cli.commands.profile.get_driver_version", return_value="535.104.05"),
                patch(
                    "llama_cli.commands.profile.compute_driver_version_hash",
                    return_value="def456",
                ),
                patch("llama_cli.commands.profile.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.commands.profile.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.commands.profile.ProfileFlavor"),
                patch("llama_cli.commands.profile.ProfileRecord") as mock_record_cls,
                patch("llama_cli.commands.profile.write_profile") as mock_write,
                patch(
                    "llama_manager.profile_orchestrator.create_default_profile_registry",
                    return_value=registry,
                ),
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

    def test_all_flavors_accepted(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """cmd_profile should accept all valid flavors: balanced, fast, quality."""
        for flavor in ["balanced", "fast", "quality"]:
            with _build_mock_config(tmp_path) as (_, _, _, registry):
                benchmark_result = _make_benchmark_result()
                mock_record = MagicMock()
                mock_record.to_dict.return_value = {"test": "data"}

                with (
                    patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
                    patch(
                        "llama_manager.profile_orchestrator.resolve_benchmark_binary",
                        return_value="/fake/llama-bench",
                    ),
                    patch("llama_cli.commands.profile.require_executable"),
                    patch("llama_cli.commands.profile.get_gpu_identifier", return_value="test-gpu"),
                    patch("llama_cli.commands.profile.get_driver_version", return_value="unknown"),
                    patch(
                        "llama_cli.commands.profile.compute_driver_version_hash",
                        return_value="abc123",
                    ),
                    patch("llama_cli.commands.profile.build_benchmark_cmd", return_value=["bench"]),
                    patch(
                        "llama_cli.commands.profile.run_benchmark", return_value=benchmark_result
                    ),
                    patch("llama_cli.commands.profile.ProfileFlavor"),
                    patch("llama_cli.commands.profile.ProfileRecord") as mock_record_cls,
                    patch("llama_cli.commands.profile.write_profile") as mock_write,
                    patch(
                        "llama_manager.profile_orchestrator.create_default_profile_registry",
                        return_value=registry,
                    ),
                ):
                    mock_record_cls.return_value = mock_record
                    mock_write.return_value = tmp_path / "profiles" / "profile.json"

                    exit_code = cmd_profile(
                        slot_id=f"slot-{flavor}",
                        flavor=flavor,
                        json_output=False,
                    )

                    assert exit_code == 0, f"Failed for flavor: {flavor}"

    def test_write_profile_called_with_correct_dir(
        self,
        tmp_path: Path,
    ) -> None:
        """cmd_profile should call write_profile with the profiles_dir from config."""
        with _build_mock_config(tmp_path) as (_, _, _, registry):
            benchmark_result = _make_benchmark_result()
            mock_record = MagicMock()
            mock_record.to_dict.return_value = {"test": "data"}

            with (
                patch("llama_cli.commands.profile._detect_backend", return_value="sycl"),
                patch(
                    "llama_manager.profile_orchestrator.resolve_benchmark_binary",
                    return_value="/fake/llama-bench",
                ),
                patch("llama_cli.commands.profile.require_executable"),
                patch("llama_cli.commands.profile.get_gpu_identifier", return_value="test-gpu"),
                patch("llama_cli.commands.profile.get_driver_version", return_value="unknown"),
                patch(
                    "llama_cli.commands.profile.compute_driver_version_hash",
                    return_value="abc123",
                ),
                patch("llama_cli.commands.profile.build_benchmark_cmd", return_value=["bench"]),
                patch("llama_cli.commands.profile.run_benchmark", return_value=benchmark_result),
                patch("llama_cli.commands.profile.ProfileFlavor"),
                patch("llama_cli.commands.profile.ProfileRecord") as mock_record_cls,
                patch("llama_cli.commands.profile.write_profile") as mock_write,
                patch(
                    "llama_manager.profile_orchestrator.create_default_profile_registry",
                    return_value=registry,
                ),
            ):
                mock_record_cls.return_value = mock_record
                mock_write.return_value = tmp_path / "profiles" / "profile.json"

                cmd_profile(
                    slot_id="test-slot",
                    flavor="balanced",
                    json_output=False,
                )

                mock_write.assert_called_once()
                call_args = mock_write.call_args
                assert call_args[0][0] == tmp_path / "profiles"


# ---------------------------------------------------------------------------
# TestStreamToText
# ---------------------------------------------------------------------------


class TestStreamToText:
    """Tests for _stream_to_text()."""

    def test_none_returns_empty(self) -> None:
        """_stream_to_text(None) should return empty string."""
        from llama_cli.commands.profile import _stream_to_text

        assert _stream_to_text(None) == ""

    def test_string_unchanged(self) -> None:
        """_stream_to_text(str) should return unchanged."""
        from llama_cli.commands.profile import _stream_to_text

        assert _stream_to_text("already a string") == "already a string"

    def test_bytes_decodes_utf8(self) -> None:
        """_stream_to_text(bytes) should decode as utf-8."""
        from llama_cli.commands.profile import _stream_to_text

        result = _stream_to_text(b"hello world")
        assert result == "hello world"

    def test_bytes_invalid_utf8_replacement(self) -> None:
        """_stream_to_text(invalid bytes) should use replacement char."""
        from llama_cli.commands.profile import _stream_to_text

        # 0xFF is invalid UTF-8
        result = _stream_to_text(b"\xff\xfe")
        assert result == "��"


# ---------------------------------------------------------------------------
# TestProfileArgumentParser
# ---------------------------------------------------------------------------


class TestProfileArgumentParser:
    """Tests for _ProfileArgumentParser error handling."""

    def test_error_exits_with_code_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_ProfileArgumentParser should exit with code 1 on error."""
        from llama_cli.commands.profile import _build_profile_parser

        parser = _build_profile_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["only_one_arg"])

        assert exc_info.value.code == 1

    def test_error_prints_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_ProfileArgumentParser should print usage on error."""
        from llama_cli.commands.profile import _build_profile_parser

        parser = _build_profile_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["only_one_arg"])

        captured = capsys.readouterr()
        assert "usage:" in captured.err.lower()


# ---------------------------------------------------------------------------
# TestValidateSlotId
# ---------------------------------------------------------------------------


class TestValidateSlotId:
    """Tests for _validate_slot_id()."""

    def test_valid_slot_id(self) -> None:
        """Valid slot_id should return None."""
        from llama_cli.commands.profile import _validate_slot_id

        assert _validate_slot_id("slot0") is None
        assert _validate_slot_id("summary-balanced") is None
        assert _validate_slot_id("qwen35-coding") is None
        assert _validate_slot_id("my_slot-123") is None

    def test_empty_string(self) -> None:
        """Empty string should return error."""
        from llama_cli.commands.profile import _validate_slot_id

        result = _validate_slot_id("")
        assert result == "slot_id must not be empty"

    def test_path_traversal(self) -> None:
        """Path traversal '..' should return error."""
        from llama_cli.commands.profile import _validate_slot_id

        result = _validate_slot_id("../etc/passwd")
        assert result == "slot_id must not contain path traversal sequences"

    def test_path_separators(self) -> None:
        """Path separators should return error."""
        from llama_cli.commands.profile import _validate_slot_id

        assert _validate_slot_id("foo/bar") == "slot_id must not contain path separators"
        assert _validate_slot_id("foo\\bar") == "slot_id must not contain path separators"

    def test_special_chars_rejected(self) -> None:
        """Special characters should be rejected."""
        from llama_cli.commands.profile import _validate_slot_id

        assert _validate_slot_id("slot@0") is not None
        assert _validate_slot_id("slot#1") is not None
        assert _validate_slot_id("slot $name") is not None
        assert _validate_slot_id("slot;cmd") is not None

    def test_whitespace_only(self) -> None:
        """Whitespace-only slot_id should return error."""
        from llama_cli.commands.profile import _validate_slot_id

        result = _validate_slot_id("   ")
        assert result == "slot_id must not be empty"


# ---------------------------------------------------------------------------
# TestExitIfProfileCancelled
# ---------------------------------------------------------------------------


class TestExitIfProfileCancelled:
    """Tests for _exit_if_profile_cancelled."""

    def test_returns_none_when_no_cancel_event(self) -> None:
        from llama_cli.commands.profile import _exit_if_profile_cancelled

        result = _exit_if_profile_cancelled(None, "slot0", lambda *a, **kw: None)
        assert result is None

    def test_returns_none_when_event_not_set(self) -> None:
        import threading

        from llama_cli.commands.profile import _exit_if_profile_cancelled

        event = threading.Event()
        result = _exit_if_profile_cancelled(event, "slot0", lambda *a, **kw: None)
        assert result is None

    def test_returns_1_when_event_set(self) -> None:
        import threading

        from llama_cli.commands.profile import _exit_if_profile_cancelled

        event = threading.Event()
        event.set()
        messages: list[str] = []
        result = _exit_if_profile_cancelled(event, "slot0", lambda msg, **kw: messages.append(msg))
        assert result == 1
        assert any("cancelled" in m for m in messages)


# ---------------------------------------------------------------------------
# TestCheckSlotLockfile
# ---------------------------------------------------------------------------


class TestCheckSlotLockfile:
    """Tests for _check_slot_lockfile."""

    def test_no_lockfile_no_emit(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _check_slot_lockfile
        from llama_manager import Config

        config = MagicMock(spec=Config)
        config.profiles_dir = tmp_path / "profiles"
        emitted: list[str] = []
        _check_slot_lockfile("slot0", config, emitted.append)
        assert emitted == []

    def test_lockfile_exists_emits_warning(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _check_slot_lockfile
        from llama_manager import Config

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        lock = runtime_dir / "slot0.lock"
        lock.write_text("")

        config = MagicMock(spec=Config)
        config.profiles_dir = runtime_dir / "profiles"
        emitted: list[str] = []
        _check_slot_lockfile("slot0", config, emitted.append)
        assert any("lockfile" in m for m in emitted)

    def test_oserror_is_swallowed(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _check_slot_lockfile
        from llama_manager import Config

        config = MagicMock(spec=Config)
        # Make profiles_dir.parent raise OSError
        profiles_dir_mock = MagicMock()
        profiles_dir_mock.parent.__truediv__ = MagicMock(side_effect=OSError("disk error"))
        config.profiles_dir = profiles_dir_mock
        # Should not raise
        _check_slot_lockfile("slot0", config, lambda msg: None)


# ---------------------------------------------------------------------------
# TestHandleCancel
# ---------------------------------------------------------------------------


class TestHandleCancel:
    """Tests for _handle_cancel."""

    def test_returns_none_when_not_set(self) -> None:
        import subprocess
        import threading
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _handle_cancel

        proc = MagicMock(spec=subprocess.Popen)
        event = threading.Event()
        result = _handle_cancel(proc, event)
        assert result is None

    def test_returns_130_when_set(self) -> None:
        import subprocess
        import threading
        from unittest.mock import MagicMock, patch

        from llama_cli.commands.profile import _handle_cancel

        proc = MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        event = threading.Event()
        event.set()

        with (
            patch("os.getpgid", return_value=12345),
            patch("os.killpg"),
        ):
            result = _handle_cancel(proc, event)

        assert result is not None
        assert result.exit_code == 130
        assert "cancelled" in result.stderr


# ---------------------------------------------------------------------------
# TestHandleTimeout
# ---------------------------------------------------------------------------


class TestHandleTimeout:
    """Tests for _handle_timeout."""

    def test_returns_none_when_not_elapsed(self) -> None:
        import subprocess
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _handle_timeout

        proc = MagicMock(spec=subprocess.Popen)
        import time

        start = time.monotonic()
        result = _handle_timeout(proc, start, 600.0)
        assert result is None

    def test_returns_124_when_elapsed(self) -> None:
        import subprocess
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _handle_timeout

        proc = MagicMock(spec=subprocess.Popen)
        # start far in the past
        start = 0.0
        result = _handle_timeout(proc, start, 1.0)
        assert result is not None
        assert result.exit_code == 124
        assert "timed out" in result.stderr


# ---------------------------------------------------------------------------
# TestPollUntilDone
# ---------------------------------------------------------------------------


class TestPollUntilDone:
    """Tests for _poll_until_done."""

    def test_process_exits_normally(self) -> None:
        import subprocess
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _poll_until_done

        stdout_mock = MagicMock()
        stdout_mock.read.return_value = "output"
        stderr_mock = MagicMock()
        stderr_mock.read.return_value = ""

        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = 0
        proc.stdout = stdout_mock
        proc.stderr = stderr_mock

        result = _poll_until_done(proc, 600, None)
        assert result.exit_code == 0
        assert result.stdout == "output"

    def test_cancel_event_terminates(self) -> None:
        import subprocess
        import threading
        from unittest.mock import MagicMock, patch

        from llama_cli.commands.profile import _poll_until_done

        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None  # never exits on its own
        proc.pid = 12345

        cancel = threading.Event()
        cancel.set()

        with (
            patch("os.getpgid", return_value=12345),
            patch("os.killpg"),
        ):
            result = _poll_until_done(proc, 600, cancel)

        assert result.exit_code == 130

    def test_timeout_returns_124(self) -> None:
        import subprocess
        from unittest.mock import MagicMock

        from llama_cli.commands.profile import _poll_until_done

        proc = MagicMock(spec=subprocess.Popen)
        proc.poll.return_value = None  # never exits
        proc.wait.return_value = None

        # Use a very short timeout so it expires immediately
        result = _poll_until_done(proc, 0, None)
        assert result.exit_code == 124
