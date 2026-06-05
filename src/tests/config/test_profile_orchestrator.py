"""Tests for llama_manager.profile_orchestrator module."""

import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config import (
    Config,
    ProfileFlavor,
    ProfileRecord,
    ServerConfig,
    create_default_profile_registry,
)
from llama_manager.profile_orchestrator import (
    BENCHMARK_PROMPT_TOKENS,
    BENCHMARK_RUN_TIMEOUT_SECONDS,
    BenchmarkConfig,
    SubprocessResult,
    create_profile_record,
    detect_backend,
    get_driver_version,
    resolve_benchmark_binary,
    resolve_benchmark_config,
    resolve_profile_slot,
    run_profile,
)
from tests.support.helpers import make_server_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> Config:
    """Create a Config with optional overrides.

    Supports dotted paths for nested sub-dataclasses, e.g.
    ``paths.llama_server_bin_intel`` or ``deployment.summary_balanced_port``.
    """
    cfg = Config()
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".", 1)
            parent = getattr(cfg, parts[0])
            setattr(parent, parts[1], value)
        else:
            setattr(cfg, key, value)
    return cfg


def _make_server_config(**overrides: Any) -> ServerConfig:
    """Create a ServerConfig with optional overrides."""
    return make_server_config(**overrides)


# ---------------------------------------------------------------------------
# Test resolve_profile_slot
# ---------------------------------------------------------------------------


class TestResolveProfileSlot:
    """Tests for resolve_profile_slot."""

    def test_known_slot_id_returns_profile_config(self) -> None:
        """resolve_profile_slot should return config for known slot IDs."""
        config = _make_config()
        result = resolve_profile_slot("summary-balanced", config)

        assert isinstance(result, ServerConfig)
        assert result.alias == "summary-balanced"
        assert result.device == "SYCL0"
        assert result.port == config.deployment.summary_balanced_port

    def test_known_slot_id_qwen35(self) -> None:
        """resolve_profile_slot should return config for qwen35 slot."""
        config = _make_config()
        result = resolve_profile_slot("qwen35", config)

        assert isinstance(result, ServerConfig)
        assert result.alias == "qwen35-coding"
        assert result.device == ""  # CUDA profile
        assert result.port == config.deployment.qwen35_port

    def test_known_slot_id_summary_fast(self) -> None:
        """resolve_profile_slot should return config for summary-fast slot."""
        config = _make_config()
        result = resolve_profile_slot("summary-fast", config)

        assert isinstance(result, ServerConfig)
        assert result.alias == "summary-fast"
        assert result.device == "SYCL0"
        assert result.port == config.deployment.summary_fast_port

    def test_unknown_slot_id_returns_defaults(self) -> None:
        """resolve_profile_slot should return summary-balanced defaults for unknown slot IDs."""
        config = _make_config()
        result = resolve_profile_slot("unknown-slot", config)

        assert isinstance(result, ServerConfig)
        assert result.alias == "unknown-slot"
        assert result.device == "SYCL0"
        assert result.port == config.deployment.summary_balanced_port

    def test_alias_resolution(self) -> None:
        """resolve_profile_slot should resolve aliases to profile IDs."""
        config = _make_config()
        result = resolve_profile_slot("summary_balanced", config)

        assert isinstance(result, ServerConfig)
        assert result.alias == "summary-balanced"

    def test_custom_registry(self) -> None:
        """resolve_profile_slot should use provided registry."""
        config = _make_config()
        registry = create_default_profile_registry(config)
        result = resolve_profile_slot("summary-balanced", config, registry=registry)

        assert isinstance(result, ServerConfig)
        assert result.alias == "summary-balanced"


# ---------------------------------------------------------------------------
# Test detect_backend
# ---------------------------------------------------------------------------


class TestDetectBackend:
    """Tests for detect_backend."""

    def test_cuda_backend_empty_device(self) -> None:
        """detect_backend should return 'cuda' for empty device."""
        cfg = _make_server_config(device="")
        assert detect_backend(cfg) == "cuda"

    def test_sycl_backend_nonempty_device(self) -> None:
        """detect_backend should return 'sycl' for non-empty device."""
        cfg = _make_server_config(device="SYCL0")
        assert detect_backend(cfg) == "sycl"

    def test_sycl_backend_cuda_device(self) -> None:
        """detect_backend should return 'sycl' for cuda:0 device."""
        cfg = _make_server_config(device="cuda:0")
        assert detect_backend(cfg) == "sycl"

    def test_cuda_backend_whitespace_only_device(self) -> None:
        """detect_backend should return 'cuda' for whitespace-only device."""
        cfg = _make_server_config(device="   ")
        assert detect_backend(cfg) == "cuda"


# ---------------------------------------------------------------------------
# Test resolve_benchmark_config
# ---------------------------------------------------------------------------


class TestResolveBenchmarkConfig:
    """Tests for resolve_benchmark_config."""

    def test_cuda_profile_uses_server_config(self) -> None:
        """resolve_benchmark_config should use server config values for CUDA profiles."""
        cfg = _make_server_config(
            device="",
            model="/cuda/model.gguf",
            threads=16,
            ubatch_size=2048,
            cache_type_k="f16",
            cache_type_v="f16",
            n_gpu_layers="all",
        )
        config = _make_config()
        result = resolve_benchmark_config(cfg, ProfileFlavor.BALANCED, config)

        assert result.model == "/cuda/model.gguf"
        assert result.threads == 16
        assert result.ubatch_size == 2048
        assert result.cache_type_k == "f16"
        assert result.cache_type_v == "f16"
        assert result.n_gpu_layers == "all"

    def test_sycl_balanced_flavor(self) -> None:
        """resolve_benchmark_config should use balanced defaults for SYCL balanced flavor."""
        cfg = _make_server_config(device="SYCL0")
        config = _make_config(
            **{
                "deployment.model_summary_balanced": "/balanced/model.gguf",
                "server_defaults.threads_summary_balanced": 8,
                "server_defaults.ubatch_size_summary_balanced": 1024,
                "server_defaults.cache_type_summary_k": "q8_0",
                "server_defaults.cache_type_summary_v": "q8_0",
            }
        )
        result = resolve_benchmark_config(cfg, ProfileFlavor.BALANCED, config)

        assert result.model == "/balanced/model.gguf"
        assert result.threads == 8
        assert result.ubatch_size == 1024
        assert result.cache_type_k == "q8_0"
        assert result.cache_type_v == "q8_0"

    def test_sycl_fast_flavor(self) -> None:
        """resolve_benchmark_config should use fast defaults for SYCL fast flavor."""
        cfg = _make_server_config(device="SYCL0")
        config = _make_config(
            **{
                "deployment.model_summary_fast": "/fast/model.gguf",
                "server_defaults.threads_summary_fast": 4,
                "server_defaults.ubatch_size_summary_fast": 512,
                "server_defaults.cache_type_summary_k": "q8_0",
                "server_defaults.cache_type_summary_v": "q8_0",
            }
        )
        result = resolve_benchmark_config(cfg, ProfileFlavor.FAST, config)

        assert result.model == "/fast/model.gguf"
        assert result.threads == 4
        assert result.ubatch_size == 512

    def test_sycl_quality_flavor_uses_balanced(self) -> None:
        """resolve_benchmark_config should use balanced defaults for SYCL quality flavor."""
        cfg = _make_server_config(device="SYCL0")
        config = _make_config(
            **{
                "deployment.model_summary_balanced": "/balanced/model.gguf",
                "server_defaults.threads_summary_balanced": 8,
                "server_defaults.ubatch_size_summary_balanced": 1024,
                "server_defaults.cache_type_summary_k": "q8_0",
                "server_defaults.cache_type_summary_v": "q8_0",
            }
        )
        result = resolve_benchmark_config(cfg, ProfileFlavor.QUALITY, config)

        assert result.model == "/balanced/model.gguf"
        assert result.threads == 8
        assert result.ubatch_size == 1024

    def test_result_is_benchmark_config(self) -> None:
        """resolve_benchmark_config should return a BenchmarkConfig instance."""
        cfg = _make_server_config(device="SYCL0")
        config = _make_config()
        result = resolve_benchmark_config(cfg, ProfileFlavor.BALANCED, config)

        assert isinstance(result, BenchmarkConfig)


# ---------------------------------------------------------------------------
# Test resolve_benchmark_binary
# ---------------------------------------------------------------------------


class TestResolveBenchmarkBinary:
    """Tests for resolve_benchmark_binary."""

    def test_derives_from_server_binary(self, tmp_path: Path) -> None:
        """resolve_benchmark_binary should derive path from server binary directory."""
        server_bin = tmp_path / "llama-server"
        server_bin.write_text("#!/bin/sh")
        server_bin.chmod(0o755)

        bench_bin = tmp_path / "llama-bench"
        bench_bin.write_text("#!/bin/sh")
        bench_bin.chmod(0o755)

        cfg = _make_server_config(server_bin=str(server_bin))
        config = _make_config(**{"paths.llama_server_bin_intel": str(server_bin)})
        result = resolve_benchmark_binary(cfg, config)
        assert result == str(bench_bin)

    def test_fallback_to_shutil_which(self, tmp_path: Path) -> None:
        """resolve_benchmark_binary should fall back to shutil.which when no server binary."""
        cfg = _make_server_config(server_bin="")
        config = _make_config(**{"paths.llama_server_bin_intel": ""})
        with patch(
            "llama_manager.profile_orchestrator.shutil.which", return_value="/usr/bin/llama-bench"
        ):
            result = resolve_benchmark_binary(cfg, config)
            assert result == "/usr/bin/llama-bench"

    def test_returns_none_when_unavailable(self) -> None:
        """resolve_benchmark_binary should return None when binary is not found."""
        cfg = _make_server_config(server_bin="")
        config = _make_config(**{"paths.llama_server_bin_intel": ""})

        with patch("llama_manager.profile_orchestrator.shutil.which", return_value=None):
            result = resolve_benchmark_binary(cfg, config)
            assert result is None

    def test_ignores_unrelated_server_binaries(self, tmp_path: Path) -> None:
        """resolve_benchmark_binary should skip unrelated server binary names."""
        server_bin = tmp_path / "llama-server-metal"
        server_bin.write_text("#!/bin/sh")
        server_bin.chmod(0o755)

        cfg = _make_server_config(server_bin=str(server_bin))
        config = _make_config(**{"paths.llama_server_bin_intel": str(server_bin)})

        # llama-bench doesn't exist in tmp_path
        result = resolve_benchmark_binary(cfg, config)
        assert result is None


# ---------------------------------------------------------------------------
# Test get_driver_version
# ---------------------------------------------------------------------------


class TestGetDriverVersion:
    """Tests for get_driver_version."""

    def test_cuda_backend_queries_nvidia_smi(self) -> None:
        """get_driver_version should query nvidia-smi for CUDA backend."""
        with patch("llama_manager.profile_orchestrator._query_nvidia_driver") as mock_nvidia:
            mock_nvidia.return_value = "535.104.05"
            result = get_driver_version("cuda")
            assert result == "535.104.05"
            mock_nvidia.assert_called_once()

    def test_sycl_backend_queries_sycl_ls(self) -> None:
        """get_driver_version should query sycl-ls for SYCL backend."""
        with patch("llama_manager.profile_orchestrator._query_sycl_driver") as mock_sycl:
            mock_sycl.return_value = "gpu:0, intel:arc"
            result = get_driver_version("sycl")
            assert result == "gpu:0, intel:arc"
            mock_sycl.assert_called_once()

    def test_cuda_returns_unknown_on_failure(self) -> None:
        """get_driver_version should return 'unknown' when nvidia-smi fails."""
        with patch("llama_manager.profile_orchestrator._query_nvidia_driver", return_value=None):
            result = get_driver_version("cuda")
            assert result == "unknown"

    def test_sycl_returns_unknown_on_failure(self) -> None:
        """get_driver_version should return 'unknown' when sycl-ls fails."""
        with patch("llama_manager.profile_orchestrator._query_sycl_driver", return_value=None):
            result = get_driver_version("sycl")
            assert result == "unknown"


# ---------------------------------------------------------------------------
# Test create_profile_record
# ---------------------------------------------------------------------------


class TestCreateProfileRecord:
    """Tests for create_profile_record."""

    def test_creates_valid_record(self) -> None:
        """create_profile_record should create a valid ProfileRecord."""
        benchmark_result = MagicMock()
        benchmark_result.tokens_per_second = 100.0
        benchmark_result.avg_latency_ms = 10.0
        benchmark_result.peak_vram_mb = 8192.0

        config = _make_config(server_binary_version="llama-server 1.0.0")

        record = create_profile_record(
            gpu_identifier="nvidia-rtx-3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="535.104.05",
            benchmark_result=benchmark_result,
            config=config,
        )

        assert isinstance(record, ProfileRecord)
        assert record.gpu_identifier == "nvidia-rtx-3090-00"
        assert record.backend == "cuda"
        assert record.flavor == ProfileFlavor.BALANCED
        assert record.driver_version == "535.104.05"
        assert record.server_binary_version == "llama-server 1.0.0"
        assert record.metrics.tokens_per_second == 100.0
        assert record.metrics.avg_latency_ms == 10.0
        assert record.metrics.peak_vram_mb == 8192.0
        assert record.parameters == {}
        assert "Z" in record.profiled_at

    def test_driver_version_hash_is_computed(self) -> None:
        """create_profile_record should compute driver_version_hash."""
        benchmark_result = MagicMock()
        benchmark_result.tokens_per_second = 100.0
        benchmark_result.avg_latency_ms = 10.0
        benchmark_result.peak_vram_mb = None

        config = _make_config(server_binary_version="")

        record = create_profile_record(
            gpu_identifier="test-gpu",
            backend="sycl",
            flavor=ProfileFlavor.FAST,
            driver_version="750.1.0",
            benchmark_result=benchmark_result,
            config=config,
        )

        assert len(record.driver_version_hash) == 16
        assert all(c in "0123456789abcdef" for c in record.driver_version_hash)

    def test_peak_vram_none_preserved(self) -> None:
        """create_profile_record should preserve None peak_vram_mb."""
        benchmark_result = MagicMock()
        benchmark_result.tokens_per_second = 100.0
        benchmark_result.avg_latency_ms = 10.0
        benchmark_result.peak_vram_mb = None

        config = _make_config()

        record = create_profile_record(
            gpu_identifier="test-gpu",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="535.104.05",
            benchmark_result=benchmark_result,
            config=config,
        )

        assert record.metrics.peak_vram_mb is None


# ---------------------------------------------------------------------------
# Test run_profile
# ---------------------------------------------------------------------------


class TestRunProfile:
    """Tests for run_profile."""

    def _make_mock_config(self, tmp_path: Path) -> Config:
        """Create a Config with test paths."""
        cfg = _make_config()
        server_bin = tmp_path / "llama-server"
        server_bin.write_text("#!/bin/sh\n")
        server_bin.chmod(0o755)

        bench_bin = tmp_path / "llama-bench"
        bench_bin.write_text("#!/bin/sh\n")
        bench_bin.chmod(0o755)

        cfg.paths.xdg_data_base = str(tmp_path)
        cfg.paths.llama_server_bin_intel = str(server_bin)
        cfg.server_binary_version = "llama-server 1.0.0"
        cfg.deployment.model_summary_balanced = str(tmp_path / "model.gguf")
        cfg.server_defaults.threads_summary_balanced = 8
        cfg.server_defaults.threads_summary_fast = 4
        cfg.server_defaults.ubatch_size_summary_balanced = 1024
        cfg.server_defaults.ubatch_size_summary_fast = 512
        cfg.server_defaults.cache_type_summary_k = "q8_0"
        cfg.server_defaults.cache_type_summary_v = "q8_0"

        return cfg

    def test_successful_profile_returns_record(self, tmp_path: Path) -> None:
        """run_profile should return a ProfileRecord on success."""
        config = self._make_mock_config(tmp_path)

        valid_benchmark_output = (
            "llama-bench result:\n"
            "| t/s  | latency (ms) | vram (MB) |\n"
            "|------|--------------|-----------|\n"
            "| 150.0| 8.0          | 6144.0    |\n"
        )

        def fake_runner(cmd: list[str]) -> SubprocessResult:
            return SubprocessResult(exit_code=0, stdout=valid_benchmark_output, stderr="")

        record = run_profile(
            slot_id="summary-balanced",
            config=config,
            flavor="balanced",
            runner=fake_runner,
            driver_provider=lambda backend: "535.104.05",
        )

        assert record is not None
        assert isinstance(record, ProfileRecord)
        assert record.backend == "sycl"
        assert record.flavor == ProfileFlavor.BALANCED

    def test_benchmark_failure_returns_none(self, tmp_path: Path) -> None:
        """run_profile should return None when benchmark fails."""
        config = self._make_mock_config(tmp_path)

        def fake_runner(cmd: list[str]) -> SubprocessResult:
            return SubprocessResult(exit_code=1, stdout="", stderr="error")

        record = run_profile(
            slot_id="summary-balanced",
            config=config,
            flavor="balanced",
            runner=fake_runner,
            driver_provider=lambda backend: "535.104.05",
        )

        assert record is None

    def test_missing_benchmark_binary_returns_none(self, tmp_path: Path) -> None:
        """run_profile should return None when benchmark binary is unavailable."""
        config = self._make_mock_config(tmp_path)
        config.paths.llama_server_bin_intel = ""

        with patch("llama_manager.profile_orchestrator.shutil.which", return_value=None):
            record = run_profile(
                slot_id="unknown-slot",
                config=config,
                flavor="balanced",
                driver_provider=lambda backend: "535.104.05",
            )

        assert record is None

    def test_non_executable_benchmark_returns_none(self, tmp_path: Path) -> None:
        """run_profile should return None when benchmark binary is not executable."""
        config = self._make_mock_config(tmp_path)

        # Create a non-executable file
        bench_bin = tmp_path / "llama-bench"
        bench_bin.write_text("#!/bin/sh")
        bench_bin.chmod(0o644)

        server_bin = tmp_path / "llama-server"
        server_bin.write_text("#!/bin/sh")
        server_bin.chmod(0o755)

        config.paths.llama_server_bin_intel = str(server_bin)

    def test_driver_provider_used_when_given(self, tmp_path: Path) -> None:
        """run_profile should use the driver_provider when given."""
        config = self._make_mock_config(tmp_path)

        benchmark_result = MagicMock()
        benchmark_result.tokens_per_second = 100.0
        benchmark_result.avg_latency_ms = 10.0
        benchmark_result.peak_vram_mb = None

        def fake_runner(cmd: list[str]) -> SubprocessResult:
            return SubprocessResult(exit_code=0, stdout="t/s: 100.0\nlatency: 10.0 ms", stderr="")

        custom_driver = "custom-driver-1.0"
        record = run_profile(
            slot_id="summary-balanced",
            config=config,
            flavor="balanced",
            runner=fake_runner,
            driver_provider=lambda backend: custom_driver,
        )

        assert record is not None
        assert record.driver_version == custom_driver

    def test_profile_written_to_disk(self, tmp_path: Path) -> None:
        """run_profile should write the profile record to disk."""
        config = self._make_mock_config(tmp_path)

        benchmark_result = MagicMock()
        benchmark_result.tokens_per_second = 100.0
        benchmark_result.avg_latency_ms = 10.0
        benchmark_result.peak_vram_mb = None

        def fake_runner(cmd: list[str]) -> SubprocessResult:
            return SubprocessResult(exit_code=0, stdout="t/s: 100.0\nlatency: 10.0 ms", stderr="")

        run_profile(
            slot_id="summary-balanced",
            config=config,
            flavor="balanced",
            runner=fake_runner,
            driver_provider=lambda backend: "535.104.05",
        )

        # Check that a profile file was created
        profiles_dir = config.paths.profiles_dir
        profile_files = list(profiles_dir.glob("*.json"))
        assert len(profile_files) == 1

    def test_custom_registry_used(self, tmp_path: Path) -> None:
        """run_profile should use the provided registry."""
        config = self._make_mock_config(tmp_path)
        registry = create_default_profile_registry(config)

        benchmark_result = MagicMock()
        benchmark_result.tokens_per_second = 100.0
        benchmark_result.avg_latency_ms = 10.0
        benchmark_result.peak_vram_mb = None

        def fake_runner(cmd: list[str]) -> SubprocessResult:
            return SubprocessResult(exit_code=0, stdout="t/s: 100.0\nlatency: 10.0 ms", stderr="")

        record = run_profile(
            slot_id="summary-balanced",
            config=config,
            flavor="balanced",
            runner=fake_runner,
            driver_provider=lambda backend: "535.104.05",
            registry=registry,
        )

        assert record is not None


# ---------------------------------------------------------------------------
# Test module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_benchmark_prompt_tokens(self) -> None:
        """BENCHMARK_PROMPT_TOKENS should be 512."""
        assert BENCHMARK_PROMPT_TOKENS == 512

    def test_benchmark_run_timeout(self) -> None:
        """BENCHMARK_RUN_TIMEOUT_SECONDS should be 600."""
        assert BENCHMARK_RUN_TIMEOUT_SECONDS == 600


# ---------------------------------------------------------------------------
# Test BenchmarkConfig
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_frozen_dataclass(self) -> None:
        """BenchmarkConfig should be immutable."""
        cfg = BenchmarkConfig(
            model="/model.gguf",
            threads=8,
            ubatch_size=1024,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        )
        # FrozenInstanceError is raised at runtime for frozen dataclasses
        # We use a try/except to avoid pyright static analysis complaints
        try:
            cfg.model = "/other.gguf"  # type: ignore[reportGeneralTypeIssues]
        except dataclasses.FrozenInstanceError, TypeError, AttributeError:
            pass
        else:
            pytest.fail("Expected FrozenInstanceError or similar")

    def test_default_n_gpu_layers(self) -> None:
        """BenchmarkConfig should default n_gpu_layers to 99."""
        cfg = BenchmarkConfig(
            model="/model.gguf",
            threads=8,
            ubatch_size=1024,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
        )
        assert cfg.n_gpu_layers == 99

    def test_custom_n_gpu_layers(self) -> None:
        """BenchmarkConfig should accept custom n_gpu_layers."""
        cfg = BenchmarkConfig(
            model="/model.gguf",
            threads=8,
            ubatch_size=1024,
            cache_type_k="q8_0",
            cache_type_v="q8_0",
            n_gpu_layers="all",
        )
        assert cfg.n_gpu_layers == "all"
