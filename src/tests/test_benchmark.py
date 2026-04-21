"""Tests for llama_manager.benchmark — command building, parsing, and runner."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from llama_manager.benchmark import (
    SubprocessResult,
    build_benchmark_cmd,
    parse_benchmark_output,
    run_benchmark,
)


class TestBuildBenchmarkCmd:
    """Tests for build_benchmark_cmd."""

    def _make_temp_bin(self, executable: bool = True) -> str:
        """Create a temporary file to serve as a fake llama-bench binary."""
        fd, path = tempfile.mkstemp(prefix="llama-bench-")
        os.close(fd)
        if executable:
            os.chmod(path, 0o755)  # noqa: S103
        return path

    def test_contains_required_flags(self) -> None:
        """build_benchmark_cmd should contain all required flags."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/models/test.gguf",
                port=8080,
                threads=4,
                ctx_size=4096,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )
        finally:
            os.unlink(bin_path)

        assert "-m" in cmd
        assert "/models/test.gguf" in cmd
        assert "-p" in cmd
        assert "8080" in cmd
        assert "-t" in cmd
        assert "4" in cmd
        assert "-c" in cmd
        assert "4096" in cmd
        assert "--ubatch-size" in cmd
        assert "512" in cmd
        assert "--cache-type-k" in cmd
        assert "F16" in cmd
        assert "--cache-type-v" in cmd
        assert "F16" in cmd
        assert "-ngl" in cmd

    def test_returns_list_of_strings(self) -> None:
        """build_benchmark_cmd should return a list[str]."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/models/test.gguf",
                port=8080,
                threads=4,
                ctx_size=4096,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )
        finally:
            os.unlink(bin_path)

        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)

    def test_n_gpu_layers_default_all(self) -> None:
        """n_gpu_layers should default to 'all'."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/models/test.gguf",
                port=8080,
                threads=4,
                ctx_size=4096,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )
        finally:
            os.unlink(bin_path)

        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "all"

    def test_n_gpu_layers_custom_value(self) -> None:
        """n_gpu_layers should accept a custom int value."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/models/test.gguf",
                port=8080,
                threads=4,
                ctx_size=4096,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
                n_gpu_layers=33,
            )
        finally:
            os.unlink(bin_path)

        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "33"

    def test_nonexistent_binary_raises_file_not_found(self) -> None:
        """build_benchmark_cmd should raise FileNotFoundError for missing binary."""
        with pytest.raises(FileNotFoundError, match="llama-bench binary not found"):
            build_benchmark_cmd(
                bench_bin="/nonexistent/llama-bench",
                model="/models/test.gguf",
                port=8080,
                threads=4,
                ctx_size=4096,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )

    def test_nonexecutable_binary_raises_permission_error(self) -> None:
        """build_benchmark_cmd should raise PermissionError for non-executable binary."""
        fd, path = tempfile.mkstemp(prefix="llama-bench-noexec-")
        os.close(fd)
        try:
            os.chmod(path, 0o644)
            with pytest.raises(PermissionError, match="llama-bench binary is not executable"):
                build_benchmark_cmd(
                    bench_bin=path,
                    model="/models/test.gguf",
                    port=8080,
                    threads=4,
                    ctx_size=4096,
                    ubatch_size=512,
                    cache_type_k="F16",
                    cache_type_v="F16",
                )
        finally:
            os.unlink(path)


class TestParseBenchmarkOutput:
    """Tests for parse_benchmark_output."""

    def test_success_with_valid_output(self) -> None:
        """parse_benchmark_output should return BenchmarkResult from valid output."""
        output = (
            "llama-bench: I llama-bench version: 1.0.0\n"
            "tokens per second: 123.45\n"
            "avg latency: 45.67 ms\n"
            "peak memory: 2048.0 MB\n"
        )
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(123.45)
        assert result.avg_latency_ms == pytest.approx(45.67)
        assert result.peak_vram_mb == pytest.approx(2048.0)

    def test_empty_output_returns_none(self) -> None:
        """parse_benchmark_output should return None for empty output."""
        assert parse_benchmark_output("") is None
        assert parse_benchmark_output("   ") is None
        assert parse_benchmark_output("\n\n") is None

    def test_partial_output_only_tokens_per_second(self) -> None:
        """parse_benchmark_output should return partial result with tokens/s only."""
        output = "tokens per second: 999.99\n"
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(999.99)
        assert result.avg_latency_ms == pytest.approx(0.0)
        assert result.peak_vram_mb is None

    def test_partial_output_tokens_and_latency(self) -> None:
        """parse_benchmark_output should return partial result without VRAM."""
        output = "tokens per second: 500.0\navg latency: 20.0 ms\n"
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(500.0)
        assert result.avg_latency_ms == pytest.approx(20.0)
        assert result.peak_vram_mb is None

    def test_various_tokens_per_second_formats(self) -> None:
        """parse_benchmark_output should handle different tokens/s formats."""
        # t/s format
        result = parse_benchmark_output("t/s: 100.0\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(100.0)

        # tokens/s format
        result = parse_benchmark_output("tokens/s: 200.0\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(200.0)

        # tok/s format
        result = parse_benchmark_output("tok/s: 300.0\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(300.0)

    def test_various_latency_formats(self) -> None:
        """parse_benchmark_output should handle different latency formats."""
        # avg latency
        result = parse_benchmark_output("avg latency: 10.5 ms\n")
        assert result is not None
        assert result.avg_latency_ms == pytest.approx(10.5)

        # latency
        result = parse_benchmark_output("latency: 20.3 ms\n")
        assert result is not None
        assert result.avg_latency_ms == pytest.approx(20.3)

    def test_various_vram_formats(self) -> None:
        """parse_benchmark_output should handle different VRAM formats."""
        # peak memory (with tokens/s so result is not None)
        result = parse_benchmark_output("tokens per second: 100.0\npeak memory: 4096.0 MB\n")
        assert result is not None
        assert result.peak_vram_mb == pytest.approx(4096.0)

        # peak vram (with tokens/s so result is not None)
        result = parse_benchmark_output("tokens per second: 100.0\npeak vram: 8192.0 mb\n")
        assert result is not None
        assert result.peak_vram_mb == pytest.approx(8192.0)

    def test_no_matching_metrics_returns_none(self) -> None:
        """parse_benchmark_output should return None when no metrics match."""
        output = "This is just some random text without any metrics.\n"
        assert parse_benchmark_output(output) is None

    def test_output_with_only_vram_returns_none(self) -> None:
        """parse_benchmark_output should return None if only VRAM is present (no tokens/s or latency)."""
        output = "peak memory: 1024.0 MB\n"
        assert parse_benchmark_output(output) is None


class TestRunBenchmark:
    """Tests for run_benchmark."""

    def test_calls_runner_with_cmd(self) -> None:
        """run_benchmark should pass cmd to runner."""
        runner = MagicMock()
        cmd = ["llama-bench", "-m", "model.gguf"]
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 100.0\n",
            stderr="",
        )

        run_benchmark(cmd, runner)

        runner.assert_called_once_with(cmd)

    def test_returns_none_on_nonzero_exit(self) -> None:
        """run_benchmark should return None when runner returns nonzero exit code."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=1,
            stdout="",
            stderr="error occurred\n",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is None

    def test_returns_none_on_empty_stdout(self) -> None:
        """run_benchmark should return None when runner returns empty stdout."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="",
            stderr="",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is None

    def test_returns_none_on_whitespace_stdout(self) -> None:
        """run_benchmark should return None when runner returns whitespace-only stdout."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="   \n\n  ",
            stderr="",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is None

    def test_returns_parsed_result_on_success(self) -> None:
        """run_benchmark should return parsed BenchmarkResult on success."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 42.0\navg latency: 1.0 ms\n",
            stderr="",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(42.0)
        assert result.avg_latency_ms == pytest.approx(1.0)
