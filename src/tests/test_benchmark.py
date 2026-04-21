"""Tests for llama_manager.benchmark — command building, parsing, and runner."""

from __future__ import annotations

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.benchmark import (
    BenchmarkResult,
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

    def test_non_finite_tokens_per_second(self) -> None:
        """Test that inf/nan tokens_per_second is handled gracefully."""
        # "inf" does not match the numeric regex in parse_benchmark_output,
        # but avg latency still parses, so a partial result is returned.
        output = "tok/s: inf\navg latency: 100.0 ms\npeak memory: 2048"
        result = parse_benchmark_output(output)
        assert result is not None
        # tokens_per_second defaults to 0.0 when the regex doesn't match
        assert result.tokens_per_second == pytest.approx(0.0)
        assert result.avg_latency_ms == pytest.approx(100.0)

    def test_nan_tokens_per_second(self) -> None:
        """Test that nan tokens_per_second is handled gracefully."""
        output = "tokens per second: nan\navg latency: 50.0 ms\n"
        result = parse_benchmark_output(output)
        assert result is not None
        # "nan" does not match the numeric regex, so defaults to 0.0
        assert result.tokens_per_second == pytest.approx(0.0)
        assert result.avg_latency_ms == pytest.approx(50.0)


class TestBuildBenchmarkCmdEdgeCases:
    """Edge-case tests for build_benchmark_cmd."""

    def _make_temp_bin(self, executable: bool = True) -> str:
        """Create a temporary file to serve as a fake llama-bench binary."""
        fd, path = tempfile.mkstemp(prefix="llama-bench-")
        os.close(fd)
        if executable:
            os.chmod(path, 0o755)  # noqa: S103
        return path

    def test_n_gpu_layers_explicit_string_all(self) -> None:
        """Test build_benchmark_cmd with n_gpu_layers='all' as explicit string."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/tmp/model.gguf",
                port=8080,
                threads=4,
                ctx_size=2048,
                ubatch_size=512,
                cache_type_k="q4_0",
                cache_type_v="q4_0",
                n_gpu_layers="all",  # explicit string, not default
            )
        finally:
            os.unlink(bin_path)
        assert "-ngl" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "all"

    def test_n_gpu_layers_custom_int_value(self) -> None:
        """Test build_benchmark_cmd with n_gpu_layers as an int."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/tmp/model.gguf",
                port=8080,
                threads=4,
                ctx_size=2048,
                ubatch_size=512,
                cache_type_k="q4_0",
                cache_type_v="q4_0",
                n_gpu_layers=99,
            )
        finally:
            os.unlink(bin_path)
        assert "-ngl" in cmd
        ngl_idx = cmd.index("-ngl")
        assert cmd[ngl_idx + 1] == "99"


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

    def test_stderr_ignored_for_result(self) -> None:
        """run_benchmark should ignore stderr and only use stdout for parsing."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 150.0\navg latency: 5.0 ms\n",
            stderr="warning: deprecated flag\n",
        )

        result = run_benchmark(["llama-bench"], runner)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(150.0)

    def test_runner_called_exactly_once(self) -> None:
        """run_benchmark should call runner exactly once."""
        runner = MagicMock()
        runner.return_value = SubprocessResult(
            exit_code=0,
            stdout="tokens per second: 100.0\n",
            stderr="",
        )

        run_benchmark(["llama-bench", "-m", "model.gguf"], runner)
        runner.assert_called_once()
        assert runner.call_count == 1


class TestSubprocessResult:
    """Tests for SubprocessResult dataclass."""

    def test_immutable(self) -> None:
        """SubprocessResult should be immutable (frozen dataclass)."""
        result = SubprocessResult(exit_code=0, stdout="out", stderr="err")
        with pytest.raises(AttributeError):
            result.exit_code = 1  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.stdout = "new"  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.stderr = "new"  # type: ignore[assignment]

    def test_equality(self) -> None:
        """SubprocessResult equality should compare all fields."""
        r1 = SubprocessResult(exit_code=0, stdout="out", stderr="err")
        r2 = SubprocessResult(exit_code=0, stdout="out", stderr="err")
        r3 = SubprocessResult(exit_code=1, stdout="out", stderr="err")
        assert r1 == r2
        assert r1 != r3

    def test_repr(self) -> None:
        """SubprocessResult repr should include all fields."""
        result = SubprocessResult(exit_code=42, stdout="hello", stderr="world")
        repr_str = repr(result)
        assert "exit_code=42" in repr_str
        assert "stdout='hello'" in repr_str
        assert "stderr='world'" in repr_str


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_immutable(self) -> None:
        """BenchmarkResult should be immutable (frozen dataclass)."""
        result = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        with pytest.raises(AttributeError):
            result.tokens_per_second = 200.0  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.avg_latency_ms = 20.0  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            result.peak_vram_mb = 4096.0  # type: ignore[assignment]

    def test_equality(self) -> None:
        """BenchmarkResult equality should compare all fields."""
        r1 = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        r2 = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        r3 = BenchmarkResult(
            tokens_per_second=200.0,
            avg_latency_ms=10.0,
            peak_vram_mb=2048.0,
        )
        assert r1 == r2
        assert r1 != r3

    def test_none_peak_vram(self) -> None:
        """BenchmarkResult should accept None for peak_vram_mb."""
        result = BenchmarkResult(
            tokens_per_second=100.0,
            avg_latency_ms=10.0,
            peak_vram_mb=None,
        )
        assert result.peak_vram_mb is None

    def test_zero_values(self) -> None:
        """BenchmarkResult should accept zero values."""
        result = BenchmarkResult(
            tokens_per_second=0.0,
            avg_latency_ms=0.0,
            peak_vram_mb=0.0,
        )
        assert result.tokens_per_second == 0.0
        assert result.avg_latency_ms == 0.0
        assert result.peak_vram_mb == 0.0

    def test_large_values(self) -> None:
        """BenchmarkResult should handle large values."""
        result = BenchmarkResult(
            tokens_per_second=999999.99,
            avg_latency_ms=9999.99,
            peak_vram_mb=99999.0,
        )
        assert result.tokens_per_second == pytest.approx(999999.99)
        assert result.avg_latency_ms == pytest.approx(9999.99)
        assert result.peak_vram_mb == pytest.approx(99999.0)

    def test_fractional_values(self) -> None:
        """BenchmarkResult should handle fractional values."""
        result = BenchmarkResult(
            tokens_per_second=0.5,
            avg_latency_ms=0.01,
            peak_vram_mb=123.456,
        )
        assert result.tokens_per_second == pytest.approx(0.5)
        assert result.avg_latency_ms == pytest.approx(0.01)
        assert result.peak_vram_mb == pytest.approx(123.456)


class TestParseBenchmarkOutputOnlyLatency:
    """Tests for parse_benchmark_output with only latency present."""

    def test_only_latency_parsed(self) -> None:
        """parse_benchmark_output should return result with only latency."""
        output = "avg latency: 50.0 ms\n"
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(0.0)
        assert result.avg_latency_ms == pytest.approx(50.0)
        assert result.peak_vram_mb is None

    def test_only_latency_no_ms_unit(self) -> None:
        """parse_benchmark_output should parse latency without ms unit."""
        output = "latency: 25.5\n"
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(0.0)
        assert result.avg_latency_ms == pytest.approx(25.5)
        assert result.peak_vram_mb is None


class TestParseBenchmarkOutputValueErrorBranches:
    """Tests for parse_benchmark_output ValueError handling branches.

    These tests use mocking to exercise the except ValueError blocks,
    which are unreachable with the default regex patterns since they only
    match valid numeric strings.
    """

    def test_valueerror_in_tokens_parsing(self) -> None:
        """parse_benchmark_output should handle ValueError in tokens parsing."""
        with patch("re.search") as mock_search:
            call_count = [0]

            def side_effect(pattern, string, flags=0):
                call_count[0] += 1
                # Simulate tokens pattern matching a non-numeric string
                if call_count[0] <= 4:
                    # First 4 calls are tokens patterns — return non-numeric
                    match = MagicMock()
                    match.group.return_value = "not_a_number"
                    match.__bool__ = lambda self: True
                    return match
                # Latency pattern matches valid number
                match = MagicMock()
                match.group.return_value = "100.0"
                match.__bool__ = lambda self: True
                return match

            mock_search.side_effect = side_effect
            result = parse_benchmark_output(
                "tokens per second: not_a_number\navg latency: 100.0 ms\n"
            )
            assert result is not None
            # tokens_per_second defaults to 0.0 when ValueError occurs
            assert result.tokens_per_second == pytest.approx(0.0)
            assert result.avg_latency_ms == pytest.approx(100.0)

    def test_valueerror_in_latency_parsing(self) -> None:
        """parse_benchmark_output should handle ValueError in latency parsing."""
        with patch("re.search") as mock_search:
            call_count = [0]

            def side_effect(pattern, string, flags=0):
                call_count[0] += 1
                # Tokens pattern matches valid number (first call)
                if call_count[0] == 1:
                    match = MagicMock()
                    match.group.return_value = "500.0"
                    match.__bool__ = lambda self: True
                    return match
                # All subsequent patterns (tokens 2-4, latency 1-4, vram 1-6)
                # return None (no match), so the function falls through to
                # the next pattern group. For latency, the first pattern
                # that "matches" returns a non-numeric string.
                # We simulate: tokens patterns 2-4 return None, then latency
                # pattern 1 returns a non-numeric match.
                if call_count[0] <= 4:
                    # Tokens patterns 2-4: no match
                    return None
                if call_count[0] == 5:
                    # First latency pattern "matches" but with non-numeric
                    match = MagicMock()
                    match.group.return_value = "slow"
                    match.__bool__ = lambda self: True
                    return match
                # Remaining patterns: no match
                return None

            mock_search.side_effect = side_effect
            result = parse_benchmark_output("tokens per second: 500.0\navg latency: slow\n")
            assert result is not None
            assert result.tokens_per_second == pytest.approx(500.0)
            # avg_latency_ms defaults to 0.0 when ValueError occurs
            assert result.avg_latency_ms == pytest.approx(0.0)

    def test_valueerror_in_vram_parsing(self) -> None:
        """parse_benchmark_output should handle ValueError in VRAM parsing."""
        with patch("re.search") as mock_search:
            call_count = [0]

            def side_effect(pattern, string, flags=0):
                call_count[0] += 1
                # Tokens pattern 1 matches valid number
                if call_count[0] == 1:
                    match = MagicMock()
                    match.group.return_value = "100.0"
                    match.__bool__ = lambda self: True
                    return match
                # Tokens patterns 2-4: no match
                if call_count[0] <= 4:
                    return None
                # Latency pattern 1 matches valid number
                if call_count[0] == 5:
                    match = MagicMock()
                    match.group.return_value = "50.0"
                    match.__bool__ = lambda self: True
                    return match
                # Latency patterns 2-4: no match
                if call_count[0] <= 8:
                    return None
                # VRAM pattern 1 matches non-numeric
                if call_count[0] == 9:
                    match = MagicMock()
                    match.group.return_value = "gibberish"
                    match.__bool__ = lambda self: True
                    return match
                # Remaining VRAM patterns: no match
                return None

            mock_search.side_effect = side_effect
            result = parse_benchmark_output(
                "tokens per second: 100.0\navg latency: 50.0 ms\npeak memory: gibberish\n"
            )
            assert result is not None
            assert result.tokens_per_second == pytest.approx(100.0)
            assert result.avg_latency_ms == pytest.approx(50.0)
            # peak_vram_mb stays None when ValueError occurs
            assert result.peak_vram_mb is None


class TestParseBenchmarkOutputIsfiniteBranches:
    """Tests for parse_benchmark_output math.isfinite validation branches.

    These tests use mocking to exercise the math.isfinite=False paths,
    which are unreachable with the default regex patterns since they only
    match valid finite numeric strings.
    """

    def test_isfinite_false_for_tokens(self) -> None:
        """parse_benchmark_output should set tokens_per_second to None when isfinite returns False."""
        with patch.object(
            math,
            "isfinite",
            side_effect=[False, True],  # first call (tokens) False, second (latency) True
        ):
            result = parse_benchmark_output("tokens per second: 123.45\navg latency: 10.0 ms\n")
            assert result is not None
            # tokens_per_second is invalidated by isfinite check
            assert result.tokens_per_second == pytest.approx(0.0)
            # avg_latency_ms is still valid
            assert result.avg_latency_ms == pytest.approx(10.0)

    def test_isfinite_false_for_latency(self) -> None:
        """parse_benchmark_output should set avg_latency_ms to None when isfinite returns False."""
        with patch.object(
            math,
            "isfinite",
            side_effect=[True, False],  # first call (tokens) True, second (latency) False
        ):
            result = parse_benchmark_output("tokens per second: 100.0\navg latency: 50.0 ms\n")
            assert result is not None
            # avg_latency_ms is invalidated by isfinite check
            assert result.avg_latency_ms == pytest.approx(0.0)
            # tokens_per_second is still valid
            assert result.tokens_per_second == pytest.approx(100.0)

    def test_isfinite_false_for_both_returns_none(self) -> None:
        """parse_benchmark_output should return None when both metrics fail isfinite."""
        with patch.object(math, "isfinite", return_value=False):
            result = parse_benchmark_output("tokens per second: 100.0\navg latency: 50.0 ms\n")
            # Both tokens and latency invalidated → return None
            assert result is None

    def test_isfinite_true_preserves_values(self) -> None:
        """parse_benchmark_output should preserve values when isfinite returns True."""
        with patch.object(math, "isfinite", return_value=True):
            result = parse_benchmark_output("tokens per second: 42.0\navg latency: 3.14 ms\n")
            assert result is not None
            assert result.tokens_per_second == pytest.approx(42.0)
            assert result.avg_latency_ms == pytest.approx(3.14)


class TestParseBenchmarkOutputMixedFormats:
    """Tests for parse_benchmark_output with mixed format strings."""

    def test_uppercase_tokens_format(self) -> None:
        """parse_benchmark_output should handle uppercase tokens format."""
        result = parse_benchmark_output("TOKENS PER SECOND: 100.0\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(100.0)

    def test_mixed_case_latency_format(self) -> None:
        """parse_benchmark_output should handle mixed case latency format."""
        result = parse_benchmark_output("AvG lAtEnCy: 25.0 ms\n")
        assert result is not None
        assert result.avg_latency_ms == pytest.approx(25.0)

    def test_mixed_case_vram_format(self) -> None:
        """parse_benchmark_output should handle mixed case VRAM format."""
        result = parse_benchmark_output("tokens per second: 100.0\nPEAK VRAM: 8192.0 MB\n")
        assert result is not None
        assert result.peak_vram_mb == pytest.approx(8192.0)

    def test_multiple_metrics_same_line(self) -> None:
        """parse_benchmark_output should handle metrics on same line."""
        result = parse_benchmark_output("tokens per second: 100.0 avg latency: 5.0 ms\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(100.0)
        assert result.avg_latency_ms == pytest.approx(5.0)

    def test_multiline_with_header_footer(self) -> None:
        """parse_benchmark_output should handle realistic multi-line benchmark output."""
        output = (
            "llama-bench: I llama-bench version: 1.0.0 (git: abc123)\n"
            "llama-bench: built on Apr 21 2026\n"
            "model: /models/qwen2.5-7b-q4_k_m.gguf\n"
            "params: 7.0B\n"
            "tokens per second: 145.678\n"
            "avg latency: 6.89 ms\n"
            "peak memory: 4096.5 MB\n"
            "llama-bench: benchmark complete\n"
        )
        result = parse_benchmark_output(output)
        assert result is not None
        assert result.tokens_per_second == pytest.approx(145.678)
        assert result.avg_latency_ms == pytest.approx(6.89)
        assert result.peak_vram_mb == pytest.approx(4096.5)

    def test_whitespace_around_colon(self) -> None:
        """parse_benchmark_output should handle various whitespace around colons."""
        # Space before colon
        result = parse_benchmark_output("tokens per second : 100.0\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(100.0)

    def test_no_space_after_colon(self) -> None:
        """parse_benchmark_output should handle no space after colon."""
        result = parse_benchmark_output("tokens per second:100.0\n")
        assert result is not None
        assert result.tokens_per_second == pytest.approx(100.0)


class TestBuildBenchmarkCmdAllParams:
    """Tests for build_benchmark_cmd with all parameter combinations."""

    def _make_temp_bin(self, executable: bool = True) -> str:
        """Create a temporary file to serve as a fake llama-bench binary."""
        fd, path = tempfile.mkstemp(prefix="llama-bench-")
        os.close(fd)
        if executable:
            os.chmod(path, 0o755)  # noqa: S103
        return path

    def test_all_parameters_included(self) -> None:
        """build_benchmark_cmd should include all parameters in the command."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/models/qwen2.5-7b-q4_k_m.gguf",
                port=8080,
                threads=16,
                ctx_size=8192,
                ubatch_size=2048,
                cache_type_k="F16",
                cache_type_v="F32",
                n_gpu_layers=99,
            )
        finally:
            os.unlink(bin_path)

        assert cmd[0] == bin_path
        assert "-m" in cmd
        assert "/models/qwen2.5-7b-q4_k_m.gguf" in cmd
        assert "-p" in cmd
        assert "8080" in cmd
        assert "-t" in cmd
        assert "16" in cmd
        assert "-c" in cmd
        assert "8192" in cmd
        assert "--ubatch-size" in cmd
        assert "2048" in cmd
        assert "--cache-type-k" in cmd
        assert "F16" in cmd
        assert "--cache-type-v" in cmd
        assert "F32" in cmd
        assert "-ngl" in cmd
        assert "99" in cmd

    def test_cmd_length(self) -> None:
        """build_benchmark_cmd should produce a command with expected length."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/model.gguf",
                port=8080,
                threads=4,
                ctx_size=2048,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )
        finally:
            os.unlink(bin_path)

        # bench_bin(1) + 9 flag-value pairs(18) = 19 elements
        # Wait, let me count: bin, -m, model, -p, port, -t, threads, -c, ctx,
        # --ubatch-size, size, --cache-type-k, type_k, --cache-type-v, type_v,
        # -ngl, layers = 17 elements
        assert len(cmd) == 17

    def test_cmd_order(self) -> None:
        """build_benchmark_cmd should maintain correct argument order."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/model.gguf",
                port=8080,
                threads=4,
                ctx_size=2048,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )
        finally:
            os.unlink(bin_path)

        # Verify order: bin, -m model, -p port, -t threads, -c ctx, --ubatch-size size,
        # --cache-type-k type, --cache-type-v type, -ngl layers
        assert cmd[0] == bin_path
        assert cmd[1] == "-m"
        assert cmd[3] == "-p"
        assert cmd[5] == "-t"
        assert cmd[7] == "-c"
        assert cmd[9] == "--ubatch-size"
        assert cmd[11] == "--cache-type-k"
        assert cmd[13] == "--cache-type-v"
        assert cmd[15] == "-ngl"

    def test_special_characters_in_model_path(self) -> None:
        """build_benchmark_cmd should handle special characters in model path."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/models/qwen2.5-7b-q4_k_m.gguf",
                port=8080,
                threads=4,
                ctx_size=2048,
                ubatch_size=512,
                cache_type_k="F16",
                cache_type_v="F16",
            )
        finally:
            os.unlink(bin_path)

        assert "/models/qwen2.5-7b-q4_k_m.gguf" in cmd

    def test_cache_type_q8_0(self) -> None:
        """build_benchmark_cmd should accept Q8_0 cache types."""
        bin_path = self._make_temp_bin()
        try:
            cmd = build_benchmark_cmd(
                bench_bin=bin_path,
                model="/model.gguf",
                port=8080,
                threads=4,
                ctx_size=2048,
                ubatch_size=512,
                cache_type_k="Q8_0",
                cache_type_v="Q8_0",
            )
        finally:
            os.unlink(bin_path)

        assert "Q8_0" in cmd
