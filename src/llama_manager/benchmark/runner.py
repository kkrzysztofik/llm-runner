"""Benchmark runner — subprocess result types and execution protocol."""

import os
from collections.abc import Callable
from dataclasses import dataclass

from .parser import BenchmarkResult, parse_benchmark_output


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    """Result of a benchmark subprocess execution.

    Attributes:
        exit_code: Non-negative exit code returned by the subprocess.
        stdout: Captured standard output from the benchmark process.
        stderr: Captured standard error from the benchmark process.
    """

    exit_code: int
    stdout: str
    stderr: str


# Callable type alias for benchmark subprocess runners.
# Concrete implementations execute a command list and return a SubprocessResult.
BenchmarkRunner = Callable[[list[str]], SubprocessResult]


def build_benchmark_cmd(
    bench_bin: str,
    model: str,
    n_prompt: int,
    threads: int,
    ubatch_size: int,
    cache_type_k: str,
    cache_type_v: str,
    n_gpu_layers: int | str = "all",
) -> list[str]:
    """Build llama-bench command arguments.

    Constructs a subprocess-safe command list for running llama-bench with the
    specified configuration.

    Args:
        bench_bin: Path to the llama-bench binary.
        model: Path to the model file to benchmark.
        n_prompt: Number of prompt tokens for benchmarking.
        threads: Number of threads to use.
        ubatch_size: Ubatch size for the benchmark.
        cache_type_k: Cache type for K attention (e.g. "F16", "F32", "Q8_0").
        cache_type_v: Cache type for V attention (e.g. "F16", "F32", "Q8_0").
        n_gpu_layers: Number of layers to offload to GPU, or "all" for full offload.
            Defaults to "all" per spec.

    Returns:
        List of command arguments for subprocess.

    Raises:
        FileNotFoundError: If bench_bin does not exist.
        PermissionError: If bench_bin exists but is not executable.

    """
    if not os.path.exists(bench_bin):
        raise FileNotFoundError(f"llama-bench binary not found: {bench_bin}")
    if not os.access(bench_bin, os.X_OK):
        raise PermissionError(f"llama-bench binary is not executable: {bench_bin}")

    cmd = [
        bench_bin,
        "-m",
        model,
        "-p",
        str(n_prompt),
        "-t",
        str(threads),
        "--ubatch-size",
        str(ubatch_size),
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
        "-ngl",
        str(n_gpu_layers),
    ]

    return cmd


def run_benchmark(cmd: list[str], runner: BenchmarkRunner) -> BenchmarkResult | None:
    """Run a benchmark subprocess via the injectable runner and parse results.

    Delegates actual execution to *runner*, keeping this module pure — no
    subprocess calls at module level.

    Args:
        cmd: Subprocess-safe command list to execute.
        runner: Callable that executes *cmd* and returns a
            :class:`SubprocessResult`.

    Returns:
        A :class:`BenchmarkResult` with parsed metrics, or ``None`` when the
        subprocess exited with a non-zero code, produced no stdout, or the
        output could not be parsed.

    """
    result = runner(cmd)

    if result.exit_code != 0:
        return None

    if not result.stdout or not result.stdout.strip():
        return None

    return parse_benchmark_output(result.stdout)
