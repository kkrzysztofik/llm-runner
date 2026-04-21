# Benchmark result types and runner protocol

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregated results from a single benchmark run.

    Attributes:
        tokens_per_second: Throughput measured in tokens per second.
        avg_latency_ms: Average inference latency in milliseconds.
        peak_vram_mb: Peak VRAM usage in megabytes, or ``None`` if unavailable.
    """

    tokens_per_second: float
    avg_latency_ms: float
    peak_vram_mb: float | None


# Callable type alias for benchmark subprocess runners.
# Concrete implementations execute a command list and return a SubprocessResult.
BenchmarkRunner = Callable[[list[str]], SubprocessResult]


def parse_benchmark_output(output: str) -> BenchmarkResult | None:
    """Parse benchmark stdout for performance metrics.

    Extracts tokens/s, avg latency (ms), and peak VRAM (MB) from benchmark
    output using regex matching. Returns a :class:`BenchmarkResult` with
    whatever valid metrics it can find.

    The function is forgiving — it returns ``None`` only when the output is
    empty, no metrics are found at all, or parsed values are not valid floats.
    Partial results (e.g. only tokens/s and latency, but no VRAM) are still
    returned with ``peak_vram_mb`` set to ``None``.

    Args:
        output: Raw stdout string from a benchmark subprocess.

    Returns:
        A :class:`BenchmarkResult` with extracted metrics, or ``None`` if
        the output is empty or no metrics could be parsed.

    """
    if not output or not output.strip():
        return None

    tokens_per_second: float | None = None
    avg_latency_ms: float | None = None
    peak_vram_mb: float | None = None

    # tokens/s — match "tokens per second", "t/s", "tokens/s", "tok/s"
    tokens_patterns = [
        r"tokens?\s+per\s+second[:\s]+([0-9]*\.?[0-9]+)",
        r"t/s[:\s]+([0-9]*\.?[0-9]+)",
        r"tokens?/s[:\s]+([0-9]*\.?[0-9]+)",
        r"tok/s[:\s]+([0-9]*\.?[0-9]+)",
    ]
    for pattern in tokens_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                tokens_per_second = float(match.group(1))
            except ValueError:
                pass
            else:
                break

    # avg latency in ms — match "avg latency", "latency", "ms"
    latency_patterns = [
        r"avg\s+latency[:\s]+([0-9]*\.?[0-9]+)\s*ms",
        r"latency[:\s]+([0-9]*\.?[0-9]+)\s*ms",
        r"avg\s+latency[:\s]+([0-9]*\.?[0-9]+)",
        r"latency[:\s]+([0-9]*\.?[0-9]+)",
    ]
    for pattern in latency_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                avg_latency_ms = float(match.group(1))
            except ValueError:
                pass
            else:
                break

    # peak VRAM in MB — match "peak memory", "vram", "memory", "MB"
    vram_patterns = [
        r"peak\s+memory[:\s]+([0-9]*\.?[0-9]+)\s*mb",
        r"peak\s+vram[:\s]+([0-9]*\.?[0-9]+)\s*mb",
        r"vram[:\s]+([0-9]*\.?[0-9]+)\s*mb",
        r"peak\s+memory[:\s]+([0-9]*\.?[0-9]+)",
        r"memory[:\s]+([0-9]*\.?[0-9]+)\s*mb",
        r"memory[:\s]+([0-9]*\.?[0-9]+)",
    ]
    for pattern in vram_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                peak_vram_mb = float(match.group(1))
            except ValueError:
                pass
            else:
                break

    # Return None only if no metrics were found at all
    if tokens_per_second is None and avg_latency_ms is None:
        return None

    # Validate that required metrics are valid floats
    if tokens_per_second is not None and not math.isfinite(tokens_per_second):
        tokens_per_second = None
    if avg_latency_ms is not None and not math.isfinite(avg_latency_ms):
        avg_latency_ms = None

    if tokens_per_second is None or avg_latency_ms is None:
        return None

    return BenchmarkResult(
        tokens_per_second=tokens_per_second,
        avg_latency_ms=avg_latency_ms,
        peak_vram_mb=peak_vram_mb,
    )


def build_benchmark_cmd(
    bench_bin: str,
    model: str,
    n_prompt: int,
    threads: int,
    ctx_size: int,
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
        ctx_size: Context size (number of tokens).
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
        "-c",
        str(ctx_size),
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
