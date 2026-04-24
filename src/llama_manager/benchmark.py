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


def _extract_first_float(text: str) -> float | None:
    match = re.search(r"(\d*\.?\d+)", text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


# ── Markdown table parsing helpers ──────────────────────────────────────


def _find_column_indices(
    header_cells: list[str],
) -> tuple[int | None, int | None, int | None]:
    """Find column indices for tokens/s, latency, and VRAM in a table header.

    Args:
        header_cells: Stripped, lowercased header cell values.

    Returns:
        Tuple of (tokens_idx, latency_idx, vram_idx).
    """
    tokens_idx: int | None = None
    latency_idx: int | None = None
    vram_idx: int | None = None

    for idx, header in enumerate(header_cells):
        if tokens_idx is None and any(token in header for token in ("t/s", "tok/s", "tokens/s")):
            tokens_idx = idx
        if latency_idx is None and "latency" in header:
            latency_idx = idx
        if vram_idx is None and any(token in header for token in ("vram", "memory")):
            vram_idx = idx

    return tokens_idx, latency_idx, vram_idx


def _parse_data_row(
    cells: list[str],
    tokens_idx: int,
    latency_idx: int,
    vram_idx: int | None,
) -> tuple[float | None, float | None, float | None]:
    """Parse a single data row from a markdown table.

    Args:
        cells: Stripped cell values from the row.
        tokens_idx: Column index for tokens/s.
        latency_idx: Column index for latency.
        vram_idx: Column index for VRAM, or ``None`` if not present.

    Returns:
        Tuple of (tokens_per_second, avg_latency_ms, peak_vram_mb).
    """
    tokens_per_second = _extract_first_float(cells[tokens_idx])
    avg_latency_ms = _extract_first_float(cells[latency_idx])
    peak_vram_mb = (
        _extract_first_float(cells[vram_idx])
        if vram_idx is not None and len(cells) > vram_idx
        else None
    )
    return tokens_per_second, avg_latency_ms, peak_vram_mb


def _parse_markdown_table_metrics(
    output: str,
) -> tuple[float | None, float | None, float | None]:
    """Extract metrics from a markdown table in benchmark output.

    Looks for a table with headers containing tokens/s, latency, and optionally
    VRAM/memory columns. Returns the first valid data row found.

    Args:
        output: Raw stdout string from a benchmark subprocess.

    Returns:
        Tuple of (tokens_per_second, avg_latency_ms, peak_vram_mb).
    """
    table_lines = [line for line in output.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return None, None, None

    header_cells = [cell.strip().lower() for cell in table_lines[0].strip().strip("|").split("|")]
    if not header_cells:
        return None, None, None

    tokens_idx, latency_idx, vram_idx = _find_column_indices(header_cells)

    if tokens_idx is None or latency_idx is None:
        return None, None, None

    for line in table_lines[2:]:
        if not line.strip() or set(line.replace("|", "").strip()) <= {"-", ":", " "}:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) <= max(tokens_idx, latency_idx):
            continue

        return _parse_data_row(cells, tokens_idx, latency_idx, vram_idx)

    return None, None, None


# ── Regex-based extractors ─────────────────────────────────────────────

_TOKENS_PATTERNS: list[str] = [
    r"tokens?\s+per\s+second[:\s]+(\d*\.?\d+)",
    r"t/s[:\s]+(\d*\.?\d+)",
    r"tokens?/s[:\s]+(\d*\.?\d+)",
    r"tok/s[:\s]+(\d*\.?\d+)",
]

_LATENCY_PATTERNS: list[str] = [
    r"avg\s+latency[:\s]+(\d*\.?\d+)\s*ms",
    r"latency[:\s]+(\d*\.?\d+)\s*ms",
    r"avg\s+latency[:\s]+(\d*\.?\d+)",
    r"latency[:\s]+(\d*\.?\d+)",
]

_VRAM_PATTERNS: list[str] = [
    r"peak\s+memory[:\s]+(\d*\.?\d+)\s*mb",
    r"peak\s+vram[:\s]+(\d*\.?\d+)\s*mb",
    r"vram[:\s]+(\d*\.?\d+)\s*mb",
    r"peak\s+memory[:\s]+(\d*\.?\d+)",
    r"memory[:\s]+(\d*\.?\d+)\s*mb",
    r"memory[:\s]+(\d*\.?\d+)",
]


def _extract_tokens_per_second(output: str) -> float | None:
    """Search all tokens-per-second patterns in output.

    Args:
        output: Raw stdout string from a benchmark subprocess.

    Returns:
        Parsed tokens/s value or ``None`` if not found.
    """
    for pattern in _TOKENS_PATTERNS:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def _extract_latency(output: str) -> float | None:
    """Search all latency patterns in output.

    Args:
        output: Raw stdout string from a benchmark subprocess.

    Returns:
        Parsed latency value or ``None`` if not found.
    """
    for pattern in _LATENCY_PATTERNS:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def _extract_vram(output: str) -> float | None:
    """Search all VRAM/memory patterns in output.

    Args:
        output: Raw stdout string from a benchmark subprocess.

    Returns:
        Parsed VRAM value or ``None`` if not found.
    """
    for pattern in _VRAM_PATTERNS:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


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

    (
        tokens_per_second,
        avg_latency_ms,
        peak_vram_mb,
    ) = _parse_markdown_table_metrics(output)

    if tokens_per_second is None:
        tokens_per_second = _extract_tokens_per_second(output)

    if avg_latency_ms is None:
        avg_latency_ms = _extract_latency(output)

    if peak_vram_mb is None:
        peak_vram_mb = _extract_vram(output)

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
