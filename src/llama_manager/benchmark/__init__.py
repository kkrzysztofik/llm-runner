"""benchmark package — benchmark result types and runner protocol."""

from .parser import (
    BenchmarkResult,
    parse_benchmark_output,
)
from .runner import (
    BenchmarkRunner,
    build_benchmark_cmd,
    run_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
    "build_benchmark_cmd",
    "parse_benchmark_output",
    "run_benchmark",
]
