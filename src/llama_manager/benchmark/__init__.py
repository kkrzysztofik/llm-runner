"""benchmark package — benchmark result types and runner protocol."""

from .parser import (
    _LATENCY_PATTERNS,
    _TOKENS_PATTERNS,
    _VRAM_PATTERNS,
    BenchmarkResult,
    _extract_first_float,
    _extract_latency,
    _extract_number_from_patterns,
    _extract_tokens_per_second,
    _extract_vram,
    _find_column_indices,
    _parse_data_row,
    _parse_markdown_table_metrics,
    _parse_table_block,
    _split_contiguous_blocks,
    parse_benchmark_output,
)
from .runner import (
    BenchmarkRunner,
    SubprocessResult,
    build_benchmark_cmd,
    run_benchmark,
)

__all__ = [
    "SubprocessResult",
    "BenchmarkResult",
    "BenchmarkRunner",
    "build_benchmark_cmd",
    "parse_benchmark_output",
    "run_benchmark",
    # Internal (exported for tests)
    "_extract_first_float",
    "_extract_number_from_patterns",
    "_TOKENS_PATTERNS",
    "_LATENCY_PATTERNS",
    "_VRAM_PATTERNS",
    "_parse_markdown_table_metrics",
    "_extract_tokens_per_second",
    "_extract_latency",
    "_extract_vram",
    "_find_column_indices",
    "_parse_data_row",
    "_split_contiguous_blocks",
    "_parse_table_block",
]
