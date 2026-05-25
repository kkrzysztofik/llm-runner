#!/usr/bin/env python3
"""Benchmark GGUF metadata extraction methods.

Usage:
    python scripts/bench_gguf_parse.py <path_to_gguf_file> [path2 ...]

Measures:
  - extract_gguf_metadata()   — daemon thread in main process
  - _extract_model_index_metadata() — subprocess worker
"""

import sys
import time
from pathlib import Path

from llama_manager.metadata import extract_gguf_metadata
from llama_manager.model_index import _extract_model_index_metadata


def bench(path: str, n: int = 3) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {Path(path).name}  ({Path(path).stat().st_size / 1e6:.1f} MB)")
    print(f"{'=' * 60}")

    # Thread-based
    times = []
    for _i in range(n):
        t0 = time.perf_counter()
        extract_gguf_metadata(path, parse_timeout_s=3600.0)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    avg_thread = sum(times) / len(times)
    print(
        f"  extract_gguf_metadata (thread):  {avg_thread:.3f}s  (min={min(times):.3f}s  max={max(times):.3f}s)"
    )

    # Subprocess-based
    times = []
    for _i in range(n):
        t0 = time.perf_counter()
        _extract_model_index_metadata(
            path, prefix_cap_bytes=32 * 1024 * 1024, parse_timeout_s=3600.0
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    avg_proc = sum(times) / len(times)
    print(
        f"  _extract_model_index_metadata:    {avg_proc:.3f}s"
        f"  (min={min(times):.3f}s  max={max(times):.3f}s)"
    )

    speedup = avg_thread / avg_proc if avg_proc > 0 else float("inf")
    print(f"  Speedup (thread vs subprocess):   {speedup:.2f}x")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_gguf> [path2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not Path(path).is_file():
            print(f"  WARNING: {path} not found, skipping")
            continue
        try:
            bench(path)
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()
