"""Backward compatibility shim. Import from llama_manager.benchmark submodules instead."""

from .benchmark import *  # noqa: F401, F403
from .benchmark.parser import (  # noqa: F401
    parse_benchmark_output,
)
from .benchmark.runner import (  # noqa: F401
    BenchmarkResult,
    BenchmarkRunner,
    SubprocessResult,
    build_benchmark_cmd,
    run_benchmark,
)
