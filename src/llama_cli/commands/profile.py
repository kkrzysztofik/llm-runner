"""Profile CLI — benchmark GPU performance and cache the results.

This module provides the ``profile`` subcommand for llm-runner. It resolves
a llama-bench binary, runs a benchmark against the specified backend, and
writes a ``ProfileRecord`` to the profile cache directory.

Usage::

    llm-runner profile <slot_id> <flavor> [--json]

Flavors: balanced, fast, quality
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from llama_manager import (
    BenchmarkResult,
    BenchmarkRunner,
    Config,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    ServerConfig,
    SubprocessResult,
    build_benchmark_cmd,
    compute_driver_version_hash,
    create_qwen35_cfg,
    create_summary_balanced_cfg,
    create_summary_fast_cfg,
    get_gpu_identifier,
    normalize_slot_id,
    run_benchmark,
    write_profile,
)

LOGGER = logging.getLogger(__name__)
BENCHMARK_RUN_TIMEOUT_SECONDS = 600
BENCHMARK_PROMPT_TOKENS = 512


def _default_subprocess_runner(
    cmd: list[str], timeout_seconds: int = BENCHMARK_RUN_TIMEOUT_SECONDS
) -> SubprocessResult:
    """Execute *cmd* via ``subprocess.run`` and return the result.

    Args:
        cmd: Command list to execute (shell=False).

    Returns:
        A :class:`SubprocessResult` with exit code and captured output.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
        )
        return SubprocessResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_stdout = exc.stdout
        if isinstance(timeout_stdout, bytes):
            timeout_stdout_text = timeout_stdout.decode("utf-8", errors="replace")
        else:
            timeout_stdout_text = timeout_stdout or ""

        timeout_stderr_detail = exc.stderr
        if isinstance(timeout_stderr_detail, bytes):
            timeout_stderr_detail = timeout_stderr_detail.decode("utf-8", errors="replace")

        timeout_stderr = f"benchmark timed out after {timeout_seconds}s"
        if timeout_stderr_detail:
            timeout_stderr = f"{timeout_stderr}: {timeout_stderr_detail}"
        return SubprocessResult(
            exit_code=124,
            stdout=timeout_stdout_text,
            stderr=timeout_stderr,
        )


def require_executable(path: str) -> None:
    """Validate that *path* exists and is executable.

    Raises:
        FileNotFoundError: If the path does not exist.
        PermissionError: If the path exists but is not executable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"not executable: {path}")


def _detect_backend(server_config: ServerConfig) -> str:
    """Resolve backend from the slot-specific server configuration.

    Args:
        server_config: Slot-resolved server configuration.

    Returns:
        ``"cuda"`` for qwen35 slot config, otherwise ``"sycl"``.
    """
    if server_config.alias == "qwen35-coding":
        return "cuda"
    return "sycl"


def _resolve_slot_server_config(slot_id: str, config: Config) -> ServerConfig:
    try:
        normalized = normalize_slot_id(slot_id)
    except ValueError:
        normalized = ""
    if normalized in {"slot0", "summary-balanced", "summary_balanced", "balanced"}:
        return create_summary_balanced_cfg(config.summary_balanced_port)
    if normalized in {"slot1", "summary-fast", "summary_fast", "fast"}:
        return create_summary_fast_cfg(config.summary_fast_port)
    if normalized in {"slot2", "qwen35", "qwen35-coding", "qwen35_coding"}:
        return create_qwen35_cfg(config.qwen35_port)
    # Unknown slot IDs default to summary-balanced profile parameters.
    server_config = create_summary_balanced_cfg(config.summary_balanced_port)
    server_config.alias = slot_id
    return server_config


def _query_nvidia_driver() -> str | None:
    """Query nvidia-smi for the NVIDIA driver version.

    Returns:
        Driver version string, or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except OSError:
        pass
    return None


def _query_sycl_driver() -> str | None:
    """Query sycl-ls for device/gpu info.

    Returns:
        First line mentioning ``gpu`` or ``device``, or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["sycl-ls"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                if "gpu" in line.lower() or "device" in line.lower():
                    return line.strip()
    except OSError:
        pass
    return None


def _get_driver_version(backend: str) -> str:
    """Query the GPU driver version for the given backend.

    Args:
        backend: Either ``"cuda"`` or ``"sycl"``.

    Returns:
        Driver version string, or ``"unknown"`` on failure.
    """
    version = _query_nvidia_driver() if backend == "cuda" else _query_sycl_driver()
    return version if version is not None else "unknown"


def _resolve_benchmark_config(
    server_config: ServerConfig,
    flavor_obj: ProfileFlavor,
    config: Config,
) -> tuple[str, int, int]:
    """Resolve model, threads, and ubatch_size based on flavor.

    For the qwen35-coding alias, the server config values are used as-is.
    For summary slots, the flavor overrides the defaults.

    Args:
        server_config: Slot-resolved server configuration.
        flavor_obj: The selected profile flavor.
        config: Global configuration.

    Returns:
        Tuple of (model, threads, ubatch_size).
    """
    if server_config.alias == "qwen35-coding":
        return server_config.model, server_config.threads, server_config.ubatch_size

    if flavor_obj == ProfileFlavor.BALANCED:
        return (
            config.model_summary_balanced,
            config.default_threads_summary_balanced,
            config.default_ubatch_size_summary_balanced,
        )
    if flavor_obj == ProfileFlavor.FAST:
        return (
            config.model_summary_fast,
            config.default_threads_summary_fast,
            config.default_ubatch_size_summary_fast,
        )
    # quality — use balanced as base
    return (
        config.model_summary_balanced,
        config.default_threads_summary_balanced,
        config.default_ubatch_size_summary_balanced,
    )


def _build_benchmark_command(
    bench_bin: str,
    model: str,
    threads: int,
    ubatch_size: int,
    cache_type_k: str,
    cache_type_v: str,
    n_gpu_layers: int | str,
) -> list[str]:
    """Build the llama-benchmark command list.

    Args:
        bench_bin: Path to the llama-bench binary.
        model: Model path to benchmark.
        threads: Number of threads.
        ubatch_size: Unified batch size.
        cache_type_k: KV cache type for K.
        cache_type_v: KV cache type for V.
        n_gpu_layers: Number of layers to offload to GPU.

    Returns:
        Command list suitable for ``subprocess.run``.
    """
    return build_benchmark_cmd(
        bench_bin=bench_bin,
        model=model,
        n_prompt=BENCHMARK_PROMPT_TOKENS,
        threads=threads,
        ubatch_size=ubatch_size,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        n_gpu_layers=n_gpu_layers,
    )


def _handle_benchmark_result(
    benchmark_result: BenchmarkResult | None,
    slot_id: str,
    cancel_event: threading.Event | None,
    _emit: Callable[..., None],
) -> int:
    """Check benchmark result and cancel event; return exit code.

    Args:
        benchmark_result: The benchmark result or ``None`` on failure.
        slot_id: Slot identifier for error messages.
        cancel_event: Optional cancellation event.
        _emit: Message emitter callable (always called with stderr=True).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    if cancel_event is not None and cancel_event.is_set():
        _emit(f"Profile '{slot_id}' cancelled.", stderr=True)
        return 1
    if benchmark_result is None:
        _emit(f"error: benchmark failed for slot '{slot_id}'", stderr=True)
        return 1
    return 0


def _create_and_save_profile(
    config: Config,
    backend: str,
    flavor: ProfileFlavor,
    flavor_str: str,
    driver_version: str,
    benchmark_result: BenchmarkResult,
    slot_id: str,
    gpu_identifier: str,
    json_output: bool,
    cancel_event: threading.Event | None,
    _emit: Callable[..., None],
) -> int:
    """Create a ProfileRecord, write it, and emit results.

    Args:
        config: Global configuration.
        backend: Backend string (cuda|sycl).
        flavor: Profile flavor enum (stored in record).
        flavor_str: Flavor string for human-readable output.
        driver_version: Driver version string.
        benchmark_result: Benchmark result with metrics.
        slot_id: Slot identifier for output messages.
        gpu_identifier: GPU identifier string.
        json_output: Whether to emit JSON output.
        cancel_event: Optional cancellation event.
        _emit: Message emitter callable (always called with stderr=True for errors).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    driver_version_hash = compute_driver_version_hash(driver_version)

    record = ProfileRecord(
        gpu_identifier=gpu_identifier,
        backend=backend,
        flavor=flavor,
        driver_version=driver_version,
        driver_version_hash=driver_version_hash,
        server_binary_version=config.server_binary_version,
        profiled_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        metrics=ProfileMetrics(
            tokens_per_second=benchmark_result.tokens_per_second,
            avg_latency_ms=benchmark_result.avg_latency_ms,
            peak_vram_mb=benchmark_result.peak_vram_mb,
        ),
        parameters={},
    )

    if cancel_event is not None and cancel_event.is_set():
        _emit(f"Profile '{slot_id}' cancelled.", stderr=True)
        return 1

    profile_path = write_profile(config.profiles_dir, record)

    if json_output:
        _emit(json.dumps(record.to_dict(), indent=2))
    else:
        _emit(f"Profile recorded for slot '{slot_id}'")
        _emit(f"  GPU: {gpu_identifier}")
        _emit(f"  Backend: {backend}")
        _emit(f"  Flavor: {flavor_str}")
        _emit(f"  Tokens/s: {benchmark_result.tokens_per_second:.2f}")
        _emit(f"  Avg latency: {benchmark_result.avg_latency_ms:.2f} ms")
        _emit(f"  Profile saved to: {profile_path}")

    return 0


def cmd_profile(
    slot_id: str,
    flavor: str,
    json_output: bool = False,
    runner: BenchmarkRunner | None = None,
    quiet: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    """Run a GPU benchmark and persist the profile record.

    Args:
        slot_id: Identifier for the profiling target (used in messages).
        flavor: Performance profile flavor (balanced|fast|quality).
        json_output: If True, print JSON instead of a human-readable message.
        runner: Optional injectable benchmark runner. Uses the default
                subprocess runner when ``None``.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config = Config()

    def _emit(message: str, *, stderr: bool = False) -> None:
        if progress_callback is not None:
            progress_callback(message)
        if not quiet:
            print(message, file=sys.stderr if stderr else sys.stdout)

    if cancel_event is not None and cancel_event.is_set():
        _emit(f"Profile '{slot_id}' cancelled.", stderr=True)
        return 1

    # Check if slot is already running (skip if we can't determine)
    try:
        runtime_dir = config.profiles_dir.parent
        lock_path = runtime_dir / f"{slot_id}.lock"
        if lock_path.exists():
            _emit(
                f"warning: slot '{slot_id}' appears to be running (lockfile exists), proceeding anyway",
                stderr=True,
            )
    except OSError as exc:
        LOGGER.warning("Unable to inspect slot lockfile for %s", slot_id, exc_info=exc)

    server_config = _resolve_slot_server_config(slot_id, config)
    backend = _detect_backend(server_config)
    _emit(f"Detected backend: {backend}", stderr=True)

    # Resolve benchmark binary path
    server_bin = server_config.server_bin or config.llama_server_bin_intel
    bench_bin = server_bin.replace("llama-server", "llama-bench") if server_bin else "llama-bench"

    try:
        require_executable(bench_bin)
    except (FileNotFoundError, PermissionError) as exc:
        _emit(f"error: benchmark binary unavailable: {exc}", stderr=True)
        return 1

    # Get GPU identifier
    gpu_identifier = get_gpu_identifier(backend)

    # Get driver version
    driver_version = _get_driver_version(backend)

    # Resolve flavor-based config
    flavor_obj = ProfileFlavor(flavor)
    model, threads, ubatch_size = _resolve_benchmark_config(server_config, flavor_obj, config)

    # Determine n_gpu_layers based on backend
    n_gpu_layers = (
        config.default_n_gpu_layers_qwen35 if backend == "cuda" else server_config.n_gpu_layers
    )

    # Build benchmark command
    cmd = _build_benchmark_command(
        bench_bin=bench_bin,
        model=model,
        threads=threads,
        ubatch_size=ubatch_size,
        cache_type_k=server_config.cache_type_k,
        cache_type_v=server_config.cache_type_v,
        n_gpu_layers=n_gpu_layers,
    )

    # Run benchmark
    effective_runner: BenchmarkRunner = runner if runner is not None else _default_subprocess_runner
    benchmark_result = run_benchmark(cmd, effective_runner)

    # Handle benchmark result or cancellation
    if exit_code := _handle_benchmark_result(benchmark_result, slot_id, cancel_event, _emit):
        return exit_code

    # _handle_benchmark_result returned 0, so benchmark_result is non-None
    benchmark_result = cast(BenchmarkResult, benchmark_result)

    # Create profile record, write, and emit results
    return _create_and_save_profile(
        config=config,
        backend=backend,
        flavor=flavor_obj,
        flavor_str=flavor,
        driver_version=driver_version,
        benchmark_result=benchmark_result,
        slot_id=slot_id,
        gpu_identifier=gpu_identifier,
        json_output=json_output,
        cancel_event=cancel_event,
        _emit=_emit,
    )


def main(args: list[str] | None = None) -> int:
    """CLI entry point for the profile subcommand.

    Parses ``<slot_id> <flavor> [--json]`` from *args* and invokes
    :func:`cmd_profile`.

    Args:
        args: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    argv = args if args is not None else sys.argv[1:]

    if len(argv) < 2:
        print(
            "error: profile requires a slot_id and a flavor (balanced|fast|quality)",
            file=sys.stderr,
        )
        return 1

    slot_id = argv[0]
    flavor_str = argv[1]
    remaining = argv[2:]

    # Validate slot_id
    if not slot_id or not slot_id.strip():
        print("error: slot_id must not be empty", file=sys.stderr)
        return 1

    if ".." in slot_id:
        print("error: slot_id must not contain path traversal sequences", file=sys.stderr)
        return 1

    # Validate flavor
    try:
        flavor = ProfileFlavor(flavor_str)
    except ValueError:
        print(
            f"error: invalid flavor '{flavor_str}'. Valid flavors: balanced, fast, quality",
            file=sys.stderr,
        )
        return 1

    # Parse remaining flags
    json_output = "--json" in remaining

    return cmd_profile(slot_id, flavor.value, json_output=json_output)
