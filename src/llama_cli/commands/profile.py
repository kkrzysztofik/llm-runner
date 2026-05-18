"""Profile CLI — benchmark GPU performance and cache the results.

This module provides the ``profile`` subcommand for llm-runner. It resolves
a llama-bench binary, runs a benchmark against the specified backend, and
writes a ``ProfileRecord`` to the profile cache directory.

Usage::

    llm-runner profile <slot_id> <flavor> [--json]

Flavors: balanced, fast, quality
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import typing
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from loguru import logger

from llama_cli.commands._output import emit_json
from llama_cli.commands._subprocess import run_capture_command, stream_to_text
from llama_cli.ui_output import emit_error, emit_plain
from llama_manager import (
    BenchmarkResult,
    BenchmarkRunner,
    Config,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    RunProfileRegistry,
    ServerConfig,
    SubprocessResult,
    build_benchmark_cmd,
    compute_driver_version_hash,
    create_default_profile_registry,
    create_summary_balanced_cfg,
    get_gpu_identifier,
    resolve_backend_from_profile,
    resolve_profile_config,
    resolve_profile_id,
    run_benchmark,
    write_profile,
)

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
        result = run_capture_command(cmd, timeout_seconds=timeout_seconds)
        return SubprocessResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_stdout_text = stream_to_text(exc.stdout)
        timeout_stderr_detail = stream_to_text(exc.stderr)
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

    Uses the profile's device field to determine backend:
    - Empty device → 'cuda' (NVIDIA backend)
    - Non-empty device → 'sycl' (Intel SYCL backend)

    Args:
        server_config: Slot-resolved server configuration.

    Returns:
        Backend string: 'cuda' or 'sycl'.
    """
    # Create a temporary profile spec from server_config for resolution
    from llama_manager.config.profiles import RunProfileSpec

    temp_profile = RunProfileSpec(
        profile_id=server_config.alias,
        model=server_config.model,
        alias=server_config.alias,
        device=server_config.device,
        port=server_config.port,
        ctx_size=server_config.ctx_size,
        ubatch_size=server_config.ubatch_size,
        threads=server_config.threads,
        backend=server_config.backend,
    )
    return resolve_backend_from_profile(temp_profile)


def _resolve_slot_server_config(
    slot_id: str,
    config: Config,
    registry: RunProfileRegistry | None = None,
) -> ServerConfig:
    """Resolve a slot_id to a ServerConfig using registry-based resolution.

    Uses the profile registry to resolve slot IDs and aliases to their
    corresponding profile definitions, then creates a ServerConfig from
    the resolved profile.

    Args:
        slot_id: Slot identifier (e.g. 'slot0', 'summary-balanced', 'qwen35').
        config: Global configuration with port and model defaults.
        registry: Optional pre-built profile registry. When omitted, a fresh
            registry is created via ``create_default_profile_registry(config)``.

    Returns:
        ServerConfig for the resolved profile.
    """
    if registry is None:
        registry = create_default_profile_registry(config)

    if profile_id := resolve_profile_id(registry, slot_id):
        return resolve_profile_config(registry, profile_id)

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
        result = run_capture_command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout_seconds=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except OSError, subprocess.TimeoutExpired:
        pass
    return None


def _query_sycl_driver() -> str | None:
    """Query sycl-ls for device/gpu info.

    Returns:
        First line mentioning ``gpu`` or ``device``, or ``None`` on failure.
    """
    try:
        result = run_capture_command(
            ["sycl-ls"],
            timeout_seconds=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.splitlines():
                if "gpu" in line.lower() or "device" in line.lower():
                    return line.strip()
    except OSError, subprocess.TimeoutExpired:
        pass
    return None


def get_driver_version(backend: str) -> str:
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
    """Resolve model, threads, and ubatch_size based on flavor and profile.

    For CUDA profiles (empty device field), the server config values are
    used as-is. For SYCL profiles (non-empty device), the flavor overrides
    the defaults.

    Args:
        server_config: Slot-resolved server configuration.
        flavor_obj: The selected profile flavor.
        config: Global configuration.

    Returns:
        Tuple of (model, threads, ubatch_size).
    """
    # CUDA profiles use their own config values
    if not server_config.device.strip():
        return server_config.model, server_config.threads, server_config.ubatch_size

    # SYCL profiles use flavor-based overrides
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
        emit_json(record.to_dict())
    else:
        _emit(f"Profile recorded for slot '{slot_id}'")
        _emit(f"  GPU: {gpu_identifier}")
        _emit(f"  Backend: {backend}")
        _emit(f"  Flavor: {flavor_str}")
        _emit(f"  Tokens/s: {benchmark_result.tokens_per_second:.2f}")
        _emit(f"  Avg latency: {benchmark_result.avg_latency_ms:.2f} ms")
        _emit(f"  Profile saved to: {profile_path}")

    return 0


def _check_slot_lockfile(slot_id: str, config: Config, _emit: Callable[[str], None]) -> None:
    """Warn if slot appears to be running (lockfile exists)."""
    try:
        runtime_dir = config.profiles_dir.parent
        lock_path = runtime_dir / f"{slot_id}.lock"
        if lock_path.exists():
            _emit(
                f"warning: slot '{slot_id}' appears to be running (lockfile exists), proceeding anyway",
            )
    except OSError:
        logger.opt(exception=True).warning("Unable to inspect slot lockfile for {}", slot_id)


def _resolve_bench_bin(server_config: ServerConfig, config: Config) -> str | None:
    """Resolve benchmark binary path, return None if unavailable."""
    import shutil

    server_bin = server_config.server_bin or config.llama_server_bin_intel
    if not server_bin:
        return shutil.which("llama-bench")
    # Safely swap basename only when it matches known server binary names
    base = os.path.basename(server_bin)
    if base in ("llama-server", "llama-server.exe", "llama-server-metal"):
        bench_path = os.path.join(os.path.dirname(server_bin), "llama-bench")
        if os.path.exists(bench_path):
            return bench_path
    return shutil.which("llama-bench")


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
            emit_plain(message, err=stderr)

    if cancel_event is not None and cancel_event.is_set():
        _emit(f"Profile '{slot_id}' cancelled.", stderr=True)
        return 1

    _check_slot_lockfile(slot_id, config, lambda msg: _emit(msg, stderr=True))

    server_config = _resolve_slot_server_config(slot_id, config)
    backend = _detect_backend(server_config)
    _emit(f"Detected backend: {backend}", stderr=True)

    bench_bin = _resolve_bench_bin(server_config, config)
    if bench_bin is None:
        _emit("error: benchmark binary unavailable", stderr=True)
        return 1

    try:
        require_executable(bench_bin)
    except (FileNotFoundError, PermissionError) as exc:
        _emit(f"error: benchmark binary unavailable: {exc}", stderr=True)
        return 1

    # Get GPU identifier
    gpu_identifier = get_gpu_identifier(backend)

    # Get driver version
    driver_version = get_driver_version(backend)

    # Resolve flavor-based config
    flavor_obj = ProfileFlavor(flavor)
    model, threads, ubatch_size = _resolve_benchmark_config(server_config, flavor_obj, config)

    # Determine n_gpu_layers based on backend
    n_gpu_layers = server_config.n_gpu_layers

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


class _ProfileArgumentParser(argparse.ArgumentParser):
    """ArgumentParser subclass that exits with code 1 on errors.

    The default argparse.ArgumentParser exits with code 2 on errors.
    This project convention requires exit code 1 for user-input validation
    failures.
    """

    def error(self, message: str) -> typing.NoReturn:
        """Override to exit with code 1 instead of 2."""
        self.print_usage(sys.stderr)
        emit_error(message)
        sys.exit(1)


def _build_profile_parser() -> _ProfileArgumentParser:
    """Build the argument parser for the profile subcommand."""
    parser = _ProfileArgumentParser(
        prog="profile",
        description="Benchmark GPU performance and cache the results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s slot0 balanced      Profile slot0 with balanced flavor
  %(prog)s summary-fast fast   Profile summary-fast with fast flavor
  %(prog)s qwen35-coding quality  Profile qwen35-coding with quality flavor
  %(prog)s slot0 balanced --json  Output results as JSON
        """,
    )

    parser.add_argument(
        "slot_id",
        help="Slot identifier (e.g., slot0, summary-balanced, qwen35-coding)",
    )
    parser.add_argument(
        "flavor",
        choices=["balanced", "fast", "quality"],
        help="Performance profile flavor",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    return parser


def _validate_slot_id(slot_id: str) -> str | None:
    """Validate slot_id and return error message or None if valid.

    Args:
        slot_id: The slot identifier to validate.

    Returns:
        Error message string if invalid, None if valid.
    """
    if not slot_id or not slot_id.strip():
        return "slot_id must not be empty"
    if ".." in slot_id:
        return "slot_id must not contain path traversal sequences"
    if "/" in slot_id or "\\" in slot_id:
        return "slot_id must not contain path separators"
    if not re.match(r"^[a-zA-Z0-9_-]+$", slot_id):
        return "slot_id must only contain ASCII letters, digits, hyphens, and underscores"
    return None


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

    parser = _build_profile_parser()
    parsed = parser.parse_args(argv)

    validation_error = _validate_slot_id(parsed.slot_id)
    if validation_error is not None:
        emit_error(validation_error)
        return 1

    return cmd_profile(parsed.slot_id, parsed.flavor, json_output=bool(parsed.json))
