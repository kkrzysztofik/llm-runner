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
from datetime import UTC, datetime

from llama_manager import (
    BenchmarkRunner,
    Config,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    SubprocessResult,
    build_benchmark_cmd,
    compute_driver_version_hash,
    get_gpu_identifier,
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
        timeout_stderr = f"benchmark timed out after {timeout_seconds}s"
        if exc.stderr:
            timeout_stderr = f"{timeout_stderr}: {exc.stderr}"
        return SubprocessResult(
            exit_code=124,
            stdout=exc.stdout or "",
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


def _detect_backend(config: Config) -> str:
    """Auto-detect the GPU backend by checking which llama-server binary exists.

    Args:
        config: The application configuration.

    Returns:
        ``"cuda"`` if the NVIDIA binary exists, otherwise ``"sycl"``.
    """
    if config.llama_server_bin_nvidia and os.path.exists(config.llama_server_bin_nvidia):
        return "cuda"
    return "sycl"


def _get_driver_version(backend: str) -> str:
    """Query the GPU driver version for the given backend.

    Args:
        backend: Either ``"cuda"`` or ``"sycl"``.

    Returns:
        Driver version string, or ``"unknown"`` on failure.
    """
    try:
        if backend == "cuda":
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                shell=False,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0].strip()
        elif backend == "sycl":
            result = subprocess.run(
                ["sycl-ls"],
                capture_output=True,
                text=True,
                shell=False,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                # sycl-ls outputs device selectors; extract version info
                for line in result.stdout.splitlines():
                    if "gpu" in line.lower() or "device" in line.lower():
                        return line.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


def cmd_profile(
    slot_id: str,
    flavor: str,
    json_output: bool = False,
    runner: BenchmarkRunner | None = None,
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

    # Check if slot is already running (skip if we can't determine)
    try:
        runtime_dir = config.profiles_dir.parent
        lock_path = runtime_dir / f"{slot_id}.lock"
        if lock_path.exists():
            print(
                f"warning: slot '{slot_id}' appears to be running (lockfile exists), "
                "proceeding anyway",
                file=sys.stderr,
            )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        LOGGER.warning("Unable to inspect slot lockfile for %s", slot_id, exc_info=exc)

    # Auto-detect backend
    backend = _detect_backend(config)
    print(f"Detected backend: {backend}", file=sys.stderr)

    # Resolve benchmark binary path
    server_bin = config.llama_server_bin_intel
    bench_bin = server_bin.replace("llama-server", "llama-bench") if server_bin else "llama-bench"

    try:
        require_executable(bench_bin)
    except (FileNotFoundError, PermissionError) as exc:
        print(f"error: benchmark binary unavailable: {exc}", file=sys.stderr)
        return 1

    # Get GPU identifier
    gpu_identifier = get_gpu_identifier(backend)

    # Get driver version
    driver_version = _get_driver_version(backend)

    # Compute driver version hash
    driver_version_hash = compute_driver_version_hash(driver_version)

    # Flavor-based config selection
    flavor_obj = ProfileFlavor(flavor)
    if flavor_obj == ProfileFlavor.BALANCED:
        model = config.model_summary_balanced
        threads = config.default_threads_summary_balanced
        ubatch_size = config.default_ubatch_size_summary_balanced
    elif flavor_obj == ProfileFlavor.FAST:
        model = config.model_summary_fast
        threads = config.default_threads_summary_fast
        ubatch_size = config.default_ubatch_size_summary_fast
    else:  # quality — use balanced as base
        model = config.model_summary_balanced
        threads = config.default_threads_summary_balanced
        ubatch_size = config.default_ubatch_size_summary_balanced

    # Build benchmark command
    cmd = build_benchmark_cmd(
        bench_bin=bench_bin,
        model=model,
        n_prompt=BENCHMARK_PROMPT_TOKENS,
        threads=threads,
        ctx_size=config.default_ctx_size_summary,
        ubatch_size=ubatch_size,
        cache_type_k=config.default_cache_type_summary_k,
        cache_type_v=config.default_cache_type_summary_v,
        n_gpu_layers=config.default_n_gpu_layers_qwen35
        if backend == "cuda"
        else config.default_n_gpu_layers,
    )

    # Run benchmark
    effective_runner: BenchmarkRunner = runner if runner is not None else _default_subprocess_runner
    benchmark_result = run_benchmark(cmd, effective_runner)

    if benchmark_result is None:
        print(
            f"error: benchmark failed for slot '{slot_id}'",
            file=sys.stderr,
        )
        return 1

    # Create profile record
    record = ProfileRecord(
        gpu_identifier=gpu_identifier,
        backend=backend,
        flavor=ProfileFlavor(flavor),
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

    # Write profile
    profile_path = write_profile(config.profiles_dir, record)

    # Output result
    if json_output:
        print(json.dumps(record.to_dict(), indent=2))
    else:
        print(f"Profile recorded for slot '{slot_id}'")
        print(f"  GPU: {gpu_identifier}")
        print(f"  Backend: {backend}")
        print(f"  Flavor: {flavor}")
        print(f"  Tokens/s: {benchmark_result.tokens_per_second:.2f}")
        print(f"  Avg latency: {benchmark_result.avg_latency_ms:.2f} ms")
        print(f"  Profile saved to: {profile_path}")

    return 0


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
