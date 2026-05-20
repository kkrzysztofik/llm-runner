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
import sys
import threading
import typing
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from llama_cli.commands._output import emit_json
from llama_cli.ui_output import emit_error, emit_plain
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
    get_gpu_identifier,
    resolve_backend_from_profile,
    run_benchmark,
    write_profile,
)

# Re-export manager-level functions for backward compatibility
from llama_manager.config.profile_cache import compute_driver_version_hash
from llama_manager.profile_orchestrator import (
    get_driver_version,
    resolve_benchmark_binary,
    resolve_benchmark_config,
    resolve_profile_slot,
)


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


def _check_slot_lockfile(slot_id: str, config: Config, _emit: Callable[[str], None]) -> None:
    """Warn if slot appears to be running (lockfile exists)."""
    from loguru import logger

    try:
        runtime_dir = config.profiles_dir.parent
        lock_path = runtime_dir / f"{slot_id}.lock"
        if lock_path.exists():
            _emit(
                f"warning: slot '{slot_id}' appears to be running (lockfile exists), proceeding anyway",
            )
    except OSError:
        logger.opt(exception=True).warning("Unable to inspect slot lockfile for {}", slot_id)


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

    # Resolve slot to ServerConfig using manager function
    server_config = resolve_profile_slot(slot_id, config)
    backend = _detect_backend(server_config)
    _emit(f"Detected backend: {backend}", stderr=True)

    # Resolve benchmark binary using manager function
    bench_bin = resolve_benchmark_binary(server_config, config)
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

    # Resolve flavor-based config using manager function
    flavor_obj = ProfileFlavor(flavor)
    bench_cfg = resolve_benchmark_config(server_config, flavor_obj, config)

    # Build benchmark command
    cmd = build_benchmark_cmd(
        bench_bin=bench_bin,
        model=bench_cfg.model,
        n_prompt=512,
        threads=bench_cfg.threads,
        ubatch_size=bench_cfg.ubatch_size,
        cache_type_k=bench_cfg.cache_type_k,
        cache_type_v=bench_cfg.cache_type_v,
        n_gpu_layers=bench_cfg.n_gpu_layers,
    )

    # Run benchmark
    if runner is not None:
        # Wrap user-provided runner to forward cancel_event
        def _wrapped_runner(cmd: list[str]) -> SubprocessResult:
            return runner(cmd)

        effective_runner: BenchmarkRunner = _wrapped_runner
    else:

        def _default_runner(cmd: list[str]) -> SubprocessResult:
            return _default_subprocess_runner(cmd, cancel_event=cancel_event)

        effective_runner = _default_runner

    benchmark_result = run_benchmark(cmd, effective_runner)

    # Handle benchmark result or cancellation
    if cancel_event is not None and cancel_event.is_set():
        _emit(f"Profile '{slot_id}' cancelled.", stderr=True)
        return 1
    if benchmark_result is None:
        _emit(f"error: benchmark failed for slot '{slot_id}'", stderr=True)
        return 1

    # _handle_benchmark_result returned 0, so benchmark_result is non-None
    benchmark_result = cast(BenchmarkResult, benchmark_result)

    # Create profile record, write, and emit results
    driver_version_hash = compute_driver_version_hash(driver_version)

    record = ProfileRecord(
        gpu_identifier=gpu_identifier,
        backend=backend,
        flavor=flavor_obj,
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
        _emit(f"  Flavor: {flavor}")
        _emit(f"  Tokens/s: {benchmark_result.tokens_per_second:.2f}")
        _emit(f"  Avg latency: {benchmark_result.avg_latency_ms:.2f} ms")
        _emit(f"  Profile saved to: {profile_path}")

    return 0


def _default_subprocess_runner(
    cmd: list[str],
    timeout_seconds: int = 600,
    cancel_event: threading.Event | None = None,
) -> SubprocessResult:
    """Execute *cmd* via ``subprocess.Popen`` with cancellation support.

    Checks *cancel_event* during execution and terminates the child process
    when set.

    Args:
        cmd: Command list to execute (shell=False).
        timeout_seconds: Subprocess timeout in seconds.
        cancel_event: Optional event; when set the child is terminated.

    Returns:
        A :class:`SubprocessResult` with exit code and captured output.
    """
    import subprocess

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            while True:
                ret = proc.poll()
                if ret is not None:
                    return SubprocessResult(
                        exit_code=ret,
                        stdout=proc.stdout.read() if proc.stdout else "",
                        stderr=proc.stderr.read() if proc.stderr else "",
                    )
                if cancel_event is not None and cancel_event.is_set():
                    import os
                    import signal

                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                    return SubprocessResult(exit_code=130, stdout="", stderr="benchmark cancelled")
                proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return SubprocessResult(
                exit_code=124,
                stdout="",
                stderr=f"benchmark timed out after {timeout_seconds}s",
            )
    except subprocess.TimeoutExpired as exc:
        timeout_stdout_text = _stream_to_text(exc.stdout)
        timeout_stderr_detail = _stream_to_text(exc.stderr)
        timeout_stderr = f"benchmark timed out after {timeout_seconds}s"
        if timeout_stderr_detail:
            timeout_stderr = f"{timeout_stderr}: {timeout_stderr_detail}"
        return SubprocessResult(
            exit_code=124,
            stdout=timeout_stdout_text,
            stderr=timeout_stderr,
        )


def _stream_to_text(data: bytes | str | None) -> str:
    """Convert bytes (or None) to a string.

    When *data* is already a ``str`` (e.g. from ``subprocess.run(text=True)``),
    returns it unchanged.  When *data* is ``bytes``, decodes it.

    Args:
        data: Bytes, string, or ``None``.

    Returns:
        Decoded string, or empty string.
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")


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
