"""Profile orchestration — backend logic for GPU profiling.

This module provides the core profile orchestration functions that resolve
slot configurations, detect backends, select benchmark parameters, and
execute the full profiling pipeline. It is a pure library — no I/O except
``sys.stderr`` for errors.

All public functions return structured results and never print to the user.
The CLI layer owns user-facing messages, progress reporting, and output
formatting.
"""

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger

from .benchmark import BenchmarkResult, BenchmarkRunner, SubprocessResult, run_benchmark
from .config import (
    Config,
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    RunProfileRegistry,
    ServerConfig,
    create_default_profile_registry,
    resolve_profile_config,
    resolve_profile_id,
)
from .config.profile_cache import compute_driver_version_hash, write_profile
from .gpu_telemetry import get_gpu_identifier

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BENCHMARK_RUN_TIMEOUT_SECONDS: int = 600
BENCHMARK_PROMPT_TOKENS: int = 512


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DriverVersionProvider = Callable[[str], str]


# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Resolved benchmark parameters for a profiling run.

    Attributes:
        model: Path to the model file to benchmark.
        threads: Number of CPU threads.
        ubatch_size: Unified batch size.
        cache_type_k: KV-cache key type.
        cache_type_v: KV-cache value type.
        n_gpu_layers: Number of layers to offload to GPU (``"all"`` for CUDA).
    """

    model: str
    threads: int
    ubatch_size: int
    cache_type_k: str
    cache_type_v: str
    n_gpu_layers: int | str = 99


# ---------------------------------------------------------------------------
# Slot resolution
# ---------------------------------------------------------------------------


def resolve_profile_slot(
    slot_id: str,
    config: Config,
    registry: RunProfileRegistry | None = None,
) -> ServerConfig:
    """Resolve a slot_id to a ServerConfig.

    Uses the profile registry to resolve slot IDs and aliases to their
    corresponding profile definitions, then creates a ServerConfig from
    the resolved profile.

    Unknown slot IDs default to summary-balanced profile parameters.

    Args:
        slot_id: Slot identifier (e.g. ``'slot0'``, ``'summary-balanced'``, ``'qwen35'``).
        config: Global configuration with port and model defaults.
        registry: Optional pre-built profile registry. When omitted, a fresh
            registry is created via :func:`create_default_profile_registry`.

    Returns:
        ServerConfig for the resolved profile.
    """
    if registry is None:
        registry = create_default_profile_registry(config)

    if profile_id := resolve_profile_id(registry, slot_id):
        return resolve_profile_config(registry, profile_id)

    # Unknown slot IDs default to summary-balanced profile parameters.
    server_config = ServerConfig(
        model=config.model_summary_balanced,
        alias=slot_id,
        device="SYCL0",
        port=config.summary_balanced_port,
        ctx_size=config.default_ctx_size_summary,
        ubatch_size=config.default_ubatch_size_summary_balanced,
        threads=config.default_threads_summary_balanced,
        cache_type_k=config.default_cache_type_summary_k,
        cache_type_v=config.default_cache_type_summary_v,
        n_gpu_layers=config.default_n_gpu_layers,
    )
    return server_config


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def detect_backend(server_config: ServerConfig) -> str:
    """Detect backend from the slot-specific server configuration.

    Uses the profile's device field to determine backend:

    - Empty device → ``'cuda'`` (NVIDIA backend)
    - Non-empty device → ``'sycl'`` (Intel SYCL backend)

    Args:
        server_config: Slot-resolved server configuration.

    Returns:
        Backend string: ``'cuda'`` or ``'sycl'``.
    """
    return "cuda" if not server_config.device.strip() else "sycl"


# ---------------------------------------------------------------------------
# Benchmark config resolution
# ---------------------------------------------------------------------------


def resolve_benchmark_config(
    server_config: ServerConfig,
    flavor: ProfileFlavor,
    config: Config,
) -> BenchmarkConfig:
    """Resolve benchmark parameters based on flavor and profile.

    For CUDA profiles (empty device field), the server config values are
    used as-is. For SYCL profiles (non-empty device), the flavor overrides
    the defaults.

    Args:
        server_config: Slot-resolved server configuration.
        flavor: The selected profile flavor.
        config: Global configuration.

    Returns:
        :class:`BenchmarkConfig` with resolved parameters.
    """
    # CUDA profiles use their own config values
    if not server_config.device.strip():
        return BenchmarkConfig(
            model=server_config.model,
            threads=server_config.threads,
            ubatch_size=server_config.ubatch_size,
            cache_type_k=server_config.cache_type_k,
            cache_type_v=server_config.cache_type_v,
            n_gpu_layers=server_config.n_gpu_layers,
        )

    # SYCL profiles use flavor-based overrides
    if flavor == ProfileFlavor.BALANCED:
        return BenchmarkConfig(
            model=config.model_summary_balanced,
            threads=config.default_threads_summary_balanced,
            ubatch_size=config.default_ubatch_size_summary_balanced,
            cache_type_k=config.default_cache_type_summary_k,
            cache_type_v=config.default_cache_type_summary_v,
        )
    if flavor == ProfileFlavor.FAST:
        return BenchmarkConfig(
            model=config.model_summary_fast,
            threads=config.default_threads_summary_fast,
            ubatch_size=config.default_ubatch_size_summary_fast,
            cache_type_k=config.default_cache_type_summary_k,
            cache_type_v=config.default_cache_type_summary_v,
        )
    # quality — use balanced as base
    return BenchmarkConfig(
        model=config.model_summary_balanced,
        threads=config.default_threads_summary_balanced,
        ubatch_size=config.default_ubatch_size_summary_balanced,
        cache_type_k=config.default_cache_type_summary_k,
        cache_type_v=config.default_cache_type_summary_v,
    )


# ---------------------------------------------------------------------------
# Benchmark binary resolution
# ---------------------------------------------------------------------------


def resolve_benchmark_binary(server_config: ServerConfig, config: Config) -> str | None:
    """Resolve benchmark binary path, returning ``None`` if unavailable.

    Tries to derive the path from the server binary directory (swapping
    ``llama-server`` basename for ``llama-bench``). Falls back to
    ``shutil.which('llama-bench')``.

    Args:
        server_config: Slot-resolved server configuration.
        config: Global configuration with binary path defaults.

    Returns:
        Path to the ``llama-bench`` binary, or ``None`` if not found.
    """
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


# ---------------------------------------------------------------------------
# Driver version probing
# ---------------------------------------------------------------------------


def _query_nvidia_driver(timeout_seconds: int = 10) -> str | None:
    """Query nvidia-smi for the NVIDIA driver version.

    Args:
        timeout_seconds: Subprocess timeout in seconds.

    Returns:
        Driver version string, or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except OSError, subprocess.TimeoutExpired:
        pass
    return None


def _query_sycl_driver(timeout_seconds: int = 10) -> str | None:
    """Query sycl-ls for device/gpu info.

    Args:
        timeout_seconds: Subprocess timeout in seconds.

    Returns:
        First line mentioning ``gpu`` or ``device``, or ``None`` on failure.
    """
    try:
        result = subprocess.run(
            ["sycl-ls"],
            capture_output=True,
            text=True,
            shell=False,
            timeout=timeout_seconds,
            check=False,
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


# ---------------------------------------------------------------------------
# Profile record creation
# ---------------------------------------------------------------------------


def create_profile_record(
    gpu_identifier: str,
    backend: str,
    flavor: ProfileFlavor,
    driver_version: str,
    benchmark_result: BenchmarkResult,
    config: Config,
) -> ProfileRecord:
    """Create a :class:`ProfileRecord` from benchmark results.

    Args:
        gpu_identifier: GPU identifier string.
        backend: Backend string (``'cuda'`` or ``'sycl'``).
        flavor: Profile flavor enum.
        driver_version: Driver version string.
        benchmark_result: Benchmark result with metrics.
        config: Global configuration (for binary version).

    Returns:
        A new :class:`ProfileRecord` instance.
    """
    driver_version_hash = compute_driver_version_hash(driver_version)

    return ProfileRecord(
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


# ---------------------------------------------------------------------------
# Full profile orchestration
# ---------------------------------------------------------------------------


def run_profile(
    slot_id: str,
    config: Config,
    flavor: str,
    driver_provider: DriverVersionProvider | None = None,
    runner: BenchmarkRunner | None = None,
    registry: RunProfileRegistry | None = None,
) -> ProfileRecord | None:
    """Run a full profile: resolve config, detect backend, run benchmark, record result.

    This is the main orchestration function that ties together slot resolution,
    backend detection, benchmark config selection, driver probing, benchmark
    execution, and profile record creation/persistence.

    Args:
        slot_id: Slot identifier for the profiling target.
        config: Global configuration.
        flavor: Performance profile flavor (``'balanced'``, ``'fast'``, ``'quality'``).
        driver_provider: Optional injectable driver version provider. When
            ``None``, the default system probing is used.
        runner: Optional injectable benchmark runner. Uses the default
            subprocess runner when ``None``.
        registry: Optional pre-built profile registry.

    Returns:
        A :class:`ProfileRecord` on success, or ``None`` on failure.
    """
    # Resolve slot to ServerConfig
    server_config = resolve_profile_slot(slot_id, config, registry)
    backend = detect_backend(server_config)

    # Resolve benchmark binary
    bench_bin = resolve_benchmark_binary(server_config, config)
    if bench_bin is None:
        logger.warning("benchmark binary unavailable for slot '{}'", slot_id)
        return None

    # Validate benchmark binary is executable
    if not os.access(bench_bin, os.X_OK):
        logger.warning(
            "benchmark binary not executable: {}",
            bench_bin,
        )
        return None

    # Get GPU identifier
    gpu_identifier = get_gpu_identifier(backend)

    # Get driver version (use provider if given, otherwise probe)
    if driver_provider is not None:
        driver_version = driver_provider(backend)
    else:
        driver_version = get_driver_version(backend)

    # Resolve flavor-based benchmark config
    flavor_obj: ProfileFlavor | None = None
    try:
        flavor_obj = ProfileFlavor(flavor)
    except ValueError:
        logger.error("Unknown profile flavor: {}", flavor)
        return None

    if flavor_obj is not None:
        bench_cfg = resolve_benchmark_config(server_config, flavor_obj, config)
    else:
        return None

    # Build benchmark command
    from .benchmark import build_benchmark_cmd

    cmd = build_benchmark_cmd(
        bench_bin=bench_bin,
        model=bench_cfg.model,
        n_prompt=BENCHMARK_PROMPT_TOKENS,
        threads=bench_cfg.threads,
        ubatch_size=bench_cfg.ubatch_size,
        cache_type_k=bench_cfg.cache_type_k,
        cache_type_v=bench_cfg.cache_type_v,
        n_gpu_layers=bench_cfg.n_gpu_layers,
    )

    # Run benchmark
    effective_runner: BenchmarkRunner = runner if runner is not None else _default_subprocess_runner
    benchmark_result = run_benchmark(cmd, effective_runner)

    if benchmark_result is None:
        logger.warning("benchmark failed for slot '{}'", slot_id)
        return None

    # Create profile record
    record = create_profile_record(
        gpu_identifier=gpu_identifier,
        backend=backend,
        flavor=flavor_obj,
        driver_version=driver_version,
        benchmark_result=benchmark_result,
        config=config,
    )

    # Write profile to disk
    try:
        write_profile(config.profiles_dir, record)
    except OSError, ValueError:
        logger.opt(exception=True).error(
            "Failed to write profile record for slot '{}'",
            slot_id,
        )
        return None

    return record


# ---------------------------------------------------------------------------
# Default subprocess runner
# ---------------------------------------------------------------------------


def _default_subprocess_runner(
    cmd: list[str],
    timeout_seconds: int = BENCHMARK_RUN_TIMEOUT_SECONDS,
) -> SubprocessResult:
    """Execute *cmd* via ``subprocess.run`` and return the result.

    Args:
        cmd: Command list to execute (shell=False).
        timeout_seconds: Subprocess timeout in seconds.

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
            check=False,
        )
        return SubprocessResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
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
