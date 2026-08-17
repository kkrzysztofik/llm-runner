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
from dataclasses import dataclass

from .config import (
    Config,
    ProfileFlavor,
    ServerConfig,
    SlotProfileRegistry,
    create_default_profile_registry,
    resolve_profile_config,
    resolve_profile_id,
)

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
    registry: SlotProfileRegistry | None = None,
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
        model=config.deployment.model_summary_balanced,
        alias=slot_id,
        device="SYCL0",
        port=config.deployment.summary_balanced_port,
        ctx_size=config.server_defaults.ctx_size_summary,
        ubatch_size=config.server_defaults.ubatch_size_summary_balanced,
        threads=config.server_defaults.threads_summary_balanced,
        cache_type_k=config.server_defaults.cache_type_summary_k,
        cache_type_v=config.server_defaults.cache_type_summary_v,
        n_gpu_layers=config.server_defaults.n_gpu_layers,
    )
    return server_config


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
            model=config.deployment.model_summary_balanced,
            threads=config.server_defaults.threads_summary_balanced,
            ubatch_size=config.server_defaults.ubatch_size_summary_balanced,
            cache_type_k=config.server_defaults.cache_type_summary_k,
            cache_type_v=config.server_defaults.cache_type_summary_v,
        )
    if flavor == ProfileFlavor.FAST:
        return BenchmarkConfig(
            model=config.deployment.model_summary_fast,
            threads=config.server_defaults.threads_summary_fast,
            ubatch_size=config.server_defaults.ubatch_size_summary_fast,
            cache_type_k=config.server_defaults.cache_type_summary_k,
            cache_type_v=config.server_defaults.cache_type_summary_v,
        )
    # quality — use balanced as base
    return BenchmarkConfig(
        model=config.deployment.model_summary_balanced,
        threads=config.server_defaults.threads_summary_balanced,
        ubatch_size=config.server_defaults.ubatch_size_summary_balanced,
        cache_type_k=config.server_defaults.cache_type_summary_k,
        cache_type_v=config.server_defaults.cache_type_summary_v,
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
    server_bin = server_config.server_bin or config.paths.llama_server_bin_intel
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
