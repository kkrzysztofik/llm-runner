"""Toolchain detection — find tools, parse versions, check availability."""

import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..build_pipeline import BuildBackend
from .constants import (
    CMAKE_HINT,
    CUDA_HINT,
    CUDA_REQUIRED_TOOLS,
    GCC_HINT,
    GIT_HINT,
    MAKE_HINT,
    SYCL_HINT,
    SYCL_REQUIRED_TOOLS,
    ToolchainHint,
)

if TYPE_CHECKING:
    from ..config import ErrorCode


# Fallback paths for Intel oneAPI compilers (default install location)
# Looked up lazily so tests can patch via llama_manager.toolchain._INTEL_ONEAPI_BIN
def _get_oneapi_bin() -> Path:
    """Get the Intel oneAPI bin path, allowing test overrides via package attribute."""
    toolchain_pkg = sys.modules.get("llama_manager.toolchain")
    if toolchain_pkg is not None and hasattr(toolchain_pkg, "_INTEL_ONEAPI_BIN"):
        return Path(toolchain_pkg._INTEL_ONEAPI_BIN)
    return Path("/opt/intel/oneapi/compiler/latest/bin")


# Look up detect_tool from the package so tests can patch via llama_manager.toolchain.detect_tool
def _get_detect_tool() -> Callable[[str, int | None], tuple[bool, str | None]]:
    """Get detect_tool from the package namespace for testability."""
    toolchain_pkg = sys.modules.get("llama_manager.toolchain")
    if toolchain_pkg is not None and hasattr(toolchain_pkg, "detect_tool"):
        return toolchain_pkg.detect_tool  # type: ignore[return-value]
    return detect_tool


def _extract_version(output: str, tool_name: str) -> str | None:
    """Extract version string from tool --version output.

    Returns ``None`` when *output* is empty or whitespace-only.
    """
    if tool_name == "nvcc":
        release_match = re.search(r"release\s+(\d+(?:\.\d+){0,2})", output)
        if release_match:
            return release_match.group(1)
    version_match = re.search(r"\d+(?:\.\d+){0,2}", output)
    if version_match:
        return version_match.group(0)
    # Return None for empty or whitespace-only output
    stripped = output.strip() if output else ""
    return None if not stripped else stripped.split("\n")[0]


def _try_tool(cmd: list[str], name: str, timeout: int) -> tuple[bool, str | None]:
    """Try to run a tool and extract its version."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            version = _extract_version(result.stdout.strip(), name)
            if version is not None:
                return (True, version)
    except subprocess.TimeoutExpired, OSError:
        pass
    return (False, None)


_INTEL_ONEAPI_TOOLS = frozenset({"icpx", "icx", "dpcpp"})

# (field_name, tool_name) for common build tools checked across all backends
_COMMON_MISSING_TOOLS: tuple[tuple[str, str], ...] = (
    ("gcc", "gcc"),
    ("make", "make"),
    ("git", "git"),
    ("cmake", "cmake"),
)


def detect_tool(
    tool_name: str,
    timeout: int | None = None,
) -> tuple[bool, str | None]:
    """Detect if a tool is available and return its version string.

    Uses subprocess.run with configurable timeout to check for tool presence.
    For Intel oneAPI tools (icpx, icx, dpcpp), falls back to the default
    installation path at /opt/intel/oneapi/compiler/latest/bin/.

    Args:
        tool_name: Name of the tool to detect (e.g., "gcc", "cmake", "nvcc")
        timeout: Timeout in seconds for the subprocess call. Defaults to
            Config.toolchain_timeout_seconds if available, otherwise 30s.

    Returns:
        Tuple of (found, version_string):
        - found: True if the tool is available, False otherwise
        - version_string: Version string if found (e.g., "11.3.0"), None otherwise

    Examples:
        >>> detect_tool("gcc")
        (True, "11.3.0")
        >>> detect_tool("nonexistent_tool")
        (False, None)
    """
    if timeout is None:
        timeout = 30

    found, version = _try_tool([tool_name, "--version"], tool_name, timeout)
    if found:
        return (True, version)

    if tool_name in _INTEL_ONEAPI_TOOLS:
        fallback = _get_oneapi_bin() / tool_name
        if fallback.exists():
            found, version = _try_tool([str(fallback), "--version"], tool_name, timeout)
            if found:
                return (True, version)

    return (False, None)


def get_toolchain_hints(backend: str) -> list["ToolchainErrorDetail"]:  # noqa: UP037
    """Get error details for missing toolchain tools for a specific backend.

    Args:
        backend: Backend type ("sycl" or "cuda")

    Returns:
        List of ToolchainErrorDetail for missing required tools.
        Only includes tools that are not found (not found tools are omitted).

    Raises:
        ValueError: If backend is not "sycl" or "cuda"
    """
    # Import here to avoid circular import
    from ..config import ErrorCode

    # Map backend to required tools and hints
    backend_map: dict[str, tuple[tuple[str, ...], dict[str, ToolchainHint]]] = {
        "sycl": (
            SYCL_REQUIRED_TOOLS,
            {
                "gcc": GCC_HINT,
                "make": MAKE_HINT,
                "git": GIT_HINT,
                "cmake": CMAKE_HINT,
                "dpcpp": SYCL_HINT,
                "icx": SYCL_HINT,
                "icpx": SYCL_HINT,
            },
        ),
        "cuda": (
            CUDA_REQUIRED_TOOLS,
            {
                "gcc": GCC_HINT,
                "make": MAKE_HINT,
                "git": GIT_HINT,
                "cmake": CMAKE_HINT,
                "nvcc": CUDA_HINT,
                "nvidia-smi": CUDA_HINT,
            },
        ),
    }

    if backend not in backend_map:
        raise ValueError(f"Unknown backend: {backend}. Must be 'sycl' or 'cuda'")

    required_tools, hints_map = backend_map[backend]
    missing: list[ToolchainErrorDetail] = []

    _detect_tool = _get_detect_tool()
    for tool in required_tools:
        found, _ = _detect_tool(tool, None)
        if not found:
            hint = hints_map.get(tool, ToolchainHint(tool, f"install {tool}"))
            missing.append(
                ToolchainErrorDetail(
                    error_code=ErrorCode.TOOLCHAIN_MISSING,
                    failed_check=tool,
                    why_blocked=f"Required for {backend} backend",
                    how_to_fix=hint.install_command,
                    docs_ref=hint.install_url,
                )
            )

    return missing


def parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    Handles versions like "3.20.1" or "3.20" and normalizes suffixes.

    Args:
        version_str: Version string to parse (e.g., "3.20.1", "3.20", "3.20.1ubuntu")

    Returns:
        Tuple of integers representing the version (e.g., (3, 20, 1) or (3, 20, 0))
    """
    # Normalize: strip suffixes like "ubuntu", "deb", etc.
    normalized = re.sub(
        r"[-_](?:ubuntu|deb|linux|linux-gnu).*", "", version_str, flags=re.IGNORECASE
    )
    # Extract only numeric parts and dots
    parts = re.findall(r"\d+", normalized)
    # Convert to tuple of integers, defaulting to 0 for missing components
    result: list[int] = [int(p) for p in parts]
    # Ensure at least 3 components
    while len(result) < 3:
        result.append(0)
    return tuple(result)


def version_at_least(version_str: str, min_version: str) -> bool:
    """Check if a version string is at least the minimum version.

    Args:
        version_str: Version string to check (e.g., "3.20.1")
        min_version: Minimum required version string (e.g., "3.14")

    Returns:
        True if version_str >= min_version, False otherwise
    """
    return parse_version(version_str) >= parse_version(min_version)


@dataclass
class ToolchainStatus:
    """Status of required build tools for a specific backend.

    This dataclass aggregates the availability and version information
    for all tools required to build llama.cpp for a particular backend
    (SYCL or CUDA).
    """

    gcc: str | None = None
    make: str | None = None
    git: str | None = None
    cmake: str | None = None
    sycl_compiler: str | None = None
    cuda_toolkit: str | None = None
    nvtop: str | None = None

    @property
    def is_sycl_ready(self) -> bool:
        """Check if all SYCL backend tools are available.

        Returns:
            True if gcc, make, git, cmake, and sycl_compiler are all available.
        """
        return all(
            [
                self.gcc is not None,
                self.make is not None,
                self.git is not None,
                self.cmake is not None,
                self.sycl_compiler is not None,
            ]
        )

    @property
    def is_cuda_ready(self) -> bool:
        """Check if all CUDA backend tools are available.

        Returns:
            True if gcc, make, git, cmake, and cuda_toolkit are all available.
        """
        return all(
            [
                self.gcc is not None,
                self.make is not None,
                self.git is not None,
                self.cmake is not None,
                self.cuda_toolkit is not None,
            ]
        )

    @property
    def is_complete(self) -> bool:
        """Check if all tools for both backends are available.

        Returns:
            True if both is_sycl_ready and is_cuda_ready are True.
        """
        return self.is_sycl_ready and self.is_cuda_ready

    def _collect_common_missing(self, missing: list[str]) -> None:
        """Append missing common build tools to *missing*.

        Checks gcc, make, git, and cmake — the tools shared by all backends.
        """
        for field, name in _COMMON_MISSING_TOOLS:
            if getattr(self, field) is None:
                missing.append(name)

    def missing_tools(self, backend: BuildBackend | None = None) -> list[str]:
        """Get list of missing tool names for the specified backend.

        Args:
            backend: Build backend to check tools for. If None, returns
                all missing tools (backward compatible behavior).

        Returns:
            List of tool names that are not available for the specified backend.
            If backend is None, returns all missing tools.
        """
        missing: list[str] = []

        # Always check common tools
        self._collect_common_missing(missing)

        # If no backend specified, return all missing tools (backward compatible).
        # nvtop is optional (not required for either backend) — do NOT add it
        # in the global None check; it is only surfaced for the CUDA backend.
        if backend is None:
            if self.sycl_compiler is None:
                missing.append("sycl_compiler")
            if self.cuda_toolkit is None:
                missing.append("cuda_toolkit")
            return missing

        # Check backend-specific tools
        if backend == BuildBackend.SYCL and self.sycl_compiler is None:
            missing.append("sycl_compiler")
        elif backend == BuildBackend.CUDA:
            if self.cuda_toolkit is None:
                missing.append("cuda_toolkit")
            if self.nvtop is None:
                missing.append("nvtop")

        return missing


def detect_toolchain() -> ToolchainStatus:
    """Detect all required build tools and return their status.

    Checks for the presence and versions of all tools required for
    both SYCL and CUDA backends.

    Returns:
        ToolchainStatus with version information for all detected tools.
    """
    _detect_tool = _get_detect_tool()
    status = ToolchainStatus()

    # Common tools
    _, version = _detect_tool("gcc", None)
    status.gcc = version
    _, version = _detect_tool("make", None)
    status.make = version
    _, version = _detect_tool("git", None)
    status.git = version
    _, version = _detect_tool("cmake", None)
    status.cmake = version

    # SYCL-specific tools — probe icpx first, then icx, then dpcpp;
    # set status.sycl_compiler to the version of the first found compiler.
    for _sycl_cand in ("icpx", "icx", "dpcpp"):
        _, version = _detect_tool(_sycl_cand, None)
        if version is not None:
            status.sycl_compiler = version
            break

    # CUDA-specific tools
    _, version = _detect_tool("nvcc", None)
    status.cuda_toolkit = version
    _, version = _detect_tool("nvtop", None)
    status.nvtop = version

    return status


@dataclass
class ToolchainErrorDetail:
    """FR-005 structured actionable error detail for toolchain issues.

    Extends the M1 ErrorDetail pattern with toolchain-specific fields
    to provide actionable error messages for missing or incompatible tools.

    Attributes:
        error_code: ErrorCode enum value for this error type
        failed_check: Name of the tool or check that failed
        why_blocked: Description of why the build is blocked
        how_to_fix: Actionable instructions to resolve the issue
        docs_ref: Optional documentation URL for further information
    """

    error_code: ErrorCode
    failed_check: str
    why_blocked: str
    how_to_fix: str
    docs_ref: str | None = None
