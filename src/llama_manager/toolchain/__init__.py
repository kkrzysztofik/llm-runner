"""toolchain package — build toolchain detection and status checking."""

from .constants import (
    CMAKE_HINT,
    CMAKE_MINIMUM_VERSION,
    CUDA_HINT,
    CUDA_REQUIRED_TOOLS,
    GCC_HINT,
    GIT_HINT,
    MAKE_HINT,
    NVTOP_HINT,
    SYCL_HINT,
    SYCL_REQUIRED_TOOLS,
    ToolchainHint,
)
from .detector import (
    ToolchainErrorDetail,
    ToolchainStatus,
    detect_tool,
    detect_toolchain,
    get_toolchain_hints,
    parse_version,
    version_at_least,
)

# Fallback path for Intel oneAPI compilers — exposed for test patching
_INTEL_ONEAPI_BIN: str = "/opt/intel/oneapi/compiler/latest/bin"

__all__ = [
    # Status
    "ToolchainStatus",
    "ToolchainErrorDetail",
    # Hints
    "ToolchainHint",
    "GCC_HINT",
    "MAKE_HINT",
    "GIT_HINT",
    "CMAKE_HINT",
    "SYCL_HINT",
    "CUDA_HINT",
    "NVTOP_HINT",
    # Required tools
    "SYCL_REQUIRED_TOOLS",
    "CUDA_REQUIRED_TOOLS",
    "CMAKE_MINIMUM_VERSION",
    # Detection
    "detect_tool",
    "detect_toolchain",
    "get_toolchain_hints",
    # Version parsing
    "parse_version",
    "version_at_least",
]
