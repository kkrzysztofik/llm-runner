"""Backward compatibility shim. Import from llama_manager.toolchain submodules instead."""

from .toolchain import *  # noqa: F401, F403
from .toolchain.constants import (  # noqa: F401
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
from .toolchain.detector import (  # noqa: F401
    ToolchainErrorDetail,
    ToolchainStatus,
    detect_tool,
    detect_toolchain,
    get_toolchain_hints,
    parse_version,
    version_at_least,
)
