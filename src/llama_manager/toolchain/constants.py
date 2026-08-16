"""Toolchain constants — required tools, hints, and version constraints."""

from dataclasses import dataclass

# Module constants for required tools by backend
# Using tuple instead of frozenset for deterministic iteration order (fixes test flakiness)
SYCL_REQUIRED_TOOLS: tuple[str, ...] = ("gcc", "make", "git", "cmake", "dpcpp", "icx", "icpx")

CUDA_REQUIRED_TOOLS: tuple[str, ...] = ("gcc", "make", "git", "cmake", "nvcc", "nvidia-smi")


@dataclass
class ToolchainHint:
    """Provides installation guidance for missing build tools.

    This dataclass contains actionable information to help users
    install missing tools, including package manager commands and
    download URLs.
    """

    tool_name: str
    install_command: str
    install_url: str | None = None


# Predefined toolchain hints for common tools
# These are class-level constants for reuse across the codebase

GCC_HINT: ToolchainHint = ToolchainHint(
    tool_name="gcc",
    install_command="sudo apt-get install gcc",
    install_url="https://gcc.gnu.org/download.html",
)

MAKE_HINT: ToolchainHint = ToolchainHint(
    tool_name="make",
    install_command="sudo apt-get install make",
    install_url="https://www.gnu.org/software/make/",
)

GIT_HINT: ToolchainHint = ToolchainHint(
    tool_name="git",
    install_command="sudo apt-get install git",
    install_url="https://git-scm.com/",
)

CMAKE_HINT: ToolchainHint = ToolchainHint(
    tool_name="cmake",
    install_command="sudo apt-get install cmake",
    install_url="https://cmake.org/download/",
)

SYCL_HINT: ToolchainHint = ToolchainHint(
    tool_name="Intel oneAPI DPC++ Compiler",
    install_command="# Download the Intel install script, inspect it, then run:\ncurl -sSf https://apt.repos.intel.com/install.sh -o /tmp/intel-install.sh\n# Inspect the script before execution\ncat /tmp/intel-install.sh | less\n# Then run with sudo:\nsudo sh /tmp/intel-install.sh\nsudo apt-get install oneapi-compiler-dpcpp-cpp oneapi-compiler-dpcpp-rt",
    install_url="https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html",
)

CUDA_HINT: ToolchainHint = ToolchainHint(
    tool_name="NVIDIA CUDA Toolkit",
    install_command="sudo apt-get install cuda-toolkit-12-2",
    install_url="https://developer.nvidia.com/cuda-toolkit",
)

NVTOP_HINT: ToolchainHint = ToolchainHint(
    tool_name="nvtop",
    install_command="sudo apt-get install nvtop",
    install_url="https://github.com/Syllo/nvtop",
)
