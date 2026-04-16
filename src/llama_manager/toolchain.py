# Toolchain detection and status checking for M2 build setup

import re
from dataclasses import dataclass, field

from .build_pipeline import BuildBackend
from .config import ErrorCode

# Module constants for required tools by backend
SYCL_REQUIRED_TOOLS: list[str] = [
    "sycl-ls",
    "icpx",
    "syclpp",
]

CUDA_REQUIRED_TOOLS: list[str] = [
    "nvcc",
    "nvidia-smi",
    "nvtop",
]

CMAKE_MINIMUM_VERSION: str = "3.14"


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
            True if gcc, make, git, cmake, cuda_toolkit, and nvtop are all available.
        """
        return all(
            [
                self.gcc is not None,
                self.make is not None,
                self.git is not None,
                self.cmake is not None,
                self.cuda_toolkit is not None,
                self.nvtop is not None,
            ]
        )

    @property
    def is_complete(self) -> bool:
        """Check if all tools for both backends are available.

        Returns:
            True if both is_sycl_ready and is_cuda_ready are True.
        """
        return self.is_sycl_ready and self.is_cuda_ready

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
        if self.gcc is None:
            missing.append("gcc")
        if self.make is None:
            missing.append("make")
        if self.git is None:
            missing.append("git")
        if self.cmake is None:
            missing.append("cmake")

        # If no backend specified, return all missing tools (backward compatible)
        if backend is None:
            if self.sycl_compiler is None:
                missing.append("sycl_compiler")
            if self.cuda_toolkit is None:
                missing.append("cuda_toolkit")
            if self.nvtop is None:
                missing.append("nvtop")
            return missing

        # Check backend-specific tools
        if backend == BuildBackend.SYCL:
            if self.sycl_compiler is None:
                missing.append("sycl_compiler")
        elif backend == BuildBackend.CUDA:
            if self.cuda_toolkit is None:
                missing.append("cuda_toolkit")
            if self.nvtop is None:
                missing.append("nvtop")

        return missing


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
    required_for: list[str] = field(default_factory=list)

    @property
    def is_url_available(self) -> bool:
        """Check if an installation URL is provided."""
        return self.install_url is not None

    def format_hint(self) -> str:
        """Format a human-readable installation hint.

        Returns:
            A formatted string with installation instructions.
        """
        hint = f"Install {self.tool_name}: {self.install_command}"
        if self.required_for:
            backends = ", ".join(self.required_for)
            hint += f" (required for: {backends})"
        if self.is_url_available:
            hint += f"\n  Download: {self.install_url}"
        return hint


# Predefined toolchain hints for common tools
# These are class-level constants for reuse across the codebase

GCC_HINT: ToolchainHint = ToolchainHint(
    tool_name="gcc",
    install_command="sudo apt-get install gcc",
    install_url="https://gcc.gnu.org/download.html",
    required_for=["sycl", "cuda"],
)

MAKE_HINT: ToolchainHint = ToolchainHint(
    tool_name="make",
    install_command="sudo apt-get install make",
    install_url="https://www.gnu.org/software/make/",
    required_for=["sycl", "cuda"],
)

GIT_HINT: ToolchainHint = ToolchainHint(
    tool_name="git",
    install_command="sudo apt-get install git",
    install_url="https://git-scm.com/",
    required_for=["sycl", "cuda"],
)

CMAKE_HINT: ToolchainHint = ToolchainHint(
    tool_name="cmake",
    install_command="sudo apt-get install cmake",
    install_url="https://cmake.org/download/",
    required_for=["sycl", "cuda"],
)

SYCL_HINT: ToolchainHint = ToolchainHint(
    tool_name="Intel oneAPI DPC++ Compiler",
    install_command="curl -sSf https://apt.repos.intel.com/install.sh | sudo sh && sudo apt-get install oneapi-compiler-dpcpp-cpp oneapi-compiler-dpcpp-rt",
    install_url="https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html",
    required_for=["sycl"],
)

CUDA_HINT: ToolchainHint = ToolchainHint(
    tool_name="NVIDIA CUDA Toolkit",
    install_command="sudo apt-get install cuda-toolkit-12-2",
    install_url="https://developer.nvidia.com/cuda-toolkit",
    required_for=["cuda"],
)

NVTOP_HINT: ToolchainHint = ToolchainHint(
    tool_name="nvtop",
    install_command="sudo apt-get install nvtop",
    install_url="https://github.com/Syllo/nvtop",
    required_for=["cuda"],
)


def detect_tool(
    tool_name: str,
    timeout: int | None = None,
) -> tuple[bool, str | None]:
    """Detect if a tool is available and return its version string.

    Uses subprocess.run with configurable timeout to check for tool presence.

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
    import subprocess

    if timeout is None:
        timeout = 30  # Default timeout

    try:
        result = subprocess.run(
            [tool_name, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            # Parse version from output (e.g., "gcc (GCC) 11.3.0" → "11.3.0")
            output = result.stdout.strip()
            # Try to extract version number using regex (accepts 1-3 components: "12", "12.3", "12.3.4")
            version_match = re.search(r"\d+(?:\.\d+){0,2}", output)
            if version_match:
                version = version_match.group(0)
                return (True, version)
            return (True, output.split("\n")[0])
        return (False, None)
    except subprocess.TimeoutExpired:
        return (False, None)
    except FileNotFoundError:
        return (False, None)
    except Exception:
        return (False, None)


def get_toolchain_hints(backend: str) -> list["ToolchainErrorDetail"]:
    """Get error details for missing toolchain tools for a specific backend.

    Args:
        backend: Backend type ("sycl" or "cuda")

    Returns:
        List of ToolchainErrorDetail for missing required tools.
        Only includes tools that are not found (not found tools are omitted).

    Raises:
        ValueError: If backend is not "sycl" or "cuda"
    """
    # Map backend to required tools and hints
    backend_map: dict[str, tuple[list[str], dict[str, ToolchainHint]]] = {
        "sycl": (
            SYCL_REQUIRED_TOOLS,
            {
                "sycl-ls": SYCL_HINT,
                "icpx": SYCL_HINT,
                "syclpp": SYCL_HINT,
            },
        ),
        "cuda": (
            CUDA_REQUIRED_TOOLS,
            {
                "nvcc": CUDA_HINT,
                "nvidia-smi": CUDA_HINT,
                "nvtop": NVTOP_HINT,
            },
        ),
    }

    if backend not in backend_map:
        raise ValueError(f"Unknown backend: {backend}. Must be 'sycl' or 'cuda'")

    required_tools, hints_map = backend_map[backend]
    missing: list[ToolchainErrorDetail] = []

    for tool in required_tools:
        found, _ = detect_tool(tool)
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


def detect_toolchain() -> ToolchainStatus:
    """Detect all required build tools and return their status.

    Checks for the presence and versions of all tools required for
    both SYCL and CUDA backends.

    Returns:
        ToolchainStatus with version information for all detected tools.
    """
    status = ToolchainStatus()

    # Common tools
    _, version = detect_tool("gcc")
    status.gcc = version
    _, version = detect_tool("make")
    status.make = version
    _, version = detect_tool("git")
    status.git = version
    _, version = detect_tool("cmake")
    status.cmake = version

    # SYCL-specific tools
    _, version = detect_tool("icpx")
    status.sycl_compiler = version

    # CUDA-specific tools
    _, version = detect_tool("nvcc")
    status.cuda_toolkit = version
    _, version = detect_tool("nvtop")
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

    error_code: "ErrorCode"
    failed_check: str
    why_blocked: str
    how_to_fix: str
    docs_ref: str | None = None
