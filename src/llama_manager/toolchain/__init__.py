"""toolchain package — build toolchain detection and status checking."""

from .constants import ToolchainHint
from .detector import (
    ToolchainErrorDetail,
    ToolchainStatus,
    detect_tool,
    detect_toolchain,
    get_toolchain_hints,
)

__all__ = [
    # Status
    "ToolchainStatus",
    "ToolchainErrorDetail",
    # Hints
    "ToolchainHint",
    # Detection
    "detect_tool",
    "detect_toolchain",
    "get_toolchain_hints",
]
