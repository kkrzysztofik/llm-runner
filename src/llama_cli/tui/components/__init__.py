"""Named Textual widgets for the TUI."""

from .about_modal import AboutModal
from .digital_clock import LLM_RUNNER_LOGO, DigitalClockWidget
from .gpu_stats import GPUStatsPanel
from .modal import AddSlotModal
from .server_column import ServerColumnPanel
from .server_log import ServerLogPanel
from .system_health import (
    CPUUsageWidget,
    DateTimeWidget,
    MemorySwapWidget,
    SystemHealthWidget,
    SystemInfoWidget,
)

__all__ = [
    # Widgets
    "AboutModal",
    "AddSlotModal",
    "DigitalClockWidget",
    "LLM_RUNNER_LOGO",
    "CPUUsageWidget",
    "DateTimeWidget",
    "GPUStatsPanel",
    "MemorySwapWidget",
    "ServerColumnPanel",
    "ServerLogPanel",
    "SystemHealthWidget",
    "SystemInfoWidget",
]
