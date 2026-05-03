"""llama_cli.tui.components — panel-builder functions and named widgets."""

from .alerts import (
    GPUTelemetryWidget,
    NoticesWidget,
    SystemHealthWidget,
    SystemStatusWidget,
    build_gpu_telemetry_panel,
    build_profile_status_panel,
    build_risk_panel_acknowledged,
    build_risk_panel_required,
    build_status_messages_panel,
    build_status_panel,
)
from .menu import CommandMenu, build_command_menu
from .modal import AddSlotModal
from .panels import (
    GPUStatsPanel,
    ServerColumnPanel,
    ServerLogPanel,
    SlotStatusPanel,
)

__all__ = [
    # Widgets
    "AddSlotModal",
    "CommandMenu",
    "GPUStatsPanel",
    "GPUTelemetryWidget",
    "NoticesWidget",
    "ServerColumnPanel",
    "ServerLogPanel",
    "SlotStatusPanel",
    "SystemHealthWidget",
    "SystemStatusWidget",
    # Builder functions
    "build_command_menu",
    "build_gpu_telemetry_panel",
    "build_profile_status_panel",
    "build_risk_panel_acknowledged",
    "build_risk_panel_required",
    "build_status_messages_panel",
    "build_status_panel",
]
