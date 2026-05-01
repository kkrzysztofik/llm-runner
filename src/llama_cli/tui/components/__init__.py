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
    ServerLogPanel,
    build_column_panel,
    build_placeholder_panel,
    build_slot_section,
    build_slot_status_panel,
)

__all__ = [
    # Widgets
    "AddSlotModal",
    "CommandMenu",
    "GPUTelemetryWidget",
    "NoticesWidget",
    "ServerLogPanel",
    "SystemHealthWidget",
    "SystemStatusWidget",
    # Builder functions
    "build_column_panel",
    "build_command_menu",
    "build_gpu_telemetry_panel",
    "build_placeholder_panel",
    "build_profile_status_panel",
    "build_risk_panel_acknowledged",
    "build_risk_panel_required",
    "build_slot_section",
    "build_slot_status_panel",
    "build_status_messages_panel",
    "build_status_panel",
]
