"""llama_cli.tui.components — pure panel-builder functions."""

from .alerts import (
    build_gpu_telemetry_panel,
    build_profile_status_panel,
    build_risk_panel_acknowledged,
    build_risk_panel_required,
    build_status_messages_panel,
    build_status_panel,
)
from .menu import build_command_menu
from .panels import (
    build_column_panel,
    build_placeholder_panel,
    build_slot_section,
    build_slot_status_panel,
)

__all__ = [
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
