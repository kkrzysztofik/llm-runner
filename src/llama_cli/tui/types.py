"""Typed view state shared across the TUI submodule."""

from dataclasses import dataclass
from typing import Any, Literal

from llama_manager import GPUStats, LogBuffer, ServerConfig


@dataclass(frozen=True)
class RiskPromptState:
    """Current risk prompt state."""

    kind: Literal["vram", "hardware"]
    acknowledged: bool


@dataclass(frozen=True)
class CommandMenuState:
    """State needed to render the bottom command menu."""

    profile_request: str | None
    risk_prompt: RiskPromptState | None
    build_request: bool
    smoke_request: bool


@dataclass(frozen=True)
class SystemStatusState:
    """State needed to render the top system status widgets."""

    gpu_lines: list[str]
    notices: list[str]


@dataclass(frozen=True)
class ServerColumnState:
    """State needed to render one server column."""

    config: ServerConfig
    buffer: LogBuffer
    gpu: GPUStats | None
    host: str
    stale_warning: str | None
    slot_states: dict[str, str]
    server_processes: dict[str, Any]
    is_unsaved: bool


@dataclass(frozen=True)
class SlotStatusState:
    """State needed to render the fallback slot status panel."""

    configs: list[ServerConfig]
    slot_states: dict[str, str]
    server_processes: dict[str, Any]
    log_buffers: dict[str, LogBuffer]
    host: str
    unsaved_slots: set[str]
