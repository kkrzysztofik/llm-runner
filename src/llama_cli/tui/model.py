"""Dashboard model for the Textual TUI.

The model owns mutable runtime state. It intentionally avoids Textual objects
and Rich renderables so it can be inspected independently from the UI layer.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from llama_cli.gpu_collectors import collect_nvtop_stats
from llama_manager import (
    Config,
    GPUStats,
    LaunchResult,
    LogBuffer,
    ModelSlot,
    ServerConfig,
    ServerManager,
)

from .types import RiskPromptState


class DashboardModel:
    """Mutable dashboard state shared by controller and view models."""

    STATUS_MESSAGE_LIFETIME_S: float = 30.0

    def __init__(
        self,
        configs: list[ServerConfig],
        gpu_indices: list[int],
        slots: list[ModelSlot] | None = None,
    ) -> None:
        self.config = Config()
        self.configs = configs
        self.gpu_indices = gpu_indices
        self.slots = slots or []
        self.log_buffers: dict[str, LogBuffer] = {
            cfg.alias: LogBuffer(redact_sensitive=True) for cfg in configs
        }
        self.gpu_stats: list[GPUStats] = [
            GPUStats(idx, collector=self.make_collector(idx)) for idx in gpu_indices
        ]
        self.running = True
        self.launch_result: LaunchResult | None = None
        self.risk_prompt: RiskPromptState | None = None

        self.profile_status: dict[str, str] = {}
        self.profile_flavor: dict[str, str] = {}
        self.profile_cancel_events: dict[str, threading.Event] = {}
        self.profile_lock = threading.Lock()

        self.status_messages: list[tuple[float, str]] = []
        self.status_lock = threading.Lock()

        self.profile_request: str | None = None
        self.build_request = False
        self.smoke_request = False
        self.unsaved_slots: set[str] = set()

        self.server_manager = ServerManager()
        self.slot_states: dict[str, str] = {}
        self.server_processes: dict[str, Any] = {}

    def make_collector(self, device_index: int) -> Callable[[], dict[str, Any]]:
        """Create a GPU collector bound to a device index."""

        def collector() -> dict[str, Any]:
            return collect_nvtop_stats(device_index)

        return collector

    def stop(self) -> None:
        """Stop the dashboard."""
        self.running = False

    def set_risk_prompt(
        self, kind: Literal["vram", "hardware"], acknowledged: bool = False
    ) -> None:
        """Set the active risk prompt."""
        self.risk_prompt = RiskPromptState(kind=kind, acknowledged=acknowledged)

    def clear_risk_prompt(self) -> None:
        """Clear the active risk prompt."""
        self.risk_prompt = None

    def push_status_message(self, message: str) -> None:
        """Push a status message to the bounded TUI message buffer."""
        with self.status_lock:
            self.status_messages.append((time.monotonic(), message))
            if len(self.status_messages) > 5:
                self.status_messages.pop(0)

    def get_status_messages_since(self, since_ts: float) -> list[tuple[float, str]]:
        """Return status messages newer than ``since_ts`` and not expired."""
        cutoff = time.monotonic() - self.STATUS_MESSAGE_LIFETIME_S
        with self.status_lock:
            return [(ts, msg) for ts, msg in self.status_messages if ts > since_ts and ts >= cutoff]
