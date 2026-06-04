"""Orchestration types — dataclasses and constants for launch and server management."""

import time
from dataclasses import dataclass
from typing import Any, Final

from ..config import MultiValidationError, ServerConfig, SlotState
from ..gpu_telemetry import GPUStats
from ..log_buffer import LogBuffer

# Module-local string constants (process_manager-specific).
ARTIFACT_CHECK_NAME: Final[str] = "artifact_persistence"
OWNER_ONLY_PERMISSIONS_FAILURE: Final[str] = (
    "artifact persistence failed to enforce required owner-only permissions"
)
LOCKFILE_FIX_SUGGESTION: Final[str] = "verify the owning process or clear the lockfile"
PERMISSION_SUPPORT_HINT: Final[str] = (
    "verify runtime path and permission support/chmod limitations before retry"
)
PERMISSION_WRITABILITY_HINT: Final[str] = (
    "verify runtime path writability and filesystem permission support/chmod limitations"
)
MAX_COLLISION_RETRIES: Final[int] = 10


@dataclass
class ProcessMetadata:
    """Process ownership metadata for T001 security hardening."""

    pid: int
    create_time: float


@dataclass
class LaunchResult:
    """Result of slot-based launch operation (T020)."""

    status: str
    launched: list[str] | None = None
    warnings: list[str] | None = None
    errors: MultiValidationError | None = None

    @property
    def launch_count(self) -> int:
        """Return the number of successfully launched slots."""
        return len(self.launched) if self.launched else 0

    def is_blocked(self) -> bool:
        """Check if launch was completely blocked."""
        return self.status == "blocked"

    def is_degraded(self) -> bool:
        """Check if launch was partially successful (degraded)."""
        return self.status == "degraded"

    def is_success(self) -> bool:
        """Check if launch was fully successful."""
        return self.status == "success"


@dataclass
class LaunchOrchestrationResult:
    """Structured result from launch orchestration."""

    updated_configs: list[ServerConfig]
    launch_result: LaunchResult | None
    processes: dict[str, Any]
    slot_states: dict[str, str]
    status_messages: list[str]
    risk_result: Any  # RiskAckResult | None — avoid circular import
    empty: bool = False


@dataclass
class SlotRuntime:
    """Runtime state for a single model slot."""

    slot_id: str
    state: SlotState
    pid: int | None
    start_time: float
    logs: LogBuffer
    gpu_stats: GPUStats | None = None

    def transition_to(self, new_state: SlotState) -> None:
        """Transition to a new state, updating start_time if needed."""
        self.state = new_state
        if new_state in (SlotState.LAUNCHING, SlotState.RUNNING):
            self.start_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize runtime state to a dictionary."""
        return {
            "slot_id": self.slot_id,
            "state": self.state.value,
            "pid": self.pid,
            "start_time": self.start_time,
            "gpu_stats": self.gpu_stats is not None,
        }
