"""Orchestration types — dataclasses and constants for launch and server management."""

from dataclasses import dataclass
from typing import Any, Final

from ..config import MultiValidationError, ServerConfig

# Module-local string constants (process_manager-specific).
LOCKFILE_FIX_SUGGESTION: Final[str] = "verify the owning process or clear the lockfile"


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
