"""Pure library for computing slot state transition messages and runtime liveness."""

from collections.abc import Callable
from typing import Any

import psutil

from .config.enums import SlotState

_TRANSITION_MESSAGES: dict[tuple[str, str], tuple[str, str]] = {
    (SlotState.LAUNCHING.value, SlotState.RUNNING.value): (
        "Launched",
        "green",
    ),
    (SlotState.RUNNING.value, SlotState.DEGRADED.value): (
        "Degraded",
        "yellow",
    ),
    (SlotState.RUNNING.value, SlotState.CRASHED.value): (
        "Crashed",
        "red",
    ),
    (SlotState.DEGRADED.value, SlotState.OFFLINE.value): (
        "Offline",
        "yellow",
    ),
    (SlotState.CRASHED.value, SlotState.OFFLINE.value): (
        "Offline",
        "red",
    ),
    (SlotState.OFFLINE.value, SlotState.IDLE.value): (
        "Idle",
        "dim",
    ),
}


def compute_slot_transition(
    slot_id: str,
    old_state: str | None,
    new_state: SlotState,
) -> tuple[str, str] | None:
    """Compute the status message and color for a slot state transition.

    Args:
        slot_id: The slot identifier.
        old_state: The previous state value, or None if the slot is new.
        new_state: The new slot state.

    Returns:
        A tuple of (message, color) if the transition warrants a status
        message, otherwise None.
    """
    if old_state is None and new_state == SlotState.RUNNING:
        return (f"Slot '{slot_id}' launched successfully.", "green")

    if old_state is not None:
        key = (old_state, new_state.value)
        if key in _TRANSITION_MESSAGES:
            label, color = _TRANSITION_MESSAGES[key]
            return (f"Slot '{slot_id}': {label} ({color})", color)

    return None


def resolve_slot_runtime_status(
    current_state: str,
    process: Any | None,
    pid_exists: Callable[[int], bool] = psutil.pid_exists,
) -> str:
    """Resolve the runtime liveness status of a server slot.

    When the current state is ``"running"``, this function checks whether
    the associated process is still alive.  If the process is gone the
    returned status becomes ``"crashed"``.  For all other states the
    current state is returned unchanged.

    Args:
        current_state: The slot's logical state (e.g. ``"running"``, ``"idle"``).
        process: The subprocess.Popen handle or None.
        pid_exists: Injectable callable for testing (defaults to psutil.pid_exists).

    Returns:
        The resolved status string — either the unchanged current state
        or ``"crashed"`` when a running slot's process has exited.
    """
    if current_state != SlotState.RUNNING.value:
        return current_state

    if process is None:
        return SlotState.CRASHED.value

    poll = getattr(process, "poll", None)
    if poll is not None:
        if poll() is not None:
            return SlotState.CRASHED.value
    else:
        pid = getattr(process, "pid", None)
        if not (pid is not None and pid_exists(pid)):
            return SlotState.CRASHED.value

    return SlotState.RUNNING.value
