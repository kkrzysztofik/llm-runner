"""Pure library for computing slot state transition messages."""

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
