"""Characterization tests for slot state transition logic."""

import pytest

from llama_manager.config.enums import SlotState
from llama_manager.slot_state import compute_slot_transition


def test_first_launch_running_returns_success_message() -> None:
    """First launch (old_state None -> RUNNING) should return success."""
    result = compute_slot_transition("summary", None, SlotState.RUNNING)
    assert result is not None
    message, color = result
    assert message == "Slot 'summary' launched successfully."
    assert color == "green"


def test_launching_to_running() -> None:
    """LAUNCHING -> RUNNING should return Launched message."""
    result = compute_slot_transition("summary", SlotState.LAUNCHING.value, SlotState.RUNNING)
    assert result == ("Slot 'summary': Launched (green)", "green")


def test_running_to_degraded() -> None:
    """RUNNING -> DEGRADED should return Degraded message."""
    result = compute_slot_transition("summary", SlotState.RUNNING.value, SlotState.DEGRADED)
    assert result == ("Slot 'summary': Degraded (yellow)", "yellow")


def test_running_to_crashed() -> None:
    """RUNNING -> CRASHED should return Crashed message."""
    result = compute_slot_transition("summary", SlotState.RUNNING.value, SlotState.CRASHED)
    assert result == ("Slot 'summary': Crashed (red)", "red")


def test_degraded_to_offline() -> None:
    """DEGRADED -> OFFLINE should return Offline message."""
    result = compute_slot_transition("summary", SlotState.DEGRADED.value, SlotState.OFFLINE)
    assert result == ("Slot 'summary': Offline (yellow)", "yellow")


def test_crashed_to_offline() -> None:
    """CRASHED -> OFFLINE should return Offline message."""
    result = compute_slot_transition("summary", SlotState.CRASHED.value, SlotState.OFFLINE)
    assert result == ("Slot 'summary': Offline (red)", "red")


def test_offline_to_idle() -> None:
    """OFFLINE -> IDLE should return Idle message."""
    result = compute_slot_transition("summary", SlotState.OFFLINE.value, SlotState.IDLE)
    assert result == ("Slot 'summary': Idle (dim)", "dim")


@pytest.mark.parametrize(
    "old_state,new_state",
    [
        (None, SlotState.IDLE),
        (None, SlotState.LAUNCHING),
        (None, SlotState.DEGRADED),
        (None, SlotState.CRASHED),
        (None, SlotState.OFFLINE),
        (SlotState.IDLE.value, SlotState.LAUNCHING),
        (SlotState.IDLE.value, SlotState.RUNNING),
        (SlotState.LAUNCHING.value, SlotState.IDLE),
        (SlotState.LAUNCHING.value, SlotState.DEGRADED),
        (SlotState.RUNNING.value, SlotState.IDLE),
        (SlotState.RUNNING.value, SlotState.LAUNCHING),
        (SlotState.RUNNING.value, SlotState.OFFLINE),
        (SlotState.DEGRADED.value, SlotState.IDLE),
        (SlotState.DEGRADED.value, SlotState.RUNNING),
        (SlotState.DEGRADED.value, SlotState.CRASHED),
        (SlotState.CRASHED.value, SlotState.IDLE),
        (SlotState.CRASHED.value, SlotState.RUNNING),
        (SlotState.CRASHED.value, SlotState.DEGRADED),
        (SlotState.OFFLINE.value, SlotState.RUNNING),
        (SlotState.OFFLINE.value, SlotState.LAUNCHING),
        (SlotState.OFFLINE.value, SlotState.DEGRADED),
        (SlotState.OFFLINE.value, SlotState.CRASHED),
    ],
)
def test_unmapped_transitions_return_none(old_state: str | None, new_state: SlotState) -> None:
    """Transitions not in the mapping should return None."""
    assert compute_slot_transition("summary", old_state, new_state) is None
