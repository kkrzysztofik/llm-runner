"""Phase 7 — T079: State-machine integration test.

Verifies the full SlotRuntime lifecycle:
  IDLE → LAUNCHING → RUNNING → DEGRADED → RUNNING → OFFLINE → IDLE

Tests:
  - All state transitions via SlotRuntime.transition_to()
  - start_time updates correctly for LAUNCHING/RUNNING transitions
  - start_time preserved for non-launching transitions
  - dataclass field integrity after each transition
  - Serialization (to_dict) correctness throughout
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from llama_manager.config import SlotState
from llama_manager.log_buffer import LogBuffer
from llama_manager.process_manager import SlotRuntime  # noqa: T079


class TestStateMachineLifecycle:
    """T079: Full state-machine lifecycle integration test."""

    # ------------------------------------------------------------------
    # Phase 1: IDLE → LAUNCHING
    # ------------------------------------------------------------------

    def test_lifecycle_idle_to_launching(self) -> None:
        """Verify IDLE → LAUNCHING updates state and start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.IDLE,
            pid=None,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        assert runtime.state == SlotState.IDLE
        assert runtime.pid is None

        launching_time = time.time()
        runtime.transition_to(SlotState.LAUNCHING)

        assert runtime.state == SlotState.LAUNCHING
        assert runtime.start_time >= launching_time
        assert runtime.pid is None  # pid not assigned until RUNNING

    def test_lifecycle_launching_to_running(self) -> None:
        """Verify LAUNCHING → RUNNING updates state and start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.LAUNCHING,
            pid=None,
            start_time=time.time() - 1.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        running_time = time.time()
        runtime.transition_to(SlotState.RUNNING)

        assert runtime.state == SlotState.RUNNING
        assert runtime.start_time >= running_time
        # pid is still None — the dataclass doesn't auto-assign it
        assert runtime.pid is None

    # ------------------------------------------------------------------
    # Phase 2: RUNNING → DEGRADED (start_time must NOT update)
    # ------------------------------------------------------------------

    def test_lifecycle_running_to_degraded_preserves_start_time(self) -> None:
        """DEGRADED transition must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.RUNNING,
            pid=12345,
            start_time=1000.0,
            logs=LogBuffer(),
            gpu_stats=MagicMock(),
        )

        runtime.transition_to(SlotState.DEGRADED)

        assert runtime.state == SlotState.DEGRADED
        assert runtime.start_time == 1000.0  # unchanged
        assert runtime.pid == 12345  # pid preserved

    # ------------------------------------------------------------------
    # Phase 3: DEGRADED → RUNNING (start_time MUST update)
    # ------------------------------------------------------------------

    def test_lifecycle_degraded_to_running_updates_start_time(self) -> None:
        """Recovery from DEGRADED → RUNNING must update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.DEGRADED,
            pid=12345,
            start_time=1000.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        new_running_time = time.time()
        runtime.transition_to(SlotState.RUNNING)

        assert runtime.state == SlotState.RUNNING
        assert runtime.start_time >= new_running_time
        assert runtime.pid == 12345

    # ------------------------------------------------------------------
    # Phase 4: RUNNING → OFFLINE (start_time must NOT update)
    # ------------------------------------------------------------------

    def test_lifecycle_running_to_offline_preserves_start_time(self) -> None:
        """OFFLINE transition must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.RUNNING,
            pid=12345,
            start_time=2000.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        runtime.transition_to(SlotState.OFFLINE)

        assert runtime.state == SlotState.OFFLINE
        assert runtime.start_time == 2000.0  # unchanged
        assert runtime.pid == 12345

    # ------------------------------------------------------------------
    # Phase 5: OFFLINE → IDLE (start_time must NOT update)
    # ------------------------------------------------------------------

    def test_lifecycle_offline_to_idle_preserves_start_time(self) -> None:
        """IDLE transition from OFFLINE must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test-slot",
            state=SlotState.OFFLINE,
            pid=None,
            start_time=3000.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        runtime.transition_to(SlotState.IDLE)

        assert runtime.state == SlotState.IDLE
        assert runtime.start_time == 3000.0  # unchanged
        assert runtime.pid is None

    # ------------------------------------------------------------------
    # Full chain in a single test
    # ------------------------------------------------------------------

    def test_full_lifecycle_chain(self) -> None:
        """Verify the full chain: IDLE→LAUNCHING→RUNNING→DEGRADED→RUNNING→OFFLINE→IDLE."""
        runtime = SlotRuntime(
            slot_id="gpu0-slot1",
            state=SlotState.IDLE,
            pid=None,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        # IDLE → LAUNCHING
        runtime.transition_to(SlotState.LAUNCHING)
        assert runtime.state == SlotState.LAUNCHING
        launching_st = runtime.start_time

        # LAUNCHING → RUNNING
        time.sleep(0.001)
        runtime.transition_to(SlotState.RUNNING)
        assert runtime.state == SlotState.RUNNING
        running_st = runtime.start_time
        assert running_st > launching_st

        # RUNNING → DEGRADED
        runtime.transition_to(SlotState.DEGRADED)
        assert runtime.state == SlotState.DEGRADED
        assert runtime.start_time == running_st  # preserved

        # DEGRADED → RUNNING
        runtime.pid = 12345  # pid assigned on recovery
        runtime.gpu_stats = MagicMock()
        time.sleep(0.001)
        runtime.transition_to(SlotState.RUNNING)
        assert runtime.state == SlotState.RUNNING
        running_st2 = runtime.start_time
        assert running_st2 > running_st  # updated

        # RUNNING → OFFLINE
        runtime.transition_to(SlotState.OFFLINE)
        assert runtime.state == SlotState.OFFLINE
        offline_st = runtime.start_time
        assert offline_st == running_st2  # preserved

        # OFFLINE → IDLE
        runtime.transition_to(SlotState.IDLE)
        assert runtime.state == SlotState.IDLE
        idle_st = runtime.start_time
        assert idle_st == offline_st  # preserved

    # ------------------------------------------------------------------
    # Dataclass field integrity
    # ------------------------------------------------------------------

    def test_field_integrity_after_transitions(self) -> None:
        """All dataclass fields must remain accessible and correct after transitions."""
        gpu = MagicMock()
        runtime = SlotRuntime(
            slot_id="gpu1-slot0",
            state=SlotState.RUNNING,
            pid=9999,
            start_time=5000.0,
            logs=LogBuffer(),
            gpu_stats=gpu,
        )

        # Capture start_time before transitioning to DEGRADED (RUNNING → DEGRADED preserves)
        degraded_start = runtime.start_time
        runtime.transition_to(SlotState.DEGRADED)
        assert runtime.start_time == degraded_start  # preserved

        # DEGRADED → RUNNING updates start_time
        runtime.transition_to(SlotState.RUNNING)
        running_start = runtime.start_time
        assert running_start > degraded_start  # updated

        # RUNNING → OFFLINE preserves
        runtime.transition_to(SlotState.OFFLINE)
        assert runtime.start_time == running_start

        # OFFLINE → IDLE preserves
        runtime.transition_to(SlotState.IDLE)
        assert runtime.start_time == running_start

        # Fields must still be accessible
        assert runtime.slot_id == "gpu1-slot0"
        assert runtime.state == SlotState.IDLE
        assert runtime.pid == 9999
        assert isinstance(runtime.logs, LogBuffer)
        assert runtime.gpu_stats is gpu

    # ------------------------------------------------------------------
    # Serialization (to_dict)
    # ------------------------------------------------------------------

    def test_to_dict_at_each_state(self) -> None:
        """to_dict() must produce correct dict at every state in the lifecycle."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.IDLE,
            pid=None,
            start_time=100.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        expected_states = [
            SlotState.IDLE,
            SlotState.LAUNCHING,
            SlotState.RUNNING,
            SlotState.DEGRADED,
            SlotState.OFFLINE,
            SlotState.IDLE,  # back to idle
        ]

        for _i, new_state in enumerate(expected_states):
            runtime.transition_to(new_state)
            d = runtime.to_dict()
            assert d["slot_id"] == "test"
            assert d["state"] == new_state.value
            if new_state in (SlotState.LAUNCHING, SlotState.RUNNING):
                assert d["start_time"] >= 100.0
            else:
                assert d["start_time"] == runtime.start_time

    def test_to_dict_includes_gpu_stats_flag(self) -> None:
        """to_dict() must set gpu_stats=True when gpu_stats is not None."""
        gpu = MagicMock()
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=gpu,
        )
        d = runtime.to_dict()
        assert d["gpu_stats"] is True

    def test_to_dict_excludes_gpu_stats_when_none(self) -> None:
        """to_dict() must set gpu_stats=False when gpu_stats is None."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        d = runtime.to_dict()
        assert d["gpu_stats"] is False

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_transition_to_same_state_is_noop(self) -> None:
        """Transitioning to the same state should still update start_time for LAUNCHING/RUNNING."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=100.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        old_st = runtime.start_time
        time.sleep(0.001)
        runtime.transition_to(SlotState.RUNNING)
        assert runtime.start_time > old_st  # RUNNING always updates

    def test_transition_idle_to_idle_preserves_start_time(self) -> None:
        """IDLE → IDLE must NOT update start_time (IDLE is not in the update set)."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.IDLE,
            pid=None,
            start_time=999.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.IDLE)
        assert runtime.start_time == 999.0

    def test_transition_offline_to_offline_preserves_start_time(self) -> None:
        """OFFLINE → OFFLINE must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.OFFLINE,
            pid=None,
            start_time=777.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.OFFLINE)
        assert runtime.start_time == 777.0

    def test_transition_degraded_to_degraded_preserves_start_time(self) -> None:
        """DEGRADED → DEGRADED must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.DEGRADED,
            pid=1,
            start_time=888.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.DEGRADED)
        assert runtime.start_time == 888.0

    def test_transition_to_crashed(self) -> None:
        """CRASHED transition must NOT update start_time."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.RUNNING,
            pid=1,
            start_time=500.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )
        runtime.transition_to(SlotState.CRASHED)
        assert runtime.state == SlotState.CRASHED
        assert runtime.start_time == 500.0  # unchanged

    def test_all_slot_states_covered(self) -> None:
        """Every SlotState member must be reachable via transition_to."""
        runtime = SlotRuntime(
            slot_id="test",
            state=SlotState.IDLE,
            pid=None,
            start_time=0.0,
            logs=LogBuffer(),
            gpu_stats=None,
        )

        for state in SlotState:
            runtime.transition_to(state)
            assert runtime.state == state
