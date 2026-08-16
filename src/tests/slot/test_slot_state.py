"""Tests for slot runtime liveness resolution."""

from unittest.mock import MagicMock

from llama_manager.slot_state import resolve_slot_runtime_status


class TestResolveSlotRuntimeStatus:
    """Tests for resolve_slot_runtime_status function."""

    def test_non_running_state_unchanged(self) -> None:
        """Non-running states should be returned unchanged."""
        for state in ("idle", "offline", "crashed", "degraded", "launching"):
            result = resolve_slot_runtime_status(state, None)
            assert result == state

    def test_no_process_returns_crashed(self) -> None:
        """Running state with no process should return crashed."""
        result = resolve_slot_runtime_status("running", None)
        assert result == "crashed"

    def test_process_poll_exited_returns_crashed(self) -> None:
        """Process with poll() returning non-None should return crashed."""
        proc = MagicMock()
        proc.poll.return_value = 1
        result = resolve_slot_runtime_status("running", proc)
        assert result == "crashed"

    def test_process_poll_none_returns_running(self) -> None:
        """Process with poll() returning None should return running."""
        proc = MagicMock()
        proc.poll.return_value = None
        result = resolve_slot_runtime_status("running", proc)
        assert result == "running"

    def test_idle_state_unchanged_with_process(self) -> None:
        """Idle state with a process should remain idle."""
        proc = MagicMock()
        proc.poll.return_value = None
        result = resolve_slot_runtime_status("idle", proc)
        assert result == "idle"

    def test_offline_state_unchanged_with_no_process(self) -> None:
        """Offline state with no process should remain offline."""
        result = resolve_slot_runtime_status("offline", None)
        assert result == "offline"
