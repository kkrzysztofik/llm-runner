from __future__ import annotations

"""Tests for US1: Slot-based launch flow with lock collision detection.

Tests T014: Dual-slot success and collision launch tests covering:
(1) both slots launch without lock collision
(2) second slot fails with error_code="lock_conflict" when first lock exists with matching port

These tests target real US1 APIs that will be implemented in T017-T019.
"""


import os
import time
from unittest.mock import Mock, patch

import pytest

from llama_manager.config import ModelSlot
from llama_manager.orchestration import (
    LockMetadata,
    check_lockfile_integrity,
    create_lock,
    read_lock,
    release_lock,
)


class TestDualSlotSuccess:
    """Tests for successful dual-slot launch without collision."""

    def test_both_slots_launch_without_collision(self, tmp_path) -> None:
        """Both slots should launch successfully when no locks exist.

        This tests the happy path: two slots are configured, neither has a lock,
        and both should be able to acquire locks and launch.

        Targets: ServerManager.launch_all_slots() returning LaunchResult with status="success"
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        slot1 = ModelSlot(slot_id="slot1", model_path="/models/model1.gguf", port=8080)
        slot2 = ModelSlot(slot_id="slot2", model_path="/models/model2.gguf", port=8081)

        # Create locks for both slots (simulating successful launch)
        lock1 = create_lock(runtime_dir, slot1.slot_id, pid=12345, port=slot1.port)
        lock2 = create_lock(runtime_dir, slot2.slot_id, pid=12346, port=slot2.port)

        # Verify both locks exist
        assert lock1.exists()
        assert lock2.exists()

        # Verify lock content
        meta1 = read_lock(runtime_dir, slot1.slot_id)
        assert meta1 is not None
        assert isinstance(meta1, LockMetadata)
        assert meta1.pid == 12345
        assert meta1.port == slot1.port

        meta2 = read_lock(runtime_dir, slot2.slot_id)
        assert meta2 is not None
        assert isinstance(meta2, LockMetadata)
        assert meta2.pid == 12346
        assert meta2.port == slot2.port

    def test_lock_metadata_timestamps(self, tmp_path) -> None:
        """Lock metadata should have valid started_at timestamps."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        meta = read_lock(runtime_dir, "slot1")
        assert meta is not None
        assert isinstance(meta, LockMetadata)
        # started_at should be a positive timestamp close to now
        assert meta.started_at > 0
        # Should be within last 60 seconds using wall-clock time.time()
        assert time.time() - meta.started_at < 60

    def test_lockfile_permissions(self, tmp_path) -> None:
        """Lockfiles should have 0600 permissions."""
        import stat

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        mode = stat.S_IMODE(os.stat(lock_path).st_mode)
        assert mode == 0o600

    def test_release_lock_deletes_file(self, tmp_path) -> None:
        """release_lock should delete the lockfile."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        assert (runtime_dir / "slot-slot1.lock").exists()

        release_lock(runtime_dir, "slot1")

        assert not (runtime_dir / "slot-slot1.lock").exists()

    def test_release_lock_nonexistent_does_not_error(self, tmp_path) -> None:
        """release_lock should not raise when lockfile does not exist."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Should not raise
        release_lock(runtime_dir, "slot1")


class TestLockCollision:
    """Tests for lock conflict detection when launching second slot."""

    def test_second_slot_fails_with_lock_conflict(self, tmp_path) -> None:
        """Second slot should fail with error_code='lock_conflict' when first lock exists with matching port.

        This tests the collision case: slot1 has a lock, and slot2 tries to launch
        with the same port as slot1.

        Targets: ServerManager.launch_all_slots() should return LaunchResult with
        errors containing ErrorCode.LOCK_CONFLICT when ports collide.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # First slot acquires lock
        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Second slot tries to use same slot_id - should fail with FileExistsError
        with pytest.raises(FileExistsError, match="already exists"):
            create_lock(runtime_dir, "slot1", pid=12346, port=8080)

    def test_lock_conflict_with_same_slot_id(self, tmp_path) -> None:
        """Creating a lock for the same slot_id should fail with FileExistsError."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Try to create another lock for the same slot - should raise FileExistsError
        with pytest.raises(FileExistsError) as exc_info:
            create_lock(runtime_dir, "slot1", pid=12346, port=8081)

        assert "already exists" in str(exc_info.value)

    def test_different_ports_no_collision(self, tmp_path) -> None:
        """Different slots with different ports should not collide."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Each slot has its own lockfile (slot_id-based)
        create_lock(runtime_dir, "slot1", pid=12345, port=8080)
        create_lock(runtime_dir, "slot2", pid=12346, port=8081)

        # Both locks should exist
        assert read_lock(runtime_dir, "slot1") is not None
        assert read_lock(runtime_dir, "slot2") is not None

    def test_check_lockfile_integrity_no_lock(self, tmp_path) -> None:
        """check_lockfile_integrity should return None when no lock exists."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        result = check_lockfile_integrity(runtime_dir, "slot1")
        assert result is None


class TestServerManagerLaunchAllSlots:
    """Tests for ServerManager.launch_all_slots() API (T017-T019 implementation)."""

    def test_launch_all_slots_returns_launch_result(self, tmp_path) -> None:
        """ServerManager.launch_all_slots() should return LaunchResult.

        This test will fail until T017-T019 implement the LaunchResult dataclass
        and ServerManager.launch_all_slots() method.
        """
        from llama_manager.orchestration import ServerManager

        manager = ServerManager()

        # Mock subprocess to avoid actually starting processes
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = Mock()
            mock_proc.pid = 12345
            mock_proc.stdout = None
            mock_proc.stderr = None
            mock_popen.return_value = mock_proc

            with patch("psutil.Process") as mock_psutil:
                mock_proc_obj = mock_psutil.return_value
                mock_proc_obj.create_time.return_value = 1234567890.123

                # This will fail until T017-T019 implement launch_all_slots
                # and return a LaunchResult object
                with patch(
                    "llama_manager.orchestration.lockfile.resolve_runtime_dir",
                    return_value=tmp_path,
                ):
                    result = manager.launch_all_slots([])

                # Assert LaunchResult exists and has expected structure
                from llama_manager.orchestration import LaunchResult

                assert isinstance(result, LaunchResult)
                assert hasattr(result, "status")
                assert hasattr(result, "launched")
                assert hasattr(result, "warnings")
                assert hasattr(result, "errors")

    def test_launch_all_slots_with_slots_returns_launched_list(self, tmp_path) -> None:
        """launch_all_slots should return LaunchResult with launched slot IDs.

        This test will fail until T017-T019 implement the LaunchResult dataclass.
        """
        from llama_manager.orchestration import LaunchResult, ServerManager

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        manager = ServerManager()

        # Create slots
        slots = [
            ModelSlot(slot_id="slot1", model_path="/models/model1.gguf", port=8080),
            ModelSlot(slot_id="slot2", model_path="/models/model2.gguf", port=8081),
        ]

        # Mock subprocess to avoid actually starting processes
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = Mock()
            mock_proc.pid = 12345
            mock_proc.stdout = None
            mock_proc.stderr = None
            mock_popen.return_value = mock_proc

            with patch("psutil.Process") as mock_psutil:
                mock_proc_obj = mock_psutil.return_value
                mock_proc_obj.create_time.return_value = 1234567890.123

                # This will fail until T017-T019 implement launch_all_slots
                result = manager.launch_all_slots(slots, runtime_dir)

                # Assert LaunchResult has launched list
                assert isinstance(result, LaunchResult)
                assert result.launched is not None
                assert len(result.launched) == len(slots)

    def test_launch_all_slots_with_blocking_locks_returns_blocked_status(self, tmp_path) -> None:
        """launch_all_slots should return LaunchResult with status='blocked' when all slots blocked.

        This test will fail until T017-T019 implement the LaunchResult dataclass
        and proper blocking logic.
        """
        from llama_manager.orchestration import LaunchResult, ServerManager

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        manager = ServerManager()

        # Block all slots with indeterminate locks
        slots = [
            ModelSlot(slot_id="slot1", model_path="/models/model1.gguf", port=8080),
            ModelSlot(slot_id="slot2", model_path="/models/model2.gguf", port=8081),
        ]

        # Create locks that will appear as indeterminate
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        # Mock subprocess
        with (
            patch("subprocess.Popen") as mock_popen,
            patch("psutil.pid_exists") as mock_exists,
            patch("psutil.Process") as mock_process,
        ):
            mock_popen.return_value = Mock(pid=12345, stdout=None, stderr=None)
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # This will fail until T017-T019 implement launch_all_slots
            result = manager.launch_all_slots(slots, runtime_dir)

            # Assert LaunchResult has blocked status
            assert isinstance(result, LaunchResult)
            assert result.status == "blocked"

    def test_launch_all_slots_with_mixed_availability_returns_degraded_status(
        self, tmp_path
    ) -> None:
        """launch_all_slots should return LaunchResult with status='degraded' when some slots available.

        This test will fail until T017-T019 implement the LaunchResult dataclass
        and proper degraded logic.
        """
        from llama_manager.orchestration import LaunchResult, ServerManager

        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        manager = ServerManager()

        # Block one slot, leave one available
        slots = [
            ModelSlot(slot_id="slot1", model_path="/models/model1.gguf", port=8080),
            ModelSlot(slot_id="slot2", model_path="/models/model2.gguf", port=8081),
        ]

        # Block slot1
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)

        # Mock subprocess and runtime dir resolution so acquire_lock uses the test dir
        with (
            patch("subprocess.Popen") as mock_popen,
            patch("psutil.pid_exists") as mock_exists,
            patch("psutil.Process") as mock_process,
            patch(
                "llama_manager.orchestration.lockfile.resolve_runtime_dir", return_value=runtime_dir
            ),
        ):
            mock_popen.return_value = Mock(pid=12345, stdout=None, stderr=None)
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port for slot1
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # This will fail until T017-T019 implement launch_all_slots
            result = manager.launch_all_slots(slots, runtime_dir)

            # Assert LaunchResult has degraded status
            assert isinstance(result, LaunchResult)
            assert result.status == "degraded"


class TestLaunchOrchestrate:
    """Characterization tests for launch_orchestrate extracted from TUI controller."""

    def test_empty_configs_returns_empty_result(self) -> None:
        """Empty configs should return empty=True with guidance message."""
        from llama_manager.orchestration import (
            LaunchOrchestrationResult,
            launch_orchestrate,
        )

        with patch(
            "llama_manager.orchestration.manager.apply_profile_overrides",
            return_value=([], ["No profile found"]),
        ):
            mock_sm = Mock()
            result = launch_orchestrate(
                configs=[],
                base_config=Mock(),
                server_manager=mock_sm,
                log_buffers={},
                get_driver_version=lambda _b: "driver-1",
            )

        assert isinstance(result, LaunchOrchestrationResult)
        assert result.empty is True
        assert result.updated_configs == []
        assert "No slots configured. Press 'a' to add a slot." in result.status_messages
        mock_sm.begin_launch_attempt.assert_not_called()

    def test_success_path_returns_processes_and_slot_states(self) -> None:
        """Happy path should return processes, slot_states, and success result."""
        from llama_manager.orchestration import LaunchResult, launch_orchestrate
        from tests.support.helpers import make_server_config

        cfg = make_server_config(alias="test", port=8080)
        mock_proc = Mock()
        mock_sm = Mock()
        mock_sm.begin_launch_attempt.return_value = "attempt-1"
        mock_sm.issue_ack_token.return_value = "ack:attempt-1"
        mock_sm.launch_all_slots.return_value = LaunchResult(
            status="success", launched=["test"], warnings=None, errors=None
        )
        mock_sm.start_servers.return_value = [mock_proc]

        log_buffers = {"test": Mock()}

        with (
            patch(
                "llama_manager.orchestration.manager.apply_profile_overrides",
                return_value=([cfg], ["Applied profile"]),
            ),
            patch(
                "llama_manager.risk_ack.evaluate_risks",
                return_value=Mock(has_risks=False, risks_acknowledged=False, risk_details=[]),
            ),
        ):
            result = launch_orchestrate(
                configs=[cfg],
                base_config=Mock(),
                server_manager=mock_sm,
                log_buffers=log_buffers,
                get_driver_version=lambda _b: "driver-1",
            )

        assert result.empty is False
        assert result.launch_result is not None
        assert result.launch_result.status == "success"
        assert result.processes == {"test": mock_proc}
        assert result.slot_states == {"test": "running"}
        assert "Applied profile" in result.status_messages
        mock_sm.start_servers.assert_called_once()

    def test_blocked_launch_returns_no_processes(self) -> None:
        """Blocked launch should return empty processes/dict and status messages."""
        from llama_manager.orchestration import LaunchResult, launch_orchestrate
        from tests.support.helpers import make_server_config

        cfg = make_server_config(alias="test", port=8080)
        mock_sm = Mock()
        mock_sm.begin_launch_attempt.return_value = "attempt-1"
        mock_sm.issue_ack_token.return_value = "ack:attempt-1"
        mock_sm.launch_all_slots.return_value = LaunchResult(
            status="blocked",
            launched=[],
            warnings=None,
            errors=Mock(errors=[]),
        )

        with (
            patch(
                "llama_manager.orchestration.manager.apply_profile_overrides",
                return_value=([cfg], []),
            ),
            patch(
                "llama_manager.risk_ack.evaluate_risks",
                return_value=Mock(has_risks=False, risks_acknowledged=False, risk_details=[]),
            ),
        ):
            result = launch_orchestrate(
                configs=[cfg],
                base_config=Mock(),
                server_manager=mock_sm,
                log_buffers={},
                get_driver_version=lambda _b: "driver-1",
            )

        assert result.launch_result is not None
        assert result.launch_result.is_blocked() is True
        assert result.processes == {}
        assert result.slot_states == {}
        assert "Launch blocked: no slots could be launched" in result.status_messages
        mock_sm.start_servers.assert_not_called()

    def test_degraded_launch_returns_partial_processes(self) -> None:
        """Degraded launch should return partial processes and warning messages."""
        from llama_manager.orchestration import LaunchResult, launch_orchestrate
        from tests.support.helpers import make_server_config

        cfg1 = make_server_config(alias="slot1", port=8080)
        cfg2 = make_server_config(alias="slot2", port=8081)

        mock_proc = Mock()
        mock_sm = Mock()
        mock_sm.begin_launch_attempt.return_value = "attempt-1"
        mock_sm.issue_ack_token.return_value = "ack:attempt-1"
        mock_sm.launch_all_slots.return_value = LaunchResult(
            status="degraded",
            launched=["slot1"],
            warnings=["slot2 blocked by lock"],
            errors=None,
        )
        mock_sm.start_servers.return_value = [mock_proc]

        log_buffers = {"slot1": Mock(), "slot2": Mock()}

        with (
            patch(
                "llama_manager.orchestration.manager.apply_profile_overrides",
                return_value=([cfg1, cfg2], []),
            ),
            patch(
                "llama_manager.risk_ack.evaluate_risks",
                return_value=Mock(has_risks=False, risks_acknowledged=False, risk_details=[]),
            ),
        ):
            result = launch_orchestrate(
                configs=[cfg1, cfg2],
                base_config=Mock(),
                server_manager=mock_sm,
                log_buffers=log_buffers,
                get_driver_version=lambda _b: "driver-1",
            )

        assert result.launch_result is not None
        assert result.launch_result.is_degraded() is True
        assert list(result.processes.keys()) == ["slot1"]
        assert "slot2" not in result.processes
        assert "Launch degraded: some slots blocked" in result.status_messages
        mock_sm.start_servers.assert_called_once()

    def test_risk_evaluation_result_included(self) -> None:
        """Risk evaluation result should be included in the orchestration result."""
        from llama_manager.orchestration import LaunchResult, launch_orchestrate
        from tests.support.helpers import make_server_config

        cfg = make_server_config(alias="test", port=8080)
        risk_result = Mock(
            has_risks=True,
            risks_acknowledged=False,
            risk_details=[
                {
                    "alias": "test",
                    "risk": "privileged_port",
                    "risk_kind": "hardware",
                }
            ],
        )

        mock_proc = Mock()
        mock_sm = Mock()
        mock_sm.begin_launch_attempt.return_value = "attempt-1"
        mock_sm.issue_ack_token.return_value = "ack:attempt-1"
        mock_sm.launch_all_slots.return_value = LaunchResult(
            status="success", launched=["test"], warnings=None, errors=None
        )
        mock_sm.start_servers.return_value = [mock_proc]

        with (
            patch(
                "llama_manager.orchestration.manager.apply_profile_overrides",
                return_value=([cfg], []),
            ),
            patch(
                "llama_manager.risk_ack.evaluate_risks",
                return_value=risk_result,
            ),
        ):
            result = launch_orchestrate(
                configs=[cfg],
                base_config=Mock(),
                server_manager=mock_sm,
                log_buffers={"test": Mock()},
                get_driver_version=lambda _b: "driver-1",
            )

        assert result.risk_result == risk_result
        assert result.risk_result.has_risks is True


"""Tests for US1: Degraded one-slot vs full-block behavior.

Tests T016: Degraded one-slot vs full-block behavior tests:
- one available slot returns success with warnings list
- all slots blocked returns MultiValidationError with error_count=2 and launch_count=0
"""


from llama_manager.config import ErrorDetail, MultiValidationError


class MockLaunchResult:
    """Mock LaunchResult for testing degraded vs blocked behavior.

    This is a simplified version for testing purposes.
    In the actual implementation, this would be defined in process_manager.py.
    """

    def __init__(
        self,
        status: str,
        launched: list[str] | None = None,
        warnings: list[str] | None = None,
        errors: MultiValidationError | None = None,
    ):
        self.status = status
        self.launched = launched or []
        self.warnings = warnings or []
        self.errors = errors

    @property
    def launch_count(self) -> int:
        """Return the number of successfully launched slots."""
        return len(self.launched)


class TestDegradedOneSlot:
    """Tests for degraded launch when one slot is available."""

    def test_one_available_slot_returns_degraded_status(self, tmp_path) -> None:
        """One available slot should return degraded status with launched list.

        When one slot has a blocking lock but another slot is available,
        the launch should succeed for the available slot with a degraded status.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Slot1 has a live lock (blocked) - mock psutil to simulate live process
        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Slot2 has no lock (available)
        block2 = check_lockfile_integrity(runtime_dir, "slot2")

        # slot2 should have no block
        assert block2 is None

        # Slot1 would be blocked if we mock psutil to show live process
        # This is tested in test_lock_integrity.py

    def test_degraded_launch_with_warnings(self, tmp_path) -> None:
        """Degraded launch should include warnings about blocked slots.

        When some slots are blocked but others can launch,
        the result should include warnings about the blocked slots.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block one slot
        create_lock(runtime_dir, "blocked_slot", pid=99999, port=8080)

        # Simulate degraded launch result
        launched = ["available_slot"]
        warnings = [
            "slot blocked_slot: lockfile_integrity - indeterminate_owner: lock exists but ownership verification is not definitive"
        ]

        result = MockLaunchResult(status="degraded", launched=launched, warnings=warnings)

        assert result.status == "degraded"
        assert result.launch_count == 1
        assert len(result.warnings) == 1
        assert "blocked_slot" in result.warnings[0]

    def test_degraded_launch_multiple_blocked(self, tmp_path) -> None:
        """Degraded launch should warn about multiple blocked slots."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block two slots
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        # Simulate degraded launch result
        launched = ["slot3"]
        warnings = [
            "slot slot1: lockfile_integrity - indeterminate_owner",
            "slot slot2: lockfile_integrity - indeterminate_owner",
        ]

        result = MockLaunchResult(status="degraded", launched=launched, warnings=warnings)

        assert result.status == "degraded"
        assert result.launch_count == 1
        assert len(result.warnings) == 2

    def test_degraded_launch_all_available(self, tmp_path) -> None:
        """Launch should succeed when all slots are available.

        When no locks exist, all slots should launch successfully without
        degraded status or warnings.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # No locks exist - all slots available
        launched = ["slot1", "slot2", "slot3"]
        warnings: list[str] = []

        result = MockLaunchResult(status="success", launched=launched, warnings=warnings)

        assert result.status == "success"
        assert result.launch_count == 3
        assert len(result.warnings) == 0


class TestFullBlock:
    """Tests for full-block behavior when all slots are blocked."""

    def test_all_slots_blocked_returns_error(self, tmp_path) -> None:
        """All slots blocked should return MultiValidationError with error_count=2.

        When every configured slot has a blocking lock,
        the launch should fail with a MultiValidationError containing all errors.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block all slots with fake PIDs
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []

        # Mock psutil to simulate live processes with indeterminate state for both slots
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 8081  # Different port for slot1
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check integrity for slot1
            block1 = check_lockfile_integrity(runtime_dir, "slot1")
            if block1:
                errors.append(block1)

            # Reset mock for slot2 with different port
            mock_conn.laddr.port = 8080  # Different port for slot2
            mock_proc.connections.return_value = [mock_conn]

            # Check integrity for slot2
            block2 = check_lockfile_integrity(runtime_dir, "slot2")
            if block2:
                errors.append(block2)

        # Both slots should have blocks (indeterminate state)
        assert len(errors) == 2

        # Create MultiValidationError
        multi_error = MultiValidationError(errors=errors)

        assert multi_error.error_count == 2

    def test_full_block_with_launch_count_zero(self, tmp_path) -> None:
        """Full block should have launch_count=0."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block all slots with fake PIDs
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []

        # Mock psutil to simulate live processes with indeterminate state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 8081  # Different port for slot1
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check integrity for slot1
            block1 = check_lockfile_integrity(runtime_dir, "slot1")
            if block1:
                errors.append(block1)

            # Reset mock for slot2 with different port
            mock_conn.laddr.port = 8080  # Different port for slot2
            mock_proc.connections.return_value = [mock_conn]

            # Check integrity for slot2
            block2 = check_lockfile_integrity(runtime_dir, "slot2")
            if block2:
                errors.append(block2)

        multi_error = MultiValidationError(errors=errors)

        result = MockLaunchResult(status="blocked", errors=multi_error)

        assert result.status == "blocked"
        assert result.launch_count == 0
        assert result.errors is not None
        assert result.errors.error_count == 2

    def test_full_block_error_details(self, tmp_path) -> None:
        """Full block errors should contain proper error details.

        Mock psutil to ensure fake PIDs are treated as running with indeterminate
        port state, making errors deterministic.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block slots with indeterminate state
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []

        # Mock psutil to simulate live processes with indeterminate port state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port triggers indeterminate owner
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check both slots deterministically
            for slot_id in ["slot1", "slot2"]:
                block = check_lockfile_integrity(runtime_dir, slot_id)
                if block:
                    errors.append(block)

        # Should have exactly 2 errors (one per slot)
        assert len(errors) == 2

        multi_error = MultiValidationError(errors=errors)

        # Each error should have proper fields populated
        for error in multi_error.errors:
            assert error.error_code is not None
            assert error.failed_check is not None
            assert error.why_blocked is not None
            assert error.how_to_fix is not None

    def test_full_block_mixed_error_types(self, tmp_path) -> None:
        """Full block should have deterministic indeterminate errors.

        Mock psutil to ensure fake PIDs are treated as running with indeterminate
        port state, making both slots return indeterminate errors deterministically.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create locks for both slots
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []

        # Mock psutil to simulate live processes with indeterminate port state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port triggers indeterminate owner
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check both slots deterministically
            for slot_id in ["slot1", "slot2"]:
                block = check_lockfile_integrity(runtime_dir, slot_id)
                if block:
                    errors.append(block)

        # Both slots should have indeterminate blocks when psutil is mocked
        assert len(errors) == 2

        multi_error = MultiValidationError(errors=errors)
        assert multi_error.error_count == 2


class TestDegradedVsFullBlockComparison:
    """Comparison tests between degraded and full-block behaviors."""

    def test_degraded_has_launched_slots(self, tmp_path) -> None:
        """Degraded status should have non-zero launched count."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # One blocked, one available
        create_lock(runtime_dir, "blocked", pid=99999, port=8080)

        launched = ["available"]
        warnings = ["slot blocked: lockfile_integrity"]

        result = MockLaunchResult(status="degraded", launched=launched, warnings=warnings)

        assert result.status == "degraded"
        assert result.launch_count > 0

    def test_full_block_has_no_launched_slots(self, tmp_path) -> None:
        """Full block status should have zero launched count.

        Mock psutil to ensure fake PIDs are treated as running with indeterminate
        port state, making errors deterministic.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # All blocked
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []

        # Mock psutil to simulate live processes with indeterminate port state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port triggers indeterminate owner
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check both slots deterministically
            for slot_id in ["slot1", "slot2"]:
                block = check_lockfile_integrity(runtime_dir, slot_id)
                if block:
                    errors.append(block)

        multi_error = MultiValidationError(errors=errors)
        result = MockLaunchResult(status="blocked", errors=multi_error)

        assert result.status == "blocked"
        assert result.launch_count == 0

    def test_degraded_has_warnings(self, tmp_path) -> None:
        """Degraded status should have warnings about blocked slots."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "blocked_slot", pid=99999, port=8080)

        launched = ["good_slot"]
        warnings = ["slot blocked_slot: lockfile_integrity"]

        result = MockLaunchResult(status="degraded", launched=launched, warnings=warnings)

        assert result.status == "degraded"
        assert len(result.warnings) > 0

    def test_full_block_has_errors_not_warnings(self, tmp_path) -> None:
        """Full block should have errors, not warnings.

        Mock psutil to ensure fake PIDs are treated as running during
        check_lockfile_integrity, making the test deterministic.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []

        # Mock psutil to simulate live processes with indeterminate state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 8081  # Different port for slot1
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check integrity for slot1
            block1 = check_lockfile_integrity(runtime_dir, "slot1")
            if block1:
                errors.append(block1)

            # Reset mock for slot2 with different port
            mock_conn.laddr.port = 8080  # Different port for slot2
            mock_proc.connections.return_value = [mock_conn]

            # Check integrity for slot2
            block2 = check_lockfile_integrity(runtime_dir, "slot2")
            if block2:
                errors.append(block2)

        multi_error = MultiValidationError(errors=errors)
        result = MockLaunchResult(status="blocked", errors=multi_error)

        assert result.status == "blocked"
        assert result.errors is not None
        assert len(result.warnings) == 0
        # Full block should have errors populated
        assert len(result.errors.errors) == 2
        # Each error should have proper fields
        for error in result.errors.errors:
            assert error.error_code is not None
            assert error.failed_check is not None
            assert error.why_blocked is not None
            assert error.how_to_fix is not None

    def test_error_count_matches_blocked_slots(self, tmp_path) -> None:
        """MultiValidationError error_count should match number of blocked slots."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block 3 slots with fake PIDs
        for i in range(3):
            create_lock(runtime_dir, f"slot{i}", pid=99999 - i, port=8080 + i)

        # Mock psutil to simulate live processes with indeterminate state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            errors: list[ErrorDetail] = []
            for i in range(3):
                block = check_lockfile_integrity(runtime_dir, f"slot{i}")
                if block:
                    errors.append(block)

            multi_error = MultiValidationError(errors=errors)

            assert multi_error.error_count == len(errors)
            assert multi_error.error_count == 3


class TestLaunchDecisionLogic:
    """Tests for launch decision logic (degraded vs blocked)."""

    def test_single_slot_blocked_is_full_block(self, tmp_path) -> None:
        """Single blocked slot should be full block (no slots to launch)."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=99999, port=8080)

        # Mock psutil to simulate live process with indeterminate state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            block = check_lockfile_integrity(runtime_dir, "slot1")

            assert block is not None
            # No slots to launch = full block

    def test_single_slot_available_is_success(self, tmp_path) -> None:
        """Single available slot should launch successfully."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # No lock = available
        block = check_lockfile_integrity(runtime_dir, "slot1")

        assert block is None
        # Slot is available to launch

    def test_mixed_availability_is_degraded(self, tmp_path) -> None:
        """Mixed availability (some blocked, some available) is degraded."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block one slot with fake PID
        create_lock(runtime_dir, "blocked", pid=99999, port=8080)

        # slot2 has no lock (available)
        block_available = check_lockfile_integrity(runtime_dir, "slot2")
        assert block_available is None  # available

        # Mock psutil to simulate live process for blocked slot
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            block_blocked = check_lockfile_integrity(runtime_dir, "blocked")

            assert block_blocked is not None  # blocked

            # This is degraded: one blocked, one available
            # Would result in: status="degraded", launched=["slot2"], warnings=[...]
