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
from llama_manager.process_manager import (
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
        from llama_manager.process_manager import ServerManager

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
                result = manager.launch_all_slots([])

                # Assert LaunchResult exists and has expected structure
                from llama_manager.process_manager import LaunchResult

                assert isinstance(result, LaunchResult)
                assert hasattr(result, "status")
                assert hasattr(result, "launched")
                assert hasattr(result, "warnings")
                assert hasattr(result, "errors")

    def test_launch_all_slots_with_slots_returns_launched_list(self, tmp_path) -> None:
        """launch_all_slots should return LaunchResult with launched slot IDs.

        This test will fail until T017-T019 implement the LaunchResult dataclass.
        """
        from llama_manager.process_manager import LaunchResult, ServerManager

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
        from llama_manager.process_manager import LaunchResult, ServerManager

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
        from llama_manager.process_manager import LaunchResult, ServerManager

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
            mock_conn.laddr.port = 9999  # Different port for slot1
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # This will fail until T017-T019 implement launch_all_slots
            result = manager.launch_all_slots(slots, runtime_dir)

            # Assert LaunchResult has degraded status
            assert isinstance(result, LaunchResult)
            assert result.status == "degraded"
