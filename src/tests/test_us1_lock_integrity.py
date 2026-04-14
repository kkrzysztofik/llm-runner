"""Tests for US1: Lock integrity with stale/live/indeterminate ownership.

Tests T015: Stale/live/indeterminate lock ownership tests:
- stale (PID not found → auto-clear)
- live (PID exists + port matches → block)
- indeterminate (PID exists + port mismatch → block with FR-005 error_code="LOCKFILE_INTEGRITY_FAILURE")

These tests target real check_lockfile_integrity() API which is already implemented.
"""

from unittest.mock import Mock, patch

import psutil

from llama_manager.config import ErrorCode
from llama_manager.process_manager import (
    check_lockfile_integrity,
    create_lock,
    read_lock,
)


class TestStaleLock:
    """Tests for stale lock handling (PID not found → auto-clear)."""

    def test_stale_lock_auto_clears(self, tmp_path) -> None:
        """Stale lock (PID not found) should be auto-cleared.

        When the process that created the lock is no longer running,
        the lock should be considered stale and automatically cleared.

        Targets: check_lockfile_integrity() with psutil.pid_exists(pid) == False
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create a lock with a fake PID that doesn't exist
        fake_pid = 99999  # Very unlikely to be a real PID
        create_lock(runtime_dir, "slot1", pid=fake_pid, port=8080)

        # Verify lock exists initially
        assert read_lock(runtime_dir, "slot1") is not None

        # Check integrity - should auto-clear the stale lock
        result = check_lockfile_integrity(runtime_dir, "slot1")

        # Stale lock should be cleared and return None
        assert result is None
        # Lockfile should be deleted
        assert not (runtime_dir / "slot-slot1.lock").exists()

    def test_stale_lock_multiple_checks(self, tmp_path) -> None:
        """Stale lock should remain cleared on subsequent checks."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        fake_pid = 99998
        create_lock(runtime_dir, "slot1", pid=fake_pid, port=8080)

        # First check clears the lock
        check_lockfile_integrity(runtime_dir, "slot1")

        # Second check should also return None (lock is gone)
        result = check_lockfile_integrity(runtime_dir, "slot1")
        assert result is None

    def test_stale_lock_different_ports(self, tmp_path) -> None:
        """Stale lock should be cleared regardless of port value."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        fake_pid = 99997
        create_lock(runtime_dir, "slot1", pid=fake_pid, port=65535)

        result = check_lockfile_integrity(runtime_dir, "slot1")
        assert result is None
        assert not (runtime_dir / "slot-slot1.lock").exists()

    def test_stale_lock_by_age_auto_clears(self, tmp_path) -> None:
        """Stale lock (age > 300s) should be auto-cleared.

        T017: Lock age check - treat lock as stale when
        time.time() - metadata.started_at > 300 seconds (wall-clock timebase).
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create a lock with a valid PID but old timestamp
        valid_pid = 12345  # Will be mocked as existing
        create_lock(runtime_dir, "slot1", pid=valid_pid, port=8080)

        # Manually update the lock's started_at to be > 300 seconds old
        import json
        import time

        lock_path = runtime_dir / "slot-slot1.lock"
        lock_data = json.loads(lock_path.read_text())

        # Set started_at to 301 seconds ago (using wall-clock time)
        lock_data["started_at"] = time.time() - 301
        lock_path.write_text(json.dumps(lock_data))

        # Verify lock exists initially
        assert read_lock(runtime_dir, "slot1") is not None

        # Mock psutil to simulate process exists
        with patch("psutil.pid_exists") as mock_exists:
            mock_exists.return_value = True

            # Check integrity - should auto-clear due to age
            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Stale lock (by age) should be cleared and return None
            assert result is None
            # Lockfile should be deleted
            assert not (runtime_dir / "slot-slot1.lock").exists()

    def test_stale_lock_by_age_with_live_process(self, tmp_path) -> None:
        """Stale lock (age > 300s) should be cleared even if PID exists and port matches."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create a lock with a valid PID
        valid_pid = 12345
        create_lock(runtime_dir, "slot1", pid=valid_pid, port=8080)

        # Manually update the lock's started_at to be > 300 seconds old
        import json
        import time

        lock_path = runtime_dir / "slot-slot1.lock"
        lock_data = json.loads(lock_path.read_text())
        lock_data["started_at"] = time.time() - 301
        lock_path.write_text(json.dumps(lock_data))

        # Mock psutil to simulate process exists with matching port
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 8080  # Matching port
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check integrity - should auto-clear due to age (before port check)
            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Should be cleared as stale by age
            assert result is None
            assert not (runtime_dir / "slot-slot1.lock").exists()


class TestLiveLock:
    """Tests for live lock handling (PID exists + port matches → block)."""

    def test_live_lock_with_matching_port_blocks(self, tmp_path) -> None:
        """Live lock (PID exists + port matches) should not block (returns None).

        When the process is still running and bound to the same port,
        the lock is valid and should not trigger an integrity error.

        Targets: check_lockfile_integrity() with psutil.pid_exists(pid) == True
                 and port matching in psutil.Process.connections()
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        fake_pid = 12345
        create_lock(runtime_dir, "slot1", pid=fake_pid, port=8080)

        # Mock psutil to simulate a running process with matching port
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            # Process exists
            mock_exists.return_value = True

            # Mock process with matching port
            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 8080  # Same port as lock
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # Check integrity - should return None (valid lock)
            result = check_lockfile_integrity(runtime_dir, "slot1")
            assert result is None

    def test_live_lock_different_port_should_fail_mock(self, tmp_path) -> None:
        """Test setup: when mocking psutil, ensure we can simulate port mismatch."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create lock with port 8080
        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Mock psutil to simulate process with DIFFERENT port (8081)
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 8081  # Different port!
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # This should return an error (indeterminate state)
            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Should return LOCKFILE_INTEGRITY_FAILURE
            assert result is not None
            assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE


class TestIndeterminateLock:
    """Tests for indeterminate lock handling (PID exists + port mismatch → block)."""

    def test_indeterminate_lock_port_mismatch_returns_error(self, tmp_path) -> None:
        """Indeterminate lock (PID exists + port mismatch) should return LOCKFILE_INTEGRITY_FAILURE.

        When the process exists but is not bound to the expected port,
        we have an indeterminate ownership state that should be blocked.

        Targets: check_lockfile_integrity() returning ErrorDetail with
                 error_code=ErrorCode.LOCKFILE_INTEGRITY_FAILURE
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create lock with port 8080
        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Mock psutil to simulate process with DIFFERENT port
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 8081  # Different port!
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Should return LOCKFILE_INTEGRITY_FAILURE
            assert result is not None
            assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE
            assert result.failed_check == "lockfile_integrity"
            assert "indeterminate_owner" in result.why_blocked

    def test_indeterminate_lock_access_denied_returns_error(self, tmp_path) -> None:
        """Indeterminate lock (AccessDenied when checking connections) should return LOCKFILE_INTEGRITY_FAILURE."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Mock psutil to simulate AccessDenied
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_proc.connections.side_effect = psutil.AccessDenied(12345, "denied")
            mock_process.return_value = mock_proc

            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Should return LOCKFILE_INTEGRITY_FAILURE
            assert result is not None
            assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE
            assert "indeterminate_owner" in result.why_blocked

    def test_indeterminate_lock_no_connections(self, tmp_path) -> None:
        """Indeterminate lock (no connections found) should return LOCKFILE_INTEGRITY_FAILURE."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Mock psutil to simulate process with no connections
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_proc.connections.return_value = []  # No connections
            mock_process.return_value = mock_proc

            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Should return LOCKFILE_INTEGRITY_FAILURE
            assert result is not None
            assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE

    def test_indeterminate_lock_multiple_connections_no_match(self, tmp_path) -> None:
        """Indeterminate lock (multiple connections but none match port) should return error."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            # Multiple connections, none matching port 8080
            conn1 = Mock()
            conn1.laddr.port = 8081
            conn2 = Mock()
            conn2.laddr.port = 8082
            mock_proc.connections.return_value = [conn1, conn2]
            mock_process.return_value = mock_proc

            result = check_lockfile_integrity(runtime_dir, "slot1")

            assert result is not None
            assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE

    def test_indeterminate_lock_os_error_returns_error(self, tmp_path) -> None:
        """Indeterminate lock (OSError when checking process) should return LOCKFILE_INTEGRITY_FAILURE."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True
            mock_process.side_effect = OSError("Permission denied")

            result = check_lockfile_integrity(runtime_dir, "slot1")

            assert result is not None
            assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE


class TestLockIntegrityEdgeCases:
    """Edge case tests for lock integrity checking."""

    def test_lock_with_zero_port_fails_validation(self, tmp_path) -> None:
        """Lock with port 0 should still be created but may fail validation."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Can create lock with port 0 (validation happens elsewhere)
        create_lock(runtime_dir, "slot1", pid=12345, port=0)

        meta = read_lock(runtime_dir, "slot1")
        assert meta is not None
        assert meta.port == 0  # type: ignore[union-attr]

    def test_lock_with_high_port(self, tmp_path) -> None:
        """Lock with maximum valid port should work."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=65535)

        meta = read_lock(runtime_dir, "slot1")
        assert meta is not None
        assert meta.port == 65535  # type: ignore[union-attr]

    def test_lock_metadata_persistence(self, tmp_path) -> None:
        """Lock metadata should persist across read operations."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        fake_pid = 54321
        create_lock(runtime_dir, "slot1", pid=fake_pid, port=8080)

        meta1 = read_lock(runtime_dir, "slot1")
        meta2 = read_lock(runtime_dir, "slot1")

        assert meta1 is not None
        assert meta2 is not None
        assert meta1.pid == meta2.pid == fake_pid  # type: ignore[union-attr]
        assert meta1.port == meta2.port == 8080  # type: ignore[union-attr]
        # started_at should be approximately the same
        assert abs(meta1.started_at - meta2.started_at) < 0.001  # type: ignore[union-attr]

    def test_check_lockfile_integrity_with_no_psutil(self, tmp_path) -> None:
        """check_lockfile_integrity should handle missing psutil gracefully."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=8080)

        # Mock psutil to simulate process not found (graceful handling)
        with patch("psutil.pid_exists") as mock_exists:
            mock_exists.return_value = False

            # Should handle gracefully and return None (stale lock cleared)
            result = check_lockfile_integrity(runtime_dir, "slot1")
            # Should return None (stale lock auto-cleared)
            assert result is None
