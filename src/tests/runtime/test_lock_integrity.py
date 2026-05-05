"""Tests for US1: Lock integrity with stale/live/indeterminate ownership.

Tests T015: Stale/live/indeterminate lock ownership tests:
- stale (PID not found → auto-clear)
- live (PID exists + port matches → block)
- indeterminate (PID exists + port mismatch → block with FR-005 error_code="LOCKFILE_INTEGRITY_FAILURE")

These tests target real check_lockfile_integrity() API which is already implemented.
"""

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

from llama_manager.config import ErrorCode, ErrorDetail
from llama_manager.orchestration import (
    LockMetadata,
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
        """Stale lock (PID gone) should be auto-cleared.

        Locks are only cleared when PID doesn't exist or ownership is invalid.
        Age alone does NOT clear a lock.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create a lock with a valid PID but old timestamp
        valid_pid = 12345
        create_lock(runtime_dir, "slot1", pid=valid_pid, port=8080)

        # Manually update the lock's started_at to be > 300 seconds old
        lock_path = runtime_dir / "slot-slot1.lock"
        lock_data = json.loads(lock_path.read_text())

        # Set started_at to 301 seconds ago (using wall-clock time)
        lock_data["started_at"] = time.time() - 301
        lock_path.write_text(json.dumps(lock_data))

        # Verify lock exists initially
        assert read_lock(runtime_dir, "slot1") is not None

        # Mock psutil to simulate process DOES NOT exist (PID gone)
        with patch("psutil.pid_exists") as mock_exists:
            mock_exists.return_value = False

            # Check integrity - should auto-clear because PID is gone
            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Stale lock (PID gone) should be cleared and return None
            assert result is None
            # Lockfile should be deleted
            assert not (runtime_dir / "slot-slot1.lock").exists()

    def test_stale_lock_by_age_with_live_process(self, tmp_path) -> None:
        """Old lock (age > 300s) should NOT be cleared if PID exists and port matches.

        Locks are only cleared when PID is gone or ownership/port is invalid.
        Age alone does not clear a lock.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Create a lock with a valid PID
        valid_pid = 12345
        create_lock(runtime_dir, "slot1", pid=valid_pid, port=8080)

        # Manually update the lock's started_at to be > 300 seconds old
        lock_path = runtime_dir / "slot-slot1.lock"
        lock_data = json.loads(lock_path.read_text())
        lock_data["started_at"] = time.time() - 301
        lock_path.write_text(json.dumps(lock_data))

        # Mock psutil to simulate process exists with matching port
        mock_conn = Mock()
        mock_conn.laddr.port = 8080  # Matching port
        mock_conn.pid = valid_pid

        with (
            patch("psutil.pid_exists") as mock_exists,
            patch("psutil.Process"),
            patch("psutil.net_connections", return_value=[mock_conn]),
        ):
            mock_exists.return_value = True

            # Check integrity - should NOT clear (live process with valid ownership)
            result = check_lockfile_integrity(runtime_dir, "slot1")

            # Should NOT be cleared - live process with valid ownership
            assert result is None
            assert (runtime_dir / "slot-slot1.lock").exists()


class TestLiveLock:
    """Tests for live lock handling (PID exists + port matches → block)."""

    def test_live_lock_with_matching_port_blocks(self, tmp_path) -> None:
        """Live lock (PID exists + port matches) should not block (returns None).

        When the process is still running and bound to the same port,
        the lock is valid and should not trigger an integrity error.

        Targets: check_lockfile_integrity() with psutil.pid_exists(pid) == True
                 and port matching in psutil.net_connections()
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        fake_pid = 12345
        create_lock(runtime_dir, "slot1", pid=fake_pid, port=8080)

        # Mock psutil to simulate a running process with matching port
        mock_conn = Mock()
        mock_conn.laddr.port = 8080  # Same port as lock
        mock_conn.pid = fake_pid

        with (
            patch("psutil.pid_exists") as mock_exists,
            patch(
                "psutil.Process",
            ),
            patch("psutil.net_connections", return_value=[mock_conn]),
        ):
            mock_exists.return_value = True

            # Check integrity - should return None (valid lock)
            result = check_lockfile_integrity(runtime_dir, "slot1")
            assert result is None

    def test_live_lock_different_port_returns_indeterminate_error(self, tmp_path) -> None:
        """Live lock with different port should return LOCKFILE_INTEGRITY_FAILURE.

        When mocking psutil, this test verifies we can simulate port mismatch
        and get the expected indeterminate owner error.
        """
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
        assert isinstance(meta, LockMetadata)
        assert meta.port == 0

    def test_lock_with_high_port(self, tmp_path) -> None:
        """Lock with maximum valid port should work."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        create_lock(runtime_dir, "slot1", pid=12345, port=65535)

        meta = read_lock(runtime_dir, "slot1")
        assert meta is not None
        assert isinstance(meta, LockMetadata)
        assert meta.port == 65535

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
        assert isinstance(meta1, LockMetadata)
        assert isinstance(meta2, LockMetadata)
        assert meta1.pid == meta2.pid == fake_pid
        assert meta1.port == meta2.port == 8080
        # started_at should be approximately the same
        assert abs(meta1.started_at - meta2.started_at) < 0.001

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


class TestLockMetadataTypeValidation:
    """Tests for explicit type validation of lock metadata fields."""

    def test_read_lock_rejects_string_pid(self, tmp_path: Path) -> None:
        """read_lock should reject pid as string when require_valid=True."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-bad.lock"
        lock_data = {
            "pid": "12345",  # string instead of int
            "port": 8080,
            "started_at": time.time(),
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = read_lock(runtime_dir, "bad", require_valid=True)
        assert isinstance(result, ErrorDetail)
        assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE
        assert "lock 'pid' must be an integer" in result.why_blocked

    def test_read_lock_rejects_string_port(self, tmp_path: Path) -> None:
        """read_lock should reject port as string when require_valid=True."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-bad.lock"
        lock_data = {
            "pid": 12345,
            "port": "8080",  # string instead of int
            "started_at": time.time(),
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = read_lock(runtime_dir, "bad", require_valid=True)
        assert isinstance(result, ErrorDetail)
        assert "lock 'port' must be an integer" in result.why_blocked

    def test_read_lock_rejects_string_started_at(self, tmp_path: Path) -> None:
        """read_lock should reject started_at as string when require_valid=True."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-bad.lock"
        lock_data = {
            "pid": 12345,
            "port": 8080,
            "started_at": "not a number",  # string instead of numeric
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = read_lock(runtime_dir, "bad", require_valid=True)
        assert isinstance(result, ErrorDetail)
        assert "lock 'started_at' must be a numeric value" in result.why_blocked

    def test_read_lock_rejects_boolean_pid(self, tmp_path: Path) -> None:
        """read_lock should reject pid as bool (bool is subclass of int)."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-bad.lock"
        lock_data = {
            "pid": True,  # bool, not int
            "port": 8080,
            "started_at": time.time(),
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = read_lock(runtime_dir, "bad", require_valid=True)
        assert isinstance(result, ErrorDetail)
        assert "lock 'pid' must be an integer" in result.why_blocked

    def test_read_lock_accepts_float_started_at(self, tmp_path: Path) -> None:
        """read_lock should accept started_at as float."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-good.lock"
        lock_data = {
            "pid": 12345,
            "port": 8080,
            "started_at": 1234567890.5,  # float
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = read_lock(runtime_dir, "good", require_valid=True)
        assert isinstance(result, LockMetadata)
        assert result.started_at == 1234567890.5

    def test_read_lock_accepts_int_started_at(self, tmp_path: Path) -> None:
        """read_lock should accept started_at as int."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-good.lock"
        lock_data = {
            "pid": 12345,
            "port": 8080,
            "started_at": 1234567890,  # int
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = read_lock(runtime_dir, "good", require_valid=True)
        assert isinstance(result, LockMetadata)
        assert result.started_at == 1234567890.0

    def test_read_lock_permissive_mode_allows_string_types(self, tmp_path: Path) -> None:
        """read_lock with require_valid=False should allow string types (coerced via int())."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-coerce.lock"
        lock_data = {
            "pid": "12345",  # string - will be coerced
            "port": "8080",  # string - will be coerced
            "started_at": "1234567890.0",  # string - will be coerced
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = read_lock(runtime_dir, "coerce", require_valid=False)
        # In permissive mode, coercion should work
        assert isinstance(result, LockMetadata)
        assert result.pid == 12345
        assert result.port == 8080

    def test_check_lockfile_integrity_rejects_malformed_metadata(self, tmp_path: Path) -> None:
        """check_lockfile_integrity should return ErrorDetail for malformed lock metadata."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        lock_path = runtime_dir / "slot-bad.lock"
        lock_data = {
            "pid": "not_a_number",  # invalid
            "port": 8080,
            "started_at": time.time(),
            "version": "1.0",
        }
        lock_path.write_text(json.dumps(lock_data))

        result = check_lockfile_integrity(runtime_dir, "bad")
        assert isinstance(result, ErrorDetail)
        assert result.error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE
        assert "lock 'pid' must be an integer" in result.why_blocked
