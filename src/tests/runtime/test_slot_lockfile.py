"""Tests for slot_lockfile module."""

from pathlib import Path
from unittest.mock import patch

from llama_manager.config import ErrorDetail
from llama_manager.config.enums import ErrorCode


class TestAcquireSlotLock:
    """Tests for acquire_slot_lock."""

    def test_acquire_creates_lock(self, tmp_path: Path) -> None:
        """Should create lockfile and return path."""
        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            from llama_manager.orchestration.slot_lockfile import acquire_slot_lock

            result = acquire_slot_lock("test-slot", 8080, server_pid=12345)

            assert result == str(tmp_path / "slot-test-slot.lock")
            assert (tmp_path / "slot-test-slot.lock").exists()

    def test_acquire_uses_current_pid_when_none(self, tmp_path: Path) -> None:
        """Should use current PID when server_pid is None."""
        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            from llama_manager.orchestration.slot_lockfile import acquire_slot_lock

            acquire_slot_lock("test-slot", 8080)


class TestReleaseSlotLock:
    """Tests for release_slot_lock."""

    def test_release_removes_lock(self, tmp_path: Path) -> None:
        """Should remove lockfile."""
        lockfile = tmp_path / "slot-test-slot.lock"
        lockfile.write_text('{"pid": 12345, "port": 8080}')

        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            from llama_manager.orchestration.slot_lockfile import release_slot_lock

            release_slot_lock("test-slot")

            assert not lockfile.exists()


class TestCheckLockStale:
    """Tests for check_lock_stale."""

    def test_no_lockfile_returns_false(self, tmp_path: Path) -> None:
        """Should return False when no lockfile exists."""
        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            from llama_manager.orchestration.slot_lockfile import check_lock_stale

            result = check_lock_stale("test-slot")

            assert result is False

    def test_failed_ownership_returns_false(self, tmp_path: Path) -> None:
        """Should return False when ownership verification fails."""
        lockfile = tmp_path / "slot-test-slot.lock"
        lockfile.write_text('{"pid": 12345, "port": 8080, "started_at": 1234567890}')

        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            with patch(
                "llama_manager.orchestration.slot_lockfile.verify_shutdown_ownership"
            ) as mock_verify:
                mock_verify.return_value = False
                from llama_manager.orchestration.slot_lockfile import shutdown_slot

                result = shutdown_slot("test-slot")

                assert result is False

    def test_oserror_on_kill_returns_true(self, tmp_path: Path) -> None:
        """Should return True when os.kill raises OSError."""
        lockfile = tmp_path / "slot-test-slot.lock"
        lockfile.write_text('{"pid": 12345, "port": 8080, "started_at": 1234567890}')

        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            with patch(
                "llama_manager.orchestration.slot_lockfile.verify_shutdown_ownership"
            ) as mock_verify:
                with patch("llama_manager.orchestration.slot_lockfile.os.kill") as mock_kill:
                    mock_verify.return_value = True
                    mock_kill.side_effect = OSError("process gone")
                    from llama_manager.orchestration.slot_lockfile import shutdown_slot

                    result = shutdown_slot("test-slot")

                    assert result is True

    def test_shutdown_with_valid_lock_exits_quickly(self, tmp_path: Path) -> None:
        """Should return True when process exits during SIGTERM wait."""
        lockfile = tmp_path / "slot-test-slot.lock"
        lockfile.write_text('{"pid": 12345, "port": 8080, "started_at": 1234567890}')

        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            with (
                patch(
                    "llama_manager.orchestration.slot_lockfile.verify_shutdown_ownership"
                ) as mock_verify,
                patch("llama_manager.orchestration.slot_lockfile.os.kill"),
                patch("llama_manager.orchestration.slot_lockfile.psutil.pid_exists") as mock_exists,
            ):
                mock_verify.return_value = True
                mock_exists.return_value = False
                from llama_manager.orchestration.slot_lockfile import shutdown_slot

                result = shutdown_slot("test-slot")

                assert result is True

    def test_shutdown_returns_false_after_timeout(self, tmp_path: Path) -> None:
        """Should return False when process survives SIGKILL wait."""
        lockfile = tmp_path / "slot-test-slot.lock"
        lockfile.write_text('{"pid": 12345, "port": 8080, "started_at": 1234567890}')

        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            with (
                patch(
                    "llama_manager.orchestration.slot_lockfile.verify_shutdown_ownership"
                ) as mock_verify,
                patch("llama_manager.orchestration.slot_lockfile.os.kill"),
                patch("llama_manager.orchestration.slot_lockfile.psutil.pid_exists") as mock_exists,
                patch("llama_manager.orchestration.slot_lockfile.time.monotonic") as mock_time,
                patch("llama_manager.orchestration.slot_lockfile.time.sleep"),
            ):
                mock_verify.return_value = True
                mock_exists.return_value = True
                mock_time.side_effect = [
                    0,
                    0.1,
                    10.1,
                    10.1,
                    10.2,
                    15.2,
                ]
                from llama_manager.orchestration.slot_lockfile import (
                    shutdown_slot,
                )

                result = shutdown_slot("test-slot", timeout=10.0)

                assert result is False


class TestCheckLockStaleErrorDetail:
    """Tests for check_lock_stale with ErrorDetail."""

    def test_error_detail_returns_false(self, tmp_path: Path) -> None:
        """Should return False when read_lock returns ErrorDetail."""
        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            with patch("llama_manager.orchestration.slot_lockfile.read_lock") as mock_read:
                mock_read.return_value = ErrorDetail(
                    error_code=ErrorCode.CONFIG_ERROR, why_blocked="bad lock"
                )
                from llama_manager.orchestration.slot_lockfile import check_lock_stale

                result = check_lock_stale("test-slot")

                assert result is False


class TestShutdownSlotErrorDetail:
    """Tests for shutdown_slot with ErrorDetail."""

    def test_error_detail_returns_true(self, tmp_path: Path) -> None:
        """Should return True when read_lock returns ErrorDetail."""
        with patch("llama_manager.orchestration.slot_lockfile.resolve_runtime_dir") as mock_resolve:
            mock_resolve.return_value = tmp_path
            with patch("llama_manager.orchestration.slot_lockfile.read_lock") as mock_read:
                mock_read.return_value = ErrorDetail(
                    error_code=ErrorCode.CONFIG_ERROR, why_blocked="bad lock"
                )
                from llama_manager.orchestration.slot_lockfile import shutdown_slot

                result = shutdown_slot("test-slot")

                assert result is True
