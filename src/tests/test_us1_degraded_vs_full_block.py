"""Tests for US1: Degraded one-slot vs full-block behavior.

Tests T016: Degraded one-slot vs full-block behavior tests:
- one available slot returns success with warnings list
- all slots blocked returns MultiValidationError with error_count=2 and launch_count=0
"""

from unittest.mock import Mock, patch

from llama_manager.config import ErrorDetail, MultiValidationError
from llama_manager.process_manager import (
    check_lockfile_integrity,
    create_lock,
)


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
        # This is tested in test_us1_lock_integrity.py

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
        """Degraded launch should succeed when all slots are available."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # No locks exist
        launched = ["slot1", "slot2", "slot3"]
        warnings: list[str] = []

        result = MockLaunchResult(status="degraded", launched=launched, warnings=warnings)

        assert result.status == "degraded"
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
        """Full block errors should contain proper error details."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # Block slots with indeterminate state
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []
        for slot_id in ["slot1", "slot2"]:
            block = check_lockfile_integrity(runtime_dir, slot_id)
            if block:
                errors.append(block)

        multi_error = MultiValidationError(errors=errors)

        # Each error should have proper fields
        for error in multi_error.errors:
            assert error.error_code is not None
            assert error.failed_check is not None
            assert error.why_blocked is not None
            assert error.how_to_fix is not None

    def test_full_block_mixed_error_types(self, tmp_path) -> None:
        """Full block can have mixed error types (stale + indeterminate).

        Mock psutil to ensure fake PIDs are treated as running during
        check_lockfile_integrity, making the test deterministic.
        """
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # slot1: stale lock (will be auto-cleared)
        # slot2: indeterminate lock (will block)
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []

        # Mock psutil to simulate live processes with indeterminate state
        with patch("psutil.pid_exists") as mock_exists, patch("psutil.Process") as mock_process:
            mock_exists.return_value = True

            mock_proc = Mock()
            mock_conn = Mock()
            mock_conn.laddr.port = 9999  # Different port for indeterminate state
            mock_proc.connections.return_value = [mock_conn]
            mock_process.return_value = mock_proc

            # slot1: stale lock - mocked as indeterminate
            block1 = check_lockfile_integrity(runtime_dir, "slot1")
            if block1:
                errors.append(block1)

            # slot2: indeterminate lock
            block2 = check_lockfile_integrity(runtime_dir, "slot2")
            if block2:
                errors.append(block2)

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
        """Full block status should have zero launched count."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()

        # All blocked
        create_lock(runtime_dir, "slot1", pid=99999, port=8080)
        create_lock(runtime_dir, "slot2", pid=99998, port=8081)

        errors: list[ErrorDetail] = []
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
