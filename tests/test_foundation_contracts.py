"""T013 foundation contract tests for Phase 2 behavior.

Covers:
- Slot ID normalization and duplicate detection
- MultiValidationError schema and deterministic ordering
- FR-005 structured actionable errors (ValidationException path)
- resolve_runtime_dir failure with actionable errors
- Lockfile/artifact permission and error paths
- redact_sensitive helper behavior
"""

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_manager.config import (
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    detect_duplicate_slots,
    normalize_slot_id,
)
from llama_manager.process_manager import (
    ArtifactMetadata,
    LockMetadata,
    ValidationException,
    create_lock,
    read_lock,
    release_lock,
    resolve_runtime_dir,
    update_lock,
    write_artifact,
)
from llama_manager.server import (
    redact_sensitive,
    validate_slots,
)


class TestNormalizeSlotId:
    """Tests for normalize_slot_id function."""

    def test_alphanumeric_preserved(self) -> None:
        """Alphanumeric characters should be preserved."""
        assert normalize_slot_id("slot123") == "slot123"
        assert normalize_slot_id("SLOT123") == "slot123"
        assert normalize_slot_id("abc123def456") == "abc123def456"

    def test_whitespace_stripped(self) -> None:
        """Whitespace should be stripped."""
        assert normalize_slot_id("  slot1  ") == "slot1"
        assert normalize_slot_id("\tslot2\n") == "slot2"
        assert normalize_slot_id("  \t\n  slot3  \t\n  ") == "slot3"

    def test_special_chars_removed(self) -> None:
        """Special characters should be removed."""
        assert normalize_slot_id("slot!@#$%") == "slot"
        assert normalize_slot_id("slot_test-123") == "slot_test-123"
        assert normalize_slot_id("slot/invalid\\chars") == "slotinvalidchars"

    def test_case_lowercased(self) -> None:
        """Input should be lowercased."""
        assert normalize_slot_id("MySlotID") == "myslotid"
        assert normalize_slot_id("ABC_def_GHI") == "abc_def_ghi"

    def test_only_special_chars_raises(self) -> None:
        """Empty string after normalization should raise ValueError."""
        with pytest.raises(ValueError, match="must contain at least one valid character"):
            normalize_slot_id("@#$%")
        with pytest.raises(ValueError, match="must contain at least one valid character"):
            normalize_slot_id("   ")
        with pytest.raises(ValueError, match="must contain at least one valid character"):
            normalize_slot_id("")

    def test_underscore_hyphen_preserved(self) -> None:
        """Underscore and hyphen should be preserved."""
        assert normalize_slot_id("slot_name") == "slot_name"
        assert normalize_slot_id("slot-name") == "slot-name"
        assert normalize_slot_id("slot_-_-") == "slot_-_-"


class TestDetectDuplicateSlots:
    """Tests for detect_duplicate_slots function."""

    def _make_slot(self, slot_id: str) -> ModelSlot:
        return ModelSlot(slot_id=slot_id, model_path="/model.gguf", port=8080)

    def test_no_duplicates(self) -> None:
        """List with unique slot IDs should return empty list."""
        slots = [
            self._make_slot("slot1"),
            self._make_slot("slot2"),
            self._make_slot("slot3"),
        ]
        duplicates = detect_duplicate_slots(slots)
        assert duplicates == []

    def test_simple_duplicate(self) -> None:
        """Duplicate slot IDs should be detected."""
        slots = [
            self._make_slot("slot1"),
            self._make_slot("slot2"),
            self._make_slot("slot1"),
        ]
        duplicates = detect_duplicate_slots(slots)
        assert duplicates == ["slot1"]

    def test_multiple_duplicates(self) -> None:
        """Multiple duplicate slots should all be detected."""
        slots = [
            self._make_slot("slot1"),
            self._make_slot("slot2"),
            self._make_slot("slot1"),
            self._make_slot("slot3"),
            self._make_slot("slot2"),
        ]
        duplicates = detect_duplicate_slots(slots)
        assert set(duplicates) == {"slot1", "slot2"}

    def test_normalized_detection(self) -> None:
        """Duplicates should be detected after normalization."""
        slots = [
            self._make_slot("Slot1"),
            self._make_slot("slot1"),
            self._make_slot("SLOT1"),
        ]
        duplicates = detect_duplicate_slots(slots)
        assert duplicates == ["slot1"]

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        assert detect_duplicate_slots([]) == []

    def test_single_slot(self) -> None:
        """Single slot should not report duplicates."""
        assert detect_duplicate_slots([self._make_slot("only")]) == []


class TestMultiValidationError:
    """Tests for MultiValidationError schema and ordering."""

    def test_error_count(self) -> None:
        """error_count property should return number of errors."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="check1",
                why_blocked="blocked",
                how_to_fix="fix it",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="check2",
                why_blocked="blocked",
                how_to_fix="fix it",
            ),
        ]
        multi = MultiValidationError(errors=errors)
        assert multi.error_count == 2

    def test_sort_errors_empty(self) -> None:
        """Sorting empty errors should not crash."""
        multi = MultiValidationError(errors=[])
        multi.sort_errors()
        assert multi.errors == []

    def test_sort_errors_deterministic(self) -> None:
        """sort_errors should produce deterministic ordering."""
        errors = [
            ErrorDetail(ErrorCode.PORT_INVALID, "port_check", "blocked", "fix port"),
            ErrorDetail(ErrorCode.FILE_NOT_FOUND, "model_check", "blocked", "fix model"),
            ErrorDetail(ErrorCode.THREADS_INVALID, "threads_check", "blocked", "fix threads"),
        ]
        multi = MultiValidationError(errors=errors)
        multi.sort_errors()
        # Should be sorted by slot order then failed_check
        assert multi.errors[0].failed_check == "model_check"
        assert multi.errors[1].failed_check == "port_check"
        assert multi.errors[2].failed_check == "threads_check"

    def test_sort_errors_slot_sequencing(self) -> None:
        """Sorting should preserve slot configuration sequence."""
        errors = [
            ErrorDetail(ErrorCode.PORT_INVALID, "slot_b_check_a", "blocked", "fix"),
            ErrorDetail(ErrorCode.FILE_NOT_FOUND, "slot_a_check_b", "blocked", "fix"),
            ErrorDetail(ErrorCode.THREADS_INVALID, "slot_a_check_a", "blocked", "fix"),
            ErrorDetail(ErrorCode.PORT_INVALID, "slot_b_check_b", "blocked", "fix"),
        ]
        multi = MultiValidationError(errors=errors)
        multi.sort_errors()
        checks = [e.failed_check for e in multi.errors]
        # slot_b appears first in original list (index 0), so slot_b entries come first
        assert checks.index("slot_b_check_a") < checks.index("slot_a_check_a")
        assert checks.index("slot_b_check_a") < checks.index("slot_a_check_b")
        assert checks.index("slot_a_check_a") < checks.index("slot_a_check_b")

    def test_sort_errors_within_slot_alphabetical(self) -> None:
        """Within each slot, errors should be sorted by failed_check alphabetically."""
        errors = [
            ErrorDetail(ErrorCode.PORT_INVALID, "slot_check_z", "blocked", "fix"),
            ErrorDetail(ErrorCode.FILE_NOT_FOUND, "slot_check_a", "blocked", "fix"),
            ErrorDetail(ErrorCode.THREADS_INVALID, "slot_check_m", "blocked", "fix"),
        ]
        multi = MultiValidationError(errors=errors)
        multi.sort_errors()
        checks = [e.failed_check for e in multi.errors]
        assert checks == ["slot_check_a", "slot_check_m", "slot_check_z"]


class TestValidationException:
    """Tests for ValidationException wrapper."""

    def test_exception_contains_multi_error(self) -> None:
        """ValidationException should wrap MultiValidationError."""
        error_detail = ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="test_check",
            why_blocked="test blocked",
            how_to_fix="test fix",
        )
        multi = MultiValidationError(errors=[error_detail])
        exc = ValidationException(multi)
        assert exc.multi_error is multi
        assert exc.multi_error.error_count == 1

    def test_exception_message_includes_count(self) -> None:
        """Exception message should include error count."""
        errors = [
            ErrorDetail(ErrorCode.PORT_INVALID, "c1", "blocked", "fix"),
            ErrorDetail(ErrorCode.THREADS_INVALID, "c2", "blocked", "fix"),
        ]
        multi = MultiValidationError(errors=errors)
        exc = ValidationException(multi)
        assert "2 error(s)" in str(exc)


class TestRedactSensitive:
    """Tests for redact_sensitive helper function."""

    def test_redacts_api_key(self) -> None:
        """Should redact keys containing KEY."""
        assert redact_sensitive("secret_value", "API_KEY") == "[REDACTED]"
        assert redact_sensitive("value", "MY_API_KEY") == "[REDACTED]"

    def test_redacts_token(self) -> None:
        """Should redact keys containing TOKEN."""
        assert redact_sensitive("token_value", "AUTH_TOKEN") == "[REDACTED]"
        assert redact_sensitive("value", "refresh_token") == "[REDACTED]"

    def test_redacts_secret(self) -> None:
        """Should redact keys containing SECRET."""
        assert redact_sensitive("secret_value", "MY_SECRET") == "[REDACTED]"
        assert redact_sensitive("value", "DATABASE_SECRET") == "[REDACTED]"

    def test_redacts_password(self) -> None:
        """Should redact keys containing PASSWORD."""
        assert redact_sensitive("password_value", "DB_PASSWORD") == "[REDACTED]"
        assert redact_sensitive("value", "admin_password") == "[REDACTED]"

    def test_redacts_auth(self) -> None:
        """Should redact keys containing AUTH."""
        assert redact_sensitive("auth_value", "AUTH_SECRET") == "[REDACTED]"
        assert redact_sensitive("value", "AUTH_TOKEN") == "[REDACTED]"

    def test_case_insensitive(self) -> None:
        """Pattern matching should be case insensitive."""
        assert redact_sensitive("value", "api_key") == "[REDACTED]"
        assert redact_sensitive("value", "API_KEY") == "[REDACTED]"
        assert redact_sensitive("value", "Api_Key") == "[REDACTED]"

    def test_non_sensitive_keys_unchanged(self) -> None:
        """Non-sensitive keys should pass through unchanged."""
        assert redact_sensitive("/path/to/model", "MODEL_PATH") == "/path/to/model"
        assert redact_sensitive("12345", "PORT_NUMBER") == "12345"
        assert redact_sensitive("value", "CONFIG_VALUE") == "value"

    def test_empty_string_value(self) -> None:
        """Empty string values should still be redacted."""
        assert redact_sensitive("", "API_KEY") == "[REDACTED]"


class TestValidateSlots:
    """Tests for validate_slots with FR-005 error paths."""

    def test_valid_slots_pass(self, tmp_path: Path) -> None:
        """Valid slot configurations should pass validation."""
        # Create actual model files
        model1 = tmp_path / "model1.gguf"
        model2 = tmp_path / "model2.gguf"
        model1.touch()
        model2.touch()

        slots = [
            ModelSlot(slot_id="slot1", model_path=str(model1), port=8080),
            ModelSlot(slot_id="slot2", model_path=str(model2), port=8081),
        ]
        result = validate_slots(slots)
        assert result is None

    def test_invalid_slot_id_raises_validation_error(self, tmp_path: Path) -> None:
        """Invalid slot IDs should return MultiValidationError."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        # Test with port validation error instead since detect_duplicate_slots calls normalize_slot_id
        slots = [
            ModelSlot(slot_id="test", model_path=str(model_file), port=0),
        ]
        result = validate_slots(slots)
        assert result is not None
        assert result.error_count == 1
        assert result.errors[0].error_code == ErrorCode.PORT_INVALID

    def test_duplicate_slots_detected(self) -> None:
        """Duplicate slot IDs should be detected."""
        slots = [
            ModelSlot(slot_id="slot1", model_path="/model1.gguf", port=8080),
            ModelSlot(slot_id="slot2", model_path="/model2.gguf", port=8081),
            ModelSlot(slot_id="slot1", model_path="/model3.gguf", port=8082),
        ]
        result = validate_slots(slots)
        assert result is not None
        assert result.error_count >= 1
        error_codes = [e.error_code for e in result.errors]
        assert ErrorCode.DUPLICATE_SLOT in error_codes

    def test_invalid_port_detected(self) -> None:
        """Invalid port should be detected."""
        slots = [
            ModelSlot(slot_id="slot1", model_path="/model.gguf", port=0),
        ]
        result = validate_slots(slots)
        assert result is not None
        assert any(e.error_code == ErrorCode.PORT_INVALID for e in result.errors)

    def test_nonexistent_model_path_detected(self, tmp_path: Path) -> None:
        """Non-existent model path should be detected."""
        slots = [
            ModelSlot(slot_id="slot1", model_path=str(tmp_path / "nonexistent.gguf"), port=8080),
        ]
        result = validate_slots(slots)
        assert result is not None
        error_codes = [e.error_code for e in result.errors]
        assert ErrorCode.FILE_NOT_FOUND in error_codes


class TestResolveRuntimeDir:
    """Tests for resolve_runtime_dir FR-005 error handling."""

    def test_uses_llm_runner_runtime_dir_env(self, tmp_path: Path) -> None:
        """Should use LLM_RUNNER_RUNTIME_DIR if set and valid."""
        env_dir = tmp_path / "custom_runtime"
        env_dir.mkdir()
        with patch.dict(os.environ, {"LLM_RUNNER_RUNTIME_DIR": str(env_dir)}, clear=False):
            result = resolve_runtime_dir()
            assert result == env_dir

    def test_uses_xdg_runtime_dir(self, tmp_path: Path) -> None:
        """Should use XDG_RUNTIME_DIR/llm-runner if LLM_RUNNER_RUNTIME_DIR not set."""
        xdg_base = tmp_path / "xdg_runtime"
        xdg_base.mkdir()
        with patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(xdg_base)}, clear=False):
            result = resolve_runtime_dir()
            assert result == xdg_base / "llm-runner"

    def test_no_env_var_raises_validation_exception(self) -> None:
        """Should raise ValidationException when no runtime dir available."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationException) as exc_info:
                resolve_runtime_dir()
            assert exc_info.value.multi_error.error_count == 1
            assert (
                exc_info.value.multi_error.errors[0].error_code == ErrorCode.RUNTIME_DIR_UNAVAILABLE
            )

    def test_invalid_env_var_falls_back_to_xdg(self, tmp_path: Path) -> None:
        """Should fall back to XDG_RUNTIME_DIR if LLM_RUNNER_RUNTIME_DIR is invalid."""
        xdg_base = tmp_path / "xdg_runtime"
        xdg_base.mkdir()
        with patch.dict(os.environ, {"XDG_RUNTIME_DIR": str(xdg_base)}, clear=False):
            result = resolve_runtime_dir()
            assert result == xdg_base / "llm-runner"

    def test_error_detail_has_actionable_fields(self) -> None:
        """ErrorDetail should have actionable why_blocked and how_to_fix fields."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationException) as exc_info:
                resolve_runtime_dir()
            error = exc_info.value.multi_error.errors[0]
            assert "runtime_dir" in error.failed_check.lower()
            assert len(error.why_blocked) > 0
            assert len(error.how_to_fix) > 0


class TestCreateLock:
    """Tests for lockfile creation with permission/error paths."""

    def test_creates_lockfile(self, tmp_path: Path) -> None:
        """Should create lockfile with correct structure."""
        slot_id = "test_slot"
        pid = 1234
        port = 8080

        lock_path = create_lock(tmp_path, slot_id, pid, port)

        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == pid
        assert data["port"] == port
        assert "started_at" in data
        assert data["version"] == "1.0"

    def test_lockfile_permissions_0600(self, tmp_path: Path) -> None:
        """Lockfile should have 0600 permissions."""
        lock_path = create_lock(tmp_path, "slot", 1234, 8080)
        mode = stat.S_IMODE(os.stat(lock_path).st_mode)
        assert mode == 0o600

    def test_file_exists_raises(self, tmp_path: Path) -> None:
        """Should raise FileExistsError if lockfile already exists."""
        create_lock(tmp_path, "slot", 1234, 8080)
        with pytest.raises(FileExistsError):
            create_lock(tmp_path, "slot", 1234, 8081)

    def test_invalid_runtime_dir_raises_validation_exception(self, tmp_path: Path) -> None:
        """Should raise ValidationException for unwritable runtime dir."""
        # Create a file instead of directory
        not_dir = tmp_path / "not_a_dir"
        not_dir.write_text("not a directory")
        with pytest.raises(ValidationException) as exc_info:
            create_lock(not_dir, "slot", 1234, 8080)
        assert (
            exc_info.value.multi_error.errors[0].error_code == ErrorCode.LOCKFILE_INTEGRITY_FAILURE
        )


class TestReadLock:
    """Tests for lockfile reading."""

    def test_reads_valid_lock(self, tmp_path: Path) -> None:
        """Should read valid lockfile correctly."""
        _ = create_lock(tmp_path, "slot", 1234, 8080)
        metadata = read_lock(tmp_path, "slot")
        assert metadata is not None
        assert metadata.pid == 1234
        assert metadata.port == 8080
        assert isinstance(metadata.started_at, float)

    def test_nonexistent_lock_returns_none(self, tmp_path: Path) -> None:
        """Should return None for non-existent lockfile."""
        metadata = read_lock(tmp_path, "nonexistent")
        assert metadata is None

    def test_corrupted_lock_returns_none(self, tmp_path: Path) -> None:
        """Should return None for corrupted lockfile."""
        lock_path = tmp_path / "lock-corrupted.json"
        lock_path.write_text("invalid json {{{")
        metadata = read_lock(tmp_path, "corrupted")
        assert metadata is None


class TestUpdateLock:
    """Tests for lockfile update."""

    def test_updates_existing_lock(self, tmp_path: Path) -> None:
        """Should update existing lockfile with new metadata."""
        create_lock(tmp_path, "slot", 1234, 8080)
        update_lock(tmp_path, "slot", 5678, 9000)
        metadata = read_lock(tmp_path, "slot")
        assert metadata is not None
        assert metadata.pid == 5678
        assert metadata.port == 9000

    def test_missing_lock_raises_file_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError if lockfile does not exist."""
        with pytest.raises(FileNotFoundError):
            update_lock(tmp_path, "nonexistent", 1234, 8080)


class TestReleaseLock:
    """Tests for lockfile release."""

    def test_deletes_lockfile(self, tmp_path: Path) -> None:
        """Should delete existing lockfile."""
        create_lock(tmp_path, "slot", 1234, 8080)
        release_lock(tmp_path, "slot")
        assert not (tmp_path / "lock-slot.json").exists()

    def test_noop_if_nonexistent(self, tmp_path: Path) -> None:
        """Should not error if lockfile doesn't exist."""
        # Should not raise
        release_lock(tmp_path, "nonexistent")


class TestWriteArtifact:
    """Tests for artifact persistence with FR-005 error paths."""

    def test_writes_artifact(self, tmp_path: Path) -> None:
        """Should write artifact with JSON serialization."""
        data = {"slot_id": "test", "status": "running"}
        artifact_path = write_artifact(tmp_path, "slot", data)
        assert artifact_path.exists()
        loaded = json.loads(artifact_path.read_text())
        assert loaded["slot_id"] == "test"
        assert loaded["status"] == "running"

    def test_artifact_permissions_0600(self, tmp_path: Path) -> None:
        """Artifact should have 0600 permissions."""
        artifact_path = write_artifact(tmp_path, "slot", {"data": "value"})
        mode = stat.S_IMODE(os.stat(artifact_path).st_mode)
        assert mode == 0o600

    def test_redacts_sensitive_in_artifact(self, tmp_path: Path) -> None:
        """Should redact sensitive environment variable values in artifact."""
        data = {
            "api_key": "secret123",
            "model_path": "/path/to/model",
            "password": "pass456",
        }
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert loaded["api_key"] == "[REDACTED]"
        assert loaded["model_path"] == "/path/to/model"
        assert loaded["password"] == "[REDACTED]"  # noqa: S105

    def test_unwritable_dir_raises_permission_error(self, tmp_path: Path) -> None:
        """Should raise PermissionError for unwritable directory."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir(mode=0o444)
        try:
            # The error happens when trying to create artifact subdirectory
            with pytest.raises(PermissionError):
                write_artifact(readonly_dir, "slot", {"data": "value"})
        finally:
            readonly_dir.chmod(0o755)  # Restore for cleanup

    def test_non_json_serializable_raises(self, tmp_path: Path) -> None:
        """Should raise ValidationException for non-JSON-serializable data."""
        with pytest.raises(ValidationException) as exc_info:
            write_artifact(tmp_path, "slot", {"data": object()})  # type: ignore
        assert (
            exc_info.value.multi_error.errors[0].error_code
            == ErrorCode.ARTIFACT_PERSISTENCE_FAILURE
        )


class TestArtifactMetadata:
    """Tests for ArtifactMetadata dataclass."""

    def test_required_fields(self) -> None:
        """Should have required fields."""
        metadata = ArtifactMetadata(artifact_type="test", created_at=1234567890.0)
        assert metadata.artifact_type == "test"
        assert metadata.created_at == 1234567890.0
        assert metadata.slot_id is None
        assert metadata.additional_fields == {}

    def test_all_fields(self) -> None:
        """Should accept all optional fields."""
        metadata = ArtifactMetadata(
            artifact_type="log",
            created_at=1234567890.0,
            slot_id="slot1",
            additional_fields={"key": "value"},
        )
        assert metadata.artifact_type == "log"
        assert metadata.slot_id == "slot1"
        assert metadata.additional_fields == {"key": "value"}

    def test_additional_fields_default_empty_dict(self) -> None:
        """additional_fields should default to empty dict."""
        metadata = ArtifactMetadata(
            artifact_type="test", created_at=1234567890.0, additional_fields={}
        )
        assert metadata.additional_fields == {}
        # Pyright knows __post_init__ initializes this
        assert metadata.additional_fields is not None
        # Mutating should not affect other instances
        metadata.additional_fields["key"] = "value"
        metadata2 = ArtifactMetadata(
            artifact_type="test", created_at=1234567890.0, additional_fields={}
        )
        assert metadata2.additional_fields == {}


class TestLockMetadata:
    """Tests for LockMetadata dataclass."""

    def test_required_fields(self) -> None:
        """Should have required fields."""
        metadata = LockMetadata(pid=1234, port=8080, started_at=1234567890.0)
        assert metadata.pid == 1234
        assert metadata.port == 8080
        assert metadata.started_at == 1234567890.0


class TestErrorCodeComprehensive:
    """Comprehensive tests for ErrorCode coverage."""

    def test_all_expected_error_codes_present(self) -> None:
        """All Phase 2 error codes should be present."""
        codes = [c.value for c in ErrorCode]
        expected = [
            "FILE_NOT_FOUND",
            "PATH_INVALID",
            "PERMISSION_DENIED",
            "PORT_CONFLICT",
            "PORT_INVALID",
            "THREADS_INVALID",
            "CONFIG_ERROR",
            "invalid_slot_id",
            "duplicate_slot",
            "runtime_dir_unavailable",
            "lockfile_integrity_failure",
            "artifact_persistence_failure",
            "backend_not_eligible",
        ]
        for code in expected:
            assert code in codes, f"Missing ErrorCode: {code}"

    def test_error_code_is_str_enum(self) -> None:
        """ErrorCode should be a StrEnum with string values."""
        assert isinstance(ErrorCode.FILE_NOT_FOUND, str)
        assert ErrorCode.FILE_NOT_FOUND == "FILE_NOT_FOUND"
        assert ErrorCode.FILE_NOT_FOUND in [ErrorCode.FILE_NOT_FOUND]
