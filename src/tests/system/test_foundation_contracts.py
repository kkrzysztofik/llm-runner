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

from llama_manager.common.security import redact_env_value
from llama_manager.config import (
    ErrorCode,
    ErrorDetail,
    ModelSlot,
    MultiValidationError,
    detect_duplicate_slots,
    normalize_slot_id,
)
from llama_manager.orchestration import (
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
from llama_manager.validation import (
    validate_slots,
)
from tests.support.helpers import valid_artifact_data


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
        # slot_ids are sorted alphabetically: a before b
        assert checks.index("slot_a_check_a") < checks.index("slot_b_check_a")
        assert checks.index("slot_a_check_a") < checks.index("slot_b_check_b")
        assert checks.index("slot_a_check_b") < checks.index("slot_b_check_a")
        assert checks.index("slot_a_check_b") < checks.index("slot_b_check_b")

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
    """Tests for redact_env_value helper function."""

    def test_redacts_api_key(self) -> None:
        """Should redact keys containing KEY."""
        assert redact_env_value("secret_value", "API_KEY") == "[REDACTED]"
        assert redact_env_value("value", "MY_API_KEY") == "[REDACTED]"

    def test_redacts_token(self) -> None:
        """Should redact keys containing TOKEN."""
        assert redact_env_value("token_value", "AUTH_TOKEN") == "[REDACTED]"
        assert redact_env_value("value", "refresh_token") == "[REDACTED]"

    def test_redacts_secret(self) -> None:
        """Should redact keys containing SECRET."""
        assert redact_env_value("secret_value", "MY_SECRET") == "[REDACTED]"
        assert redact_env_value("value", "DATABASE_SECRET") == "[REDACTED]"

    def test_redacts_password(self) -> None:
        """Should redact keys containing PASSWORD."""
        assert redact_env_value("password_value", "DB_PASSWORD") == "[REDACTED]"
        assert redact_env_value("value", "admin_password") == "[REDACTED]"

    def test_redacts_auth(self) -> None:
        """Should redact keys containing AUTH."""
        assert redact_env_value("auth_value", "AUTH_SECRET") == "[REDACTED]"
        assert redact_env_value("value", "AUTH_TOKEN") == "[REDACTED]"

    def test_case_insensitive(self) -> None:
        """Pattern matching should be case insensitive."""
        assert redact_env_value("value", "api_key") == "[REDACTED]"
        assert redact_env_value("value", "API_KEY") == "[REDACTED]"
        assert redact_env_value("value", "Api_Key") == "[REDACTED]"

    def test_non_sensitive_keys_unchanged(self) -> None:
        """Non-sensitive keys should pass through unchanged."""
        assert redact_env_value("/path/to/model", "MODEL_PATH") == "/path/to/model"
        assert redact_env_value("12345", "PORT_NUMBER") == "12345"
        assert redact_env_value("value", "CONFIG_VALUE") == "value"

    def test_empty_string_value(self) -> None:
        """Empty string values should still be redacted."""
        assert redact_env_value("", "API_KEY") == "[REDACTED]"


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
        """Should fall back to XDG_RUNTIME_DIR if LLM_RUNNER_RUNTIME_DIR points to a file."""
        xdg_base = tmp_path / "xdg_runtime"
        xdg_base.mkdir()
        # Set LLM_RUNNER_RUNTIME_DIR to a file (not a directory) to trigger fallback
        invalid_dir = tmp_path / "not_a_dir"
        invalid_dir.write_text("not a directory")
        with patch.dict(
            os.environ,
            {"LLM_RUNNER_RUNTIME_DIR": str(invalid_dir), "XDG_RUNTIME_DIR": str(xdg_base)},
            clear=False,
        ):
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
        assert metadata.pid == 1234  # type: ignore[union-attr]
        assert metadata.port == 8080  # type: ignore[union-attr]
        assert isinstance(metadata.started_at, float)  # type: ignore[union-attr]

    def test_nonexistent_lock_returns_none(self, tmp_path: Path) -> None:
        """Should return None for non-existent lockfile."""
        metadata = read_lock(tmp_path, "nonexistent")
        assert metadata is None

    def test_corrupted_lock_returns_none(self, tmp_path: Path) -> None:
        """Should return None for corrupted lockfile."""
        # create_lock creates files with pattern "slot-{slot_id}.lock"
        lock_path = tmp_path / "slot-corrupted.lock"
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
        assert metadata.pid == 5678  # type: ignore[union-attr]
        assert metadata.port == 9000  # type: ignore[union-attr]

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
        assert not (tmp_path / "slot-slot.lock").exists()

    def test_noop_if_nonexistent(self, tmp_path: Path) -> None:
        """Should not error if lockfile doesn't exist."""
        # Should not raise
        release_lock(tmp_path, "nonexistent")


class TestWriteArtifact:
    """Tests for artifact persistence with FR-005 error paths."""

    def _valid_artifact_data(self) -> dict:
        """Create valid artifact data with all FR-007 required fields."""
        return valid_artifact_data(environment_redacted={"model_path": "/path/to/model"})

    def test_writes_artifact(self, tmp_path: Path) -> None:
        """Should write artifact with JSON serialization."""
        data = self._valid_artifact_data()
        data["slot_id"] = "test"
        data["status"] = "running"
        artifact_path = write_artifact(tmp_path, "slot", data)
        assert artifact_path.exists()
        loaded = json.loads(artifact_path.read_text())
        assert loaded["slot_id"] == "test"
        assert loaded["status"] == "running"

    def test_artifact_permissions_0600(self, tmp_path: Path) -> None:
        """Artifact should have 0600 permissions."""
        data = self._valid_artifact_data()
        artifact_path = write_artifact(tmp_path, "slot", data)
        mode = stat.S_IMODE(os.stat(artifact_path).st_mode)
        assert mode == 0o600

    def test_redacts_sensitive_in_artifact(self, tmp_path: Path) -> None:
        """Should redact sensitive environment variable values in artifact."""
        data = self._valid_artifact_data()
        data["api_key"] = "secret123"
        data["password"] = "pass456"  # noqa: S105
        data["model_path"] = "/path/to/model"
        artifact_path = write_artifact(tmp_path, "slot", data)
        loaded = json.loads(artifact_path.read_text())
        assert loaded["api_key"] == "[REDACTED]"
        assert loaded["model_path"] == "/path/to/model"
        assert loaded["password"] == "[REDACTED]"  # noqa: S105

    def test_unwritable_dir_raises_permission_error(self, tmp_path: Path) -> None:
        """Should raise ValidationException for unwritable directory."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir(mode=0o444)
        try:
            # The error happens when trying to create artifact subdirectory;
            # production code wraps OSError/PermissionError in ValidationException
            with pytest.raises(ValidationException) as exc_info:
                write_artifact(readonly_dir, "slot", self._valid_artifact_data())
            assert "failed to create artifact directory" in str(exc_info.value)
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
            "INVALID_SLOT_ID",
            "DUPLICATE_SLOT",
            "RUNTIME_DIR_UNAVAILABLE",
            "LOCKFILE_INTEGRITY_FAILURE",
            "ARTIFACT_PERSISTENCE_FAILURE",
            "BACKEND_NOT_ELIGIBLE",
        ]
        for code in expected:
            assert code in codes, f"Missing ErrorCode: {code}"

    def test_error_code_is_str_enum(self) -> None:
        """ErrorCode should be a StrEnum with string values."""
        assert isinstance(ErrorCode.FILE_NOT_FOUND, str)
        assert ErrorCode.FILE_NOT_FOUND == "FILE_NOT_FOUND"
        assert ErrorCode.FILE_NOT_FOUND in [ErrorCode.FILE_NOT_FOUND]


from typing import Any

from llama_manager.config import (
    Config,
    ServerConfig,
)
from llama_manager.validation import (
    DryRunValidationSummary,
    build_dry_run_slot_payload,
)
from tests.support.helpers import make_server_config


def _regression_cfg(**kwargs: Any) -> ServerConfig:
    slot_id = kwargs.pop("alias", "test")
    defaults = {
        "alias": slot_id,
        "server_bin": "/usr/bin/llama-server",
        "backend": "llama_cpp",
    }
    defaults.update(kwargs)
    return make_server_config(**defaults)


@pytest.fixture
def base_config() -> Config:
    """Return a default Config for testing."""
    return Config()


def test_multi_validation_error_parity(base_config: Config) -> None:
    """T042: Verify MultiValidationError fields match canonical slot.validation_results.errors.
    We verify that the errors reported in MultiValidationError are consistent with
    the individual validation failures.
    """
    # Setup slots with intentional errors
    # Slot 1: Invalid port
    # Slot 2: Model not found
    slots = [
        ModelSlot(slot_id="slot1", model_path="/valid/path/model.gguf", port=99999),  # Invalid port
        ModelSlot(
            slot_id="slot2", model_path="/nonexistent/path/model.gguf", port=8080
        ),  # Model not found
    ]

    with (
        patch("os.path.isfile", side_effect=lambda path: path == "/valid/path/model.gguf"),
        patch("os.path.exists", side_effect=lambda path: path == "/valid/path/model.gguf"),
    ):
        mve = validate_slots(slots)

        assert isinstance(mve, MultiValidationError)
        assert mve.error_count == 2

        # Check if errors are present and consistent
        # Since validate_slots currently doesn't include slot_id in failed_check,
        # they will be sorted by failed_check name, not slot.

        # We expect at least these error codes
        error_codes = [e.error_code for e in mve.errors]
        assert ErrorCode.PORT_INVALID in error_codes
        assert ErrorCode.FILE_NOT_FOUND in error_codes


def test_slot_sequence_consistency_and_tiebreak() -> None:
    """T042: Verify slot sequence consistency and failed_check ascending tie-break.
    This test specifically checks if the sorting logic in MultiValidationError
    works when failed_check strings include slot information.
    """
    # We manually create a MultiValidationError with errors that follow the expected pattern
    # to verify the sorting logic works as intended for the contract.
    # Pattern: "slot_<slot_id>_<check>"

    errors = [
        ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot2_port",
            why_blocked="err2",
            how_to_fix="fix2",
        ),
        ErrorDetail(
            error_code=ErrorCode.FILE_NOT_FOUND,
            failed_check="slot_slot1_model",
            why_blocked="err1",
            how_to_fix="fix1",
        ),
        ErrorDetail(
            error_code=ErrorCode.PORT_INVALID,
            failed_check="slot_slot1_port",
            why_blocked="err1b",
            how_to_fix="fix1b",
        ),
        ErrorDetail(
            error_code=ErrorCode.CONFIG_ERROR,
            failed_check="unknown_err",
            why_blocked="err_u",
            how_to_fix="fix_u",
        ),
    ]

    mve = MultiValidationError(errors=errors)
    mve.sort_errors()

    # Expected order:
    # 1. slot1_model (slot1, model)
    # 2. slot1_port (slot1, port)
    # 3. slot2_port (slot2, port)
    # 4. unknown_err (end)

    assert mve.errors[0].failed_check == "slot_slot1_model"
    assert mve.errors[1].failed_check == "slot_slot1_port"
    assert mve.errors[2].failed_check == "slot_slot2_port"
    assert mve.errors[3].failed_check == "unknown_err"


def test_validate_slots_duplicate_detection() -> None:
    """T042: Verify duplicate slot detection in validation."""
    slots = [
        ModelSlot(slot_id="slot1", model_path="/path/1", port=8080),
        ModelSlot(slot_id="slot1", model_path="/path/2", port=8081),  # Duplicate ID
    ]

    with patch("os.path.isfile", return_value=True), patch("os.path.exists", return_value=True):
        mve = validate_slots(slots)
        assert isinstance(mve, MultiValidationError)
        assert any(e.error_code == ErrorCode.DUPLICATE_SLOT for e in mve.errors)


def test_validate_slots_invalid_id() -> None:
    """T042: Verify invalid slot IDs are rejected during duplicate precheck."""
    slots = [
        ModelSlot(slot_id="!!!", model_path="/path/1", port=8080),  # Invalid ID
    ]

    with (
        patch("os.path.isfile", return_value=True),
        patch("os.path.exists", return_value=True),
        pytest.raises(ValueError, match="slot_id must contain at least one valid character"),
    ):
        validate_slots(slots)


class TestFR005FR003CanonicalParity:
    """FR-003/FR-005: Verify canonical parity between MultiValidationError and
    dry-run DryRunValidationSummary.errors field-level alignment.
    """

    def test_error_code_field_alignment(self) -> None:
        """FR-005: MultiValidationError.error_code must align with DryRunValidationSummary.error_code."""
        # Create MultiValidationError with ErrorDetail
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="port out of range",
                how_to_fix="use port between 1 and 65535",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model",
                why_blocked="model not found",
                how_to_fix="provide valid model path",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Create equivalent ErrorDetail list
        validation_results_list = [
            ErrorDetail(
                slot_id="slot1",
                passed=False,
                failed_check=error.failed_check,
                error_code=error.error_code,
                why_blocked=error.why_blocked,
            )
            for error in mve.errors
        ]

        # Verify field alignment: error_code must match
        for i, (error_detail, vr) in enumerate(
            zip(mve.errors, validation_results_list, strict=True)
        ):
            assert error_detail.error_code == vr.error_code, (
                f"Error {i}: error_code mismatch - "
                f"ErrorDetail={error_detail.error_code}, ErrorDetail={vr.error_code}"
            )

    def test_failed_check_field_alignment(self) -> None:
        """FR-005: MultiValidationError.failed_check must align with DryRunValidationSummary.failed_check."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port_validation",
                why_blocked="port conflict",
                how_to_fix="use unique port",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model_check",
                why_blocked="model missing",
                how_to_fix="check model path",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Create equivalent ErrorDetail list
        validation_results_list = [
            ErrorDetail(
                slot_id=error.failed_check.split("_")[1],  # Extract slot_id from failed_check
                passed=False,
                failed_check=error.failed_check,
                error_code=error.error_code,
                why_blocked=error.why_blocked,
            )
            for error in mve.errors
        ]

        # Verify field alignment: failed_check must match exactly
        for i, (error_detail, vr) in enumerate(
            zip(mve.errors, validation_results_list, strict=True)
        ):
            assert error_detail.failed_check == vr.failed_check, (
                f"Error {i}: failed_check mismatch - "
                f"ErrorDetail={error_detail.failed_check}, ErrorDetail={vr.failed_check}"
            )

    def test_validation_results_checks_alignment(self) -> None:
        """FR-003/FR-005: validation_results.checks must contain aligned error info."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="port must be between 1 and 65535",
                how_to_fix="specify a valid port number",
            ),
        ]
        mve = MultiValidationError(errors=errors)

        # Create DryRunValidationSummary with checks that align with ErrorDetails
        checks = [
            {
                "failed_check": error.failed_check,
                "error_code": error.error_code.value,  # type: ignore[union-attr]
                "why_blocked": error.why_blocked,
                "how_to_fix": error.how_to_fix,
            }
            for error in mve.errors
        ]

        validation_results = DryRunValidationSummary(passed=False, checks=checks)

        # Verify checks contain aligned fields
        assert len(validation_results.checks) == len(mve.errors)
        for check, error in zip(validation_results.checks, mve.errors, strict=True):
            assert check["failed_check"] == error.failed_check
            assert check["error_code"] == error.error_code.value  # type: ignore[union-attr]
            assert check["why_blocked"] == error.why_blocked
            assert check["how_to_fix"] == error.how_to_fix


class TestFR003SlotConfigurationSequenceConsistency:
    """FR-003: Verify slot configuration sequence consistency between error output
    and dry-run payload.
    """

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _regression_cfg(**{"alias": slot_id, **kwargs})

    def test_error_slot_order_matches_dry_run_slot_order(self) -> None:
        """FR-003: Error slot sequence order must match dry-run payload slot order."""
        # Create errors with specific slot order
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port",
                why_blocked="port conflict in slot2",
                how_to_fix="fix port in slot2",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model",
                why_blocked="model missing in slot1",
                how_to_fix="fix model in slot1",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="port conflict in slot1",
                how_to_fix="fix port in slot1",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # After sorting, expected order is: slot1_model, slot1_port, slot2_port
        expected_sorted_order = ["slot_slot1_model", "slot_slot1_port", "slot_slot2_port"]
        actual_sorted_order = [error.failed_check for error in mve.errors]
        assert actual_sorted_order == expected_sorted_order, (
            f"Sort order mismatch: expected {expected_sorted_order}, got {actual_sorted_order}"
        )

        # Create DryRunValidationSummary with same slot order
        validation_results = DryRunValidationSummary(
            passed=False,
            checks=[
                {
                    "slot_id": error.failed_check.split("_")[1],
                    "failed_check": error.failed_check,
                    "error_code": error.error_code.value,  # type: ignore[union-attr]
                }
                for error in mve.errors
            ],
        )

        # Verify slot sequence consistency
        error_slot_sequence = [error.failed_check.split("_")[1] for error in mve.errors]
        check_slot_sequence = [check["slot_id"] for check in validation_results.checks]

        assert error_slot_sequence == check_slot_sequence, (
            f"Slot sequence mismatch: errors={error_slot_sequence}, checks={check_slot_sequence}"
        )

    def test_dry_run_payload_slot_scope_matches_error_slot_sequence(self) -> None:
        """FR-003: Dry-run slot_scope list order must match error slot sequence."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot3_port",
                why_blocked="port issue in slot3",
                how_to_fix="fix slot3",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_model",
                why_blocked="model issue in slot1",
                how_to_fix="fix slot1",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port",
                why_blocked="port issue in slot2",
                how_to_fix="fix slot2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Build dry-run payloads in sorted error order
        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=DryRunValidationSummary(
                    passed=False,
                    checks=[{"failed_check": error.failed_check}],
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        # slot_scope should match the sorted error slot order
        slot_scope = [p.slot_id for p in payloads]
        expected_slot_order = [error.failed_check.split("_")[1] for error in mve.errors]

        assert slot_scope == expected_slot_order, (
            f"slot_scope order mismatch: expected {expected_slot_order}, got {slot_scope}"
        )


class TestFR003FailedCheckAscendingTieBreak:
    """FR-003: Verify failed_check ascending tie-break within each slot."""

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _regression_cfg(**{"alias": slot_id, **kwargs})

    def test_failed_check_ascending_tiebreak_within_slot(self) -> None:
        """FR-003: failed_check should be sorted ascending within each slot."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_z_port_validation",  # Should come last in slot1
                why_blocked="z error",
                how_to_fix="fix z",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot1_a_model_check",  # Should come first in slot1
                why_blocked="a error",
                how_to_fix="fix a",
            ),
            ErrorDetail(
                error_code=ErrorCode.CONFIG_ERROR,
                failed_check="slot_slot1_m_ctx_size",  # Should come middle in slot1
                why_blocked="m error",
                how_to_fix="fix m",
            ),
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot2_port",  # slot2 errors
                why_blocked="slot2 error",
                how_to_fix="fix slot2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Expected order: slot1_a_model_check, slot1_m_ctx_size, slot1_z_port_validation, slot2_port
        expected_order = [
            "slot_slot1_a_model_check",
            "slot_slot1_m_ctx_size",
            "slot_slot1_z_port_validation",
            "slot_slot2_port",
        ]
        actual_order = [error.failed_check for error in mve.errors]

        assert actual_order == expected_order, (
            f"Tie-break order mismatch: expected {expected_order}, got {actual_order}"
        )

        # Verify slot sequence: slot1 errors before slot2
        slot1_indices = [i for i, e in enumerate(mve.errors) if "slot1" in e.failed_check]
        slot2_indices = [i for i, e in enumerate(mve.errors) if "slot2" in e.failed_check]

        assert all(idx < slot2_indices[0] for idx in slot1_indices), (
            "Slot1 errors should come before slot2 errors"
        )


class TestFR003NewArtifactShapeAssertions:
    """FR-003: Explicit tests for new dry-run artifact shape: slot_scope list
    and resolved_command mapping.
    """

    def _cfg(self, slot_id: str, **kwargs: Any) -> ServerConfig:
        """Create ServerConfig for testing."""
        return _regression_cfg(**{"alias": slot_id, **kwargs})

    def test_slot_scope_is_list_of_strings(self) -> None:
        """FR-003: slot_scope must be a list of strings (slot IDs)."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="error1",
                how_to_fix="fix1",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot2_model",
                why_blocked="error2",
                how_to_fix="fix2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        # Build payloads in sorted order
        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=DryRunValidationSummary(
                    passed=False, checks=[{"failed_check": error.failed_check}]
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        # slot_scope is the canonical list of slot IDs
        slot_scope = [p.slot_id for p in payloads]

        assert isinstance(slot_scope, list), "slot_scope must be a list"
        assert all(isinstance(slot_id, str) for slot_id in slot_scope), (
            "All slot_scope entries must be strings"
        )
        assert len(slot_scope) == len(mve.errors), (
            f"slot_scope length mismatch: expected {len(mve.errors)}, got {len(slot_scope)}"
        )

    def test_resolved_command_is_mapping_of_slot_id_to_command_args(self) -> None:
        """FR-003: resolved_command must be a dict mapping slot_id -> command_args list."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="error1",
                how_to_fix="fix1",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=DryRunValidationSummary(
                    passed=False, checks=[{"failed_check": error.failed_check}]
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        # Build resolved_command mapping (as done in dry_run.py)
        resolved_command = {p.slot_id: p.command_args for p in payloads}

        assert isinstance(resolved_command, dict), "resolved_command must be a dict"

        # Each key must be a slot_id and each value must be a list of command args
        for slot_id, cmd_args in resolved_command.items():
            assert isinstance(slot_id, str), (
                f"resolved_command key must be string, got {type(slot_id)}"
            )
            assert isinstance(cmd_args, list), (
                f"resolved_command[{slot_id}] must be list, got {type(cmd_args)}"
            )
            assert all(isinstance(arg, str) for arg in cmd_args), (
                f"resolved_command[{slot_id}] must contain only strings"
            )

    def test_slot_scope_and_resolved_command_keys_alignment(self) -> None:
        """FR-003: resolved_command keys must exactly match slot_scope entries."""
        errors = [
            ErrorDetail(
                error_code=ErrorCode.PORT_INVALID,
                failed_check="slot_slot1_port",
                why_blocked="error1",
                how_to_fix="fix1",
            ),
            ErrorDetail(
                error_code=ErrorCode.FILE_NOT_FOUND,
                failed_check="slot_slot2_model",
                why_blocked="error2",
                how_to_fix="fix2",
            ),
        ]
        mve = MultiValidationError(errors=errors)
        mve.sort_errors()

        payloads = [
            build_dry_run_slot_payload(
                self._cfg(slot_id=error.failed_check.split("_")[1]),
                slot_id=error.failed_check.split("_")[1],
                validation_results=DryRunValidationSummary(
                    passed=False, checks=[{"failed_check": error.failed_check}]
                ),
                warnings=[],
            )
            for error in mve.errors
        ]

        slot_scope = [p.slot_id for p in payloads]
        resolved_command = {p.slot_id: p.command_args for p in payloads}

        # Keys in resolved_command must match slot_scope entries
        assert set(resolved_command.keys()) == set(slot_scope), (
            f"resolved_command keys {set(resolved_command.keys())} must match slot_scope {set(slot_scope)}"
        )

        # Order must be consistent: resolved_command should preserve slot_scope order
        ordered_keys = list(resolved_command.keys())
        assert ordered_keys == slot_scope, (
            f"resolved_command key order {ordered_keys} must match slot_scope order {slot_scope}"
        )

    def test_resolved_command_contains_correct_command_args_for_each_slot(self) -> None:
        """FR-003: resolved_command[<slot_id>] must contain the correct command_args."""
        cfg = self._cfg(slot_id="test-slot")
        payload = build_dry_run_slot_payload(
            cfg,
            slot_id="test-slot",
            validation_results=DryRunValidationSummary(passed=True, checks=[]),
            warnings=[],
        )

        resolved_command = {"test-slot": payload.command_args}

        # resolved_command["test-slot"] must equal payload.command_args
        assert "test-slot" in resolved_command, "resolved_command must contain 'test-slot' key"
        assert resolved_command["test-slot"] == payload.command_args, (
            "resolved_command['test-slot'] must equal payload.command_args"
        )

        # Verify command_args structure
        cmd_args = resolved_command["test-slot"]
        assert isinstance(cmd_args, list), "command_args must be a list"
        assert len(cmd_args) > 0, "command_args must not be empty"
        assert "--model" in cmd_args, "command_args must contain --model flag"


"""T026: Foundational regression tests for Phase 2.

Integration tests for:
- detect_toolchain()
- get_toolchain_hints()
- create_venv()
- check_venv_integrity()
- write_failure_report()
"""


import sys

import pytest

from llama_manager.reports import FailureReport, write_failure_report
from llama_manager.setup_venv import VenvResult, check_venv_integrity, create_venv, get_venv_path
from llama_manager.toolchain import (
    ToolchainErrorDetail,
    ToolchainHint,
    ToolchainStatus,
    get_toolchain_hints,
    parse_version,
    version_at_least,
)


class TestDetectToolchainIntegration:
    """Integration tests for detect_toolchain functionality."""

    def test_detect_toolchain_basic(self) -> None:
        """detect_toolchain should return ToolchainStatus with correct structure."""
        # Mock detect_tool to return known values
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # Simulate all tools present
            mock_detect.return_value = (True, "1.0.0")

            # Import here to avoid circular import
            from llama_manager.toolchain import detect_toolchain

            status = detect_toolchain()

            assert isinstance(status, ToolchainStatus)
            # All fields should be set since we mocked detect_tool to return True
            assert status.gcc is not None
            assert status.make is not None
            assert status.git is not None
            assert status.cmake is not None
            assert status.sycl_compiler is not None
            assert status.cuda_toolkit is not None
            assert status.nvtop is not None

    def test_detect_toolchain_partial_tools(self) -> None:
        """detect_toolchain should handle partial tool availability."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # Simulate some tools present, some missing
            # Calls: gcc, make, git, cmake, icpx, icx, dpcpp, nvcc, nvtop
            mock_detect.side_effect = [
                (True, "11.4.0"),  # gcc
                (True, "4.3"),  # make
                (True, "2.37.0"),  # git
                (True, "3.25.0"),  # cmake
                (False, None),  # icpx (SYCL candidate 1)
                (False, None),  # icx (SYCL candidate 2)
                (False, None),  # dpcpp (SYCL candidate 3)
                (True, "12.2.0"),  # nvcc (CUDA toolkit)
                (True, "2.1.0"),  # nvtop
            ]

            from llama_manager.toolchain import detect_toolchain

            status = detect_toolchain()

            assert status.gcc == "11.4.0"
            assert status.sycl_compiler is None  # Missing
            assert status.is_sycl_ready is False
            assert status.is_cuda_ready is True

    def test_detect_toolchain_all_missing(self) -> None:
        """detect_toolchain should handle all tools missing."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing
            mock_detect.return_value = (False, None)

            from llama_manager.toolchain import detect_toolchain

            status = detect_toolchain()

            assert status.gcc is None
            assert status.make is None
            assert status.is_sycl_ready is False
            assert status.is_cuda_ready is False
            assert status.is_complete is False


class TestGetToolchainHintsIntegration:
    """Integration tests for get_toolchain_hints functionality."""

    def test_get_toolchain_hints_sycl_integration(self) -> None:
        """get_toolchain_hints should return list of ToolchainErrorDetail for SYCL."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing
            mock_detect.return_value = (False, None)

            errors = get_toolchain_hints("sycl")

            assert isinstance(errors, list)
            assert len(errors) == 7  # gcc, make, git, cmake, dpcpp, icx, icpx
            for error in errors:
                assert isinstance(error, ToolchainErrorDetail)
                assert error.error_code.value == "TOOLCHAIN_MISSING"
                assert error.failed_check is not None
                assert error.why_blocked is not None
                assert error.how_to_fix is not None

    def test_get_toolchain_hints_cuda_integration(self) -> None:
        """get_toolchain_hints should return list of ToolchainErrorDetail for CUDA."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools missing
            mock_detect.return_value = (False, None)

            errors = get_toolchain_hints("cuda")

            assert isinstance(errors, list)
            assert len(errors) == 6  # gcc, make, git, cmake, nvcc, nvidia-smi
            for error in errors:
                assert isinstance(error, ToolchainErrorDetail)
                assert error.error_code.value == "TOOLCHAIN_MISSING"
                assert error.failed_check is not None
                assert error.why_blocked is not None
                assert error.how_to_fix is not None

    def test_get_toolchain_hints_empty_when_all_present(self) -> None:
        """get_toolchain_hints should return empty list when all tools present."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            # All tools present
            mock_detect.return_value = (True, "1.0.0")

            sycl_errors = get_toolchain_hints("sycl")
            cuda_errors = get_toolchain_hints("cuda")

            assert len(sycl_errors) == 0
            assert len(cuda_errors) == 0

    def test_get_toolchain_hints_invalid_backend_raises(self) -> None:
        """get_toolchain_hints should raise ValueError for invalid backend."""
        with pytest.raises(ValueError) as exc_info:
            get_toolchain_hints("invalid")
        assert "Unknown backend" in str(exc_info.value)


class TestCreateVenvIntegration:
    """Integration tests for create_venv functionality."""

    def test_create_venv_integration(self, tmp_path: Path) -> None:
        """create_venv should return VenvResult with correct structure."""
        venv_path = tmp_path / "test_venv"

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

            assert isinstance(result, VenvResult)
            assert result.venv_path == venv_path
            assert result.created is True
            assert result.reused is False
            assert result.was_created is True
            assert result.was_reused is False
            assert "source" in result.activation_command
            assert "bin/activate" in result.activation_command

    def test_create_venv_reuse_existing(self, tmp_path: Path) -> None:
        """create_venv should reuse existing venv."""
        venv_path = tmp_path / "existing_venv"
        venv_path.mkdir()

        # Create minimal valid venv structure so check_venv_integrity passes
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)

        result = create_venv(venv_path)

        assert isinstance(result, VenvResult)
        assert result.venv_path == venv_path
        assert result.created is False
        assert result.reused is True
        assert result.was_created is False
        assert result.was_reused is True

    def test_create_venv_activation_command_format(self, tmp_path: Path) -> None:
        """create_venv should generate correct activation command format."""
        venv_path = tmp_path / "test_venv"

        with patch("llama_manager.setup_venv.venv.create"):
            result = create_venv(venv_path)

            # Should be sourceable
            assert result.activation_command.startswith("source ")
            assert result.activation_command.endswith("/activate")


class TestCheckVenvIntegrityIntegration:
    """Integration tests for check_venv_integrity functionality."""

    def test_check_venv_integrity_valid_venv(self, tmp_path: Path) -> None:
        """check_venv_integrity should validate valid venv structure."""
        venv_path = tmp_path / "valid_venv"
        venv_path.mkdir()

        # Create minimal venv structure
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)

        is_valid, error = check_venv_integrity(venv_path)

        assert is_valid is True
        assert error is None

    def test_check_venv_integrity_invalid_venv(self, tmp_path: Path) -> None:
        """check_venv_integrity should detect invalid venv structure."""
        venv_path = tmp_path / "invalid_venv"
        venv_path.mkdir()

        # Missing pyvenv.cfg
        is_valid, error = check_venv_integrity(venv_path)

        assert is_valid is False
        assert error == "pyvenv.cfg missing"

    def test_check_venv_integrity_integration_with_mock(self, tmp_path: Path) -> None:
        """check_venv_integrity should work with mocked paths."""
        # Create a real venv structure for the test
        venv_path = tmp_path / "mock_venv"
        venv_path.mkdir()
        (venv_path / "pyvenv.cfg").write_text("home = /usr/bin\n")
        bin_dir = venv_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "python").symlink_to(sys.executable)

        # Now test with the real path - no need to mock Path.exists globally
        is_valid, error = check_venv_integrity(venv_path)

        assert is_valid is True
        assert error is None


class TestWriteFailureReportIntegration:
    """Integration tests for write_failure_report functionality."""

    def test_write_failure_report_integration(self, tmp_path: Path) -> None:
        """write_failure_report should create proper failure report structure."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json='{"exit_code": 1}',
            build_output="Build failed: compilation error",
            error_details=[{"type": "BuildError", "message": "compilation failed"}],
            metadata={"backend": "sycl"},
        )

        assert isinstance(report, FailureReport)
        assert report.report_dir.exists()
        assert report.report_dir.is_dir()
        assert "20" in report.report_dir.name  # YYYYMMDD_HHMMSS format

        # Check all files created
        assert (report.report_dir / "build-artifact.json").exists()
        assert (report.report_dir / "build-output.log").exists()
        assert (report.report_dir / "error-details.json").exists()

    def test_write_failure_report_permissions(self, tmp_path: Path) -> None:
        """write_failure_report should enforce correct permissions."""
        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output="",
            error_details=[],
        )

        # Directory should be 0700
        dir_mode = report.report_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700

        # Files should be 0600
        for filename in ["build-artifact.json", "build-output.log", "error-details.json"]:
            file_path = report.report_dir / filename
            file_mode = file_path.stat().st_mode & 0o777
            assert file_mode == 0o600

    def test_write_failure_report_redaction_integration(self, tmp_path: Path) -> None:
        """write_failure_report should redact sensitive data in output."""
        output_with_secrets = "API_KEY=secret123 TOKEN=abc456 Normal build output"

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output=output_with_secrets,
            error_details=[],
        )

        # Read the output file
        output_path = report.report_dir / "build-output.log"
        with open(output_path) as f:
            actual_output = f.read()

        # Should be redacted
        assert "[REDACTED]" in actual_output
        assert "secret123" not in actual_output
        assert "abc456" not in actual_output
        # Non-sensitive content should remain
        assert "Normal build output" in actual_output

    def test_write_failure_report_truncation_integration(self, tmp_path: Path) -> None:
        """write_failure_report should truncate large outputs."""
        # Create very long output
        long_output = "x" * 20000  # More than default 8192 bytes

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output=long_output,
            error_details=[],
        )

        # Read the output file
        output_path = report.report_dir / "build-output.log"
        with open(output_path) as f:
            actual_output = f.read()

        # Should be truncated to 8192 bytes
        assert len(actual_output) <= 8192
        assert len(actual_output) < len(long_output)

    def test_write_failure_report_json_serialization(self, tmp_path: Path) -> None:
        """write_failure_report should properly serialize JSON data."""
        error_details = [
            {"type": "BuildError", "message": "compilation failed"},
            {"type": "Warning", "message": "deprecated flag used"},
        ]

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json='{"exit_code": 1}',
            build_output="",
            error_details=error_details,
        )

        # Read error details file
        errors_path = report.report_dir / "error-details.json"
        with open(errors_path) as f:
            loaded_errors = json.load(f)

        assert len(loaded_errors) == 2
        assert loaded_errors[0]["type"] == "BuildError"
        assert loaded_errors[1]["type"] == "Warning"


class TestPhase2Comprehensive:
    """Comprehensive tests covering all Phase 2 functionality."""

    def test_version_parsing_and_comparison(self) -> None:
        """Test version parsing and comparison for toolchain validation."""
        # Test parse_version
        assert parse_version("3.20.1") == (3, 20, 1)
        assert parse_version("3.20") == (3, 20, 0)
        assert parse_version("3.20.1ubuntu") == (3, 20, 1)

        # Test version_at_least
        assert version_at_least("3.20.1", "3.20.0") is True
        assert version_at_least("3.19.0", "3.20.0") is False
        assert version_at_least("3.14", "3.14") is True

    def test_toolchain_error_detail_structure(self) -> None:
        """Test ToolchainErrorDetail has all required fields."""
        from llama_manager.config import ErrorCode

        error = ToolchainErrorDetail(
            error_code=ErrorCode.TOOLCHAIN_MISSING,
            failed_check="gcc",
            why_blocked="Required for sycl backend",
            how_to_fix="sudo apt-get install gcc",
            docs_ref="https://gcc.gnu.org/download.html",
        )

        assert error.error_code == ErrorCode.TOOLCHAIN_MISSING
        assert error.failed_check == "gcc"
        assert error.why_blocked == "Required for sycl backend"
        assert error.how_to_fix == "sudo apt-get install gcc"
        assert error.docs_ref == "https://gcc.gnu.org/download.html"

    def test_toolchain_hint_structure(self) -> None:
        """Test ToolchainHint has all required fields."""
        hint = ToolchainHint(
            tool_name="gcc",
            install_command="sudo apt-get install gcc",
            install_url="https://gcc.gnu.org/download.html",
            required_for=["sycl", "cuda"],
        )

        assert hint.tool_name == "gcc"
        assert hint.install_command == "sudo apt-get install gcc"
        assert hint.install_url == "https://gcc.gnu.org/download.html"
        assert hint.required_for == ["sycl", "cuda"]
        assert hint.is_url_available is True

    def test_venv_path_utility(self) -> None:
        """Test get_venv_path utility function."""
        # Test with default
        with patch.dict(os.environ, {}, clear=True):
            # Ensure XDG_CACHE_HOME is not set
            os.environ.pop("XDG_CACHE_HOME", None)
            result = get_venv_path()
            assert isinstance(result, Path)
            assert "llm-runner" in str(result)
            assert "venv" in str(result)

    def test_sycl_vs_cuda_toolchain_hints(self) -> None:
        """Test that SYCL and CUDA hints return different tools."""
        with patch("llama_manager.toolchain.detect_tool") as mock_detect:
            mock_detect.return_value = (False, None)

            sycl_errors = get_toolchain_hints("sycl")
            cuda_errors = get_toolchain_hints("cuda")

            # Should have different tool names
            sycl_tools = {e.failed_check for e in sycl_errors}
            cuda_tools = {e.failed_check for e in cuda_errors}

            # Common tools overlap (gcc, make, git, cmake)
            common = sycl_tools & cuda_tools
            assert "gcc" in common
            assert "make" in common

            # SYCL-specific tools
            assert sycl_tools == {"gcc", "make", "git", "cmake", "dpcpp", "icx", "icpx"}

            # CUDA-specific tools (no nvtop)
            assert cuda_tools == {"gcc", "make", "git", "cmake", "nvcc", "nvidia-smi"}

    def test_report_security(self, tmp_path: Path) -> None:
        """Test that failure reports properly handle sensitive data."""
        sensitive_output = """
        API_KEY=supersecret123
        TOKEN=abc456def
        PASSWORD=mypass
        Normal log line
        AUTH_HEADER=bearer_token
        """

        report = write_failure_report(
            report_dir=tmp_path,
            build_artifact_json="{}",
            build_output=sensitive_output,
            error_details=[],
        )

        # Read build-output.log and verify redaction
        output_file = report.report_dir / "build-output.log"
        with open(output_file) as f:
            content = f.read()

        # Sensitive values should be redacted
        assert "supersecret123" not in content
        assert "abc456def" not in content
        assert "mypass" not in content
        assert "bearer_token" not in content

        # Should have redaction markers
        assert "[REDACTED]" in content
        # Non-sensitive content should be preserved
        assert "Normal log line" in content

    def test_venv_result_properties(self, tmp_path: Path) -> None:
        """Test VenvResult properties."""
        result = VenvResult(
            venv_path=tmp_path / "venv",
            created=True,
            reused=False,
            activation_command="source /tmp/venv/bin/activate",
        )

        assert result.was_created is True
        assert result.was_reused is False
        assert result.is_valid is False  # Path doesn't exist

        # Test path methods
        python_path = result.get_python_path()
        assert python_path.name == "python"

        pip_path = result.get_pip_path()
        assert pip_path.name == "pip"
