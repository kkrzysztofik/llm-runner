"""Tests for llama_manager.profile_cache utility functions."""

import hashlib
import json
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.profile_cache import (
    ProfileFlavor,
    ProfileMetrics,
    ProfileRecord,
    _sanitize_filename_component,
    compute_driver_version_hash,
    compute_gpu_identifier,
    ensure_profiles_dir,
    get_profile_path,
)


class TestSanitizeFilenameComponent:
    """Tests for _sanitize_filename_component."""

    def test_alphanumeric_passthrough(self) -> None:
        """Alphanumeric strings pass through unchanged."""
        assert _sanitize_filename_component("hello123") == "hello123"

    def test_dashes_preserved(self) -> None:
        """Dashes are preserved in the output."""
        assert _sanitize_filename_component("hello-world") == "hello-world"

    def test_underscores_preserved(self) -> None:
        """Underscores are preserved in the output."""
        assert _sanitize_filename_component("hello_world") == "hello_world"

    def test_dots_preserved(self) -> None:
        """Dots are preserved in the output."""
        assert _sanitize_filename_component("hello.world") == "hello.world"

    def test_uppercase_lowercased(self) -> None:
        """Uppercase letters are lowercased."""
        assert _sanitize_filename_component("HelloWorld") == "helloworld"

    def test_spaces_replaced_with_underscore(self) -> None:
        """Spaces are replaced with underscores."""
        assert _sanitize_filename_component("hello world") == "hello_world"

    def test_special_chars_replaced_with_underscore(self) -> None:
        """Special characters are replaced with underscores."""
        assert _sanitize_filename_component("GeForce RTX 3090") == "geforce_rtx_3090"

    def test_leading_trailing_stripped(self) -> None:
        """Leading and trailing whitespace is stripped."""
        assert _sanitize_filename_component("  hello  ") == "hello"

    def test_mixed_case_and_special(self) -> None:
        """Mixed case with special chars is fully sanitized."""
        assert _sanitize_filename_component("GeForce RTX 3090 Ti!") == "geforce_rtx_3090_ti_"

    def test_numbers_preserved(self) -> None:
        """Numbers are preserved."""
        assert _sanitize_filename_component("model_v2_1") == "model_v2_1"

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            _sanitize_filename_component("")

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            _sanitize_filename_component(None)  # type: ignore[arg-type]

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="at least one valid character"):
            _sanitize_filename_component("   ")

    def test_all_special_chars_becomes_underscores(self) -> None:
        """String with only special chars becomes underscores (not empty)."""
        # @#$% all become underscores -> "____" which is non-empty
        result = _sanitize_filename_component("@#$%")
        assert result == "____"

    def test_mixed_valid_invalid(self) -> None:
        """Mixed valid and invalid chars produces sanitized result."""
        result = _sanitize_filename_component("GPU@Device#1")
        assert result == "gpu_device_1"

    def test_numeric_string(self) -> None:
        """Pure numeric string passes through."""
        assert _sanitize_filename_component("12345") == "12345"

    def test_underscore_hyphen_mixed(self) -> None:
        """Mixed underscores and hyphens are preserved."""
        assert _sanitize_filename_component("a_b-c_d") == "a_b-c_d"


class TestComputeGpuIdentifier:
    """Tests for compute_gpu_identifier."""

    def test_cuda_format(self) -> None:
        """CUDA backend produces nvidia-prefixed identifier."""
        result = compute_gpu_identifier("cuda", "GeForce RTX 3090", 0)
        assert result == "nvidia-geforce_rtx_3090-00"

    def test_sycl_format(self) -> None:
        """SYCL backend produces intel-prefixed identifier."""
        result = compute_gpu_identifier("sycl", "Arc B580", 0)
        assert result == "intel-arc_b580-00"

    def test_cuda_multi_device(self) -> None:
        """CUDA identifier includes device index."""
        result = compute_gpu_identifier("cuda", "GeForce RTX 3090", 2)
        assert result == "nvidia-geforce_rtx_3090-02"

    def test_sycl_multi_device(self) -> None:
        """SYCL identifier includes device index."""
        result = compute_gpu_identifier("sycl", "Arc B580", 1)
        assert result == "intel-arc_b580-01"

    def test_gpu_name_with_dots(self) -> None:
        """GPU names with dots are preserved."""
        result = compute_gpu_identifier("cuda", "GeForce.RTX.3090", 0)
        assert result == "nvidia-geforce.rtx.3090-00"

    def test_gpu_name_already_sanitized(self) -> None:
        """Already-sanitized GPU names pass through."""
        result = compute_gpu_identifier("sycl", "arc-b580", 0)
        assert result == "intel-arc-b580-00"

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="unsupported backend"):
            compute_gpu_identifier("opencl", "Test GPU", 0)

    def test_empty_gpu_name_raises(self) -> None:
        """Empty GPU name raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            compute_gpu_identifier("cuda", "", 0)

    def test_gpu_name_with_all_special_chars_becomes_underscores(self) -> None:
        """GPU name with only special chars becomes underscores."""
        # @#$% each become underscores -> "____" which is non-empty
        result = compute_gpu_identifier("cuda", "@#$%", 0)
        assert result == "nvidia-____-00"

    def test_result_is_lowercase(self) -> None:
        """Result is always lowercase."""
        result = compute_gpu_identifier("cuda", "GeForce RTX 3090 Ti", 0)
        assert result == result.lower()

    def test_result_is_list_of_strings(self) -> None:
        """Result is a string."""
        result = compute_gpu_identifier("sycl", "Arc B580", 0)
        assert isinstance(result, str)

    def test_cuda_with_complex_gpu_name(self) -> None:
        """CUDA with complex GPU name produces expected format."""
        result = compute_gpu_identifier(
            "cuda",
            "NVIDIA GeForce RTX 4090 SUPER",
            3,
        )
        assert result == "nvidia-nvidia_geforce_rtx_4090_super-03"

    def test_sycl_with_complex_gpu_name(self) -> None:
        """SYCL with complex GPU name produces expected format."""
        result = compute_gpu_identifier(
            "sycl",
            "Intel(R) Arc(TM) B580",
            0,
        )
        # (R) -> _r_, (TM) -> _tm_, spaces -> underscores
        assert result == "intel-intel_r__arc_tm__b580-00"


class TestComputeDriverVersionHash:
    """Tests for compute_driver_version_hash."""

    def test_deterministic_hash(self) -> None:
        """Same input always produces the same hash."""
        h1 = compute_driver_version_hash("545.23.08")
        h2 = compute_driver_version_hash("545.23.08")
        assert h1 == h2

    def test_different_inputs_different_hashes(self) -> None:
        """Different inputs produce different hashes."""
        h1 = compute_driver_version_hash("545.23.08")
        h2 = compute_driver_version_hash("546.01.00")
        assert h1 != h2

    def test_hash_length_is_16(self) -> None:
        """Hash is truncated to 16 hex characters."""
        result = compute_driver_version_hash("545.23.08")
        assert len(result) == 16

    def test_hash_is_hex(self) -> None:
        """Hash contains only hexadecimal characters."""
        result = compute_driver_version_hash("545.23.08")
        assert all(c in "0123456789abcdef" for c in result)

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            compute_driver_version_hash("")

    def test_none_raises(self) -> None:
        """None raises ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            compute_driver_version_hash(None)  # type: ignore[arg-type]

    def test_whitespace_only_produces_hash(self) -> None:
        """Whitespace-only strings produce a valid hash (not stripped)."""
        result = compute_driver_version_hash("   ")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_known_hash(self) -> None:
        """Known input produces expected SHA-256 prefix."""
        # SHA-256 of "545.23.08" starts with these 16 hex chars
        expected = hashlib.sha256(b"545.23.08").hexdigest()[:16]
        assert compute_driver_version_hash("545.23.08") == expected

    def test_unicode_driver_version(self) -> None:
        """Unicode driver version strings are handled."""
        result = compute_driver_version_hash("v1.0β")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_numeric_string(self) -> None:
        """Numeric-only driver version produces valid hash."""
        result = compute_driver_version_hash("12345")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_long_driver_version(self) -> None:
        """Long driver version strings are handled."""
        long_version = "368.69-0ubuntu0.16.04.1~src1"
        result = compute_driver_version_hash(long_version)
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# Test get_profile_path
# ---------------------------------------------------------------------------


class TestGetProfilePath:
    """Tests for get_profile_path function."""

    def test_get_profile_path_traversal_blocked(self, tmp_path: Path) -> None:
        """Path traversal attempts should raise ValueError."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        # Mock _sanitize_filename_component to return a path with ..
        # This tests the traversal protection in get_profile_path
        with (
            patch(
                "llama_manager.profile_cache._sanitize_filename_component",
                side_effect=lambda s: "../../../etc" if s == "gpu" else s,
            ),
            pytest.raises(ValueError, match="escapes profiles_dir"),
        ):
            get_profile_path(profiles_dir, "gpu", "cuda", ProfileFlavor.BALANCED)

    def test_get_profile_path_actual_path_within_dir(self, tmp_path: Path) -> None:
        """Verified path is within profiles_dir."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        result = get_profile_path(profiles_dir, "nvidia-gpu0", "cuda", ProfileFlavor.BALANCED)
        assert result.parent == profiles_dir

    def test_get_profile_path_sycl_backend(self, tmp_path: Path) -> None:
        """Path includes sanitized sycl backend."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        result = get_profile_path(profiles_dir, "intel-arc-b580", "sycl", ProfileFlavor.FAST)
        assert "intel-arc-b580-sycl-fast.json" in str(result)

    def test_get_profile_path_cuda_backend(self, tmp_path: Path) -> None:
        """Path includes sanitized cuda backend."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        result = get_profile_path(profiles_dir, "nvidia-gpu0", "cuda", ProfileFlavor.BALANCED)
        assert "nvidia-gpu0-cuda-balanced.json" in str(result)


# ---------------------------------------------------------------------------
# Test ensure_profiles_dir permissions
# ---------------------------------------------------------------------------


class TestEnsureProfilesDirPermissions:
    """Tests for ensure_profiles_dir function."""

    def test_creates_with_owner_only(self, tmp_path: Path) -> None:
        """Directory should be created with 0o700 permissions."""
        profiles_dir = tmp_path / "profiles"
        ensure_profiles_dir(profiles_dir)
        # Check directory exists and has correct permissions
        assert profiles_dir.exists()
        import stat

        mode = profiles_dir.stat().st_mode
        assert stat.S_IMODE(mode) == 0o700

    def test_does_not_overwrite_existing_dir(self, tmp_path: Path) -> None:
        """Existing directory is not modified."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        original_mtime = profiles_dir.stat().st_mtime

        ensure_profiles_dir(profiles_dir)

        assert profiles_dir.exists()
        # mtime should be unchanged since directory wasn't recreated
        assert profiles_dir.stat().st_mtime == original_mtime

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        profiles_dir = tmp_path / "deep" / "nested" / "profiles"
        ensure_profiles_dir(profiles_dir)
        assert profiles_dir.exists()
        assert profiles_dir.is_dir()


class TestAtomicWriteJson:
    """Tests for _atomic_write_json function."""

    def test_permission_mismatch_raises_oserror(self, tmp_path: Path) -> None:
        """OSError raised when permissions don't match after rename."""
        from llama_manager.profile_cache import _atomic_write_json

        profile_path = tmp_path / "test.json"

        with patch("llama_manager.profile_cache.os.stat") as mock_stat:
            # Return a mode that doesn't match 0o600 (e.g. 0o644)
            fake_stat = MagicMock()
            fake_stat.st_mode = 0o100644  # regular file with 0644 permissions
            mock_stat.return_value = fake_stat

            with pytest.raises(OSError, match="permissions mismatch"):
                _atomic_write_json(profile_path, {"key": "value"})

            # The target file exists (os.replace succeeded), but OSError was raised
            # The temp file was cleaned up by the except block
            assert profile_path.exists()

    def test_successful_write_returns_none(self, tmp_path: Path) -> None:
        """Successful write returns None and creates file with correct permissions."""
        from llama_manager.profile_cache import FILE_MODE_OWNER_ONLY, _atomic_write_json

        profile_path = tmp_path / "test.json"
        result = _atomic_write_json(profile_path, {"key": "value"})

        assert result is None
        assert profile_path.exists()
        mode = profile_path.stat().st_mode & 0o777
        assert mode == FILE_MODE_OWNER_ONLY

    def test_write_content_is_valid_json(self, tmp_path: Path) -> None:
        """Written content is valid JSON with correct data."""
        from llama_manager.profile_cache import _atomic_write_json

        profile_path = tmp_path / "test.json"
        data = {"name": "test", "count": 42, "nested": {"a": 1}}
        _atomic_write_json(profile_path, data)

        loaded = json.loads(profile_path.read_text(encoding="utf-8"))
        assert loaded == data

    def test_write_ends_with_newline(self, tmp_path: Path) -> None:
        """Written file ends with a newline character."""
        from llama_manager.profile_cache import _atomic_write_json

        profile_path = tmp_path / "test.json"
        _atomic_write_json(profile_path, {"a": 1})

        content = profile_path.read_text(encoding="utf-8")
        assert content.endswith("\n")


# ---------------------------------------------------------------------------
# Test write_profile
# ---------------------------------------------------------------------------


class TestWriteProfile:
    """Tests for write_profile function."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """write_profile should create file."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            write_profile,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = write_profile(tmp_path, record)
        assert result.exists()

    def test_write_returns_path(self, tmp_path: Path) -> None:
        """Should return Path to written file."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            write_profile,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = write_profile(tmp_path, record)
        assert isinstance(result, Path)
        assert "nvidia-geforce_rtx_3090-00-cuda-balanced.json" in str(result)

    def test_write_creates_profiles_dir(self, tmp_path: Path) -> None:
        """write_profile creates profiles_dir if it doesn't exist."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            write_profile,
        )

        profiles_dir = tmp_path / "new_profiles"
        assert not profiles_dir.exists()

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        write_profile(profiles_dir, record)
        assert profiles_dir.exists()
        assert profiles_dir.is_dir()

    def test_write_file_has_owner_only_permissions(self, tmp_path: Path) -> None:
        """Written profile file has 0o600 permissions."""
        import stat
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            write_profile,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = write_profile(tmp_path, record)
        mode = result.stat().st_mode
        assert stat.S_IMODE(mode) == 0o600

    def test_write_sycl_profile(self, tmp_path: Path) -> None:
        """write_profile works with SYCL backend."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            write_profile,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="intel-arc_b580-00",
            backend="sycl",
            flavor=ProfileFlavor.FAST,
            driver_version="750.1.0",
            driver_version_hash=compute_driver_version_hash("750.1.0"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=200.0,
                avg_latency_ms=5.0,
                peak_vram_mb=4096.0,
            ),
        )

        result = write_profile(tmp_path, record)
        assert result.exists()
        assert "intel-arc_b580-00-sycl-fast.json" in str(result)


# ---------------------------------------------------------------------------
# Test read_profile
# ---------------------------------------------------------------------------


class TestReadProfile:
    """Tests for read_profile function."""

    def test_file_not_found_returns_none(self, tmp_path: Path) -> None:
        """Returns None when profile file does not exist."""
        from llama_manager.profile_cache import (
            read_profile,
        )

        result = read_profile(
            tmp_path,
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
        )
        assert result is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        """Returns None when JSON file is corrupted."""
        from llama_manager.profile_cache import (
            read_profile,
        )

        profile_path = tmp_path / "nvidia-geforce_rtx_3090-00-cuda-balanced.json"
        profile_path.write_text("not valid json{{{", encoding="utf-8")

        result = read_profile(
            tmp_path,
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
        )
        assert result is None

    def test_missing_required_field_returns_none(self, tmp_path: Path) -> None:
        """Returns None when required fields are missing."""
        from llama_manager.profile_cache import (
            read_profile,
        )

        profile_path = tmp_path / "nvidia-geforce_rtx_3090-00-cuda-balanced.json"
        profile_path.write_text(
            json.dumps({"gpu_identifier": "test"}),  # missing most fields
            encoding="utf-8",
        )

        result = read_profile(
            tmp_path,
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
        )
        assert result is None

    def test_unsupported_schema_version_returns_none(self, tmp_path: Path) -> None:
        """Returns None when schema version is not current."""
        from llama_manager.profile_cache import (
            read_profile,
        )

        profile_path = tmp_path / "nvidia-geforce_rtx_3090-00-cuda-balanced.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema_version": "0.9",
                    "gpu_identifier": "test",
                    "backend": "cuda",
                    "flavor": "balanced",
                    "driver_version": "1.0",
                    "driver_version_hash": "abc123",
                    "server_binary_version": "v1",
                    "profiled_at": "2025-01-01T00:00:00Z",
                    "metrics": {
                        "tokens_per_second": 100.0,
                        "avg_latency_ms": 10.0,
                        "peak_vram_mb": 8192.0,
                    },
                }
            ),
            encoding="utf-8",
        )

        result = read_profile(
            tmp_path,
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
        )
        assert result is None

    def test_valid_profile_returns_record(self, tmp_path: Path) -> None:
        """Returns a ProfileRecord when the file is valid."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            read_profile,
            write_profile,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )
        write_profile(tmp_path, record)

        result = read_profile(
            tmp_path,
            gpu_identifier=record.gpu_identifier,
            backend=record.backend,
            flavor=record.flavor,
        )
        assert result is not None
        assert result.gpu_identifier == record.gpu_identifier
        assert result.backend == record.backend
        assert result.flavor == record.flavor
        assert result.metrics.tokens_per_second == 100.0

    def test_non_dict_json_returns_none(self, tmp_path: Path) -> None:
        """Returns None when JSON is not a dict (e.g. a list)."""
        from llama_manager.profile_cache import (
            read_profile,
        )

        profile_path = tmp_path / "nvidia-geforce_rtx_3090-00-cuda-balanced.json"
        profile_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        result = read_profile(
            tmp_path,
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
        )
        assert result is None

    def test_invalid_metrics_returns_none(self, tmp_path: Path) -> None:
        """Returns None when metrics dict is malformed."""
        from llama_manager.profile_cache import (
            read_profile,
        )

        profile_path = tmp_path / "nvidia-geforce_rtx_3090-00-cuda-balanced.json"
        profile_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "gpu_identifier": "test",
                    "backend": "cuda",
                    "flavor": "balanced",
                    "driver_version": "1.0",
                    "driver_version_hash": "abc123",
                    "server_binary_version": "v1",
                    "profiled_at": "2025-01-01T00:00:00Z",
                    "metrics": {"bad": "data"},  # missing required keys
                }
            ),
            encoding="utf-8",
        )

        result = read_profile(
            tmp_path,
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test check_staleness
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    """Tests for check_staleness function."""

    def test_fresh_profile_not_stale(self) -> None:
        """Fresh profile with matching driver and binary is not stale."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            check_staleness,
            compute_driver_version_hash,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="545.23.08",
            current_binary_version="llama-server 1.0.0",
            staleness_days=30,
        )
        assert result.is_stale is False
        assert result.reasons == []
        assert result.driver_version_display == "545.23.08"
        assert 0 <= result.age_days <= 2

    def test_driver_changed_is_stale(self) -> None:
        """Profile is stale when driver version changed."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            StalenessReason,
            check_staleness,
            compute_driver_version_hash,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="546.01.00",  # different driver
            current_binary_version="llama-server 1.0.0",
            staleness_days=30,
        )
        assert result.is_stale is True
        assert StalenessReason.DRIVER_CHANGED in result.reasons
        assert len(result.reasons) == 1

    def test_binary_changed_is_stale(self) -> None:
        """Profile is stale when binary version changed."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            StalenessReason,
            check_staleness,
            compute_driver_version_hash,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="545.23.08",
            current_binary_version="llama-server 2.0.0",  # different binary
            staleness_days=30,
        )
        assert result.is_stale is True
        assert StalenessReason.BINARY_CHANGED in result.reasons
        assert len(result.reasons) == 1

    def test_age_exceeded_is_stale(self) -> None:
        """Profile is stale when age exceeds staleness_days."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            StalenessReason,
            check_staleness,
            compute_driver_version_hash,
        )

        old_date = datetime.now(UTC) - timedelta(days=60)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=old_date.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="545.23.08",
            current_binary_version="llama-server 1.0.0",
            staleness_days=30,
        )
        assert result.is_stale is True
        assert StalenessReason.AGE_EXCEEDED in result.reasons
        assert result.age_days > 30

    def test_all_three_conditions_stale(self) -> None:
        """Profile is stale when all three conditions fail."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            StalenessReason,
            check_staleness,
            compute_driver_version_hash,
        )

        old_date = datetime.now(UTC) - timedelta(days=60)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=old_date.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="546.01.00",  # driver changed
            current_binary_version="llama-server 2.0.0",  # binary changed
            staleness_days=30,
        )
        assert result.is_stale is True
        assert len(result.reasons) == 3
        assert StalenessReason.DRIVER_CHANGED in result.reasons
        assert StalenessReason.BINARY_CHANGED in result.reasons
        assert StalenessReason.AGE_EXCEEDED in result.reasons

    def test_naive_datetime_treated_as_utc(self) -> None:
        """Naive datetime strings are treated as UTC."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            check_staleness,
            compute_driver_version_hash,
        )

        now = datetime.now() - timedelta(days=1)  # naive
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),  # naive ISO string
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="545.23.08",
            current_binary_version="llama-server 1.0.0",
            staleness_days=30,
        )
        assert result.is_stale is False

    def test_invalid_profiled_at_treated_as_stale(self) -> None:
        """Invalid profiled_at timestamp is treated as stale."""
        from llama_manager.profile_cache import (
            StalenessReason,
            check_staleness,
            compute_driver_version_hash,
        )

        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at="not-a-date",
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="545.23.08",
            current_binary_version="llama-server 1.0.0",
            staleness_days=30,
        )
        assert result.is_stale is True
        assert StalenessReason.AGE_EXCEEDED in result.reasons

    def test_warning_message_format(self) -> None:
        """StalenessResult.warning_message formats reasons correctly."""
        from llama_manager.profile_cache import (
            StalenessReason,
            StalenessResult,
        )

        result = StalenessResult(
            is_stale=True,
            reasons=[
                StalenessReason.DRIVER_CHANGED,
                StalenessReason.AGE_EXCEEDED,
            ],
            driver_version_display="545.23.08",
            age_days=45.0,
        )
        assert "Driver Changed" in result.warning_message
        assert "Age Exceeded" in result.warning_message
        assert "Binary Changed" not in result.warning_message

    def test_non_stale_warning_message_empty(self) -> None:
        """Non-stale result has empty warning message."""
        from llama_manager.profile_cache import (
            StalenessResult,
        )

        result = StalenessResult(is_stale=False)
        assert result.warning_message == ""

    def test_staleness_days_zero_never_age_exceeded(self) -> None:
        """When staleness_days is 0, age is never considered exceeded."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            check_staleness,
            compute_driver_version_hash,
        )

        old_date = datetime.now(UTC) - timedelta(days=365)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=old_date.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )

        result = check_staleness(
            record,
            current_driver_version="545.23.08",
            current_binary_version="llama-server 1.0.0",
            staleness_days=0,
        )
        assert result.is_stale is False


# ---------------------------------------------------------------------------
# Test load_profile_with_staleness
# ---------------------------------------------------------------------------


class TestLoadProfileWithStaleness:
    """Tests for load_profile_with_staleness function."""

    def test_no_file_returns_none_tuple(self, tmp_path: Path) -> None:
        """Returns (None, None) when profile file does not exist."""
        from llama_manager.profile_cache import (
            load_profile_with_staleness,
        )

        record, staleness = load_profile_with_staleness(
            tmp_path,
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            current_driver_version="545.23.08",
            current_binary_version="llama-server 1.0.0",
            staleness_days=30,
        )
        assert record is None
        assert staleness is None

    def test_valid_file_returns_record_and_staleness(self, tmp_path: Path) -> None:
        """Returns (record, staleness) when profile exists and is valid."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            load_profile_with_staleness,
            write_profile,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )
        write_profile(tmp_path, record)

        loaded_record, staleness = load_profile_with_staleness(
            tmp_path,
            gpu_identifier=record.gpu_identifier,
            backend=record.backend,
            flavor=record.flavor,
            current_driver_version=record.driver_version,
            current_binary_version=record.server_binary_version,
            staleness_days=30,
        )
        assert loaded_record is not None
        assert loaded_record.gpu_identifier == record.gpu_identifier
        assert staleness is not None
        assert staleness.is_stale is False

    def test_stale_profile_returns_staleness_result(self, tmp_path: Path) -> None:
        """Returns staleness result with reasons when profile is stale."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            load_profile_with_staleness,
            write_profile,
        )

        old_date = datetime.now(UTC) - timedelta(days=60)
        stale_record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=old_date.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
        )
        write_profile(tmp_path, stale_record)

        loaded_record, staleness = load_profile_with_staleness(
            tmp_path,
            gpu_identifier=stale_record.gpu_identifier,
            backend=stale_record.backend,
            flavor=stale_record.flavor,
            current_driver_version="546.01.00",  # driver changed
            current_binary_version="llama-server 2.0.0",  # binary changed
            staleness_days=30,
        )
        assert loaded_record is not None
        assert staleness is not None
        assert staleness.is_stale is True
        assert len(staleness.reasons) == 3


# ---------------------------------------------------------------------------
# Test profile_to_override_dict
# ---------------------------------------------------------------------------


class TestProfileToOverrideDict:
    """Tests for profile_to_override_dict function."""

    def test_filters_to_whitelist_only(self) -> None:
        """Only whitelisted fields are included in the result."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            profile_to_override_dict,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
            parameters={
                "threads": 8,
                "ctx_size": 4096,
                "ubatch_size": 512,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "n_gpu_layers": 99,  # not in whitelist
                "model_alias": "test",  # not in whitelist
            },
        )

        result = profile_to_override_dict(record)

        # Whitelisted fields present
        assert result["threads"] == 8
        assert result["ctx_size"] == 4096
        assert result["ubatch_size"] == 512
        assert result["cache_type_k"] == "f16"
        assert result["cache_type_v"] == "f16"

        # Non-whitelisted fields excluded
        assert "n_gpu_layers" not in result
        assert "model_alias" not in result

    def test_empty_parameters_returns_empty_dict(self) -> None:
        """Returns empty dict when parameters is empty."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            profile_to_override_dict,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
            parameters={},
        )

        result = profile_to_override_dict(record)
        assert result == {}

    def test_only_non_whitelisted_returns_empty_dict(self) -> None:
        """Returns empty dict when all parameters are non-whitelisted."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            profile_to_override_dict,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
            parameters={
                "n_gpu_layers": "all",
                "model_alias": "test",
            },
        )

        result = profile_to_override_dict(record)
        assert result == {}

    def test_values_preserved_as_is(self) -> None:
        """Parameter values are preserved without modification."""
        from datetime import datetime, timedelta

        from llama_manager.profile_cache import (
            compute_driver_version_hash,
            profile_to_override_dict,
        )

        now = datetime.now(UTC) - timedelta(days=1)
        record = ProfileRecord(
            schema_version="1.0",
            gpu_identifier="nvidia-geforce_rtx_3090-00",
            backend="cuda",
            flavor=ProfileFlavor.BALANCED,
            driver_version="545.23.08",
            driver_version_hash=compute_driver_version_hash("545.23.08"),
            server_binary_version="llama-server 1.0.0",
            profiled_at=now.isoformat(),
            metrics=ProfileMetrics(
                tokens_per_second=100.0,
                avg_latency_ms=10.0,
                peak_vram_mb=8192.0,
            ),
            parameters={
                "threads": 16,
                "cache_type_k": "bf16",
            },
        )

        result = profile_to_override_dict(record)
        assert result["threads"] == 16
        assert result["cache_type_k"] == "bf16"
