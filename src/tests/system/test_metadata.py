"""Tests for llama_manager.metadata — GGUF metadata extraction.

Covers:
- Valid GGUF v3 metadata extraction (T045)
- Missing general.name fallback (T046)
- Corrupt file handling (T047)
- Truncated file handling (T048)
- GGUF v4 unsupported error (T049)
- Parse timeout (T050)
- Filename NFKC normalization (T051)
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from llama_manager.metadata import (
    extract_gguf_metadata,
    normalize_filename,
)
from llama_manager.metadata._reader import (
    _extract_architecture_from_reader,
    _extract_field_from_reader,
    _extract_from_gguf_reader,
    _extract_int_field_from_reader,
)
from tests.support.helpers import fixture_path

# ---------------------------------------------------------------------------
# T045 — Valid GGUF v3 metadata extraction
# ---------------------------------------------------------------------------


class TestValidMetadataExtraction:
    """T045: Verify metadata extraction from a valid GGUF v3 file."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_extract_all_fields_from_valid_v3(self) -> None:
        """extract_gguf_metadata should return a complete record for a valid GGUF v3 file."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path, prefix_cap_bytes=65536)

        assert record.raw_path == path
        # architecture detected from "llama" pattern in binary data
        assert record.architecture == "llama"
        # Derived fields
        assert record.normalized_stem == "gguf_v3_valid"
        # Metadata fields
        assert isinstance(record.parse_timestamp, str)
        assert record.parse_timeout_s == 5.0
        assert record.prefix_cap_bytes == 65536

    def test_extract_custom_prefix_cap(self) -> None:
        """extract_gguf_metadata should respect custom prefix_cap_bytes."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path, prefix_cap_bytes=1024)
        assert record.prefix_cap_bytes == 1024

    def test_extract_custom_timeout(self) -> None:
        """extract_gguf_metadata should store parse_timeout_s in the record."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path, parse_timeout_s=10.0)
        assert record.parse_timeout_s == 10.0

    def test_record_is_dataclass(self) -> None:
        """GGUFMetadataRecord should be a dataclass with expected fields."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path)
        # Should have __dataclass_fields__
        assert hasattr(record, "__dataclass_fields__")
        expected_fields = {
            "raw_path",
            "normalized_stem",
            "general_name",
            "architecture",
            "tokenizer_type",
            "embedding_length",
            "block_count",
            "context_length",
            "max_context_length",
            "attention_head_count",
            "attention_head_count_kv",
            "file_type",
            "quantization_type",
            "parse_timestamp",
            "parse_timeout_s",
            "prefix_cap_bytes",
        }
        assert set(record.__dataclass_fields__.keys()) == expected_fields

    def test_file_type_extracted(self) -> None:
        """Extracted record should have file_type (int or None)."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path)
        assert record.file_type is None or isinstance(record.file_type, int)

    def test_quantization_type_extracted(self) -> None:
        """Extracted record should have quantization_type (str or None)."""
        path = str(self._fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path)
        assert record.quantization_type is None or isinstance(record.quantization_type, str)


# ---------------------------------------------------------------------------
# T046 — Missing general.name (fallback to normalized filename stem)
# ---------------------------------------------------------------------------


class TestMissingGeneralName:
    """T046: Verify fallback behavior when general.name is absent."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_general_name_none_when_missing(self) -> None:
        """general_name should be None when not present in GGUF file."""
        path = str(self._fixture_path("gguf_v3_no_name.gguf"))
        record = extract_gguf_metadata(path)
        assert record.general_name is None

    def test_normalized_stem_used_as_fallback(self) -> None:
        """normalized_stem should derive from filename when general.name is missing."""
        path = str(self._fixture_path("gguf_v3_no_name.gguf"))
        record = extract_gguf_metadata(path)
        assert record.normalized_stem == "gguf_v3_no_name"

    def test_other_fields_still_populated(self) -> None:
        """Other fields should still be populated even without general.name."""
        path = str(self._fixture_path("gguf_v3_no_name.gguf"))
        record = extract_gguf_metadata(path)
        assert record.architecture == "llama"
        assert record.context_length is None  # regex doesn't match binary format
        assert record.max_context_length is None
        assert record.attention_head_count is None


# ---------------------------------------------------------------------------
# T047 — Corrupt file (bad magic bytes)
# ---------------------------------------------------------------------------


class TestCorruptFile:
    """T047: Verify error handling for corrupt GGUF files."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_bad_magic_raises_value_error(self) -> None:
        """extract_gguf_metadata should raise ValueError for bad magic bytes."""
        path = str(self._fixture_path("gguf_corrupt.gguf"))
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata(path)
        assert "ggufreader" in str(exc_info.value).lower()

    def test_truncated_file_raises_value_error(self) -> None:
        """Truncated file without valid KV store should raise ValueError."""
        path = str(self._fixture_path("gguf_truncated.gguf"))
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata(path)
        assert "ggufreader" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# T049 — GGUF v4 unsupported version error
# ---------------------------------------------------------------------------


class TestGGUFv4Unsupported:
    """T049: Verify that GGUF v4 files produce an unsupported error."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_v4_file_raises_value_error(self) -> None:
        """extract_gguf_metadata should raise ValueError for GGUF v4 files."""
        path = str(self._fixture_path("gguf_v4_unsupported.gguf"))
        with pytest.raises(ValueError):
            extract_gguf_metadata(path)


# ---------------------------------------------------------------------------
# T050 — Parse timeout
# ---------------------------------------------------------------------------


class TestParseTimeout:
    """T050: Verify timeout handling when parsing takes too long."""

    def test_timeout_raises_timeout_error(self) -> None:
        """extract_gguf_metadata should raise TimeoutError when parse exceeds timeout."""

        def _slow_read(*args, **kwargs):
            time.sleep(2)
            return None

        with patch(
            "llama_manager.metadata._try_gguf_reader",
            side_effect=_slow_read,
        ):
            with pytest.raises(TimeoutError) as exc_info:
                extract_gguf_metadata(
                    "/fake/model.gguf",
                    prefix_cap_bytes=1024,
                    parse_timeout_s=0.1,
                )
            assert "timed out" in str(exc_info.value).lower()

    def test_timeout_message_contains_path(self) -> None:
        """TimeoutError message should include the model path."""
        model_path = "/models/my-model.gguf"

        def _slow_read(*args, **kwargs):
            time.sleep(2)
            return None

        with patch(
            "llama_manager.metadata._try_gguf_reader",
            side_effect=_slow_read,
        ):
            with pytest.raises(TimeoutError) as exc_info:
                extract_gguf_metadata(
                    model_path,
                    prefix_cap_bytes=1024,
                    parse_timeout_s=0.1,
                )
            assert model_path in str(exc_info.value)

    def test_timeout_does_not_affect_fast_files(self) -> None:
        """Fast files should not trigger timeout."""
        path = str(fixture_path("gguf_v3_valid.gguf"))
        record = extract_gguf_metadata(path, prefix_cap_bytes=65536, parse_timeout_s=0.5)
        assert record.architecture == "llama"

    def test_thread_exception_propagated(self) -> None:
        """Exceptions in the parse thread should propagate to the caller."""

        def _fail_read(*args, **kwargs):
            raise OSError("disk full")

        with patch(
            "llama_manager.metadata._try_gguf_reader",
            side_effect=_fail_read,
        ):
            with pytest.raises(OSError) as exc_info:
                extract_gguf_metadata("/fake/model.gguf")
            assert "disk full" in str(exc_info.value)


# ---------------------------------------------------------------------------
# T051 — Filename NFKC normalization
# ---------------------------------------------------------------------------


class TestFilenameNormalization:
    """T051: Verify NFKC normalization of filenames."""

    def test_simple_ascii(self) -> None:
        """Simple ASCII filename should pass through unchanged."""
        assert normalize_filename("my-model") == "my-model"

    def test_whitespace_replaced_with_underscore(self) -> None:
        """Whitespace sequences should be replaced with single underscore."""
        assert normalize_filename("my model") == "my_model"
        assert normalize_filename("my  model") == "my_model"
        assert normalize_filename("my   model") == "my_model"

    def test_leading_trailing_underscore_stripped(self) -> None:
        """Leading and trailing underscores should be removed."""
        assert normalize_filename("_my_model_") == "my_model"
        assert normalize_filename("__my_model__") == "my_model"

    def test_multiple_underscores_collapsed(self) -> None:
        """Multiple consecutive underscores should be collapsed."""
        assert normalize_filename("my__model") == "my_model"
        assert normalize_filename("my___model") == "my_model"

    def test_nfkc_combining_characters(self) -> None:
        """NFKC normalization should decompose compatibility characters."""
        # ﬃ (U+FB03, LATIN SMALL LIGATURE FFI) → ffi under NFKC
        # The ligature alone should normalize to "ffi"
        assert normalize_filename("\ufb03") == "ffi"

    def test_invalid_chars_replaced(self) -> None:
        """Invalid filename characters should be replaced with underscore."""
        # Control characters
        result = normalize_filename("my\x00model")
        assert "my" in result and "model" in result
        # Forward slash
        assert normalize_filename("my/model") == "my_model"
        # Backslash
        assert normalize_filename("my\\model") == "my_model"
        # Colon
        assert normalize_filename("my:model") == "my_model"

    def test_uppercase_preserved(self) -> None:
        """Uppercase letters should be preserved."""
        assert normalize_filename("MyModel") == "MyModel"

    def test_numbers_preserved(self) -> None:
        """Numbers should be preserved in the normalized name."""
        assert normalize_filename("model-v2") == "model-v2"
        assert normalize_filename("model_v2") == "model_v2"

    def test_empty_string_returns_unknown(self) -> None:
        """Empty string should return 'unknown'."""
        assert normalize_filename("") == "unknown"

    def test_whitespace_only_returns_unknown(self) -> None:
        """Whitespace-only string should return 'unknown'."""
        assert normalize_filename("   ") == "unknown"
        assert normalize_filename("\t\n") == "unknown"

    def test_underscore_only_returns_unknown(self) -> None:
        """Underscore-only string should return 'unknown' after stripping."""
        assert normalize_filename("___") == "unknown"

    def test_unicode_special_chars(self) -> None:
        """Unicode characters outside ASCII range should be handled."""
        # Emoticon and other non-filename-safe chars
        result = normalize_filename("model\x01\x02test")
        assert "model" in result and "test" in result

    def test_mixed_whitespace_and_underscores(self) -> None:
        """Mixed whitespace and underscores should be normalized."""
        # Spaces become underscores, then collapse
        assert normalize_filename("my model name") == "my_model_name"

    def test_pipe_character_replaced(self) -> None:
        """Pipe character (|) is invalid in filenames and should be replaced."""
        assert normalize_filename("my|model") == "my_model"

    def test_question_mark_replaced(self) -> None:
        """Question mark is invalid in filenames and should be replaced."""
        assert normalize_filename("my?model") == "my_model"

    def test_asterisk_replaced(self) -> None:
        """Asterisk is invalid in filenames and should be replaced."""
        assert normalize_filename("my*model") == "my_model"

    def test_greater_less_than_replaced(self) -> None:
        """Angle brackets are invalid in filenames and should be replaced."""
        assert normalize_filename("my<model>") == "my_model"

    def test_double_quoted_replaced(self) -> None:
        """Double quotes are invalid in filenames and should be replaced."""
        assert normalize_filename('my"model') == "my_model"

    def test_complex_mixed_input(self) -> None:
        """Complex mixed input should be fully normalized."""
        result = normalize_filename("  My  Model  v2  ")
        assert result == "My_Model_v2"

    def test_newlines_and_tabs(self) -> None:
        """Newlines and tabs are whitespace and should be replaced."""
        result = normalize_filename("my\nmodel\tname")
        assert result == "my_model_name"

    def test_null_byte_replaced(self) -> None:
        """Null byte is invalid and should be replaced."""
        result = normalize_filename("my\x00model")
        # Should contain parts of the original string
        assert "my" in result
        assert "model" in result


# ---------------------------------------------------------------------------
# Private helper function tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# prefix_cap_bytes / parse_timeout_s validation
# ---------------------------------------------------------------------------


class TestExtractGgufMetadataValidation:
    """Validation of prefix_cap_bytes and parse_timeout_s parameters."""

    def test_prefix_cap_bytes_zero_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for prefix_cap_bytes=0."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", prefix_cap_bytes=0)
        assert "prefix_cap_bytes" in str(exc_info.value).lower()
        assert "positive" in str(exc_info.value).lower()

    def test_prefix_cap_bytes_negative_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for negative prefix_cap_bytes."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", prefix_cap_bytes=-100)
        assert "prefix_cap_bytes" in str(exc_info.value).lower()

    def test_parse_timeout_zero_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for parse_timeout_s=0."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", parse_timeout_s=0.0)
        assert "parse_timeout_s" in str(exc_info.value).lower()
        assert "positive" in str(exc_info.value).lower()

    def test_parse_timeout_negative_raises(self) -> None:
        """extract_gguf_metadata should raise ValueError for negative parse_timeout_s."""
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/fake/model.gguf", parse_timeout_s=-1.0)
        assert "parse_timeout_s" in str(exc_info.value).lower()

    def test_validation_before_any_io(self) -> None:
        """Validation errors should occur before any file reads or thread creation."""
        # Should raise ValueError without touching the filesystem
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata("/nonexistent/model.gguf", prefix_cap_bytes=0)
        # If we get here, ValueError was raised before any IO attempt
        assert exc_info.value is not None

    def test_valid_parameters_accepted(self) -> None:
        """Valid positive parameters should not raise."""
        # Use a fixture that exists
        path = str(fixture_path("gguf_v3_valid.gguf"))
        # Should succeed without raising
        record = extract_gguf_metadata(path, prefix_cap_bytes=1024, parse_timeout_s=1.0)
        assert record is not None
        assert record.prefix_cap_bytes == 1024
        assert record.parse_timeout_s == 1.0


"""Generate GGUF test fixtures as a side-effect test.

This test generates synthetic GGUF binary files for metadata extraction tests.
It is idempotent and safe to run repeatedly.
"""


import struct

# Value type constants
_GGUF_TYPE_UINT32: int = 4
_GGUF_TYPE_STRING: int = 8

# Required keys for a minimal valid llama model GGUF v3 file
_REQUIRED_KEYS: dict[str, tuple[int, bytes]] = {
    "general.architecture": (_GGUF_TYPE_STRING, b"llama"),
    "tokenizer.type": (_GGUF_TYPE_STRING, b"bpe"),
    "llama.embedding_length": (_GGUF_TYPE_UINT32, struct.pack("<I", 4096)),
    "llama.block_count": (_GGUF_TYPE_UINT32, struct.pack("<I", 32)),
    "llama.context_length": (_GGUF_TYPE_UINT32, struct.pack("<I", 8192)),
    "llama.attention.head_count": (_GGUF_TYPE_UINT32, struct.pack("<I", 32)),
    "llama.attention.head_count_kv": (_GGUF_TYPE_UINT32, struct.pack("<I", 8)),
}

_GENERAL_NAME_VALUE: bytes = struct.pack("<Q", len(b"test-model-v1")) + b"test-model-v1"


def _pack_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    key_bytes = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_bytes)) + key_bytes + struct.pack("<I", value_type) + value_bytes
    )


def _build_kv_section(keys_with_values: dict[str, tuple[int, bytes]]) -> bytes:
    return b"".join(_pack_kv(k, vt, vb) for k, (vt, vb) in keys_with_values.items())


def _count_kv_pairs(kv_section: bytes) -> int:
    count = 0
    offset = 0
    while offset < len(kv_section):
        if offset + 8 > len(kv_section):
            break
        key_len = struct.unpack_from("<Q", kv_section, offset)[0]
        if key_len == 0 or offset + 8 + key_len > len(kv_section):
            break
        offset += 8 + key_len
        if offset + 4 > len(kv_section):
            break
        value_type = struct.unpack_from("<I", kv_section, offset)[0]
        offset += 4
        if value_type == _GGUF_TYPE_STRING:
            if offset + 8 > len(kv_section):
                break
            str_len = struct.unpack_from("<Q", kv_section, offset)[0]
            offset += 8 + str_len
        elif value_type == _GGUF_TYPE_UINT32:
            offset += 4
        else:
            break
        count += 1
    return count


def _write_gguf_v3(
    path: Path,
    kv_section: bytes,
    magic: bytes = b"GGUF",
    version: int = 3,
) -> None:
    kv_count = struct.pack("<Q", _count_kv_pairs(kv_section))
    # GGUF v3 header: magic(4) + version(4) + tensor_count(8) + kv_count(8)
    tensor_count = struct.pack("<Q", 0)
    header = magic + struct.pack("<I", version) + tensor_count + kv_count
    path.write_bytes(header + kv_section)


def _generate_valid_v3(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME_VALUE) + kv
    _write_gguf_v3(path, kv)


def _generate_valid_v3_no_name(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    _write_gguf_v3(path, kv)


def _generate_corrupt(path: Path) -> None:
    path.write_bytes(b"XXXX\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def _generate_truncated(path: Path) -> None:
    path.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def _generate_v4_unsupported(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME_VALUE) + kv
    _write_gguf_v3(path, kv, magic=b"GGUF", version=4)


# ---------------------------------------------------------------------------
# TestReaderHelpers — _reader.py GGUFReader-based helpers
# ---------------------------------------------------------------------------


class TestExtractArchitectureFromReader:
    """Tests for _extract_architecture_from_reader."""

    def test_returns_none_when_field_missing(self) -> None:

        result = _extract_architecture_from_reader({})
        assert result is None

    def test_returns_none_on_value_error(self) -> None:
        from unittest.mock import MagicMock

        field = MagicMock()
        field.contents.side_effect = ValueError("bad")
        from gguf.constants import Keys

        result = _extract_architecture_from_reader({Keys.General.ARCHITECTURE: field})
        assert result is None

    def test_returns_string_value(self) -> None:
        from unittest.mock import MagicMock

        field = MagicMock()
        field.contents.return_value = "llama"
        from gguf.constants import Keys

        result = _extract_architecture_from_reader({Keys.General.ARCHITECTURE: field})
        assert result == "llama"


class TestExtractFieldFromReader:
    """Tests for _extract_field_from_reader."""

    def test_returns_none_when_key_missing(self) -> None:

        assert _extract_field_from_reader({}, "general.name") is None

    def test_returns_none_on_value_error(self) -> None:
        from unittest.mock import MagicMock

        field = MagicMock()
        field.contents.side_effect = ValueError("bad")
        assert _extract_field_from_reader({"general.name": field}, "general.name") is None

    def test_returns_string_value(self) -> None:
        from unittest.mock import MagicMock

        field = MagicMock()
        field.contents.return_value = "MyModel"
        assert _extract_field_from_reader({"general.name": field}, "general.name") == "MyModel"


class TestExtractIntFieldFromReader:
    """Tests for _extract_int_field_from_reader."""

    def test_returns_none_when_key_missing(self) -> None:

        assert _extract_int_field_from_reader({}, "llama.context_length") is None

    def test_returns_none_on_value_error(self) -> None:
        from unittest.mock import MagicMock

        field = MagicMock()
        field.contents.side_effect = ValueError("bad")
        assert _extract_int_field_from_reader({"k": field}, "k") is None

    def test_returns_none_on_type_error(self) -> None:
        from unittest.mock import MagicMock

        field = MagicMock()
        field.contents.return_value = None  # int(None) raises TypeError
        assert _extract_int_field_from_reader({"k": field}, "k") is None

    def test_returns_int_value(self) -> None:
        from unittest.mock import MagicMock

        field = MagicMock()
        field.contents.return_value = 4096
        assert _extract_int_field_from_reader({"k": field}, "k") == 4096


class TestExtractFromGGUFReader:
    """Tests for _extract_from_gguf_reader — no-architecture and cancel paths."""

    def test_no_architecture_returns_none_fields(self) -> None:

        record = _extract_from_gguf_reader(
            model_path="/fake/model.gguf",
            fields={},
            parse_timeout_s=30.0,
            prefix_cap_bytes=65536,
        )
        assert record.architecture is None
        assert record.context_length is None
        assert record.max_context_length is None
        assert record.embedding_length is None
        assert record.block_count is None

    def test_cancel_event_raises_interrupted(self) -> None:
        import threading
        from unittest.mock import MagicMock

        from gguf.constants import Keys

        cancel = threading.Event()
        cancel.set()

        arch_field = MagicMock()
        arch_field.contents.return_value = "llama"

        with pytest.raises(InterruptedError, match="parse cancelled"):
            _extract_from_gguf_reader(
                model_path="/fake/model.gguf",
                fields={Keys.General.ARCHITECTURE: arch_field},
                parse_timeout_s=30.0,
                prefix_cap_bytes=65536,
                cancel_event=cancel,
            )


def test_generate_gguf_fixtures(tmp_path: Path) -> None:
    """Generate all GGUF test fixtures for metadata extraction tests.

    Creates 5 synthetic GGUF files under tmp_path / "fixtures/" (ephemeral):
    - gguf_v3_valid.gguf: valid GGUF v3 with all required keys
    - gguf_v3_no_name.gguf: valid GGUF v3 missing general.name
    - gguf_corrupt.gguf: corrupt file (bad magic bytes)
    - gguf_truncated.gguf: truncated file (valid header, no KV data)
    - gguf_v4_unsupported.gguf: valid GGUF v4 (unsupported version)

    All fixtures are under 10 KiB and contain no tensor data.
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("gguf_v3_valid.gguf", _generate_valid_v3),
        ("gguf_v3_no_name.gguf", _generate_valid_v3_no_name),
        ("gguf_corrupt.gguf", _generate_corrupt),
        ("gguf_truncated.gguf", _generate_truncated),
        ("gguf_v4_unsupported.gguf", _generate_v4_unsupported),
    ]

    for name, gen_fn in generators:
        path = fixtures_dir / name
        gen_fn(path)
        size = path.stat().st_size
        assert size > 0, f"Fixture {name} is empty"
        assert size < 10240, f"Fixture {name} exceeds 10 KiB ({size} bytes)"
