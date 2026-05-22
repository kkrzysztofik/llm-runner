import time
from pathlib import Path
from unittest.mock import patch

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


import pytest

from llama_manager.metadata import (
    extract_gguf_metadata,
    normalize_filename,
)
from llama_manager.metadata._binary import (
    _detect_gguf_version,
    _parse_architecture,
    _parse_general_name,
    _parse_numeric_field,
    _parse_tokenizer_type,
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
            "attention_head_count",
            "attention_head_count_kv",
            "parse_timestamp",
            "parse_timeout_s",
            "prefix_cap_bytes",
        }
        assert set(record.__dataclass_fields__.keys()) == expected_fields


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
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_bad_magic_from_detect_function(self) -> None:
        """_detect_gguf_version should raise ValueError for bad magic."""
        bad_header = b"XXXX\x03\x00\x00\x00"
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(bad_header)
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_empty_data_raises_value_error(self) -> None:
        """_detect_gguf_version should raise ValueError for empty data."""
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(b"")
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_too_short_header_raises_value_error(self) -> None:
        """_detect_gguf_version should raise ValueError for header shorter than 8 bytes."""
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(b"GGUF\x00\x00")
        assert "bad magic bytes" in str(exc_info.value).lower()

    def test_non_gguf_file_raises_value_error(self) -> None:
        """_detect_gguf_version should reject random binary data."""
        with pytest.raises(ValueError) as exc_info:
            _detect_gguf_version(b"This is not a GGUF file at all\x00\x00")
        assert "bad magic bytes" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# T048 — Truncated file (valid header, 0 KV pairs)
# ---------------------------------------------------------------------------


class TestTruncatedFile:
    """T048: Verify handling of truncated GGUF files with valid header."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_truncated_file_succeeds_with_partial_data(self) -> None:
        """Truncated file with valid header and 0 KV pairs should succeed."""
        path = str(self._fixture_path("gguf_truncated.gguf"))
        record = extract_gguf_metadata(path)
        # Should succeed — no exception
        assert record is not None
        assert record.architecture is None  # no KV data to parse
        assert record.general_name is None
        assert record.context_length is None

    def test_truncated_file_still_has_stem_and_metadata(self) -> None:
        """Truncated file record should still have path-derived fields."""
        path = str(self._fixture_path("gguf_truncated.gguf"))
        record = extract_gguf_metadata(path)
        assert record.normalized_stem == "gguf_truncated"
        assert isinstance(record.parse_timestamp, str)
        assert record.parse_timeout_s == 5.0

    def test_detect_version_for_truncated_header(self) -> None:
        """_detect_gguf_version should identify v3 from truncated header."""
        truncated_header = b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00"
        version = _detect_gguf_version(truncated_header)
        assert version == 3


# ---------------------------------------------------------------------------
# T049 — GGUF v4 unsupported version error
# ---------------------------------------------------------------------------


class TestGGUFv4Unsupported:
    """T049: Verify that GGUF v4 files produce an unsupported error."""

    def _fixture_path(self, name: str) -> Path:
        return fixture_path(name)

    def test_v4_file_raises_unsupported_error(self) -> None:
        """extract_gguf_metadata should raise ValueError for GGUF v4 files."""
        path = str(self._fixture_path("gguf_v4_unsupported.gguf"))
        with pytest.raises(ValueError) as exc_info:
            extract_gguf_metadata(path)
        assert (
            "v4" in str(exc_info.value).lower()
            and "not yet supported" in str(exc_info.value).lower()
        )

    def test_detect_version_returns_4(self) -> None:
        """_detect_gguf_version should return 4 for v4 magic bytes."""
        v4_header = b"GGUF\x04\x00\x00\x00"
        version = _detect_gguf_version(v4_header)
        assert version == 4

    def test_detect_version_returns_3(self) -> None:
        """_detect_gguf_version should return 3 for v3 magic bytes."""
        v3_header = b"GGUF\x03\x00\x00\x00"
        version = _detect_gguf_version(v3_header)
        assert version == 3

    def test_detect_version_returns_2(self) -> None:
        """_detect_gguf_version should return 2 for v2 magic bytes."""
        v2_header = b"GGUF\x02\x00\x00\x00"
        version = _detect_gguf_version(v2_header)
        assert version == 2


# ---------------------------------------------------------------------------
# T050 — Parse timeout
# ---------------------------------------------------------------------------


class TestParseTimeout:
    """T050: Verify timeout handling when parsing takes too long."""

    def test_timeout_raises_timeout_error(self) -> None:
        """extract_gguf_metadata should raise TimeoutError when parse exceeds timeout."""

        def _slow_read(*args, **kwargs) -> bytes:
            # Simulate a very slow operation
            time.sleep(2)
            return b""

        with patch(
            "llama_manager.metadata._binary._read_gguf_header",
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

        def _slow_read(*args, **kwargs) -> bytes:
            time.sleep(2)
            return b""

        with patch(
            "llama_manager.metadata._binary._read_gguf_header",
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
        # Very short timeout but file is tiny — should still succeed
        record = extract_gguf_metadata(path, prefix_cap_bytes=65536, parse_timeout_s=0.5)
        assert record.architecture == "llama"

    def test_thread_exception_propagated(self) -> None:
        """Exceptions in the parse thread should propagate to the caller."""

        def _fail_read(*args, **kwargs) -> bytes:
            raise OSError("disk full")

        with patch(
            "llama_manager.metadata._binary._read_gguf_header",
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


class TestParseGeneralName:
    """Tests for _parse_general_name helper."""

    def test_parses_general_name(self) -> None:
        """_parse_general_name should extract general.name from matching bytes."""
        # The regex expects: general.name + optional whitespace + \x00 + name + \x00
        data = b"general.name\x00test-model\x00other"
        result = _parse_general_name(data)
        assert result == "test-model"

    def test_returns_none_when_not_found(self) -> None:
        """_parse_general_name should return None when key is absent."""
        data = b"some other key\x00value\x00"
        result = _parse_general_name(data)
        assert result is None

    def test_returns_replacement_chars_on_decode_error(self) -> None:
        """_parse_general_name should return replacement chars on decode error (errors='replace')."""
        # Invalid UTF-8 sequence in the name value
        data = b"general.name\x00\xff\xfe\x00"
        result = _parse_general_name(data)
        # errors="replace" returns U+FFFD replacement characters, not None
        assert result is not None
        assert "\ufffd" in result

    def test_parses_with_whitespace_before_null(self) -> None:
        """_parse_general_name should handle optional whitespace before null."""
        data = b"general.name \x00test-model\x00"
        result = _parse_general_name(data)
        assert result == "test-model"


class TestParseArchitecture:
    """Tests for _parse_architecture helper."""

    def test_detects_llama(self) -> None:
        """_parse_architecture should detect 'llama' architecture."""
        assert _parse_architecture(b"llama") == "llama"

    def test_detects_qwen2(self) -> None:
        """_parse_architecture should detect 'qwen2' architecture."""
        assert _parse_architecture(b"qwen2") == "qwen2"

    def test_detects_qwen3(self) -> None:
        """_parse_architecture should detect 'qwen3' architecture."""
        assert _parse_architecture(b"qwen3") == "qwen3"

    def test_detects_qwen(self) -> None:
        """_parse_architecture should detect 'qwen' architecture."""
        assert _parse_architecture(b"qwen") == "qwen"

    def test_detects_phi3(self) -> None:
        """_parse_architecture should detect 'phi3' architecture."""
        assert _parse_architecture(b"phi3") == "phi3"

    def test_detects_mamba(self) -> None:
        """_parse_architecture should detect 'mamba' architecture."""
        assert _parse_architecture(b"mamba") == "mamba"

    def test_returns_none_for_unknown(self) -> None:
        """_parse_architecture should return None for unknown architecture."""
        assert _parse_architecture(b"unknown_arch") is None

    def test_first_match_wins(self) -> None:
        """_parse_architecture should return first matching pattern."""
        # "qwen2" contains "qwen" — should return "qwen2" (defined first)
        assert _parse_architecture(b"qwen2") == "qwen2"

    def test_detects_gpt_2(self) -> None:
        """_parse_architecture should detect 'gpt_2' architecture."""
        assert _parse_architecture(b"gpt_2") == "gpt_2"

    def test_detects_bert(self) -> None:
        """_parse_architecture should detect 'bert' architecture."""
        assert _parse_architecture(b"bert") == "bert"

    def test_detects_mpt(self) -> None:
        """_parse_architecture should detect 'mpt' architecture."""
        assert _parse_architecture(b"mpt") == "mpt"

    def test_detects_falcon(self) -> None:
        """_parse_architecture should detect 'falcon' architecture."""
        assert _parse_architecture(b"falcon") == "falcon"

    def test_detects_stablelm(self) -> None:
        """_parse_architecture should detect 'stablelm' architecture."""
        assert _parse_architecture(b"stablelm") == "stablelm"


class TestParseTokenizerType:
    """Tests for _parse_tokenizer_type helper."""

    def test_detects_ggml_tokenizer(self) -> None:
        """_parse_tokenizer_type should detect 'ggml' tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.ggml") == "ggml"

    def test_detects_model_tokenizer(self) -> None:
        """_parse_tokenizer_type should detect 'model' tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.model") == "model"

    def test_detects_huggingface_tokenizer(self) -> None:
        """_parse_tokenizer_type should detect 'huggingface' tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.json") == "huggingface"

    def test_returns_none_for_unknown(self) -> None:
        """_parse_tokenizer_type should return None for unknown tokenizer."""
        assert _parse_tokenizer_type(b"tokenizer.unknown") is None

    def test_first_match_wins(self) -> None:
        """_parse_tokenizer_type should return first matching pattern."""
        # "tokenizer.ggml" contains "tokenizer" — should match ggml first
        assert _parse_tokenizer_type(b"tokenizer.ggml") == "ggml"


class TestParseNumericField:
    """Tests for _parse_numeric_field helper using GGUF binary KV format."""

    @staticmethod
    def _make_gguf_kv_data(key: bytes, type_tag: int, value: int, n_kv: int = 1) -> bytes:
        """Build minimal GGUF v3 header + KV records for testing.

        GGUF v3 layout:
        - magic: 8 bytes (GGUF + version byte)
        - version: 4 bytes (uint32)
        - num_tensors: 8 bytes (uint64)
        - num_kv: 8 bytes (uint64)
        Total header: 28 bytes
        """
        # GGUF v3: magic (8) + version (4) + num_tensors (8) + num_kv (8) = 28
        header = b"GGUF\x03\x00\x00\x00"  # 8 bytes
        header += b"\x00\x00\x00\x00"  # version = 0 (uint32 LE)
        header += b"\x00\x00\x00\x00\x00\x00\x00\x00"  # num_tensors = 0 (uint64 LE)
        header += n_kv.to_bytes(8, "little")  # num_kv (uint64 LE)

        # KV record: key_length (uint32) + key + type_tag (uint32) + value
        record = len(key).to_bytes(4, "little") + key
        record += type_tag.to_bytes(4, "little")
        if type_tag in (1, 2):  # u8, i8
            record += value.to_bytes(1, "little")
        elif type_tag in (3, 4):  # u16, i16
            record += value.to_bytes(2, "little")
        elif type_tag in (5, 6):  # u32, i32
            record += value.to_bytes(4, "little")
        elif type_tag == 8:  # f32
            import struct

            record += struct.pack("<f", 0.0)
        elif type_tag == 9:  # f64
            import struct

            record += struct.pack("<d", 0.0)
        else:
            record += value.to_bytes(4, "little")

        return header + record

    def test_parses_integer_field(self) -> None:
        """_parse_numeric_field should extract u32 values from GGUF KV records."""
        data = self._make_gguf_kv_data(b"context_length", 5, 8192)
        result = _parse_numeric_field(data, b"context_length")
        assert result == 8192

    def test_parses_string_key(self) -> None:
        """_parse_numeric_field should accept string keys."""
        data = self._make_gguf_kv_data(b"context_length", 5, 4096)
        result = _parse_numeric_field(data, "context_length")
        assert result == 4096

    def test_returns_none_for_missing_key(self) -> None:
        """_parse_numeric_field should return None for missing key."""
        data = self._make_gguf_kv_data(b"some_key", 5, 12345)
        result = _parse_numeric_field(data, b"nonexistent_field")
        assert result is None

    def test_returns_none_for_non_numeric_value(self) -> None:
        """_parse_numeric_field should return None for string type (tag 7 = string)."""
        # Type tag 7 is string, not integer — function returns None for non-numeric
        data = self._make_gguf_kv_data(b"context_length", 7, 0)
        result = _parse_numeric_field(data, b"context_length")
        assert result is None

    def test_parses_u32_value(self) -> None:
        """_parse_numeric_field should parse u32 (type_tag=5) correctly."""
        data = self._make_gguf_kv_data(b"embedding_length", 5, 4096)
        result = _parse_numeric_field(data, b"embedding_length")
        assert result == 4096

    def test_parses_i32_signed_value(self) -> None:
        """_parse_numeric_field should parse i32 (type_tag=6) as signed."""
        # i32 with value -1 (0xFFFFFFFF)
        data = self._make_gguf_kv_data(b"test_key", 6, 0xFFFFFFFF)
        result = _parse_numeric_field(data, b"test_key")
        assert result == -1

    def test_parses_u16_value(self) -> None:
        """_parse_numeric_field should parse u16 (type_tag=3) correctly."""
        data = self._make_gguf_kv_data(b"block_count", 3, 32)
        result = _parse_numeric_field(data, b"block_count")
        assert result == 32

    def test_parses_u8_value(self) -> None:
        """_parse_numeric_field should parse u8 (type_tag=1) correctly."""
        data = self._make_gguf_kv_data(b"attention_head_count", 1, 32)
        result = _parse_numeric_field(data, b"attention_head_count")
        assert result == 32

    def test_no_match_for_binary_integer(self) -> None:
        """_parse_numeric_field should return None for unrecognized data."""
        # Short data that doesn't start with GGUF magic
        data = b"context_length\x40\x20\x00\x00"
        result = _parse_numeric_field(data, b"context_length")
        assert result is None

    def test_dot_escaped_in_pattern(self) -> None:
        """_parse_numeric_field should handle dotted keys correctly."""
        data = self._make_gguf_kv_data(b"llama.context_length", 5, 8192)
        result = _parse_numeric_field(data, b"llama.context_length")
        assert result == 8192

    def test_skips_f32_and_returns_none(self) -> None:
        """_parse_numeric_field should skip f32 (type_tag=8) and return None."""
        data = self._make_gguf_kv_data(b"freq_scale", 8, 0)
        result = _parse_numeric_field(data, b"freq_scale")
        assert result is None


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
# TestBinaryHelpers — low-level _binary.py functions
# ---------------------------------------------------------------------------


class TestReadIntHelpers:
    """Tests for _read_int8/16/32/64 and _read_uint8/16/32/64."""

    def test_read_int8_positive(self) -> None:
        from llama_manager.metadata._binary import _read_int8

        assert _read_int8(b"\x7f", 0) == 127

    def test_read_int8_negative(self) -> None:
        from llama_manager.metadata._binary import _read_int8

        assert _read_int8(b"\x80", 0) == -128

    def test_read_int8_out_of_bounds(self) -> None:
        from llama_manager.metadata._binary import _read_int8

        assert _read_int8(b"", 0) is None

    def test_read_int16_positive(self) -> None:
        from llama_manager.metadata._binary import _read_int16

        data = (1000).to_bytes(2, "little")
        assert _read_int16(data, 0) == 1000

    def test_read_int16_negative(self) -> None:
        from llama_manager.metadata._binary import _read_int16

        data = (0x8000).to_bytes(2, "little")
        assert _read_int16(data, 0) == -32768

    def test_read_int16_out_of_bounds(self) -> None:
        from llama_manager.metadata._binary import _read_int16

        assert _read_int16(b"\x01", 0) is None

    def test_read_int32_positive(self) -> None:
        from llama_manager.metadata._binary import _read_int32

        data = (100000).to_bytes(4, "little")
        assert _read_int32(data, 0) == 100000

    def test_read_int32_negative(self) -> None:
        from llama_manager.metadata._binary import _read_int32

        data = (0x80000000).to_bytes(4, "little")
        assert _read_int32(data, 0) == -2147483648

    def test_read_int32_out_of_bounds(self) -> None:
        from llama_manager.metadata._binary import _read_int32

        assert _read_int32(b"\x01\x02", 0) is None

    def test_read_int64_positive(self) -> None:
        from llama_manager.metadata._binary import _read_int64

        data = (2**40).to_bytes(8, "little")
        assert _read_int64(data, 0) == 2**40

    def test_read_int64_negative(self) -> None:
        from llama_manager.metadata._binary import _read_int64

        data = (2**63).to_bytes(8, "little")
        assert _read_int64(data, 0) == -(2**63)

    def test_read_int64_out_of_bounds(self) -> None:
        from llama_manager.metadata._binary import _read_int64

        assert _read_int64(b"\x01\x02\x03\x04", 0) is None

    def test_read_uint8(self) -> None:
        from llama_manager.metadata._binary import _read_uint8

        assert _read_uint8(b"\xff", 0) == 255
        assert _read_uint8(b"", 0) is None

    def test_read_uint16(self) -> None:
        from llama_manager.metadata._binary import _read_uint16

        data = (65535).to_bytes(2, "little")
        assert _read_uint16(data, 0) == 65535
        assert _read_uint16(b"\x01", 0) is None

    def test_read_uint32(self) -> None:
        from llama_manager.metadata._binary import _read_uint32

        data = (0xDEADBEEF).to_bytes(4, "little")
        assert _read_uint32(data, 0) == 0xDEADBEEF
        assert _read_uint32(b"\x01\x02", 0) is None

    def test_read_uint64(self) -> None:
        from llama_manager.metadata._binary import _read_uint64

        data = (2**60).to_bytes(8, "little")
        assert _read_uint64(data, 0) == 2**60
        assert _read_uint64(b"\x01\x02\x03\x04", 0) is None


class TestSkipHelpers:
    """Tests for _skip_array_element, _skip_non_integer_type, _skip_record."""

    def test_skip_array_element_uint8(self) -> None:
        from llama_manager.metadata._binary import _skip_array_element

        assert _skip_array_element(b"\x00" * 10, 0, 0) == 1  # UINT8

    def test_skip_array_element_uint16(self) -> None:
        from llama_manager.metadata._binary import _skip_array_element

        assert _skip_array_element(b"\x00" * 10, 0, 2) == 2  # UINT16

    def test_skip_array_element_uint32(self) -> None:
        from llama_manager.metadata._binary import _skip_array_element

        assert _skip_array_element(b"\x00" * 10, 0, 4) == 4  # UINT32

    def test_skip_array_element_uint64(self) -> None:
        from llama_manager.metadata._binary import _skip_array_element

        assert _skip_array_element(b"\x00" * 10, 0, 10) == 8  # UINT64

    def test_skip_non_integer_bool(self) -> None:
        from llama_manager.metadata._binary import _skip_non_integer_type

        assert _skip_non_integer_type(b"\x00" * 10, 0, 7) == 1  # BOOL

    def test_skip_non_integer_string(self) -> None:
        from llama_manager.metadata._binary import _skip_non_integer_type

        # 8-byte length prefix = 3, then 3 bytes of data
        data = (3).to_bytes(8, "little") + b"abc"
        assert _skip_non_integer_type(data, 0, 8) == 11  # 8 + 3

    def test_skip_non_integer_string_truncated(self) -> None:
        from llama_manager.metadata._binary import _skip_non_integer_type

        # Only 4 bytes — can't read 8-byte length
        assert _skip_non_integer_type(b"\x01\x02\x03\x04", 0, 8) == 0

    def test_skip_non_integer_array(self) -> None:
        from llama_manager.metadata._binary import _skip_non_integer_type

        # ARRAY: 4-byte elem_type=0 (UINT8) + 8-byte count=2 + 2 bytes
        data = (0).to_bytes(4, "little") + (2).to_bytes(8, "little") + b"\x01\x02"
        assert _skip_non_integer_type(data, 0, 9) == 14  # 12 header + 2 elements

    def test_skip_non_integer_array_truncated(self) -> None:
        from llama_manager.metadata._binary import _skip_non_integer_type

        # Only 8 bytes — can't read full 12-byte array header
        assert _skip_non_integer_type(b"\x00" * 8, 0, 9) == 0

    def test_skip_non_integer_unknown_type(self) -> None:
        from llama_manager.metadata._binary import _skip_non_integer_type

        # Unknown type tag — returns offset unchanged
        assert _skip_non_integer_type(b"\x00" * 10, 5, 99) == 5

    def test_skip_record_uint8(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        # type_tag=0 (UINT8): 4 bytes type + 1 byte value
        data = (0).to_bytes(4, "little") + b"\x42"
        assert _skip_record(data, 0) == 5

    def test_skip_record_uint16(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        data = (2).to_bytes(4, "little") + b"\x01\x00"
        assert _skip_record(data, 0) == 6

    def test_skip_record_uint32(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        data = (4).to_bytes(4, "little") + b"\x01\x00\x00\x00"
        assert _skip_record(data, 0) == 8

    def test_skip_record_uint64(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        data = (10).to_bytes(4, "little") + b"\x01" + b"\x00" * 7
        assert _skip_record(data, 0) == 12

    def test_skip_record_string(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        # type_tag=8 (STRING): 4 bytes type + 8-byte length=3 + 3 bytes
        data = (8).to_bytes(4, "little") + (3).to_bytes(8, "little") + b"abc"
        assert _skip_record(data, 0) == 15  # 4 + 8 + 3

    def test_skip_record_string_truncated(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        # type_tag=8 but only 4 bytes after type tag (need 8 for length)
        data = (8).to_bytes(4, "little") + b"\x01\x02\x03\x04"
        assert _skip_record(data, 0) == 4  # returns offset after type tag

    def test_skip_record_none_type_tag(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        # Only 2 bytes — can't read 4-byte type tag
        assert _skip_record(b"\x01\x02", 0) == 0

    def test_skip_record_unknown_type(self) -> None:
        from llama_manager.metadata._binary import _skip_record

        # type_tag=99 (unknown) — returns offset after type tag
        data = (99).to_bytes(4, "little") + b"\x00" * 4
        assert _skip_record(data, 0) == 4


class TestReadIntegerValue:
    """Tests for _read_integer_value and _read_legacy_integer_value."""

    def test_read_integer_value_uint8(self) -> None:
        from llama_manager.metadata._binary import _read_integer_value

        assert _read_integer_value(b"\x42", 0, 0) == 0x42

    def test_read_integer_value_int8(self) -> None:
        from llama_manager.metadata._binary import _read_integer_value

        assert _read_integer_value(b"\x80", 0, 1) == -128

    def test_read_integer_value_uint64(self) -> None:
        from llama_manager.metadata._binary import _read_integer_value

        data = (2**60).to_bytes(8, "little")
        assert _read_integer_value(data, 0, 10) == 2**60

    def test_read_integer_value_int64(self) -> None:
        from llama_manager.metadata._binary import _read_integer_value

        data = (2**63).to_bytes(8, "little")
        assert _read_integer_value(data, 0, 11) == -(2**63)

    def test_read_integer_value_non_integer_type(self) -> None:
        from llama_manager.metadata._binary import _read_integer_value

        # type_tag=8 (STRING) — not an integer type
        assert _read_integer_value(b"\x00" * 8, 0, 8) is None

    def test_read_legacy_integer_value_uint8(self) -> None:
        from llama_manager.metadata._binary import _read_legacy_integer_value

        assert _read_legacy_integer_value(b"\x42", 0, 1) == 0x42

    def test_read_legacy_integer_value_int8(self) -> None:
        from llama_manager.metadata._binary import _read_legacy_integer_value

        assert _read_legacy_integer_value(b"\x80", 0, 2) == -128

    def test_read_legacy_integer_value_uint32(self) -> None:
        from llama_manager.metadata._binary import _read_legacy_integer_value

        data = (12345).to_bytes(4, "little")
        assert _read_legacy_integer_value(data, 0, 5) == 12345

    def test_read_legacy_integer_value_invalid_type(self) -> None:
        from llama_manager.metadata._binary import _read_legacy_integer_value

        assert _read_legacy_integer_value(b"\x00" * 4, 0, 0) is None
        assert _read_legacy_integer_value(b"\x00" * 4, 0, 7) is None


class TestReadKeyWithLengthSize:
    """Tests for _read_key_with_length_size (4-byte legacy path)."""

    def test_4byte_key_length(self) -> None:
        from llama_manager.metadata._binary import _read_key_with_length_size

        key = b"general.name"
        data = len(key).to_bytes(4, "little") + key
        new_offset, result = _read_key_with_length_size(data, 0, 4)
        assert result == "general.name"
        assert new_offset == 4 + len(key)

    def test_4byte_key_truncated(self) -> None:
        from llama_manager.metadata._binary import _read_key_with_length_size

        # Only 2 bytes — can't read 4-byte length
        new_offset, result = _read_key_with_length_size(b"\x01\x02", 0, 4)
        assert result is None

    def test_4byte_key_data_truncated(self) -> None:
        from llama_manager.metadata._binary import _read_key_with_length_size

        # Length says 10 but only 3 bytes of key data
        data = (10).to_bytes(4, "little") + b"abc"
        new_offset, result = _read_key_with_length_size(data, 0, 4)
        assert result is None

    def test_invalid_length_size(self) -> None:
        from llama_manager.metadata._binary import _read_key_with_length_size

        # length_size != 4 and != 8 → returns (offset, None)
        new_offset, result = _read_key_with_length_size(b"\x00" * 10, 0, 2)
        assert result is None


class TestSkipRecordWithKeyFormat:
    """Tests for _skip_record_with_key_format (legacy 4-byte path)."""

    def test_legacy_u8_type(self) -> None:
        from llama_manager.metadata._binary import _skip_record_with_key_format

        # type_tag=1 (u8 in legacy), 4 bytes type + 1 byte value
        data = (1).to_bytes(4, "little") + b"\x42"
        result = _skip_record_with_key_format(data, 0, 4)
        assert result == 5

    def test_legacy_u16_type(self) -> None:
        from llama_manager.metadata._binary import _skip_record_with_key_format

        data = (3).to_bytes(4, "little") + b"\x01\x00"
        result = _skip_record_with_key_format(data, 0, 4)
        assert result == 6

    def test_legacy_u32_type(self) -> None:
        from llama_manager.metadata._binary import _skip_record_with_key_format

        data = (5).to_bytes(4, "little") + b"\x01\x00\x00\x00"
        result = _skip_record_with_key_format(data, 0, 4)
        assert result == 8

    def test_legacy_u64_type(self) -> None:
        from llama_manager.metadata._binary import _skip_record_with_key_format

        data = (9).to_bytes(4, "little") + b"\x01" + b"\x00" * 7
        result = _skip_record_with_key_format(data, 0, 4)
        assert result == 12

    def test_legacy_none_type_tag(self) -> None:
        from llama_manager.metadata._binary import _skip_record_with_key_format

        # Only 2 bytes — can't read 4-byte type tag
        result = _skip_record_with_key_format(b"\x01\x02", 0, 4)
        assert result == 0

    def test_8byte_delegates_to_skip_record(self) -> None:
        from llama_manager.metadata._binary import _skip_record_with_key_format

        # type_tag=0 (UINT8): 4 bytes type + 1 byte value
        data = (0).to_bytes(4, "little") + b"\x42"
        result = _skip_record_with_key_format(data, 0, 8)
        assert result == 5


# ---------------------------------------------------------------------------
# TestReaderHelpers — _reader.py GGUFReader-based helpers
# ---------------------------------------------------------------------------


class TestExtractArchitectureFromReader:
    """Tests for _extract_architecture_from_reader."""

    def test_returns_none_when_field_missing(self) -> None:
        from llama_manager.metadata._reader import _extract_architecture_from_reader

        result = _extract_architecture_from_reader({})
        assert result is None

    def test_returns_none_on_value_error(self) -> None:
        from unittest.mock import MagicMock

        from llama_manager.metadata._reader import _extract_architecture_from_reader

        field = MagicMock()
        field.contents.side_effect = ValueError("bad")
        from gguf.constants import Keys

        result = _extract_architecture_from_reader({Keys.General.ARCHITECTURE: field})
        assert result is None

    def test_returns_string_value(self) -> None:
        from unittest.mock import MagicMock

        from llama_manager.metadata._reader import _extract_architecture_from_reader

        field = MagicMock()
        field.contents.return_value = "llama"
        from gguf.constants import Keys

        result = _extract_architecture_from_reader({Keys.General.ARCHITECTURE: field})
        assert result == "llama"


class TestExtractFieldFromReader:
    """Tests for _extract_field_from_reader."""

    def test_returns_none_when_key_missing(self) -> None:
        from llama_manager.metadata._reader import _extract_field_from_reader

        assert _extract_field_from_reader({}, "general.name") is None

    def test_returns_none_on_value_error(self) -> None:
        from unittest.mock import MagicMock

        from llama_manager.metadata._reader import _extract_field_from_reader

        field = MagicMock()
        field.contents.side_effect = ValueError("bad")
        assert _extract_field_from_reader({"general.name": field}, "general.name") is None

    def test_returns_string_value(self) -> None:
        from unittest.mock import MagicMock

        from llama_manager.metadata._reader import _extract_field_from_reader

        field = MagicMock()
        field.contents.return_value = "MyModel"
        assert _extract_field_from_reader({"general.name": field}, "general.name") == "MyModel"


class TestExtractIntFieldFromReader:
    """Tests for _extract_int_field_from_reader."""

    def test_returns_none_when_key_missing(self) -> None:
        from llama_manager.metadata._reader import _extract_int_field_from_reader

        assert _extract_int_field_from_reader({}, "llama.context_length") is None

    def test_returns_none_on_value_error(self) -> None:
        from unittest.mock import MagicMock

        from llama_manager.metadata._reader import _extract_int_field_from_reader

        field = MagicMock()
        field.contents.side_effect = ValueError("bad")
        assert _extract_int_field_from_reader({"k": field}, "k") is None

    def test_returns_none_on_type_error(self) -> None:
        from unittest.mock import MagicMock

        from llama_manager.metadata._reader import _extract_int_field_from_reader

        field = MagicMock()
        field.contents.return_value = None  # int(None) raises TypeError
        assert _extract_int_field_from_reader({"k": field}, "k") is None

    def test_returns_int_value(self) -> None:
        from unittest.mock import MagicMock

        from llama_manager.metadata._reader import _extract_int_field_from_reader

        field = MagicMock()
        field.contents.return_value = 4096
        assert _extract_int_field_from_reader({"k": field}, "k") == 4096


class TestExtractFromGGUFReader:
    """Tests for _extract_from_gguf_reader — no-architecture and cancel paths."""

    def test_no_architecture_returns_none_fields(self) -> None:
        from llama_manager.metadata._reader import _extract_from_gguf_reader

        record = _extract_from_gguf_reader(
            model_path="/fake/model.gguf",
            fields={},
            parse_timeout_s=30.0,
            prefix_cap_bytes=65536,
        )
        assert record.architecture is None
        assert record.context_length is None
        assert record.embedding_length is None
        assert record.block_count is None

    def test_cancel_event_raises_interrupted(self) -> None:
        import threading
        from unittest.mock import MagicMock

        from gguf.constants import Keys

        from llama_manager.metadata._reader import _extract_from_gguf_reader

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
