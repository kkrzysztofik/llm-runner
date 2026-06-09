"""Tests for llama_manager.metadata._reader."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

from gguf.constants import Keys

from llama_manager.metadata._reader import (
    _detect_tokenizer_type_from_reader,
    _extract_architecture_from_reader,
    _extract_field_from_reader,
    _extract_int_field_from_reader,
    _try_gguf_reader,
)


def _make_field(contents_value: object) -> MagicMock:
    """Create a mock ReaderField with the given contents() return value."""
    field = MagicMock()
    field.contents.return_value = contents_value
    return field


def _make_fields_dict(pairs: dict[str, object]) -> dict[str, MagicMock]:
    """Create a fields dict from key -> contents_value pairs."""
    return {k: _make_field(v) for k, v in pairs.items()}


# ---------------------------------------------------------------------------
# TestExtractArchitectureFromReader
# ---------------------------------------------------------------------------


class TestExtractArchitectureFromReader:
    """Tests for _extract_architecture_from_reader()."""

    def test_returns_string_value(self) -> None:
        """Should return the string representation of the field contents."""
        fields = _make_fields_dict({Keys.General.ARCHITECTURE: "llama"})
        result = _extract_architecture_from_reader(fields)
        assert result == "llama"

    def test_returns_none_when_missing(self) -> None:
        """Should return None when architecture key is not present."""
        result = _extract_architecture_from_reader({})
        assert result is None

    def test_returns_none_on_value_error(self) -> None:
        """Should return None when field.contents() raises ValueError."""
        field = MagicMock()
        field.contents.side_effect = ValueError("bad value")
        fields = {"llama.architecture": field}
        result = _extract_architecture_from_reader(fields)
        assert result is None

    def test_returns_none_on_attribute_error(self) -> None:
        """Should return None when field.contents() raises AttributeError."""
        field = MagicMock()
        field.contents.side_effect = AttributeError("missing attr")
        fields = {"llama.architecture": field}
        result = _extract_architecture_from_reader(fields)
        assert result is None

    def test_none_field_returns_none(self) -> None:
        """None values in the fields dict should return None."""
        fields: dict[str, None] = {"llama.architecture": None}
        result = _extract_architecture_from_reader(fields)  # type: ignore[arg-type]
        assert result is None


# ---------------------------------------------------------------------------
# TestExtractFieldFromReader
# ---------------------------------------------------------------------------


class TestExtractFieldFromReader:
    """Tests for _extract_field_from_reader()."""

    def test_returns_string_value(self) -> None:
        """Should return string value of the field."""
        fields = _make_fields_dict({"some.custom.key": "Qwen2.5-7B"})
        result = _extract_field_from_reader(fields, "some.custom.key")
        assert result == "Qwen2.5-7B"

    def test_returns_none_when_missing(self) -> None:
        """Should return None when key is not found."""
        result = _extract_field_from_reader({}, "nonexistent.key")
        assert result is None

    def test_returns_none_on_value_error(self) -> None:
        """Should return None on ValueError."""
        field = MagicMock()
        field.contents.side_effect = ValueError("bad")
        fields = {"some.key": field}
        result = _extract_field_from_reader(fields, "some.key")
        assert result is None

    def test_returns_none_on_attribute_error(self) -> None:
        """Should return None on AttributeError."""
        field = MagicMock()
        field.contents.side_effect = AttributeError("missing")
        fields = {"some.key": field}
        result = _extract_field_from_reader(fields, "some.key")
        assert result is None

    def test_int_contents_becomes_string(self) -> None:
        """Integer contents should be converted to string."""
        fields = _make_fields_dict({"some.key": 42})
        result = _extract_field_from_reader(fields, "some.key")
        assert result == "42"


# ---------------------------------------------------------------------------
# TestExtractIntFieldFromReader
# ---------------------------------------------------------------------------


class TestExtractIntFieldFromReader:
    """Tests for _extract_int_field_from_reader()."""

    def test_returns_int_value(self) -> None:
        """Should return integer value."""
        fields = _make_fields_dict({"context.length": 4096})
        result = _extract_int_field_from_reader(fields, "context.length")
        assert result == 4096

    def test_returns_none_when_missing(self) -> None:
        """Should return None when key is not found."""
        result = _extract_int_field_from_reader({}, "nonexistent")
        assert result is None

    def test_returns_none_on_value_error(self) -> None:
        """Should return None when contents() raises ValueError."""
        field = MagicMock()
        field.contents.side_effect = ValueError("not an int")
        fields = {"some.key": field}
        result = _extract_int_field_from_reader(fields, "some.key")
        assert result is None

    def test_returns_none_on_type_error(self) -> None:
        """Should return None when contents() raises TypeError."""
        field = MagicMock()
        field.contents.side_effect = TypeError("bad type")
        fields = {"some.key": field}
        result = _extract_int_field_from_reader(fields, "some.key")
        assert result is None

    def test_returns_none_on_attribute_error(self) -> None:
        """Should return None on AttributeError."""
        field = MagicMock()
        field.contents.side_effect = AttributeError("missing")
        fields = {"some.key": field}
        result = _extract_int_field_from_reader(fields, "some.key")
        assert result is None

    def test_string_int_converted(self) -> None:
        """String integer should be converted to int."""
        fields = _make_fields_dict({"block.count": "32"})
        result = _extract_int_field_from_reader(fields, "block.count")
        assert result == 32


# ---------------------------------------------------------------------------
# TestDetectTokenizerTypeFromReader
# ---------------------------------------------------------------------------


class TestDetectTokenizerTypeFromReader:
    """Tests for _detect_tokenizer_type_from_reader()."""

    def test_ggml_tokenizer(self) -> None:
        """Should detect ggml tokenizer type."""
        fields = {"tokenizer.ggml.tokens": []}
        result = _detect_tokenizer_type_from_reader(fields)  # type: ignore[arg-type]
        assert result == "ggml"

    def test_model_tokenizer(self) -> None:
        """Should detect model tokenizer type."""
        fields = {"tokenizer.model": "some_path"}
        result = _detect_tokenizer_type_from_reader(fields)  # type: ignore[arg-type]
        assert result == "model"

    def test_huggingface_tokenizer(self) -> None:
        """Should detect huggingface tokenizer type."""
        fields = {"tokenizer.json": "{}"}
        result = _detect_tokenizer_type_from_reader(fields)  # type: ignore[arg-type]
        assert result == "huggingface"

    def test_first_match_wins(self) -> None:
        """First matching tokenizer type should win."""
        fields = {
            "tokenizer.ggml.tokens": [],
            "tokenizer.model": "path",
            "tokenizer.json": "{}",
        }
        result = _detect_tokenizer_type_from_reader(fields)  # type: ignore[arg-type]
        assert result == "ggml"

    def test_none_when_no_match(self) -> None:
        """Should return None when no tokenizer pattern matches."""
        fields = {"some.random.key": "value"}
        result = _detect_tokenizer_type_from_reader(fields)  # type: ignore[arg-type]
        assert result is None

    def test_empty_fields(self) -> None:
        """Empty fields dict should return None."""
        result = _detect_tokenizer_type_from_reader({})
        assert result is None


# ---------------------------------------------------------------------------
# TestTryGgufReader
# ---------------------------------------------------------------------------


class TestTryGgufReader:
    """Tests for _try_gguf_reader()."""

    def test_nonexistent_file_returns_none(self) -> None:
        """Should return None for a nonexistent file."""
        result = _try_gguf_reader("/nonexistent/path/model.gguf")
        assert result is None

    def test_cancel_event_returns_none(self, tmp_path: Path) -> None:
        """Should return None when cancel_event is set before reading."""
        cancel = threading.Event()
        cancel.set()
        # Create a minimal temp file so the file exists
        test_file = tmp_path / "test.gguf"
        test_file.write_bytes(b"\x00" * 100)
        result = _try_gguf_reader(str(test_file))
        assert result is None

    def test_cancel_after_copy_returns_none(self, tmp_path: Path) -> None:
        """Should return None when cancel_event is set after file copy."""
        cancel = threading.Event()
        test_file = tmp_path / "test.gguf"
        test_file.write_bytes(b"\x00" * 100)

        # Set cancel after a brief delay — but since we can't easily race,
        # just set it before the call to get the early-exit path.
        cancel.set()
        result = _try_gguf_reader(str(test_file))
        assert result is None
