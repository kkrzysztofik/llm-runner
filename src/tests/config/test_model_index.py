"""Tests for llama_manager.model_index — GGUF model index cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llama_manager.config.defaults import Config
from llama_manager.metadata import GGUFMetadataRecord
from llama_manager.model_index import (
    ModelIndexEntry,
    load_model_index,
    model_index_path,
    refresh_model_index,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_xdg_config(tmp_path: Path):
    """Set XDG_CACHE_HOME to tmp_path and return tmp_path for cleanup."""
    original = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(tmp_path)
    yield tmp_path
    if original is None:
        os.environ.pop("XDG_CACHE_HOME", None)
    else:
        os.environ["XDG_CACHE_HOME"] = original


@pytest.fixture()
def mock_model_dir(tmp_path: Path) -> Path:
    """Create a directory with fake .gguf files."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_a.gguf").write_bytes(
        b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    (models_dir / "model_b.gguf").write_bytes(
        b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    (models_dir / "readme.txt").write_text("not a model")
    return models_dir


@pytest.fixture()
def sample_config(tmp_path: Path, mock_model_dir: Path) -> MagicMock:
    """Return a Config mock with models_dir pointing to mock_model_dir."""
    cfg = MagicMock(spec=Config)
    cfg.models_dir = str(mock_model_dir)
    cfg.gguf_metadata_prefix_cap_bytes = 1024
    cfg.gguf_metadata_parse_timeout_s = 1.0
    return cfg


# ---------------------------------------------------------------------------
# model_index_path
# ---------------------------------------------------------------------------


def test_returns_path_under_xdg_cache(tmp_xdg_config: Path) -> None:
    """model_index_path should return path under XDG_CACHE_HOME/llm-runner/.

    We need to create a minimal Config mock since model_index_path uses Config.
    """
    from unittest.mock import MagicMock

    from llama_manager.config.defaults import Config

    cfg = MagicMock(spec=Config)
    cfg.models_dir = str(tmp_xdg_config / "models")

    idx_path = model_index_path(cfg)
    assert str(idx_path).endswith("llm-runner/model-index.json")


def test_creates_parent_dir(tmp_xdg_config: Path) -> None:
    """model_index_path should create the llm-runner parent directory."""
    from unittest.mock import MagicMock

    from llama_manager.config.defaults import Config

    cfg = MagicMock(spec=Config)
    cfg.models_dir = str(tmp_xdg_config / "models")

    idx_path = model_index_path(cfg)
    assert idx_path.parent.exists()


# ---------------------------------------------------------------------------
# ModelIndexEntry serialization
# ---------------------------------------------------------------------------


def test_to_dict_roundtrip() -> None:
    """ModelIndexEntry.to_dict() + from_dict() should round-trip."""
    entry = ModelIndexEntry(
        path="/models/test.gguf",
        normalized_stem="test",
        general_name="Test Model",
        architecture="llama",
        file_type=15,
        quantization_type="Q4_K_M",
        context_length=8192,
        embedding_length=4096,
        block_count=32,
        file_size_bytes=4294967296,
        parse_error=None,
        mtime_iso="2024-01-01T00:00:00+00:00",
    )
    d = entry.to_dict()
    restored = ModelIndexEntry.from_dict(d)
    assert restored.path == entry.path
    assert restored.normalized_stem == entry.normalized_stem
    assert restored.architecture == entry.architecture
    assert restored.file_type == entry.file_type
    assert restored.quantization_type == entry.quantization_type
    assert restored.context_length == entry.context_length
    assert restored.file_size_bytes == entry.file_size_bytes
    assert restored.parse_error is None


def test_from_dict_missing_fields() -> None:
    """ModelIndexEntry.from_dict() should ignore unknown keys."""
    full = {
        "path": "/models/test.gguf",
        "normalized_stem": "test",
        "general_name": None,
        "architecture": "llama",
        "file_type": None,
        "quantization_type": None,
        "context_length": 8192,
        "embedding_length": None,
        "block_count": None,
        "file_size_bytes": 1000,
        "parse_error": None,
        "mtime_iso": "2024-01-01T00:00:00+00:00",
        "extra_key": "should_be_ignored",
    }
    entry = ModelIndexEntry.from_dict(full)
    assert entry.path == "/models/test.gguf"
    assert entry.architecture == "llama"
    assert entry.context_length == 8192
    assert entry.file_size_bytes == 1000


# ---------------------------------------------------------------------------
# load_model_index
# ---------------------------------------------------------------------------


def test_load_returns_empty_list_on_missing(tmp_xdg_config: Path, sample_config: MagicMock) -> None:
    """load_model_index should return [] when cache file doesn't exist."""
    result = load_model_index(sample_config)
    assert result == []


def test_load_returns_empty_list_on_corrupt_json(
    tmp_xdg_config: Path, sample_config: MagicMock
) -> None:
    """load_model_index should return [] on corrupt JSON."""
    idx_path = model_index_path(sample_config)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text("not json {{{")

    result = load_model_index(sample_config)
    assert result == []


def test_load_returns_cached_entries(tmp_xdg_config: Path, sample_config: MagicMock) -> None:
    """load_model_index should return entries from valid cache."""
    idx_path = model_index_path(sample_config)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "path": "/models/a.gguf",
            "normalized_stem": "a",
            "general_name": None,
            "architecture": "llama",
            "file_type": None,
            "quantization_type": None,
            "context_length": 8192,
            "embedding_length": None,
            "block_count": None,
            "file_size_bytes": 1000,
            "parse_error": None,
            "mtime_iso": "2024-01-01T00:00:00+00:00",
        }
    ]
    idx_path.write_text(json.dumps(entries))

    result = load_model_index(sample_config)
    assert len(result) == 1
    assert result[0].path == "/models/a.gguf"
    assert result[0].normalized_stem == "a"


# ---------------------------------------------------------------------------
# refresh_model_index
# ---------------------------------------------------------------------------


def test_refresh_scans_directory(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    mock_model_dir: Path,
) -> None:
    """refresh_model_index should scan .gguf files and return entries."""
    with patch("llama_manager.model_index.extract_gguf_metadata") as mock_extract:
        from llama_manager.metadata import GGUFMetadataRecord

        mock_extract.return_value = GGUFMetadataRecord(
            raw_path="/models/model_a.gguf",
            normalized_stem="model_a",
            general_name=None,
            architecture="llama",
            tokenizer_type=None,
            file_type=None,
            quantization_type=None,
            embedding_length=None,
            block_count=None,
            context_length=None,
            attention_head_count=None,
            attention_head_count_kv=None,
        )

        entries, total, errors = refresh_model_index(sample_config)

    # Should have 2 .gguf files scanned
    assert total == 2
    assert errors == 0
    assert len(entries) == 2


def test_refresh_with_errors(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    mock_model_dir: Path,
) -> None:
    """refresh_model_index should handle parse errors gracefully."""
    call_count = 0

    def _extract_with_error(*args, **kwargs) -> GGUFMetadataRecord:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("bad magic")
        from llama_manager.metadata import GGUFMetadataRecord

        return GGUFMetadataRecord(
            raw_path=str(mock_model_dir / "model_b.gguf"),
            normalized_stem="model_b",
            general_name=None,
            architecture="llama",
            tokenizer_type=None,
            file_type=None,
            quantization_type=None,
            embedding_length=None,
            block_count=None,
            context_length=None,
            attention_head_count=None,
            attention_head_count_kv=None,
        )

    with patch(
        "llama_manager.model_index.extract_gguf_metadata",
        side_effect=_extract_with_error,
    ):
        entries, total, errors = refresh_model_index(sample_config)

    assert errors == 1
    assert total == 2
    # The errored entry should have parse_error set
    errored = [e for e in entries if e.parse_error is not None]
    assert len(errored) == 1
    assert errored[0].parse_error is not None
    assert "bad magic" in errored[0].parse_error


def test_refresh_timeout_uses_raw_metadata_fallback(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    mock_model_dir: Path,
) -> None:
    """Timeouts in the primary parser should fall back to raw GGUF metadata."""
    raw_record = GGUFMetadataRecord(
        raw_path=str(mock_model_dir / "model_a.gguf"),
        normalized_stem="model_a",
        architecture="qwen3",
        file_type=18,
        quantization_type="Q6_K",
        context_length=32768,
    )

    with (
        patch(
            "llama_manager.model_index.extract_gguf_metadata",
            side_effect=TimeoutError("slow primary parser"),
        ),
        patch("llama_manager.model_index._extract_from_raw_bytes", return_value=raw_record),
    ):
        entries, total, errors = refresh_model_index(sample_config)

    assert total == 2
    assert errors == 0
    assert {entry.quantization_type for entry in entries} == {"Q6_K"}
    assert all(entry.parse_error is None for entry in entries)


def test_refresh_parse_error_uses_filename_metadata(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    tmp_path: Path,
) -> None:
    """Failed parses should still expose useful architecture and quant from filename."""
    models_dir = tmp_path / "filename-models"
    models_dir.mkdir()
    model_path = models_dir / "Qwen3-0.8B-UD-Q8_K_XL.gguf"
    model_path.write_bytes(b"broken")
    sample_config.models_dir = str(models_dir)

    with patch(
        "llama_manager.model_index.extract_gguf_metadata",
        side_effect=ValueError("bad metadata"),
    ):
        entries, total, errors = refresh_model_index(sample_config)

    assert total == 1
    assert errors == 1
    assert entries[0].architecture == "qwen3"
    assert entries[0].quantization_type == "Q8_K_XL"
    assert entries[0].parse_error == "bad metadata"


def test_refresh_missing_dir_returns_empty(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
) -> None:
    """refresh_model_index should return empty when models_dir doesn't exist."""
    sample_config.models_dir = str(tmp_xdg_config / "nonexistent")
    entries, total, errors = refresh_model_index(sample_config)
    assert entries == []
    assert total == 0
    assert errors == 0


def test_refresh_caches_mtime(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    mock_model_dir: Path,
) -> None:
    """Second refresh should skip files with unchanged mtime."""
    extract_calls = 0

    def _count_calls(*args, **kwargs) -> MagicMock:
        nonlocal extract_calls
        extract_calls += 1
        mock_record = MagicMock()
        mock_record.normalized_stem = "cached"
        mock_record.general_name = None
        mock_record.architecture = "llama"
        mock_record.file_type = None
        mock_record.quantization_type = None
        mock_record.context_length = None
        mock_record.embedding_length = None
        mock_record.block_count = None
        return mock_record

    with patch(
        "llama_manager.model_index.extract_gguf_metadata",
        side_effect=_count_calls,
    ):
        # First refresh
        refresh_model_index(sample_config)
        first_count = extract_calls

        # Second refresh — should use cache
        refresh_model_index(sample_config)
        second_count = extract_calls

    # Second refresh should not re-extract (cache hit)
    assert second_count == first_count


def test_refresh_cache_stale_mtime(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    mock_model_dir: Path,
) -> None:
    """Modifying a file's mtime should cause it to be re-scanned."""
    extract_calls = 0

    def _count_calls(*args, **kwargs) -> MagicMock:
        nonlocal extract_calls
        extract_calls += 1
        mock_record = MagicMock()
        mock_record.normalized_stem = "stale"
        mock_record.general_name = None
        mock_record.architecture = "llama"
        mock_record.file_type = None
        mock_record.quantization_type = None
        mock_record.context_length = None
        mock_record.embedding_length = None
        mock_record.block_count = None
        return mock_record

    with patch(
        "llama_manager.model_index.extract_gguf_metadata",
        side_effect=_count_calls,
    ):
        # First refresh
        refresh_model_index(sample_config)
        first_count = extract_calls

        # Touch a file to change its mtime
        (mock_model_dir / "model_a.gguf").touch()

        # Second refresh — should re-scan the touched file
        refresh_model_index(sample_config)
        second_count = extract_calls

    # Second refresh should call extract one more time (for the touched file)
    assert second_count > first_count


def test_refresh_writes_atomically(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    mock_model_dir: Path,
) -> None:
    """refresh_model_index should write the index file after scanning."""
    with patch("llama_manager.model_index.extract_gguf_metadata") as mock_extract:
        mock_extract.return_value = MagicMock(
            normalized_stem="test",
            general_name=None,
            architecture="llama",
            file_type=None,
            quantization_type=None,
            context_length=None,
            embedding_length=None,
            block_count=None,
        )
        refresh_model_index(sample_config)

    idx_path = model_index_path(sample_config)
    assert idx_path.exists()
    # Should be valid JSON
    data = json.loads(idx_path.read_text())
    assert isinstance(data, list)


def test_refresh_scans_case_insensitive_suffix(
    tmp_xdg_config: Path,
    sample_config: MagicMock,
    mock_model_dir: Path,
) -> None:
    """refresh_model_index should find files with mixed-case .Gguf suffix."""
    with patch("llama_manager.model_index.extract_gguf_metadata") as mock_extract:
        mock_extract.return_value = MagicMock(
            normalized_stem="mixed",
            general_name=None,
            architecture="llama",
            file_type=None,
            quantization_type=None,
            embedding_length=None,
            block_count=None,
            context_length=None,
            attention_head_count=None,
            attention_head_count_kv=None,
        )
        # Create a file with mixed-case suffix
        mixed_file = mock_model_dir / "model.Gguf"
        mixed_file.touch()

        entries, total, errors = refresh_model_index(sample_config)

    # Should find the new .Gguf file plus the fixture's .gguf files
    paths = {e.path for e in entries}
    assert str(mock_model_dir / "model.Gguf") in paths
    assert total == 3
    assert errors == 0
