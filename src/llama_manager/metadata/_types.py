"""GGUF metadata types, constants, and filename normalization."""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

# GGUF magic bytes (little-endian)
_GGUF_V2_MAGIC = b"GGUF\x02\x00\x00\x00"
_GGUF_V3_MAGIC = b"GGUF\x03\x00\x00\x00"
_GGUF_V4_MAGIC = b"GGUF\x04\x00\x00\x00"

# Pattern for general.name key in GGUF metadata (raw binary format)
_GENERAL_NAME_PATTERN = re.compile(
    rb"general\.name\s*\x00([^\x00]+)\x00",
)

# Invalid filename character pattern (NFKC normalization applied)
_INVALID_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f/\\:\*\?\"<>\|]")

# Pre-compiled patterns for normalize_filename to avoid recompilation
_WHITESPACE_PATTERN = re.compile(r"\s+")
_MULTI_UNDERSCORE_PATTERN = re.compile(r"_+")


@dataclass
class GGUFMetadataRecord:
    """Extracted metadata from a GGUF model file header.

    Only the fields that were found in the file header are populated;
    missing fields are ``None`` (except for fields derived from the
    file path which are always set).
    """

    raw_path: str
    normalized_stem: str
    general_name: str | None = None
    architecture: str | None = None
    tokenizer_type: str | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    context_length: int | None = None
    attention_head_count: int | None = None
    attention_head_count_kv: int | None = None
    parse_timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    parse_timeout_s: float = 0.0
    prefix_cap_bytes: int = 0


def normalize_filename(filename: str) -> str:
    """Normalize a filename stem for use as a model identifier.

    Applies Unicode NFKC normalization, replaces whitespace sequences
    with a single underscore, and removes invalid filename characters.

    Args:
        filename: Raw filename stem (without extension).

    Returns:
        Normalized filename stem suitable for use as an identifier.

    """
    # NFKC normalization
    normalized = unicodedata.normalize("NFKC", filename)

    # Replace whitespace sequences with underscore
    normalized = _WHITESPACE_PATTERN.sub("_", normalized)

    # Remove invalid filename characters
    normalized = _INVALID_FILENAME_CHARS.sub("_", normalized)

    # Collapse multiple underscores
    normalized = _MULTI_UNDERSCORE_PATTERN.sub("_", normalized)

    # Strip leading/trailing underscores
    normalized = normalized.strip("_")

    return normalized or "unknown"
