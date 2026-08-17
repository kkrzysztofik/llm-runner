"""GGUF metadata types, constants, and filename normalization."""

import re
import unicodedata
from dataclasses import dataclass

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

    normalized_stem: str
    general_name: str | None = None
    architecture: str | None = None
    file_type: int | None = None
    quantization_type: str | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    context_length: int | None = None
    max_context_length: int | None = None


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
