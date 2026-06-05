"""GGUF metadata extraction without loading full model weights.

Parses GGUF file headers to extract model metadata (architecture,
context length, attention heads, etc.) using the ``gguf`` library's
key constants and optional timeout.
"""

from ._types import GGUFMetadataRecord, normalize_filename
from .extractor import extract_gguf_metadata

__all__ = [
    "GGUFMetadataRecord",
    "extract_gguf_metadata",
    "normalize_filename",
]
