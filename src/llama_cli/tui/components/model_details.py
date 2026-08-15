"""Shared model-index detail formatting for profile TUI surfaces."""

from __future__ import annotations

from llama_manager.model_index import ModelIndexEntry


def model_detail_parts(entry: ModelIndexEntry) -> list[str]:
    """Return compact model metadata parts for profile UI labels."""
    parts: list[str] = []
    if entry.architecture:
        parts.append(f"Arch: {entry.architecture}")
    if entry.quantization_type:
        parts.append(f"Quant: {entry.quantization_type}")
    max_context_length = entry.max_context_length or entry.context_length
    if max_context_length:
        parts.append(f"Max Ctx: {max_context_length}")
    if entry.file_size_bytes:
        size_gib = entry.file_size_bytes / (1024**3)
        parts.append(f"Size: {size_gib:.1f} GiB")
    if entry.parse_error:
        parts.append(f"Metadata: {short_parse_error(entry.parse_error)}")
    return parts


def short_parse_error(error: str) -> str:
    """Convert long parse exceptions into a UI-sized message."""
    if "timed out after" in error:
        return "parse timed out; using filename/cache fallback"
    return error.split(" for ", maxsplit=1)[0]
