"""GGUF metadata extraction without loading full model weights.

Parses GGUF file headers to extract model metadata (architecture,
context length, attention heads, etc.) using the ``gguf`` library's
key constants and optional timeout.
"""

from queue import Empty, Queue
from threading import Thread

from ._binary import (
    _detect_gguf_version,
    _extract_from_raw_bytes,
    _parse_architecture,
    _parse_general_name,
    _parse_numeric_field,
    _parse_tokenizer_type,
)
from ._reader import _extract_from_gguf_reader, _try_gguf_reader
from ._types import GGUFMetadataRecord, normalize_filename

__all__ = [
    "GGUFMetadataRecord",
    "_detect_gguf_version",
    "_parse_architecture",
    "_parse_general_name",
    "_parse_numeric_field",
    "_parse_tokenizer_type",
    "extract_gguf_metadata",
    "normalize_filename",
]


def extract_gguf_metadata(
    model_path: str,
    prefix_cap_bytes: int = 32 * 1024 * 1024,
    parse_timeout_s: float = 5.0,
) -> GGUFMetadataRecord:
    """Extract metadata from a GGUF file without loading full weights.

    Attempts to use the ``gguf`` library's ``GGUFReader`` for robust
    parsing.  Falls back to raw bytes parsing using the ``gguf``
    library's key constants for compatibility with existing test
    fixtures.

    The entire operation is wrapped in a timeout to prevent
    indefinite hangs.

    Args:
        model_path: Path to the GGUF model file.
        prefix_cap_bytes: Maximum bytes to read from the file
            (default 32 MiB).
        parse_timeout_s: Maximum seconds to allow for the read+parse
            operation (default 5.0).

    Returns:
        A ``GGUFMetadataRecord`` with extracted metadata.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file is not a valid GGUF file or parameters
            are invalid.
        TimeoutError: If the parse exceeds ``parse_timeout_s``.

    """
    # Validate parameters before any reads or thread creation
    if not isinstance(prefix_cap_bytes, int) or prefix_cap_bytes <= 0:
        raise ValueError(
            f"prefix_cap_bytes must be a positive integer, got: {prefix_cap_bytes}",
        )
    if not isinstance(parse_timeout_s, int | float) or parse_timeout_s <= 0:
        raise ValueError(
            f"parse_timeout_s must be a positive number, got: {parse_timeout_s}",
        )

    result_queue: Queue[GGUFMetadataRecord | BaseException] = Queue(maxsize=1)

    def _parse() -> None:
        try:
            # Attempt GGUFReader first
            reader_result = _try_gguf_reader(model_path, prefix_cap_bytes)

            if reader_result is not None:
                fields, version = reader_result

                # Check version
                if version == 4:
                    raise ValueError("GGUF v4 format not yet supported")

                record = _extract_from_gguf_reader(
                    model_path,
                    fields,
                    parse_timeout_s,
                    prefix_cap_bytes,
                )
            else:
                # Fall back to raw bytes parsing
                record = _extract_from_raw_bytes(
                    model_path,
                    prefix_cap_bytes,
                    parse_timeout_s,
                )

            result_queue.put(record, block=False)
        except Exception as exc:
            result_queue.put(exc, block=False)

    # Run parse in a thread with timeout
    thread = Thread(target=_parse, daemon=True)
    thread.start()
    thread.join(timeout=parse_timeout_s)

    if thread.is_alive():
        raise TimeoutError(
            f"GGUF metadata parse timed out after {parse_timeout_s}s for {model_path}",
        )

    try:
        item = result_queue.get(block=False)
    except Empty:
        raise RuntimeError("parse completed without producing a result") from None

    if isinstance(item, BaseException):
        raise item

    return item
