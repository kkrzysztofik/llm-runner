"""GGUF metadata extraction without loading full model weights.

Parses GGUF file headers to extract model metadata (architecture,
context length, attention heads, etc.) using the ``gguf`` library's
key constants and optional timeout.
"""

from queue import Empty, Queue
from threading import Event, Thread

from ._binary import _extract_from_raw_bytes
from ._reader import _extract_from_gguf_reader, _try_gguf_reader
from ._types import GGUFMetadataRecord, normalize_filename

__all__ = [
    "GGUFMetadataRecord",
    "extract_gguf_metadata",
    "normalize_filename",
]


def _parse_gguf_in_thread(
    model_path: str,
    prefix_cap_bytes: int,
    parse_timeout_s: float,
    result_queue: Queue[GGUFMetadataRecord | BaseException],
    cancel_event: Event,
) -> None:
    """Run GGUF parse logic inside a worker thread.

    Puts the resulting ``GGUFMetadataRecord`` (or an exception) into
    *result_queue*.  Aborts silently if *cancel_event* is set before the
    result is ready.
    """
    try:
        reader_result = _try_gguf_reader(model_path, prefix_cap_bytes, cancel_event)

        if cancel_event.is_set():
            return

        if reader_result is not None:
            fields, version = reader_result
            if version == 4:
                raise ValueError("GGUF v4 format not yet supported")
            record = _extract_from_gguf_reader(
                model_path,
                fields,
                parse_timeout_s,
                prefix_cap_bytes,
                cancel_event,
            )
        else:
            record = _extract_from_raw_bytes(
                model_path,
                prefix_cap_bytes,
                parse_timeout_s,
                cancel_event,
            )

        if not cancel_event.is_set():
            result_queue.put(record, block=False)
    except Exception as exc:
        if not cancel_event.is_set():
            result_queue.put(exc, block=False)


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
    if (
        isinstance(prefix_cap_bytes, bool)
        or not isinstance(prefix_cap_bytes, int)
        or prefix_cap_bytes <= 0
    ):
        raise ValueError(
            f"prefix_cap_bytes must be a positive integer, got: {prefix_cap_bytes}",
        )
    if (
        isinstance(parse_timeout_s, bool)
        or not isinstance(parse_timeout_s, int | float)
        or parse_timeout_s <= 0
    ):
        raise ValueError(
            f"parse_timeout_s must be a positive number, got: {parse_timeout_s}",
        )

    result_queue: Queue[GGUFMetadataRecord | BaseException] = Queue(maxsize=1)
    cancel_event = Event()

    # Run parse in a thread with timeout
    thread = Thread(
        target=_parse_gguf_in_thread,
        args=(model_path, prefix_cap_bytes, parse_timeout_s, result_queue, cancel_event),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=parse_timeout_s)

    if thread.is_alive():
        cancel_event.set()
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
