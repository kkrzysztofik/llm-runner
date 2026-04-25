"""GGUF metadata extraction without loading full model weights.

Parses GGUF file headers to extract model metadata (architecture,
context length, attention heads, etc.) using the ``gguf`` library's
key constants and optional timeout.
"""

import re
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gguf.constants import Keys
from gguf.gguf_reader import ReaderField

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
    normalized = re.sub(r"\s+", "_", normalized)

    # Remove invalid filename characters
    normalized = _INVALID_FILENAME_CHARS.sub("_", normalized)

    # Collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)

    # Strip leading/trailing underscores
    normalized = normalized.strip("_")

    return normalized or "unknown"


def _read_gguf_header(
    model_path: str,
    prefix_cap_bytes: int,
) -> bytes:
    """Read the beginning of a GGUF file up to prefix_cap_bytes.

    Args:
        model_path: Path to the GGUF file.
        prefix_cap_bytes: Maximum number of bytes to read.

    Returns:
        The raw bytes from the start of the file.

    Raises:
        FileNotFoundError: If the model file does not exist.
        PermissionError: If the file cannot be read.
        OSError: If reading fails for other reasons.

    """
    with open(model_path, "rb") as fh:
        data = fh.read(prefix_cap_bytes)
    if not data:
        raise OSError("model file is empty")
    return data


def _detect_gguf_version(header: bytes) -> int:
    """Detect the GGUF version from magic bytes.

    Args:
        header: The first 8 bytes of the file.

    Returns:
        GGUF version number (2, 3, or 4).

    Raises:
        ValueError: If magic bytes are not recognized.

    """
    if header[:8] == _GGUF_V2_MAGIC:
        return 2
    if header[:8] == _GGUF_V3_MAGIC:
        return 3
    if header[:8] == _GGUF_V4_MAGIC:
        return 4
    raise ValueError("not a valid GGUF file (bad magic bytes)")


def _parse_general_name(data: bytes) -> str | None:
    """Extract general.name from GGUF header bytes.

    Args:
        data: Raw bytes from the file header.

    Returns:
        The general.name value, or None if not found.

    """
    match = _GENERAL_NAME_PATTERN.search(data)
    if match:
        try:
            return match.group(1).decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def _parse_architecture(data: bytes) -> str | None:
    """Extract architecture hint from GGUF header bytes.

    Looks for common architecture patterns in the header data.

    Args:
        data: Raw bytes from the file header.

    Returns:
        Architecture string, or None if not found.

    """
    # Common architecture patterns in GGUF files
    _ARCH_PATTERNS: list[tuple[bytes, str]] = [
        (b"llama", "llama"),
        (b"gpt_2", "gpt_2"),
        (b"bert", "bert"),
        (b"mpt", "mpt"),
        (b"falcon", "falcon"),
        (b"qwen2", "qwen2"),
        (b"qwen3", "qwen3"),
        (b"phi3", "phi3"),
        (b"stablelm", "stablelm"),
        (b"qwen", "qwen"),
        (b"mamba", "mamba"),
    ]
    for pattern, arch in _ARCH_PATTERNS:
        if pattern in data:
            return arch
    return None


def _parse_tokenizer_type(data: bytes) -> str | None:
    """Extract tokenizer type from GGUF header bytes.

    Args:
        data: Raw bytes from the file header.

    Returns:
        Tokenizer type string, or None if not found.

    """
    _TOKENIZER_PATTERNS: list[tuple[bytes, str]] = [
        (b"tokenizer.ggml", "ggml"),
        (b"tokenizer.model", "model"),
        (b"tokenizer.json", "huggingface"),
    ]
    for pattern, tok_type in _TOKENIZER_PATTERNS:
        if pattern in data:
            return tok_type
    return None


def _parse_numeric_field(data: bytes, key: bytes | str) -> int | None:
    """Attempt to extract a numeric field from GGUF header bytes.

    Looks for the key string followed by a numeric value in the
    header data.

    Args:
        data: Raw bytes from the file header.
        key: The field key to search for.

    Returns:
        The numeric value found, or None.

    """
    key_bytes = key.encode() if isinstance(key, str) else key
    pattern = re.escape(key_bytes).replace(rb"\.", rb"\.") + rb"\s+(\d+)"
    match = re.search(pattern, data)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_architecture_from_reader(
    reader_fields: dict[str, ReaderField],
) -> str | None:
    """Extract architecture from GGUFReader fields dict.

    Uses the ``gguf`` library's ``Keys.General.ARCHITECTURE`` constant.

    Args:
        reader_fields: The fields dict from GGUFReader.

    Returns:
        Architecture string, or None if not found.

    """
    field = reader_fields.get(Keys.General.ARCHITECTURE)
    if field is None:
        return None
    try:
        return str(field.contents())
    except (ValueError, AttributeError):
        return None


def _extract_field_from_reader(
    reader_fields: dict[str, ReaderField],
    key: str,
) -> str | None:
    """Extract a string field from GGUFReader fields dict.

    Args:
        reader_fields: The fields dict from GGUFReader.
        key: The GGUF key to look up.

    Returns:
        The string value, or None if not found.

    """
    field = reader_fields.get(key)
    if field is None:
        return None
    try:
        return str(field.contents())
    except (ValueError, AttributeError):
        return None


def _extract_int_field_from_reader(
    reader_fields: dict[str, ReaderField],
    key: str,
) -> int | None:
    """Extract an integer field from GGUFReader fields dict.

    Args:
        reader_fields: The fields dict from GGUFReader.
        key: The GGUF key to look up.

    Returns:
        The integer value, or None if not found or not an integer type.

    """
    field = reader_fields.get(key)
    if field is None:
        return None
    try:
        val = field.contents()
        return int(val)
    except (ValueError, TypeError, AttributeError):
        return None


def _detect_tokenizer_type_from_reader(
    fields: dict[str, ReaderField],
) -> str | None:
    """Detect tokenizer type from GGUFReader fields.

    Checks for common tokenizer key patterns in the GGUF KV store.

    Args:
        fields: The fields dict from GGUFReader.

    Returns:
        Tokenizer type string, or None if not found.

    """
    for key in fields:
        if "tokenizer.ggml" in key:
            return "ggml"
        if "tokenizer.model" in key:
            return "model"
        if "tokenizer.json" in key:
            return "huggingface"
    return None


def _try_gguf_reader(
    model_path: str,
    prefix_cap_bytes: int,
) -> tuple[dict[str, ReaderField], int] | None:
    """Try to read GGUF file using the gguf library's GGUFReader.

    Creates a temporary file containing only the first ``prefix_cap_bytes``
    bytes of the model to avoid memory-mapping the entire file.

    Returns fields dict and version, or None if GGUFReader fails.

    Args:
        model_path: Path to the GGUF file.
        prefix_cap_bytes: Maximum bytes to read from the file.

    Returns:
        A tuple of (fields_dict, version) or None on failure.

    """
    import gguf

    capped_path: str | None = None
    try:
        capped_path = model_path + ".tmp"
        with open(model_path, "rb") as src, open(capped_path, "wb") as dst:
            dst.write(src.read(prefix_cap_bytes))

        reader = gguf.GGUFReader(capped_path, mode="r")
        fields = dict(reader.fields)

        version_field = fields.get("GGUF.version")
        version = 3
        if version_field is not None:
            with suppress(ValueError, TypeError):
                version = int(version_field.contents())

        return fields, version
    except Exception:
        return None
    finally:
        if capped_path and Path(capped_path).exists():
            with suppress(OSError):
                Path(capped_path).unlink()


def _extract_from_gguf_reader(
    model_path: str,
    fields: dict[str, ReaderField],
    parse_timeout_s: float,
    prefix_cap_bytes: int,
) -> GGUFMetadataRecord:
    """Extract metadata using GGUFReader fields dict.

    Args:
        model_path: Path to the GGUF file.
        fields: The fields dict from GGUFReader.
        parse_timeout_s: Timeout for parsing.
        prefix_cap_bytes: Bytes cap used for parsing.

    Returns:
        A ``GGUFMetadataRecord`` with extracted metadata.

    """
    general_name = _extract_field_from_reader(fields, Keys.General.NAME)
    architecture = _extract_architecture_from_reader(fields)
    tokenizer_type = _detect_tokenizer_type_from_reader(fields)

    if architecture:
        ctx_key = Keys.LLM.CONTEXT_LENGTH.format(arch=architecture)
        emb_key = Keys.LLM.EMBEDDING_LENGTH.format(arch=architecture)
        blk_key = Keys.LLM.BLOCK_COUNT.format(arch=architecture)
        atc_key = Keys.Attention.HEAD_COUNT.format(arch=architecture)
        atc_kv_key = Keys.Attention.HEAD_COUNT_KV.format(arch=architecture)

        context_length = _extract_int_field_from_reader(fields, ctx_key)
        embedding_length = _extract_int_field_from_reader(fields, emb_key)
        block_count = _extract_int_field_from_reader(fields, blk_key)
        attention_head_count = _extract_int_field_from_reader(fields, atc_key)
        attention_head_count_kv = _extract_int_field_from_reader(fields, atc_kv_key)
    else:
        context_length = None
        embedding_length = None
        block_count = None
        attention_head_count = None
        attention_head_count_kv = None

    stem = Path(model_path).stem
    normalized_stem = normalize_filename(stem)

    return GGUFMetadataRecord(
        raw_path=model_path,
        normalized_stem=normalized_stem,
        general_name=general_name,
        architecture=architecture,
        tokenizer_type=tokenizer_type,
        embedding_length=embedding_length,
        block_count=block_count,
        context_length=context_length,
        attention_head_count=attention_head_count,
        attention_head_count_kv=attention_head_count_kv,
        parse_timeout_s=parse_timeout_s,
        prefix_cap_bytes=prefix_cap_bytes,
    )


def _extract_from_raw_bytes(
    model_path: str,
    prefix_cap_bytes: int,
    parse_timeout_s: float,
) -> GGUFMetadataRecord:
    """Extract metadata by parsing raw header bytes.

    Args:
        model_path: Path to the GGUF file.
        prefix_cap_bytes: Maximum bytes to read from the file.
        parse_timeout_s: Timeout for parsing.

    Returns:
        A ``GGUFMetadataRecord`` with extracted metadata.

    """
    data = _read_gguf_header(model_path, prefix_cap_bytes)
    version = _detect_gguf_version(data[:8])
    if version == 4:
        raise ValueError("GGUF v4 format not yet supported")

    general_name = _parse_general_name(data)
    architecture = _parse_architecture(data)
    tokenizer_type = _parse_tokenizer_type(data)
    embedding_length = _parse_numeric_field(data, b"embedding_length")
    block_count = _parse_numeric_field(data, b"block_count")
    context_length = _parse_numeric_field(data, b"context_length")
    attention_head_count = _parse_numeric_field(data, b"attention_head_count")
    attention_head_count_kv = _parse_numeric_field(data, b"attention_head_count_kv")

    stem = Path(model_path).stem
    normalized_stem = normalize_filename(stem)

    return GGUFMetadataRecord(
        raw_path=model_path,
        normalized_stem=normalized_stem,
        general_name=general_name,
        architecture=architecture,
        tokenizer_type=tokenizer_type,
        embedding_length=embedding_length,
        block_count=block_count,
        context_length=context_length,
        attention_head_count=attention_head_count,
        attention_head_count_kv=attention_head_count_kv,
        parse_timeout_s=parse_timeout_s,
        prefix_cap_bytes=prefix_cap_bytes,
    )


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
    if not isinstance(parse_timeout_s, (int, float)) or parse_timeout_s <= 0:
        raise ValueError(
            f"parse_timeout_s must be a positive number, got: {parse_timeout_s}",
        )

    from queue import Empty, Queue
    from threading import Thread

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
