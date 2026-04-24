"""GGUF metadata extraction without loading full model weights.

Parses GGUF file headers to extract model metadata (architecture,
context length, attention heads, etc.) using a bounded prefix read
and optional timeout.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

# GGUF magic bytes (little-endian)
_GGUF_V2_MAGIC = b"GGUF\x02\x00\x00\x00"
_GGUF_V3_MAGIC = b"GGUF\x03\x00\x00\x00"
_GGUF_V4_MAGIC = b"GGUF\x04\x00\x00\x00"

# Pattern for general.name key in GGUF metadata
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


def extract_gguf_metadata(
    model_path: str,
    prefix_cap_bytes: int = 32 * 1024 * 1024,
    parse_timeout_s: float = 5.0,
) -> GGUFMetadataRecord:
    """Extract metadata from a GGUF file without loading full weights.

    Reads only the first ``prefix_cap_bytes`` of the file and parses
    the GGUF header to extract available metadata fields.

    The file is read synchronously in a single ``read()`` call bounded
    by ``prefix_cap_bytes``.  The entire operation is wrapped in a
    timeout to prevent indefinite hangs.

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
        ValueError: If the file is not a valid GGUF file.
        TimeoutError: If the parse exceeds ``parse_timeout_s``.

    """
    from threading import Thread

    result: GGUFMetadataRecord | None = None
    exception: BaseException | None = None
    done = False

    def _parse() -> None:
        nonlocal result, exception, done
        try:
            data = _read_gguf_header(model_path, prefix_cap_bytes)
            version = _detect_gguf_version(data[:8])

            # GGUF v4 is not yet widely supported
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

            # Derive normalized stem from path
            from pathlib import Path

            stem = Path(model_path).stem
            normalized_stem = normalize_filename(stem)

            result = GGUFMetadataRecord(
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
        except BaseException as exc:
            exception = exc
        finally:
            done = True

    # Run parse in a thread with timeout
    thread = Thread(target=_parse, daemon=True)
    thread.start()
    thread.join(timeout=parse_timeout_s)

    if not done:
        raise TimeoutError(
            f"GGUF metadata parse timed out after {parse_timeout_s}s for {model_path}"
        )

    if exception is not None:
        raise exception

    if result is None:
        raise RuntimeError("parse completed without setting result")
    return result
