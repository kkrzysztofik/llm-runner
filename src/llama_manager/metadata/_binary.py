"""Raw binary GGUF header parsing — fallback path when GGUFReader is unavailable."""

from pathlib import Path

from ._types import (
    _GENERAL_NAME_PATTERN,
    _GGUF_V2_MAGIC,
    _GGUF_V3_MAGIC,
    _GGUF_V4_MAGIC,
    GGUFMetadataRecord,
    normalize_filename,
)


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
    # Sorted longest-first so more specific patterns match before shorter prefixes
    # (e.g. "qwen3" before "qwen", "phi3" before "phi").
    _ARCH_PATTERNS: list[tuple[bytes, str]] = [
        (b"stablelm", "stablelm"),
        (b"falcon", "falcon"),
        (b"llama", "llama"),
        (b"mamba", "mamba"),
        (b"gpt_2", "gpt_2"),
        (b"qwen3", "qwen3"),
        (b"qwen2", "qwen2"),
        (b"qwen", "qwen"),
        (b"phi3", "phi3"),
        (b"bert", "bert"),
        (b"mpt", "mpt"),
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


def _read_int8(data: bytes, offset: int) -> int | None:
    """Read int8 from data at offset."""
    if offset + 1 > len(data):
        return None
    val = int.from_bytes(data[offset : offset + 1], "little")
    return val - 256 if val > 127 else val


def _read_int16(data: bytes, offset: int) -> int | None:
    """Read int16 from data at offset."""
    if offset + 2 > len(data):
        return None
    val = int.from_bytes(data[offset : offset + 2], "little")
    return val - 65536 if val > 32767 else val


def _read_int32(data: bytes, offset: int) -> int | None:
    """Read int32 from data at offset."""
    if offset + 4 > len(data):
        return None
    val = int.from_bytes(data[offset : offset + 4], "little")
    return val - 4294967296 if val > 2147483647 else val


def _skip_non_integer_type(data: bytes, offset: int, type_tag: int) -> int:
    """Skip non-integer GGUF types, return new offset."""
    # GGUF type tags: 7=BOOL (1 byte), 8=STRING (8-byte length + data), 9=ARRAY (complex)
    if type_tag == 7:  # BOOL
        return offset + 1
    if type_tag == 8:  # STRING: 8-byte little-endian length prefix + data
        if offset + 8 > len(data):
            return offset
        str_len = int.from_bytes(data[offset : offset + 8], "little")
        return offset + 8 + str_len
    if type_tag == 9:  # ARRAY: 1-byte elem type + 8-byte count + elements
        if offset + 9 > len(data):
            return offset
        elem_type = data[offset]
        elem_count = int.from_bytes(data[offset + 1 : offset + 9], "little")
        offset += 9
        for _ in range(elem_count):
            if elem_type in (1, 2, 3, 4, 5, 6):
                # Integer types: advance by fixed sizes
                int_sizes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4}
                offset += int_sizes.get(elem_type, 0)
            else:
                offset = _skip_non_integer_type(data, offset, elem_type)
        return offset
    # Unknown type — cannot safely skip
    return offset


def _read_key(data: bytes, offset: int) -> tuple[int, str | None]:
    """Read key from GGUF record, return (new_offset, key_string)."""
    if offset + 4 > len(data):
        return offset, None
    key_length = int.from_bytes(data[offset : offset + 4], "little")
    offset += 4
    if offset + key_length > len(data):
        return offset, None
    record_key = data[offset : offset + key_length].decode("utf-8", errors="replace")
    return offset + key_length, record_key


def _read_type_tag(data: bytes, offset: int) -> int | None:
    """Read type tag from GGUF record."""
    if offset + 4 > len(data):
        return None
    return int.from_bytes(data[offset : offset + 4], "little")


def _skip_record(data: bytes, offset: int) -> int:
    """Skip a GGUF record, return new offset."""
    type_tag = _read_type_tag(data, offset)
    if type_tag is None:
        return offset
    offset += 4
    if type_tag in (1, 2):
        return offset + 1
    if type_tag in (3, 4):
        return offset + 2
    if type_tag in (5, 6):
        return offset + 4
    if type_tag == 8:  # string (GGUF_TYPE_STRING)
        if offset + 4 > len(data):
            return offset
        str_len = int.from_bytes(data[offset : offset + 4], "little")
        return offset + 4 + str_len
    return offset


def _read_integer_value(data: bytes, offset: int, type_tag: int) -> int | None:
    """Read integer value if type is integer, else None."""
    if type_tag not in (1, 2, 3, 4, 5, 6):
        return None
    readers = {
        1: _read_int8,
        2: _read_int8,
        3: _read_int16,
        4: _read_int16,
        5: _read_int32,
        6: _read_int32,
    }
    reader = readers.get(type_tag)
    return reader(data, offset) if reader else None


def _parse_numeric_field(data: bytes, key: bytes | str) -> int | None:
    """Attempt to extract a numeric field from GGUF header bytes.

    Walks the GGUF binary KV record format to find the key and read
    its numeric value using struct unpacking. GGUF v2/v3 stores
    key-value pairs as:
        - key: length-prefixed UTF-8 string
        - type: uint32 type tag
        - value: typed data (u8/u16/u32/u64 for numeric types)

    Args:
        data: Raw bytes from the file header.
        key: The field key to search for.

    Returns:
        The numeric value found, or None.
    """
    key_bytes = key.encode() if isinstance(key, str) else key
    target_key = key_bytes.decode("utf-8", errors="replace")

    # GGUF v2/v3: 8 magic + 4 version + 8 num_tensors + 8 num_kv = 28 bytes
    kv_start = 28
    if len(data) < kv_start:
        return None

    offset = kv_start
    while offset + 8 <= len(data):
        offset, record_key = _read_key(data, offset)
        if record_key is None:
            break
        if record_key != target_key:
            offset = _skip_record(data, offset)
            continue
        type_tag = _read_type_tag(data, offset)
        if type_tag is None:
            break
        offset += 4
        result = _read_integer_value(data, offset, type_tag)
        if result is not None:
            return result
        offset = _skip_non_integer_type(data, offset, type_tag)
    return None


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
