"""GGUFReader-based metadata extraction — primary parsing path."""

from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from threading import Event

from gguf.constants import Keys
from gguf.gguf_reader import ReaderField

from ._types import GGUFMetadataRecord, normalize_filename

# GGUF file_type → human-readable quantization string (from gguf-file.h)
_FILE_TYPE_QUANT_MAP: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q5_0",
    5: "Q5_1",
    6: "Q8_0",
    7: "Q8_1",
    8: "F16",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "Q8_K",
    20: "IQ2_XXS",
    21: "IQ2_XS",
    22: "IQ3_XXS",
    23: "IQ1_S",
    24: "IQ4_NL",
    25: "IQ3_S",
    26: "IQ3_M",
    27: "IQ2_S",
    28: "IQ2_M",
    29: "IQ4_XS",
    30: "IQ1_M",
    31: "IQ4_XS",
    32: "IQ4_NL",
}


def _file_type_to_quant(file_type: int | None) -> str | None:
    """Convert a GGUF file_type integer to a human-readable quantization string.

    Args:
        file_type: The raw GGUF general.file_type integer.

    Returns:
        Human-readable quantization name, or ``f"type_{file_type}\"`` if unmapped.
    """
    if file_type is None:
        return None
    return _FILE_TYPE_QUANT_MAP.get(file_type, f"type_{file_type}")


def _extract_file_type(fields: Mapping[str, ReaderField]) -> int | None:
    """Extract the ``general.file_type`` integer from GGUFReader fields.

    Args:
        fields: The fields dict from GGUFReader.

    Returns:
        The file_type integer, or None if not found.
    """
    return _extract_int_field_from_reader(fields, Keys.General.FILE_TYPE)


def _extract_architecture_from_reader(
    reader_fields: Mapping[str, ReaderField],
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
    except ValueError, AttributeError:
        return None


def _extract_field_from_reader(
    reader_fields: Mapping[str, ReaderField],
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
    except ValueError, AttributeError:
        return None


def _extract_int_field_from_reader(
    reader_fields: Mapping[str, ReaderField],
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
    except ValueError, TypeError, AttributeError:
        return None


def _detect_tokenizer_type_from_reader(
    fields: Mapping[str, ReaderField],
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
) -> tuple[dict[str, ReaderField], int] | None:
    """Try to read GGUF file using the gguf library's GGUFReader.

    Uses the full file directly. ``gguf.GGUFReader`` uses ``np.memmap`` which
    maps the file into virtual address space without consuming physical RAM
    until data is accessed. We only read key/value metadata fields, not tensor
    data, so physical memory impact is minimal regardless of file size.

    Returns fields dict and version, or None if GGUFReader fails.

    Args:
        model_path: Path to the GGUF file.

    Returns:
        A tuple of (fields_dict, version) or None on failure.

    """
    import gguf

    try:
        reader = gguf.GGUFReader(model_path, mode="r")
        fields = dict(reader.fields)

        version_field = fields.get("GGUF.version")
        version = 3
        if version_field is not None:
            with suppress(ValueError, TypeError):
                version = int(version_field.contents())

        return fields, version
    except Exception:
        return None


def _extract_from_gguf_reader(
    model_path: str,
    fields: Mapping[str, ReaderField],
    parse_timeout_s: float,
    prefix_cap_bytes: int,
    cancel_event: Event | None = None,
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

    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("parse cancelled")

    tokenizer_type = _detect_tokenizer_type_from_reader(fields)
    file_type = _extract_file_type(fields)
    quantization_type = _file_type_to_quant(file_type)

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
        file_type=file_type,
        quantization_type=quantization_type,
        embedding_length=embedding_length,
        block_count=block_count,
        context_length=context_length,
        max_context_length=context_length,
        attention_head_count=attention_head_count,
        attention_head_count_kv=attention_head_count_kv,
        parse_timeout_s=parse_timeout_s,
        prefix_cap_bytes=prefix_cap_bytes,
    )
