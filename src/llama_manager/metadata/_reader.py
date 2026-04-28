"""GGUFReader-based metadata extraction — primary parsing path."""

import os
from contextlib import suppress
from pathlib import Path

from gguf.constants import Keys
from gguf.gguf_reader import ReaderField

from ._types import GGUFMetadataRecord, normalize_filename


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
    Uses ``tempfile.mkstemp`` for collision-safe temp file creation.

    Returns fields dict and version, or None if GGUFReader fails.

    Args:
        model_path: Path to the GGUF file.
        prefix_cap_bytes: Maximum bytes to read from the file.

    Returns:
        A tuple of (fields_dict, version) or None on failure.

    """
    import tempfile

    import gguf

    capped_fd: int | None = None
    capped_path: str | None = None
    try:
        # Create collision-safe temp file in system temp directory
        capped_fd, capped_path = tempfile.mkstemp(prefix="gguf_", suffix=".gguf", dir=None)
        try:
            with os.fdopen(capped_fd, "wb") as capped_file:
                capped_fd = None  # Already transferred ownership
                with open(model_path, "rb") as src:
                    capped_file.write(src.read(prefix_cap_bytes))
        except Exception:
            # If writing fails, close and clean up
            if capped_fd is not None:
                os.close(capped_fd)
                capped_fd = None
            return None

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
        # Close fd if still open (shouldn't happen after fdopen transfer)
        if capped_fd is not None:
            with suppress(OSError):
                os.close(capped_fd)
        # Remove temp file
        if capped_path:
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
