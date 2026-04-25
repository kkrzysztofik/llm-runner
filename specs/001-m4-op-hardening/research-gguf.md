# Research: gguf Library Best Practices for Metadata Extraction

**Date**: 2026-04-23
**Feature**: M4 — Operational Hardening and Smoke Verification
**Status**: Complete

---

## Decision 1: Use `gguf.GGUFReader` for Header-Only Parsing

### Decision

Use the `GGUFReader` class from `gguf.gguf_reader` to read only the GGUF header (magic, version, key-value pairs) without loading tensor data. The reader uses `numpy.memmap` for memory-efficient file access, and tensor data is read lazily (only if accessed).

```python
import gguf
from gguf.gguf_reader import GGUFReader

# Open file — reads only header, does NOT load tensor data
reader = GGUFReader(model_path)

# Access version and key-value pairs
version: int = reader.fields["GGUF.version"].contents()
kv_count: int = reader.fields["GGUF.kv_count"].contents()

# Extract individual KV fields
arch_field = reader.get_field("general.architecture")
arch: str = arch_field.contents() if arch_field else None
```

### Rationale

- **Memory efficiency**: `GGUFReader` uses `numpy.memmap` which maps the file into virtual memory without loading it fully into RAM. Tensor data is only read when explicitly accessed via `reader.tensors`.
- **Header-only access**: The `reader.fields` OrderedDict contains all key-value pairs from the header. We never need to iterate `reader.tensors` for metadata extraction.
- **Official library**: The `gguf` package is the canonical Python GGUF reader maintained by GGML (the creators of llama.cpp). It is the same library used by `convert_hf_to_gguf.py` in the llama.cpp repository.
- **Type safety**: The library is PEP 561 compatible (has `py.typed` marker), so `pyright` type checking works out of the box.
- **Version support**: `READER_SUPPORTED_VERSIONS = [2, 3]` means the library supports GGUF v2 and v3. v4+ will raise `ValueError` with a clear message — exactly what the spec requires (FR-007).

### Version Detection

```python
from gguf.constants import GGUF_MAGIC, GGUF_VERSION

# GGUF_VERSION = 3 (current stable)
# READER_SUPPORTED_VERSIONS = [2, 3]

def check_gguf_version(reader: GGUFReader) -> int:
    """Return the GGUF version, raising if unsupported."""
    version_field = reader.fields.get("GGUF.version")
    if version_field is None:
        raise ValueError("Missing GGUF.version field")
    version = version_field.contents()
    return version
```

### Alternatives Considered

- **Manual header parsing**: Implement the GGUF binary format parser from scratch (magic bytes, version uint32, kv_count uint64, then KV pairs). **Rejected** — the `gguf` library already does this correctly, handles endianness, and is maintained upstream. Manual parsing risks subtle bugs with string length prefixes, array types, and alignment padding.
- **`gguf.GGUFWriter`**: This is for *writing* GGUF files, not reading. Not applicable.
- **`gguf.metadata.Metadata.load()`**: This is a high-level convenience class that loads metadata from HuggingFace model cards and config files, then writes to a GGUF writer. It does not read existing GGUF files. Not applicable.

---

## Decision 2: Extract Required Fields via `reader.get_field()` with Type-Safe Accessor

### Decision

Provide a typed accessor function that extracts specific KV fields from the `GGUFReader`, converting `ReaderField.contents()` to the expected Python type. Use `Keys` constants from `gguf.constants` for key name references.

```python
import gguf
from gguf.gguf_reader import GGUFReader
from gguf.constants import Keys

class GgufMetadata:
    """Extracted GGUF metadata fields for a model file."""

    name: str | None
    architecture: str | None
    tokenizer_type: str | None
    embedding_length: int | None
    block_count: int | None
    context_length: int | None
    attention_head_count: int | None
    attention_head_count_kv: int | None

    def __init__(self, reader: GGUFReader, architecture: str) -> None:
        self.name = _extract_str(reader, Keys.General.NAME)
        self.architecture = _extract_str(reader, Keys.General.ARCHITECTURE)
        self.tokenizer_type = _extract_str(reader, Keys.Tokenizer.MODEL)

        # Architecture-specific LLM fields use {arch} placeholders
        self.embedding_length = _extract_int(reader, Keys.LLM.EMBEDDING_LENGTH.format(arch=architecture))
        self.block_count = _extract_int(reader, Keys.LLM.BLOCK_COUNT.format(arch=architecture))
        self.context_length = _extract_int(reader, Keys.LLM.CONTEXT_LENGTH.format(arch=architecture))
        self.attention_head_count = _extract_int(reader, Keys.Attention.HEAD_COUNT.format(arch=architecture))
        self.attention_head_count_kv = _extract_int(reader, Keys.Attention.HEAD_COUNT_KV.format(arch=architecture))


def _extract_str(reader: GGUFReader, key: str) -> str | None:
    """Extract a string KV field, returning None if missing."""
    field = reader.get_field(key)
    if field is None:
        return None
    return str(field.contents())


def _extract_int(reader: GGUFReader, key: str) -> int | None:
    """Extract an integer KV field (uint32/uint64), returning None if missing."""
    field = reader.get_field(key)
    if field is None:
        return None
    val = field.contents()
    # contents() returns numpy scalar or Python int
    return int(val) if val is not None else None
```

### Rationale

- **Architecture-dependent keys**: The `Keys.LLM.*`, `Keys.Attention.*`, etc. classes use `{arch}` placeholders that must be formatted with the actual architecture string (e.g., `llama`, `qwen35`). The `_extract_*` helpers handle this formatting.
- **None for missing fields**: The spec (FR-007) requires graceful handling of missing `general.name` — fall back to normalized filename stem. Returning `None` for missing fields enables this fallback logic.
- **Type safety**: The accessor functions have explicit return types (`str | None`, `int | None`), satisfying `pyright` strict mode.
- **`Keys` constants**: Using `gguf.constants.Keys` ensures key names match what llama.cpp expects, avoiding typos.

### Why Not `reader.fields[key].contents()` Directly?

Direct access to `reader.fields[key]` is fragile:
1. `KeyError` is raised if the key is missing — we want `None`.
2. `contents()` returns a `numpy` scalar for numeric types, which must be converted to Python `int`/`float`.
3. String values require explicit `str()` conversion.

The accessor functions centralize these conversions.

---

## Decision 3: Prefix-Only Reading via `numpy.memmap` with Byte Limit

### Decision

The `GGUFReader` constructor opens the file with `numpy.memmap`, which maps the *entire* file into virtual memory. To enforce the 32 MiB prefix cap, create the reader from a memory-mapped slice of the file instead:

```python
import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.constants import GGUF_MAGIC, GGUF_VERSION

PREFIX_CAP = 32 * 1024 * 1024  # 32 MiB
PARSE_TIMEOUT_S = 5.0

def read_gguf_metadata(
    model_path: str | Path,
    prefix_cap: int = PREFIX_CAP,
    timeout: float = PARSE_TIMEOUT_S,
) -> GgufMetadata:
    """Read GGUF metadata from the header region only.

    Args:
        model_path: Path to the GGUF file.
        prefix_cap: Maximum bytes to read from file start (default 32 MiB).
        timeout: Wall-clock timeout in seconds (default 5s).

    Returns:
        GgufMetadata with extracted header fields.

    Raises:
        ValueError: If the file is not a valid GGUF file or version is unsupported.
        TimeoutError: If parsing exceeds the timeout.
    """
    import threading
    import time

    with open(model_path, "rb") as f:
        header = f.read(prefix_cap)

    if len(header) < 16:  # magic(4) + version(4) + kv_count(8) minimum
        raise ValueError("File too small to be a valid GGUF file")

    # Check magic bytes
    magic = int.from_bytes(header[0:4], byteorder="little")
    if magic != GGUF_MAGIC:
        raise ValueError(
            f"Invalid GGUF magic: expected {GGUF_MAGIC:#010x}, got {magic:#010x}. "
            "File may be corrupt or not a GGUF file."
        )

    # Check version
    version = int.from_bytes(header[4:8], byteorder="little")
    if version > GGUF_VERSION:
        raise ValueError(
            f"Unsupported GGUF format version {version}. "
            f"Expected version {GGUF_VERSION} or earlier. "
            "This file format is not supported."
        )

    # For files <= prefix_cap, use the full header
    # For files > prefix_cap, write the header slice to a temp file
    import tempfile

    if len(header) == prefix_cap:
        # File is at least prefix_cap bytes — write header to temp file
        # GGUFReader needs a real file path for memmap
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            tmp.write(header)
            tmp_path = tmp.name
    else:
        tmp_path = model_path  # File fits within prefix_cap

    try:
        # Use threading for timeout
        result: GgufMetadata | None = None
        exc_info: Exception | None = None

        def _read() -> None:
            nonlocal result, exc_info
            try:
                reader = GGUFReader(tmp_path)
                version = reader.fields["GGUF.version"].contents()
                if version > GGUF_VERSION:
                    raise ValueError(
                        f"Unsupported GGUF format version {version}. "
                        "This file format is not supported."
                    )
                arch_field = reader.get_field("general.architecture")
                architecture = str(arch_field.contents()) if arch_field else "unknown"
                result = GgufMetadata(reader, architecture)
            except Exception as e:
                exc_info = e

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"GGUF metadata extraction timed out after {timeout}s. "
                "The file may be corrupt or extremely large."
            )

        if exc_info is not None:
            raise exc_info

        if result is None:
            raise ValueError("Failed to extract GGUF metadata (unexpected state)")

        return result

    finally:
        if tmp_path != model_path:
            import os
            os.unlink(tmp_path)
```

### Rationale

- **GGUFReader requires a file path**: The `GGUFReader` constructor takes a `path: os.PathLike[str] | str` and uses `numpy.memmap` internally. It does not accept a `bytes` or `io.BytesIO` object. Therefore, for prefix-capped reading of large files, we write the header slice to a temporary file.
- **Temporary file cleanup**: The `tempfile.NamedTemporaryFile` with `delete=False` ensures the file persists after the `with` block. The `finally` block guarantees cleanup.
- **Threading for timeout**: Python's `threading.Timer` or `signal.alarm` (POSIX-only) are alternatives, but `thread.join(timeout)` is cross-platform and simple. The daemon thread ensures no resource leak if the main thread exits.
- **Version check before reading**: The version is checked directly from the header bytes before invoking `GGUFReader`, avoiding unnecessary parsing of large files with unsupported versions.
- **Magic bytes check**: The magic bytes are verified before calling `GGUFReader`, enabling a clear "corrupt file" error message (distinguishing from parser/spec mismatch).

### Why Not Pass Bytes Directly to GGUFReader?

`GGUFReader.__init__` signature:
```python
def __init__(self, path: os.PathLike[str] | str, mode: Literal['r', 'r+', 'c'] = 'r'):
    self.data = np.memmap(path, mode=mode)
```

It uses `np.memmap(path, ...)` which requires a filesystem path. There is no `BytesIO` or `bytes` constructor. The temp-file workaround is the only option.

### Alternatives Considered

- **`io.BytesIO` + manual parsing**: Write a custom parser that reads from `io.BytesIO`. **Rejected** — re-implements the GGUF binary format. The `gguf` library already handles endianness, all value types (uint32, uint64, float32, string, bool, array), and alignment padding correctly.
- **`mmap` module with offset/length**: Python's `mmap.mmap` supports offset and length parameters, but `GGUFReader` uses `np.memmap` which does not support offset/length in its constructor. Converting `mmap` to `numpy` would require an additional copy, defeating the memory efficiency goal.
- **`GGUFReader` with full file, then ignore tensors**: Opening the full file with `GGUFReader` and simply not accessing `reader.tensors` would work for most cases, but the `np.memmap` still maps the entire file, which could be gigabytes. The prefix cap is a hard requirement (FR-007), so we must enforce it.

---

## Decision 4: Error Classification — Distinguish Corruption from Spec Mismatch

### Decision

Classify GGUF parse errors into three categories:

1. **Corrupt file**: Invalid magic bytes, truncated header (file too small), or binary data that cannot be parsed as GGUF structure.
2. **Unsupported version**: Valid GGUF header but version > 3 (v4+).
3. **Parser/spec mismatch**: Valid GGUF v3 file but missing required fields or malformed KV pairs.

```python
from enum import Enum, auto

class GgufParseErrorKind(Enum):
    CORRUPT_FILE = auto()        # Invalid magic, truncated, binary garbage
    UNSUPPORTED_VERSION = auto() # Valid header, version > 3
    MISSING_FIELD = auto()       # Valid GGUF v3, but required KV field absent
    PARSE_ERROR = auto()         # Valid GGUF v3, but KV pair malformed

class GgufParseError(ValueError):
    """Error reading GGUF metadata with a specific classification."""

    def __init__(self, kind: GgufParseErrorKind, message: str, path: str) -> None:
        self.kind = kind
        self.path = path
        super().__init__(f"[{kind.name}] {path}: {message}")
```

### Rationale

- **FR-007 requires distinction**: The spec explicitly states: "`doctor` MUST distinguish a corrupt file from a parser/spec mismatch (e.g. GGUF version)."
- **Magic bytes check is first**: If the first 4 bytes are not `GGUF` (0x46554747), the file is corrupt or not a GGUF file. This is the clearest corruption signal.
- **Version check is second**: If magic is valid but version > 3, the file is a valid GGUF file with an unsupported version. This is a spec mismatch, not corruption.
- **Field validation is third**: If the header parses successfully but required fields (e.g., `general.architecture`) are missing, this is a parser/spec mismatch — the file is valid GGUF v3 but does not conform to the expected model metadata schema.

### Error Handling Flow

```python
def read_gguf_metadata_safe(
    model_path: str | Path,
    prefix_cap: int = PREFIX_CAP,
    timeout: float = PARSE_TIMEOUT_S,
) -> GgufMetadata:
    """Read GGUF metadata with safe error classification."""
    path = str(model_path)

    # Step 1: Read header bytes
    try:
        with open(path, "rb") as f:
            header = f.read(prefix_cap)
    except OSError as e:
        raise GgufParseError(
            GgufParseErrorKind.CORRUPT_FILE,
            f"Cannot read file: {e}",
            path,
        )

    # Step 2: Check file size
    if len(header) < 16:
        raise GgufParseError(
            GgufParseErrorKind.CORRUPT_FILE,
            f"File too small ({len(header)} bytes) to be a valid GGUF file. "
            "Expected at least 16 bytes (magic + version + kv_count).",
            path,
        )

    # Step 3: Check magic bytes
    magic = int.from_bytes(header[0:4], byteorder="little")
    if magic != GGUF_MAGIC:
        actual = header[0:4].hex()
        raise GgufParseError(
            GgufParseErrorKind.CORRUPT_FILE,
            f"Invalid magic bytes: expected 'GGUF' (47475547), got 0x{actual}. "
            "File may be corrupt or not a GGUF file.",
            path,
        )

    # Step 4: Check version
    version = int.from_bytes(header[4:8], byteorder="little")
    if version > GGUF_VERSION:
        raise GgufParseError(
            GgufParseErrorKind.UNSUPPORTED_VERSION,
            f"GGUF format version {version} is not supported. "
            f"Expected version {GGUF_VERSION} or earlier.",
            path,
        )

    # Step 5: Parse with GGUFReader (via temp file if needed)
    try:
        metadata = read_gguf_metadata(path, prefix_cap, timeout)
    except TimeoutError:
        raise GgufParseError(
            GgufParseErrorKind.PARSE_ERROR,
            f"Metadata extraction timed out after {timeout}s.",
            path,
        )
    except ValueError as e:
        msg = str(e)
        if "Unsupported GGUF format version" in msg:
            v = int(msg.split()[-1])
            raise GgufParseError(
                GgufParseErrorKind.UNSUPPORTED_VERSION,
                f"GGUF format version {v} is not supported.",
                path,
            )
        raise GgufParseError(
            GgufParseErrorKind.PARSE_ERROR,
            f"Failed to parse GGUF header: {e}",
            path,
        )

    # Step 6: Validate required fields
    if metadata.architecture is None:
        raise GgufParseError(
            GgufParseErrorKind.MISSING_FIELD,
            "Missing required field 'general.architecture'. "
            "This file is a valid GGUF file but does not contain model metadata.",
            path,
        )

    return metadata
```

### Alternatives Considered

- **Single `ValueError` for all errors**: Too coarse — the spec requires distinction between corrupt, unsupported version, and missing fields. A single error type loses this information.
- **Custom exception hierarchy**: `class CorruptFileError(ValueError): ...` etc. **Rejected** — the `GgufParseError` with `GgufParseErrorKind` enum is cleaner and easier to match on (single `except GgufParseError` with `exc.kind` check).

---

## Decision 5: Unicode Normalization with `unicodedata.normalize('NFKC', ...)`

### Decision

Apply NFKC normalization to all string fields extracted from GGUF metadata, particularly `general.name` and tokenizer-related fields.

```python
import unicodedata

def normalize_gguf_string(value: str | None) -> str | None:
    """Normalize a GGUF string field using NFKC normalization.

    NFKC (Compatibility Decomposition followed by Canonical Composition)
    ensures that compatibility-equivalent characters (e.g., fullwidth
    Latin letters, composed vs. decomposed accented characters) are
    mapped to their canonical forms.
    """
    if value is None:
        return None
    return unicodedata.normalize("NFKC", value)
```

### Rationale

- **FR-008 requirement**: The spec explicitly states: "Unicode normalization uses NFKC."
- **NFKC vs NFC**: NFKC is preferred over NFC because it normalizes compatibility characters (e.g., fullwidth Latin letters to ASCII equivalents). This is important for model names that may contain non-standard Unicode characters.
- **Applied to extracted strings**: The normalization should be applied when extracting string fields from the GGUF reader, not when writing to the file (the file owns its encoding).

### Alternatives Considered

- **NFC normalization**: Simpler but does not handle compatibility characters. NFKC is the spec requirement.
- **No normalization**: Would leave model names in their raw form, potentially causing comparison issues (e.g., `Qwen3.5-2B` vs. `Qwen3.5-2B` with different Unicode representations).

---

## Decision 6: Synthetic Test Fixture Generation

### Decision

Generate minimal valid GGUF test fixtures using the `gguf.GGUFWriter` class. Each fixture contains only header metadata (no tensor data) and is under 10 KiB.

```python
#!/usr/bin/env python3
"""Generate minimal GGUF test fixtures for unit testing.

Generates files under tests/fixtures/ containing only header metadata
(no tensor data). All files are under 10 KiB.

Usage:
    python src/scripts/generate_gguf_fixtures.py
"""

import struct
import sys
from pathlib import Path

# Add the source tree to the path so we can import gguf
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gguf
from gguf import GGUFWriter
from gguf.constants import GGUF_MAGIC, GGUF_VERSION


FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def write_fixture(
    name: str,
    *,
    architecture: str = "llama",
    extra_keys: dict[str, object] | None = None,
    missing_keys: set[str] | None = None,
) -> Path:
    """Write a minimal GGUF v3 fixture with header metadata only."""
    path = FIXTURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)

    with GGUFWriter(path, "test-fixture") as w:
        # Required general fields
        w.add_general_architecture(architecture)
        w.add_name("Test Model")
        w.add_tokenizer_model("llama")

        # Architecture-specific fields (llama)
        w.add_context_length(4096)
        w.add_embedding_length(4096)
        w.add_block_count(32)
        w.add_attention_head_count(32)
        w.add_attention_head_count_kv(8)

        # Add any extra keys
        if extra_keys:
            for key, value in extra_keys.items():
                w.add_string(key, str(value))

    return path


def main() -> None:
    # 1. Valid GGUF v3 with all required keys
    write_fixture(
        "valid_gguf_v3_all_keys.gguf",
        architecture="llama",
    )

    # 2. Valid GGUF v3 missing general.name
    write_fixture(
        "valid_gguf_v3_missing_name.gguf",
        architecture="llama",
        extra_keys={"tokenizer.ggml.model": "llama"},
    )
    # Delete the name field after writing
    # (GGUFWriter doesn't support deletion, so we patch the file manually)

    # 3. Corrupt file (bad magic bytes)
    corrupt_path = FIXTURES_DIR / "corrupt_bad_magic.gguf"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corrupt_path, "wb") as f:
        f.write(b"NOTG")  # Invalid magic
        f.write(struct.pack("<I", GGUF_VERSION))  # Version
        f.write(struct.pack("<Q", 0))  # 0 KV pairs

    # 4. Truncated file (valid header, no KV data)
    truncated_path = FIXTURES_DIR / "truncated_no_kv.gguf"
    truncated_path.parent.mkdir(parents=True, exist_ok=True)
    with open(truncated_path, "wb") as f:
        f.write(b"GGUF")  # Magic
        f.write(struct.pack("<I", GGUF_VERSION))  # Version
        f.write(struct.pack("<Q", 5))  # Claims 5 KV pairs but no data follows

    # 5. Valid GGUF v4 file (unsupported version)
    v4_path = FIXTURES_DIR / "valid_gguf_v4.gguf"
    v4_path.parent.mkdir(parents=True, exist_ok=True)
    with open(v4_path, "wb") as f:
        f.write(b"GGUF")  # Magic
        f.write(struct.pack("<I", 4))  # Version 4 (unsupported)
        f.write(struct.pack("<Q", 1))  # 1 KV pair
        # Write one KV pair so the header is structurally valid
        key = b"general.architecture"
        f.write(struct.pack("<Q", len(key)))  # Key length
        f.write(key)
        f.write(struct.pack("<I", 0))  # String value type
        value = b"llama"
        f.write(struct.pack("<Q", len(value)))  # String length
        f.write(value)

    print(f"Generated {5} fixtures in {FIXTURES_DIR}")
    for p in sorted(FIXTURES_DIR.glob("*.gguf")):
        print(f"  {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

### Rationale

- **`GGUFWriter` for valid files**: The `gguf.GGUFWriter` class creates valid GGUF v3 files with proper structure. Using it for the "valid" fixtures ensures they are structurally correct.
- **Manual binary writes for edge cases**: For corrupt/truncated/unsupported-version fixtures, manual `struct`-based binary writes are simpler and more precise than trying to trick `GGUFWriter`.
- **No tensor data**: All fixtures contain only header metadata (no tensors), keeping them under 10 KiB. The `GGUFWriter` context manager handles alignment padding correctly.
- **CI compatibility**: Fixture files are committed to the repository. CI consumes static bytes only — no generation step needed in CI.

### Fixture File Sizes

| Fixture | Approx. Size |
|---------|-------------|
| `valid_gguf_v3_all_keys.gguf` | ~500 bytes |
| `valid_gguf_v3_missing_name.gguf` | ~450 bytes |
| `corrupt_bad_magic.gguf` | 16 bytes |
| `truncated_no_kv.gguf` | 16 bytes |
| `valid_gguf_v4.gguf` | ~40 bytes |

### Alternatives Considered

- **Copy from real model files**: Would produce larger fixtures and require access to actual GGUF files. Synthetic fixtures are faster, deterministic, and don't depend on external files.
- **Use `gguf` library for all fixtures**: The library doesn't support writing invalid files (bad magic, truncated). Manual binary writes are necessary for corruption tests.

---

## File Structure Recommendations

Based on the M4 spec, existing architecture, and the `gguf` library API:

```text
src/llama_manager/
  gguf_metadata.py       # Pure library: GGUF header parsing, field extraction,
                         # error classification, unicode normalization
                         # (no imports from llama_cli)

src/llama_cli/
  # No new CLI file needed — doctor command already exists in cli_parser.py
  # The doctor command will call llama_manager.gguf_metadata for metadata extraction

src/tests/
  test_gguf_metadata.py  # Unit tests: fixture-based, mocked file I/O
  fixtures/
    valid_gguf_v3_all_keys.gguf
    valid_gguf_v3_missing_name.gguf
    corrupt_bad_magic.gguf
    truncated_no_kv.gguf
    valid_gguf_v4.gguf
scripts/
  generate_gguf_fixtures.py  # Maintainer script to regenerate fixtures
```

### Module Boundaries

**`llama_manager/gguf_metadata.py`** (pure library):
- `GgufParseErrorKind` enum — error classification
- `GgufParseError` exception — typed error with classification
- `GgufMetadata` dataclass — extracted metadata fields
- `read_gguf_metadata()` — core parsing function (prefix cap, timeout)
- `read_gguf_metadata_safe()` — wrapper with safe error classification
- `normalize_gguf_string()` — NFKC normalization
- `_extract_str()` / `_extract_int()` — typed field accessors
- No argparse, no Rich, no subprocess

**`src/tests/test_gguf_metadata.py`** (unit tests):
- `TestValidGgufV3` — tests with `valid_gguf_v3_all_keys.gguf` fixture
- `TestMissingName` — tests with `valid_gguf_v3_missing_name.gguf` fixture
- `TestCorruptFile` — tests with `corrupt_bad_magic.gguf` fixture
- `TestUnsupportedVersion` — tests with `valid_gguf_v4.gguf` fixture
- `TestTruncatedFile` — tests with `truncated_no_kv.gguf` fixture
- `TestTimeout` — tests with a large synthetic file (mocked `time.sleep`)
- `TestNormalizeString` — tests NFKC normalization edge cases

---

## Testing Strategy

### Unit Tests (in `tests/test_gguf_metadata.py`)

```python
import pytest
from pathlib import Path
from llama_manager.gguf_metadata import (
    GgufParseError,
    GgufParseErrorKind,
    read_gguf_metadata_safe,
    normalize_gguf_string,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestValidGgufV3:
    def test_extract_all_required_fields(self) -> None:
        metadata = read_gguf_metadata_safe(FIXTURES_DIR / "valid_gguf_v3_all_keys.gguf")
        assert metadata.architecture == "llama"
        assert metadata.name == "Test Model"
        assert metadata.embedding_length == 4096
        assert metadata.context_length == 4096
        assert metadata.block_count == 32
        assert metadata.attention_head_count == 32
        assert metadata.attention_head_count_kv == 8


class TestMissingName:
    def test_returns_none_for_missing_name(self) -> None:
        metadata = read_gguf_metadata_safe(FIXTURES_DIR / "valid_gguf_v3_missing_name.gguf")
        assert metadata.name is None
        assert metadata.architecture == "llama"


class TestCorruptFile:
    def test_bad_magic_raises_corrupt_error(self) -> None:
        with pytest.raises(GgufParseError) as exc_info:
            read_gguf_metadata_safe(FIXTURES_DIR / "corrupt_bad_magic.gguf")
        assert exc_info.value.kind == GgufParseErrorKind.CORRUPT_FILE
        assert "Invalid magic bytes" in str(exc_info.value)


class TestUnsupportedVersion:
    def test_gguf_v4_raises_unsupported_error(self) -> None:
        with pytest.raises(GgufParseError) as exc_info:
            read_gguf_metadata_safe(FIXTURES_DIR / "valid_gguf_v4.gguf")
        assert exc_info.value.kind == GgufParseErrorKind.UNSUPPORTED_VERSION
        assert "unsupported" in str(exc_info.value).lower()


class TestTruncatedFile:
    def test_truncated_file_raises_corrupt_error(self) -> None:
        with pytest.raises(GgufParseError) as exc_info:
            read_gguf_metadata_safe(FIXTURES_DIR / "truncated_no_kv.gguf")
        # Either CORRUPT_FILE (can't read KV pairs) or PARSE_ERROR
        assert exc_info.value.kind in (
            GgufParseErrorKind.CORRUPT_FILE,
            GgufParseErrorKind.PARSE_ERROR,
        )


class TestNormalizeString:
    def test_nfkc_normalizes_fullwidth_latin(self) -> None:
        # Fullwidth Latin 'Ａ' (U+FF21) -> ASCII 'A'
        assert normalize_gguf_string("Ａ") == "A"

    def test_nfkc_preserves_ascii(self) -> None:
        assert normalize_gguf_string("Qwen3.5-2B") == "Qwen3.5-2B"

    def test_none_passes_through(self) -> None:
        assert normalize_gguf_string(None) is None
```

### Mocking Strategy

No mocking needed for fixture-based tests — the fixture files are real GGUF binary files. The `GGUFReader` parses them correctly. The only mocking needed is for timeout tests:

```python
class TestTimeout:
    def test_timeout_on_large_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Create a large file (100 MiB) with valid header
        large_path = FIXTURES_DIR / "large_gguf.gguf"
        with open(large_path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", 1))
            key = b"general.architecture"
            f.write(struct.pack("<Q", len(key)))
            f.write(key)
            f.write(struct.pack("<I", 0))
            value = b"llama"
            f.write(struct.pack("<Q", len(value)))
            f.write(value)
            # Pad to 100 MiB
            f.write(b"\x00" * (100 * 1024 * 1024 - f.tell()))

        try:
            with pytest.raises(TimeoutError):
                read_gguf_metadata_safe(large_path, prefix_cap=32 * 1024 * 1024, timeout=0.1)
        finally:
            large_path.unlink(missing_ok=True)
```

---

## CI Quality Gates

All plans must note the CI gates:

1. **lint** — `uv run ruff check .` + `uv run ruff format --check`
2. **typecheck** — `uv run pyright`
3. **test** — `uv run pytest` with coverage
4. **security** — `uv run pip-audit` (for gguf dependency)

### Dependencies to Add

```toml
[project.dependencies]
gguf>=0.18.0  # For GGUF metadata extraction (Constraint C-003)
```

### Ruff Security Considerations

The `.s` (flake8-bandit) rules in `pyproject.toml` may flag:
- `S603` (subprocess call): Not applicable — no subprocess usage in gguf_metadata.py
- `S101` (assert): Allowed in tests via per-file-ignores
- `S310` (URL validation): Not applicable — no URL handling

### pyright Type Checking

The `gguf` package has a `py.typed` marker, so all types are available for static checking. Key types:
- `GGUFReader` — has `fields: OrderedDict[str, ReaderField]` and `tensors: list[ReaderTensor]`
- `ReaderField` — has `contents()` method returning `Any`
- `GGUFWriter` — has `add_string()`, `add_uint32()`, `add_uint64()`, etc.

---

## Implementation Sequence

1. **Phase 1**: Add `gguf>=0.18.0` dependency to `pyproject.toml`
2. **Phase 2**: Implement `llama_manager/gguf_metadata.py` (pure library)
   - `GgufParseErrorKind` enum and `GgufParseError` exception
   - `GgufMetadata` dataclass with typed field accessors
   - `read_gguf_metadata()` — core parsing with prefix cap and timeout
   - `read_gguf_metadata_safe()` — safe wrapper with error classification
   - `normalize_gguf_string()` — NFKC normalization
3. **Phase 3**: Generate test fixtures with `src/scripts/generate_gguf_fixtures.py`
4. **Phase 4**: Write unit tests in `tests/test_gguf_metadata.py`
5. **Phase 5**: Integrate with existing `doctor` command in `cli_parser.py`
6. **Phase 6**: Wire up smoke model ID resolution (FR-005) to use `general.name` from GGUF metadata

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `gguf` library API changes between versions | Pin to `>=0.18.0` and test against latest release in CI. The library is maintained by GGML and stable. |
| Large GGUF files cause memory pressure from `np.memmap` | Use prefix cap (32 MiB) and write header to temp file. The temp file is at most 32 MiB. |
| Timeout thread doesn't stop cleanly | Use daemon thread — it's killed when the main thread exits. No resource leak. |
| Fixture files drift from real GGUF structure | Regenerate fixtures with `src/scripts/generate_gguf_fixtures.py` if `gguf` library changes. CI uses committed bytes only. |
| `gguf` library reads tensor data eagerly | `GGUFReader` uses `np.memmap` and only reads tensors when `reader.tensors` is accessed. We never access tensors for metadata extraction. |
| `gguf` package adds transitive dependencies (numpy, pyyaml, requests, tqdm) | These are all lightweight, well-tested packages. `numpy` is the largest but is already used by `psutil` indirectly. `pip-audit` will catch any CVEs. |

---

## Appendix: GGUF Binary Format Reference

For reference, the GGUF v3 binary format is:

```
Offset  Size      Field
------  --------  ---------------------------
0       4 bytes   Magic: "GGUF" (0x46554747 LE)
4       4 bytes   Version: uint32 LE (3 for v3)
8       8 bytes   Tensor count: uint64 LE
16      8 bytes   KV pair count: uint64 LE

--- Key-Value Pairs (repeated kv_count times) ---
0       8 bytes   Key length: uint64 LE
8       N bytes   Key string (UTF-8)
8+N     4 bytes   Value type: uint32 LE (0=uint8, 1=int8, 2=uint16, 3=int16,
                  4=uint32, 5=int32, 6=float32, 7=bool, 8=float16,
                  9=uint64, 10=int64, 11=float64, 12=string,
                  13=array, 14=uint8 array, etc.)
12+N    Variable  Value data (depends on type)
                  - Strings: uint64 length + N bytes UTF-8
                  - Scalars: 1-8 bytes depending on type
                  - Arrays: uint32 sub-type + uint64 length + elements

--- Tensor Data (not read for metadata extraction) ---
Aligned to 32-byte boundary after KV pairs
```

Key-value types from `gguf.constants.GGUFValueType`:

```python
class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    FLOAT16 = 8
    UINT64  = 9
    INT64   = 10
    FLOAT64 = 11
    STRING  = 12
    ARRAY   = 13
    UINT8_ARRAY  = 14
    INT8_ARRAY   = 15
    UINT16_ARRAY = 16
    INT16_ARRAY  = 17
    UINT32_ARRAY = 18
    INT32_ARRAY  = 19
    FLOAT32_ARRAY = 20
    BOOL_ARRAY   = 21
    UINT64_ARRAY = 22
    INT64_ARRAY  = 23
    FLOAT64_ARRAY = 24
```
