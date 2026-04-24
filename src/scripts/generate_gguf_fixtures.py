"""Generate synthetic GGUF test fixtures for metadata extraction tests.

Creates minimal GGUF v3/v4 binary files under src/tests/fixtures/ for
testing the GGUF metadata extraction code path in
llama_manager.metadata.

Binary format (GGUF v3):
    Magic: "GGUF" (4 bytes)
    Version: uint32 LE (4 bytes)
    KV count: uint64 LE (8 bytes)
    N × (key_len: uint64 LE, key: UTF-8, value_type: uint32 LE, value: bytes)

Value type constants:
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_STRING = 8

All fixtures are under 10 KiB and contain no tensor data.
"""

import struct
import sys
from collections.abc import Callable
from pathlib import Path

# GGUF value type constants
_GGUF_TYPE_UINT32: int = 4
_GGUF_TYPE_STRING: int = 8

# Required keys for a minimal valid llama model GGUF v3 file
_REQUIRED_KEYS: dict[str, tuple[int, bytes]] = {
    "general.architecture": (_GGUF_TYPE_STRING, b"llama"),
    "tokenizer.type": (_GGUF_TYPE_STRING, b"bpe"),
    "llama.embedding_length": (_GGUF_TYPE_UINT32, struct.pack("<I", 4096)),
    "llama.block_count": (_GGUF_TYPE_UINT32, struct.pack("<I", 32)),
    "llama.context_length": (_GGUF_TYPE_UINT32, struct.pack("<I", 8192)),
    "llama.attention.head_count": (_GGUF_TYPE_UINT32, struct.pack("<I", 32)),
    "llama.attention.head_count_kv": (_GGUF_TYPE_UINT32, struct.pack("<I", 8)),
}

# general.name is optional for the no_name variant
_GENERAL_NAME: tuple[int, bytes] = (
    _GGUF_TYPE_STRING,
    struct.pack("<Q", len(b"test-model-v1")) + b"test-model-v1",
)


def _pack_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    """Pack a single GGUF key-value pair into binary form.

    Args:
        key: The key string.
        value_type: GGUF value type constant.
        value_bytes: Raw bytes for the value payload.

    Returns:
        Binary representation of the KV pair (without key length prefix).
    """
    key_bytes = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_bytes))  # key length
        + key_bytes  # key bytes
        + struct.pack("<I", value_type)  # value type
        + value_bytes  # value bytes
    )


def _build_kv_section(keys_with_values: dict[str, tuple[int, bytes]]) -> bytes:
    """Build the key-value section of a GGUF file.

    Args:
        keys_with_values: Mapping of key to (value_type, value_bytes).

    Returns:
        Concatenated binary KV section.
    """
    parts: list[bytes] = []
    for key, (value_type, value_bytes) in keys_with_values.items():
        kv = _pack_kv(key, value_type, value_bytes)
        parts.append(kv)
    return b"".join(parts)


def _write_gguf_v3(
    path: Path,
    kv_section: bytes,
    magic: bytes = b"GGUF",
    version: int = 3,
) -> None:
    """Write a minimal GGUF v3 file.

    Args:
        path: Output file path.
        kv_section: Pre-built key-value section bytes.
        magic: Magic bytes prefix (default GGUF v3).
        version: GGUF version number (default 3).
    """
    kv_count = struct.pack("<Q", _count_kv_pairs(kv_section) if kv_section else 0)

    header = magic + struct.pack("<I", version) + kv_count
    path.write_bytes(header + kv_section)


def _count_kv_pairs(kv_section: bytes) -> int:
    """Count the number of KV pairs in a GGUF KV section.

    Parses the first uint64 of each KV pair (key length) to count pairs.
    """
    count = 0
    offset = 0
    while offset < len(kv_section):
        if offset + 8 > len(kv_section):
            break
        key_len = struct.unpack_from("<Q", kv_section, offset)[0]
        if key_len == 0 or offset + 8 + key_len > len(kv_section):
            break
        offset += 8 + key_len  # skip key length + key bytes
        # Skip value type (uint32)
        if offset + 4 > len(kv_section):
            break
        value_type = struct.unpack_from("<I", kv_section, offset)[0]
        offset += 4
        # Skip value bytes based on type
        if value_type == _GGUF_TYPE_STRING:
            if offset + 8 > len(kv_section):
                break
            str_len = struct.unpack_from("<Q", kv_section, offset)[0]
            offset += 8 + str_len
        elif value_type == _GGUF_TYPE_UINT32:
            offset += 4
        elif value_type == 5:  # UINT64
            offset += 8
        elif value_type == 6:  # FLOAT32
            offset += 4
        elif value_type == 7:  # BOOL
            offset += 1
        else:
            # Unknown type — skip remaining as best effort
            break
        count += 1
    return count


def generate_valid_v3(path: Path) -> None:
    """Generate a valid GGUF v3 file with all required keys."""
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME[1]) + kv
    _write_gguf_v3(path, kv)


def generate_valid_v3_no_name(path: Path) -> None:
    """Generate a valid GGUF v3 file missing general.name."""
    kv = _build_kv_section(_REQUIRED_KEYS)
    _write_gguf_v3(path, kv)


def generate_corrupt(path: Path) -> None:
    """Generate a corrupt GGUF file with bad magic bytes."""
    # Bad magic: "XXXX" instead of "GGUF"
    path.write_bytes(b"XXXX\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def generate_truncated(path: Path) -> None:
    """Generate a truncated GGUF file with valid header but no KV data."""
    # Valid v3 header, 0 key-value pairs, no KV section
    path.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def generate_v4_unsupported(path: Path) -> None:
    """Generate a valid GGUF v4 file (unsupported version)."""
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME[1]) + kv
    _write_gguf_v3(path, kv, magic=b"GGUF", version=4)


def main() -> int:
    """Generate all GGUF test fixtures.

    Returns:
        0 on success, 1 on failure.
    """
    fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    generators: list[tuple[str, Path, Callable[[Path], None]]] = [
        ("gguf_v3_valid.gguf", fixtures_dir / "gguf_v3_valid.gguf", generate_valid_v3),
        ("gguf_v3_no_name.gguf", fixtures_dir / "gguf_v3_no_name.gguf", generate_valid_v3_no_name),
        ("gguf_corrupt.gguf", fixtures_dir / "gguf_corrupt.gguf", generate_corrupt),
        ("gguf_truncated.gguf", fixtures_dir / "gguf_truncated.gguf", generate_truncated),
        (
            "gguf_v4_unsupported.gguf",
            fixtures_dir / "gguf_v4_unsupported.gguf",
            generate_v4_unsupported,
        ),
    ]

    success = True
    for name, path, gen_fn in generators:
        try:
            gen_fn(path)
            size = path.stat().st_size
            print(f"  OK   {name} ({size} bytes)")
        except Exception as exc:
            print(f" FAIL {name}: {exc}", file=sys.stderr)
            success = False

    if success:
        print(f"\nGenerated {len(generators)} fixtures in {fixtures_dir}")
    else:
        print("\nSome fixtures failed to generate.", file=sys.stderr)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
