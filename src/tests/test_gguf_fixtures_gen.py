"""Generate GGUF test fixtures as a side-effect test.

This test generates synthetic GGUF binary files for metadata extraction tests.
It is idempotent and safe to run repeatedly.
"""

import struct
from pathlib import Path

# Value type constants
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

_GENERAL_NAME_VALUE: bytes = struct.pack("<Q", len(b"test-model-v1")) + b"test-model-v1"


def _pack_kv(key: str, value_type: int, value_bytes: bytes) -> bytes:
    key_bytes = key.encode("utf-8")
    return (
        struct.pack("<Q", len(key_bytes)) + key_bytes + struct.pack("<I", value_type) + value_bytes
    )


def _build_kv_section(keys_with_values: dict[str, tuple[int, bytes]]) -> bytes:
    return b"".join(_pack_kv(k, vt, vb) for k, (vt, vb) in keys_with_values.items())


def _count_kv_pairs(kv_section: bytes) -> int:
    count = 0
    offset = 0
    while offset < len(kv_section):
        if offset + 8 > len(kv_section):
            break
        key_len = struct.unpack_from("<Q", kv_section, offset)[0]
        if key_len == 0 or offset + 8 + key_len > len(kv_section):
            break
        offset += 8 + key_len
        if offset + 4 > len(kv_section):
            break
        value_type = struct.unpack_from("<I", kv_section, offset)[0]
        offset += 4
        if value_type == _GGUF_TYPE_STRING:
            if offset + 8 > len(kv_section):
                break
            str_len = struct.unpack_from("<Q", kv_section, offset)[0]
            offset += 8 + str_len
        elif value_type == _GGUF_TYPE_UINT32:
            offset += 4
        else:
            break
        count += 1
    return count


def _write_gguf_v3(
    path: Path,
    kv_section: bytes,
    magic: bytes = b"GGUF",
    version: int = 3,
) -> None:
    kv_count = struct.pack("<Q", _count_kv_pairs(kv_section))
    header = magic + struct.pack("<I", version) + kv_count
    path.write_bytes(header + kv_section)


def _generate_valid_v3(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME_VALUE) + kv
    _write_gguf_v3(path, kv)


def _generate_valid_v3_no_name(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    _write_gguf_v3(path, kv)


def _generate_corrupt(path: Path) -> None:
    path.write_bytes(b"XXXX\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def _generate_truncated(path: Path) -> None:
    path.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


def _generate_v4_unsupported(path: Path) -> None:
    kv = _build_kv_section(_REQUIRED_KEYS)
    kv = _pack_kv("general.name", _GGUF_TYPE_STRING, _GENERAL_NAME_VALUE) + kv
    _write_gguf_v3(path, kv, magic=b"GGUF", version=4)


def test_generate_gguf_fixtures(tmp_path: Path) -> None:
    """Generate all GGUF test fixtures for metadata extraction tests.

    Creates 5 synthetic GGUF files under src/tests/fixtures/:
    - gguf_v3_valid.gguf: valid GGUF v3 with all required keys
    - gguf_v3_no_name.gguf: valid GGUF v3 missing general.name
    - gguf_corrupt.gguf: corrupt file (bad magic bytes)
    - gguf_truncated.gguf: truncated file (valid header, no KV data)
    - gguf_v4_unsupported.gguf: valid GGUF v4 (unsupported version)

    All fixtures are under 10 KiB and contain no tensor data.
    """
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("gguf_v3_valid.gguf", _generate_valid_v3),
        ("gguf_v3_no_name.gguf", _generate_valid_v3_no_name),
        ("gguf_corrupt.gguf", _generate_corrupt),
        ("gguf_truncated.gguf", _generate_truncated),
        ("gguf_v4_unsupported.gguf", _generate_v4_unsupported),
    ]

    for name, gen_fn in generators:
        path = fixtures_dir / name
        gen_fn(path)
        size = path.stat().st_size
        assert size > 0, f"Fixture {name} is empty"
        assert size < 10240, f"Fixture {name} exceeds 10 KiB ({size} bytes)"
        print(f"  OK   {name} ({size} bytes)")

    print(f"\nGenerated {len(generators)} fixtures in {fixtures_dir}")
