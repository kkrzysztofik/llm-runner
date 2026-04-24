# GGUF Test Fixtures

## Purpose

Synthetic GGUF binary files used by unit tests for GGUF metadata extraction
(`llama_manager.metadata`).  They exercise the parser's happy path, edge cases,
and error paths without requiring real model weights.

**Note on the `gguf` dependency:** The fixture files themselves are generated
with Python's `struct` module — no external dependencies are needed to create
them.  However, `llama_manager.metadata` has module-level imports from the
`gguf` package (`from gguf.constants import Keys`,
`from gguf.gguf_reader import ReaderField`), so the `gguf` PyPI dependency is
still required at test time for any test that imports from `llama_manager`.

## How They Are Generated

All fixtures are produced by a single deterministic script:

```bash
uv run python src/scripts/generate_gguf_fixtures.py
```

The script uses Python's `struct` module to write binary data directly — no
external dependencies.  It creates minimal GGUF files containing only header
metadata (no tensor data), keeping every file under 10 KiB.

## How to Regenerate

```bash
# From the repository root
uv run python src/scripts/generate_gguf_fixtures.py
```

The script overwrites any existing fixtures in `src/tests/fixtures/`.  No
cleansing is needed before re-running.

## Fixture List

| Fixture | Description | Size |
|---------|-------------|------|
| `gguf_v3_valid.gguf` | Valid GGUF v3 with all required keys: `general.name`, `general.architecture`, `tokenizer.type`, `llama.embedding_length`, `llama.block_count`, `llama.context_length`, `llama.attention.head_count`, `llama.attention.head_count_kv` | ~200 B |
| `gguf_v3_no_name.gguf` | Valid GGUF v3 missing `general.name` (tests fallback to normalized filename stem) | ~180 B |
| `gguf_corrupt.gguf` | Corrupt file — bad magic bytes (`XXXX` instead of `GGUF`) | 16 B |
| `gguf_truncated.gguf` | Truncated file — valid v3 header with 0 key-value pairs, no KV section | 16 B |
| `gguf_v4_unsupported.gguf` | Valid GGUF v4 with all required keys (expected to produce "unsupported version" error) | ~200 B |

## Binary Format

GGUF v3 binary layout:

```
Magic:     "GGUF" (4 bytes)
Version:   uint32 LE (4 bytes) = 3
KV count:  uint64 LE (8 bytes)
N × KV:    key_len (uint64) + key (UTF-8) + value_type (uint32) + value (type-dependent)
```

Value type constants used:

| Constant | Value |
|----------|-------|
| `GGUF_TYPE_UINT32` | 4 |
| `GGUF_TYPE_STRING` | 8 |

## CI Note

CI consumes **committed bytes only** — no runtime generation.  The fixture
files are checked into version control under `src/tests/fixtures/`.  The
generator script exists for maintainers to regenerate or extend fixtures
locally.
