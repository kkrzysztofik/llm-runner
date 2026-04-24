# Research: M4 — Operational Hardening and Smoke Verification

**Date**: 2026-04-23
**Feature**: M4 — Operational Hardening and Smoke Verification
**Branch**: `003-m4-op-hardening`

---

## 1. httpx for Smoke Probes

### Decision: Per-Request `httpx.Timeout` Object with Per-Probe Client Lifecycle
Use a fine-grained `httpx.Timeout(connect=120s, read=10s)` per smoke probe, with a fresh `httpx.Client` created per slot via `with httpx.Client() as client:`. No shared client across slots — probes are sequential, so connection pooling provides no benefit.

**Rationale**:
- The M4 spec requires two distinct timeout values: `smoke_listen_timeout_s` (120s, for TCP ready-check) and `smoke_http_request_timeout_s` (10s, for HTTP requests). A single scalar timeout cannot express this distinction.
- Per-request timeout is preferred over client-level because each smoke probe targets a different port.
- Each probe may use a different API key (resolved via precedence: `--api-key` CLI > `smoke.api_key` config > `LLM_RUNNER_SMOKE_API_KEY` env).
- The spec explicitly states: "Each phase attempted exactly once — no retries." Per-request timeouts align with this no-retry semantics.

**Error Classification** (httpx exception → exit code):
| httpx Exception | Exit Code | Meaning |
|---|---|---|
| `ConnectTimeout` (listen phase) | 10 | Server not ready |
| `ConnectError`, `NetworkError`, `ReadTimeout` (HTTP phase) | 11 | HTTP/API/network error |
| `HTTPStatusError` with 401/403 | 15 | Auth failure |
| `ReadTimeout` (chat phase) | 14 | Chat timeout |
| Empty `models` array, model ID mismatch | 13 | Model not found (business logic) |

**API Key Injection**: `Authorization: Bearer <key>` at the client level. The `Bearer` scheme is the standard for OpenAI-compatible APIs.

**JSON Response Parsing**: Use `response.json()` for non-streaming responses (`stream: False`). Parse `models[0].id` for `/v1/models` and `choices[0].message.content` + `model` field for chat completion. Check HTTP status codes before parsing JSON to avoid `JSONDecodeError` on error responses.

**Alternatives Considered**:
- Client-level timeout: Rejected — requires new client per probe anyway for different timeouts.
- `httpx.Timeout(None)` (infinite): Rejected — spec mandates bounded timeouts.

### Dependencies
- Runtime: `httpx>=0.28.0` (per Constraint C-002)
- Dev: `pytest-httpx>=0.30.0` (for mocking httpx in tests)

---

## 2. gguf Library for Metadata Extraction

### Decision: `gguf.GGUFReader` with Prefix-Capped Temp File + Threading Timeout
Use `GGUFReader` from the `gguf` PyPI library for header-only parsing. For prefix-capped reading of large files, read the first N bytes (default 32 MiB) into a `tempfile.NamedTemporaryFile`, then pass the temp file path to `GGUFReader`. Use a daemon thread with `join(timeout=5.0)` for cross-platform wall-clock timeout.

**Rationale**:
- `GGUFReader` uses `numpy.memmap` for memory-efficient file access — tensor data is never loaded for metadata extraction.
- `GGUFReader` requires a file path (not bytes) because `np.memmap` only accepts filesystem paths.
- The library is maintained by GGML (llama.cpp creators) and has PEP 561 type stubs.
- `READER_SUPPORTED_VERSIONS = [2, 3]` — v4+ raises `ValueError` with clear message (per FR-007).
- Error classification: `CORRUPT_FILE` (bad magic), `UNSUPPORTED_VERSION` (v4+), `PARSE_ERROR` (malformed).

**File Structure**:
```
src/llama_manager/metadata.py      # Pure library (GGUFReader wrapper)
src/tests/fixtures/gguf/           # 5 committed synthetic fixtures
src/scripts/generate_gguf_fixtures.py  # Maintainer fixture generator
```

**Fixture Generation**:
- Valid fixtures: use `gguf.GGUFWriter` (creates proper GGUF v3 structure)
- Corrupt/truncated/unversioned fixtures: use `struct`-based binary writes
- All fixtures under 10 KiB, no tensor data

**Dependencies**:
- Runtime: `gguf>=0.18.0` (per Constraint C-003)
- Dev: `pytest-httpx>=0.30.0` (for mocking httpx in tests)

---

## 3. Lockfile Implementation for Per-Slot Access Control

### Decision: Atomic `O_CREAT | O_EXCL` File Creation + Dual-Threshold Stale Detection
The existing implementation in `process_manager.py` already uses production-ready lockfile patterns:
- **Lock creation**: `O_CREAT | O_EXCL` (atomic, zero TOCTOU window)
- **Stale detection**: Dual-threshold — PID check (via `psutil`) + 300s age threshold
- **File format**: JSON with `version` field for schema evolution
- **Permissions**: `0o600` (owner-only)
- **Read-only commands**: `smoke`/`doctor` bypass lockfiles entirely
- **Naming convention**: `slot-{slot_id}.lock` in runtime directory
- **Release strategy**: Delete on clean shutdown; stale detection handles crashes

**Alternatives Evaluated**:
- `fcntl.flock()`: Automatically released on crash, but requires holding a file descriptor and doesn't work well with read-only commands.
- `os.link()`: Atomic but doesn't work across filesystem boundaries and has no built-in staleness detection.
- `portalocker`/`filelock`: External dependencies not justified for a simple per-slot lock.

**Incremental Improvements**:
1. Add `command` field to lock metadata for better `doctor --locks` output
2. Add `hostname` field for multi-user systems
3. Add a concurrent lock creation test (spawning actual subprocesses)
4. Make `lock_stale_threshold_s` configurable via `Config`

---

## 4. Rich Live Key Polling for Hardware Warnings

### Decision: Extend Existing `_input_poller` Daemon Thread + `_keypress_queue` Pattern with Warning State Machine
Reuse the existing `_cbreak_stdin()` context manager + `_input_poller()` pattern, extended with a state machine (`_WarningState.NONE/WAITING/RESOLVED`). Zero blocking: the poller thread does all I/O; the main thread only drains a queue with `get_nowait()`.

**Rationale**:
- The existing pattern already handles POSIX (`select.select()` + `tty.setcbreak()`) and Windows (`msvcrt.kbhit()` + `msvcrt.getch()`).
- Non-TTY environments fall through without cbreak mode.
- Monotonic deadline (`time.monotonic() + 30`) checked on each render cycle (~10Hz). No extra threads needed.
- Safe default on timeout: **abort** (`n`).
- Update `self.risk_panel` in-place — it's already part of the alerts section in `render()`.
- Ctrl+C during warning = abort (consistent with existing profile abort behavior).

**Key Mapping**:
- `y` → continue
- `n` or `q` → abort
- Enter without y/n/q → re-display prompt
- No input within 30s → abort (safe default)

---

## 5. Summary of All Decisions

| # | Topic | Decision | Key Rationale |
|---|-------|----------|---------------|
| 1 | httpx timeout | Per-request `httpx.Timeout(connect, read)` | Two distinct timeout values per spec |
| 2 | httpx client | Per-probe `httpx.Client` with context manager | Different ports, different API keys |
| 3 | httpx error classification | Exception hierarchy → exit codes | Clean mapping to Appendix B |
| 4 | gguf parsing | `GGUFReader` + temp file + threading timeout | Memory-efficient, official library |
| 5 | gguf version detection | `GGUFReader` raises `ValueError` for v4+ | Clear error per FR-007 |
| 6 | Lockfile creation | `O_CREAT` + `O_EXCL` atomic creation | Zero TOCTOU window |
| 7 | Lockfile stale detection | PID check + 300s age threshold | Handles crashes and PID reuse |
| 8 | Rich Live input | Extend existing `_input_poller` daemon thread | Zero blocking, cross-platform |
| 9 | Rich Live timeout | Monotonic deadline on render cycle | No extra threads needed |
