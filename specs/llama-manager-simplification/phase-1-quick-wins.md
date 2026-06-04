# Phase 1 — Quick Wins

No dependencies. Can run in parallel.

---

## Task 1: Consolidate Redaction into `common/security.py`

**Priority**: Critical | **Effort**: ~2h | **Files**: 4

### Current State

Three distinct redaction implementations:

1. **`common/security.py`** (canonical, 147 lines)
   - `is_sensitive_key(key)` — checks if a key name is sensitive
   - `redact_env_value(value, key)` — redacts a single env value by key
   - `redact_log_line(line)` — regex-based redaction of `KEY=value` in log text
   - `safe_log(template)` — t-string aware redaction
   - Patterns: `SENSITIVE_KEY_PATTERN`, `SENSITIVE_WORD_PATTERN`,
     `SENSITIVE_KEY_NAME_PATTERN`, `_LOG_SENSITIVE_PATTERN`

2. **`orchestration/manager.py`** lines 204–225 — `_redact_sensitive(text)`
   - Reimplements regex-based redaction with additional quoted-value and bearer-token handling
   - Uses its own `SENSITIVE_WORD_PATTERN` (module-level, different from `security.py`'s)
   - Handles: `KEY="quoted"`, `KEY='single-quoted'`, `Authorization: Bearer xxx`

3. **`orchestration/artifact.py`** lines 233–258 — `_redact_sensitive_in_dict(data)`
   - Recursive dict traversal calling `_redact_value(key, value, prefix)`
   - Uses `is_sensitive_key` from `common.security` but adds its own `_redact_value` wrapper

### Steps

1. **Add to `common/security.py`**:
   - `redact_text(text: str) -> str` — merge `_redact_sensitive` from `manager.py` into a
     single function that handles plain text, quoted values, and bearer tokens. Use the
     patterns already defined in `security.py` plus add bearer-token pattern.
   - `redact_dict(data: dict, env_key_prefix: str = "") -> dict` — move
     `_redact_sensitive_in_dict` + `_redact_value` from `artifact.py`

2. **Update `orchestration/manager.py`**:
   - Delete `_redact_sensitive` function (lines 204–225) and its module-level
     `SENSITIVE_WORD_PATTERN`
   - Replace all calls with `from ..common.security import redact_text`
   - Update `_stream_pipe` to call `redact_text` instead

3. **Update `orchestration/artifact.py`**:
   - Delete `_redact_sensitive_in_dict` and `_redact_value` (lines 233–258)
   - Replace with `from ..common.security import redact_dict`

4. **Update `orchestration/__init__.py`**:
   - Remove `_redact_sensitive` and `_redact_sensitive_in_dict` from exports
   - Add `redact_text` and `redact_dict` if needed by tests

5. **Update tests**:
   - `test_audit_redaction.py` — update imports from `orchestration` to `common.security`
   - Any test importing `_redact_sensitive` or `_redact_sensitive_in_dict` — update

### Pass Criteria

- [ ] Only one redaction module: `common/security.py`
- [ ] `orchestration/manager.py` has no local redaction code
- [ ] `orchestration/artifact.py` has no local redaction code
- [ ] All existing tests pass (`uv run pytest`)
- [ ] `uv run pre-commit run --all-files` passes

---

## Task 2: Consolidate Port Validation

**Priority**: High | **Effort**: ~2h | **Files**: 4

### Current State

Four separate port validation functions with **inconsistent ranges**:

| Location | Function | Range | Return Type |
|---|---|---|---|
| `config/server.py:194` | `validate_slot_port(port, slot_id)` | 1024–65535 | `ValidationResult` |
| `validation/validators.py:15` | `validate_port(port, name)` | 1–65535 | `ErrorDetail \| None` |
| `validation/validators.py:119` | inline in `_validate_slot` | 1–65535 | `ValidationResult` (inline) |
| `config/profiles.py:144` | `_require_port(port)` | 1–65535 | raises `SlotProfileError` |

The range inconsistency (1024 vs 1) is a bug — privileged ports (< 1024) should be
blocked everywhere.

### Steps

1. **Create `common/validators.py`** (new file):
   ```python
   PORT_MIN = 1024
   PORT_MAX = 65535

   def is_valid_port(port: int) -> bool:
       return isinstance(port, int) and PORT_MIN <= port <= PORT_MAX

   def validate_port_range(port: int) -> str | None:
       """Return error message if port is invalid, None if valid."""
       if not isinstance(port, int) or port < PORT_MIN or port > PORT_MAX:
           return f"port must be between {PORT_MIN} and {PORT_MAX}, got: {port}"
       return None
   ```

2. **Update `config/server.py`** — `validate_slot_port`:
   - Import `validate_port_range` from `common.validators`
   - Delegate range check, wrap result in `ValidationResult`

3. **Update `validation/validators.py`** — `validate_port`:
   - Import `validate_port_range` from `common.validators`
   - Delegate range check, wrap result in `ErrorDetail`
   - Remove inline port check in `_validate_slot` (line 119), call `validate_port` instead

4. **Update `config/profiles.py`** — `_require_port`:
   - Import `validate_port_range` from `common.validators`
   - Delegate range check, raise `SlotProfileError` on error message

### Pass Criteria

- [ ] Single source of truth for port range: `common/validators.py`
- [ ] All 4 call sites use `PORT_MIN`/`PORT_MAX` constants
- [ ] Range is consistently 1024–65535
- [ ] All existing tests pass
- [ ] `uv run pre-commit run --all-files` passes

---

## Task 4: Move Logic Out of `metadata/__init__.py`

**Priority**: Low | **Effort**: ~30min | **Files**: 1

### Current State

`metadata/__init__.py` (134 lines) contains:
- `_parse_gguf_in_thread` (lines 21–58) — worker thread function
- `extract_gguf_metadata` (lines 61–134) — public API with timeout logic
- Parameter validation, threading, queue management

This is non-trivial logic that belongs in a dedicated module.

### Steps

1. **Create `metadata/extractor.py`**:
   - Move `_parse_gguf_in_thread` and `extract_gguf_metadata` there
   - Keep all imports and logic intact

2. **Slim `metadata/__init__.py`** to re-exports only:
   ```python
   from ._types import GGUFMetadataRecord, normalize_filename
   from .extractor import extract_gguf_metadata

   __all__ = ["GGUFMetadataRecord", "extract_gguf_metadata", "normalize_filename"]
   ```

3. **Update any direct imports** of `extract_gguf_metadata` from `metadata` — should
   still work via re-export.

### Pass Criteria

- [ ] `metadata/__init__.py` is < 20 lines (re-exports only)
- [ ] `metadata/extractor.py` contains all threading/timeout logic
- [ ] All existing tests pass
