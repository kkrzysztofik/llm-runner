# Phase 2 — API Hygiene

No dependencies. Can run in parallel with Phase 1.

---

## Task 3: Consolidate Slot ID Normalization

**Priority**: Medium | **Effort**: ~1h | **Files**: 2

### Current State

| Location | Function | Behavior |
|---|---|---|
| `config/server.py:127` | `normalize_slot_id(slot_id)` | Regex `[^a-z0-9_-]` → strip chars, raises `ValueError` on empty |
| `config/profiles.py:160` | `_normalize_alias(alias)` | Simple `strip().replace("_", "-")` — no filtering, no error |

These do different things but serve the same conceptual purpose (normalizing identifiers).

### Steps

1. **Keep `normalize_slot_id` in `config/server.py`** as the canonical slot ID normalizer
   (it's the stricter, correct one).

2. **Update `config/profiles.py`** — `_normalize_alias`:
   - Replace with a call to `normalize_slot_id` from `config/server.py`
   - OR if the underscore-to-hyphen behavior is intentional and different, rename to
     `_profile_id_from_alias` and add a docstring explaining the difference
   - Audit all callers of `_normalize_alias` to confirm expected behavior

3. **Add tests** if the behavior difference is intentional — document why.

### Pass Criteria

- [ ] Only one slot-ID normalization function, OR clearly documented distinction
- [ ] All callers updated
- [ ] All existing tests pass

---

## Task 5: Stop Exporting Private Symbols from Subpackage `__init__.py`

**Priority**: Medium | **Effort**: ~2h | **Files**: 3 + tests

### Current State

Three subpackages export `_`-prefixed internals for test access:

| Package | Private Exports | Count |
|---|---|---|
| `probe/__init__.py` | `_EXIT_CODE_MAP`, `_resolve_sha`, `_handle_models_status`, `_models_failure_result`, `_probe_chat`, `_probe_models`, `_tcp_connect` | 7 |
| `benchmark/__init__.py` | `_LATENCY_PATTERNS`, `_TOKENS_PATTERNS`, `_VRAM_PATTERNS`, `_extract_first_float`, `_extract_latency`, `_extract_number_from_patterns`, `_extract_tokens_per_second`, `_extract_vram`, `_find_column_indices`, `_parse_data_row`, `_parse_markdown_table_metrics`, `_parse_table_block`, `_split_contiguous_blocks` | 13 |
| `orchestration/__init__.py` | `_append_audit_log`, `_redact_sensitive`, `_rotate_audit_log`, `_verify_shutdown_ownership`, `_redact_sensitive_in_dict` | 5 |

### Steps

1. **Update tests** to import private symbols directly from their source modules:
   - Tests for probe internals:
     `from llama_manager.probe.smoke import _tcp_connect, _probe_models, ...`
   - Tests for benchmark internals:
     `from llama_manager.benchmark.parser import _LATENCY_PATTERNS, ...`
   - Tests for orchestration internals:
     `from llama_manager.orchestration.manager import _append_audit_log, ...`

2. **Remove private exports** from each `__init__.py`:
   - `probe/__init__.py` — remove 7 `_`-prefixed imports and `__all__` entries
   - `benchmark/__init__.py` — remove 13 `_`-prefixed imports and `__all__` entries
   - `orchestration/__init__.py` — remove 5 `_`-prefixed imports and `__all__` entries

3. **Remove `# noqa: F401`** markers on the removed imports.

### Pass Criteria

- [ ] No `_`-prefixed symbols in any subpackage `__init__.py` `__all__`
- [ ] Tests import internals directly from source modules
- [ ] All existing tests pass

---

## Task 12: Consolidate Filename Sanitization

**Priority**: Low | **Effort**: ~1h | **Files**: 2

### Current State

| Location | Function | Approach |
|---|---|---|
| `config/profile_cache.py:225` | `_sanitize_filename_component` | Regex `[^a-z0-9_\-\.]` → replace with `_`, raises `ValueError` |
| `config/profiles.py:160` | `_normalize_alias` | `strip().replace("_", "-")` — no filtering |

### Steps

1. **Move `_sanitize_filename_component` + `_FILENAME_SANITIZE_PATTERN`** to
   `common/text.py` (new file).

2. **Update `config/profile_cache.py`** — import from `common.text`.

3. **Update `config/profiles.py`** — `_normalize_alias`:
   - If it should produce filename-safe output: delegate to `sanitize_filename_component`
   - If it's intentionally different (e.g., for display names): rename to `_display_alias`
     and add docstring

4. **Audit callers** to confirm which behavior is needed.

### Pass Criteria

- [ ] Single sanitization function in `common/text.py`
- [ ] Both `profiles.py` and `profile_cache.py` use it
- [ ] All existing tests pass
