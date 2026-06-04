# Phase 3 — Error + Config Refactoring

No dependencies on earlier phases. Can start after Phase 1/2 complete.

---

## Task 6: Consolidate Error Types

**Priority**: High | **Effort**: ~4h | **Files**: ~10

### Current State

Five error-related types with overlapping purposes:

| Type | Location | Purpose |
|---|---|---|
| `ValidationResult` | `config/errors.py:8` | Slot-aware pass/fail result (slot_id, passed, failed_check, error_code, error_message) |
| `ErrorDetail` | `config/errors.py:19` | Rich error with fix instructions (error_code, failed_check, why_blocked, how_to_fix, docs_ref) |
| `MultiValidationError` | `config/errors.py:30` | Collection of `ErrorDetail` with sort/count helpers |
| `ValidationResults` | `validation/commands/builder.py:76` | Dry-run payload (passed, checks: list[dict]) — confusable name! |
| `ValidationException` | `orchestration/lockfile.py:44` | Exception wrapper for `MultiValidationError` |

### Steps

1. **Merge `ValidationResult` into `ErrorDetail`**:
   - `ValidationResult` is essentially a lighter `ErrorDetail` with a `slot_id` and
     `passed` flag
   - Add `slot_id: str = ""` to `ErrorDetail`
   - Replace all `ValidationResult` usage with `ErrorDetail` (or `ErrorDetail | None`
     for pass/fail)
   - Remove `ValidationResult` dataclass

2. **Rename `ValidationResults`** (in `validation/commands/builder.py`):
   - Rename to `DryRunValidationSummary` to eliminate name confusion with
     `ValidationResult`
   - Update all references

3. **Keep `MultiValidationError`** — it serves a clear purpose as an error collection.

4. **Move `ValidationException`** from `orchestration/lockfile.py` to `config/errors.py`:
   - It wraps `MultiValidationError` — belongs with the error types
   - Update imports in `orchestration/lockfile.py`, `orchestration/manager.py`,
     `orchestration/artifact.py`

5. **Update all consumers** (~10 files):
   - `config/server.py` — `validate_slot_id`, `validate_slot_port` return types
   - `validation/validators.py` — `_validate_slot`, `_validate_duplicate_slots`,
     `_convert_results_to_errors`
   - `validation/commands/builder.py` — `sort_validation_errors`,
     `build_dry_run_slot_payload`
   - `orchestration/manager.py` — `_make_validation_error`, `_lockfile_error`
   - `orchestration/lockfile.py` — `_make_lockfile_validation_error`
   - `orchestration/artifact.py` — `_artifact_error`
   - All test files referencing `ValidationResult`

### Pass Criteria

- [ ] `ValidationResult` removed — replaced by `ErrorDetail`
- [ ] `ValidationResults` renamed to `DryRunValidationSummary`
- [ ] `ValidationException` lives in `config/errors.py`
- [ ] Only 3 error types remain: `ErrorDetail`, `MultiValidationError`,
      `ValidationException`
- [ ] All existing tests pass

---

## Task 7: Extract Speculative Decoding into Sub-dataclass

**Priority**: High | **Effort**: ~3h | **Files**: ~6

### Current State

`ServerConfig` has 36 fields. 12 are speculative decoding (36%):
```
spec_type, spec_ngram_size_n, draft_min, draft_max,
spec_draft_n_max, spec_draft_p_min, spec_draft_cache_type_k,
spec_draft_cache_type_v, spec_draft_device,
reasoning_mode, reasoning_format, reasoning_budget
```

`SlotProfileSpec` mirrors all of these (36 fields total).
`Config` has 12 `default_spec_*` fields.

### Steps

1. **Create `config/spec_decode.py`**:
   ```python
   @dataclass
   class SpeculativeDecodingConfig:
       spec_type: str = ""
       spec_ngram_size_n: int = 0
       draft_min: int = 0
       draft_max: int = 0
       spec_draft_n_max: int = 0
       spec_draft_p_min: float = 0.0
       spec_draft_cache_type_k: str = ""
       spec_draft_cache_type_v: str = ""
       spec_draft_device: str = ""
       reasoning_mode: str = "auto"
       reasoning_format: str = "none"
       reasoning_budget: str = ""
   ```

2. **Update `ServerConfig`** — replace 12 fields with:
   ```python
   spec_decode: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)
   ```
   - Move `__post_init__` validation for spec fields into
     `SpeculativeDecodingConfig.__post_init__`

3. **Update `SlotProfileSpec`** — same replacement.

4. **Update `Config`** — replace 12 `default_spec_*` fields with:
   ```python
   default_spec_decode: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)
   ```

5. **Update `build_server_cmd`** in `validation/commands/builder.py`:
   - Access spec fields via `cfg.spec_decode.spec_type` etc.

6. **Update `config/builder.py`** factory functions and `merge_config_overrides`:
   - Handle `SpeculativeDecodingConfig` as a nested dataclass

7. **Update all consumers** that access spec fields directly on `ServerConfig`:
   - `slot_manager.py`, `slot_profile_store.py`, TUI modals

### Pass Criteria

- [ ] `ServerConfig` has ≤ 24 fields (was 36)
- [ ] `SlotProfileSpec` has ≤ 24 fields (was 36)
- [ ] `Config` has ≤ 50 fields (was 62)
- [ ] `SpeculativeDecodingConfig` is a standalone, tested dataclass
- [ ] All existing tests pass
