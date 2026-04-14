# Critical Findings Patch Checklist

**Scope:** 7 critical findings across 6 files  
**Priority:** Immediate (blocks release/merge)  
**Risk Level:** Low (targeted fixes)

---

## 1. pyproject.toml — Wheel Source Layout & Entrypoint

**Finding:** Source layout inconsistent with wheel build; entrypoint may not resolve

**Files:** `pyproject.toml`

### Patch Steps:

1. [ ] **Verify src-layout structure:**
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
   
   [project]
   name = "llm-runner"
   version = "0.1.0"
   # ...
   
   [project.scripts]
   llm-runner = "llama_cli.server_runner:main"
   run-models-tui = "llama_cli.tui_app:main"
   
   [tool.hatch.build.targets.wheel]
   packages = ["src/llama_manager", "src/llama_cli"]
   ```

2. [ ] **Verify entrypoints resolve:**
   ```bash
   uv build
   uv pip install dist/*.whl
   llm-runner --help
   ```

**Acceptance:**
- [ ] `uv build` succeeds without errors
- [ ] Installed package has working CLI entrypoints
- [ ] `import llama_manager` works from installed package

**Risk:** Low — only affects packaging, not runtime

---

## 2. process_manager.py — Lock Staleness Timebase Mismatch

**Finding:** `LockMetadata.started_at` uses `time.monotonic()` instead of `time.time()`, causing invalid staleness checks across processes/reboots

**Files:** `src/llama_manager/process_manager.py`

### Patch Steps:

1. [ ] **Update `LockMetadata.started_at` initialization:**
   ```python
   # Before
   started_at: float = field(default_factory=lambda: time.monotonic())
   
   # After
   started_at: float = field(default_factory=lambda: time.time())
   ```

2. [ ] **Update `check_lockfile_integrity()` staleness comparison:**
   ```python
   # Before
   if time.monotonic() - lock_data["started_at"] > LOCK_TIMEOUT:
   
   # After
   if time.time() - lock_data["started_at"] > LOCK_TIMEOUT:
   ```

3. [ ] **Update tests using `time.monotonic()`:**
   ```python
   # In test files, replace time.monotonic() with time.time()
   ```

**Acceptance:**
- [ ] Lock staleness correctly detected after process restart
- [ ] `uv run pytest src/tests/test_us1_lock_integrity.py -xvs` passes
- [ ] No `time.monotonic()` calls in lock-related code

**Risk:** Medium — affects lockfile behavior, requires test updates

---

## 3. data-model.md — Malformed Type Annotations

**Finding:** `openai_defaults` and `openai_flag_bundle` use incorrect closing brackets (`>` instead of `]`)

**Files:** `specs/001-prd-mvp-spec/data-model.md`

### Patch Steps:

1. [ ] **Fix `openai_defaults` (line 22):**
   ```markdown
   - \`openai_defaults: dict[str, str | bool | int]\`
   ```

2. [ ] **Fix `openai_flag_bundle` (line 89):**
   ```markdown
   - \`openai_flag_bundle: dict[str, str | int | bool]\`
   ```

**Acceptance:**
- [ ] Markdown renders correctly
- [ ] Type annotations are syntactically valid Python
- [ ] No lint errors in markdown

**Risk:** Very low — documentation-only fix

---

## 4. research.md — Section Numbering Consistency

**Finding:** Section numbering inconsistent across research document

**Files:** `specs/001-prd-mvp-spec/research.md`

### Patch Steps:

1. [ ] **Audit all section headers:**
   ```bash
   grep -n "^#" specs/001-prd-mvp-spec/research.md
   ```

2. [ ] **Ensure sequential numbering:**
   - 1 → 1.1 → 1.2 → 2 → 2.1 → 2.2 (no gaps, no duplicates)

3. [ ] **Update internal references:**
   - Check all `Section X.Y` references match actual headers

**Acceptance:**
- [ ] All sections have unique, sequential numbers
- [ ] Internal cross-references are accurate
- [ ] Document structure is hierarchical and logical

**Risk:** Very low — documentation-only fix

---

## 5. prd-spec-001-compliance-review.md — Inconsistent FR/AC Totals/Status

**Finding:** FR and AC totals/status inconsistent across document

**Files:** `docs/reviews/prd-spec-001-compliance-review.md`

### Patch Steps:

1. [ ] **Calculate actual totals:**
   ```bash
   # Count FRs
   grep -o "FR-[0-9]*" docs/reviews/prd-spec-001-compliance-review.md | sort -u | wc -l
   
   # Count ACs
   grep -o "AC-[0-9]*" docs/reviews/prd-spec-001-compliance-review.md | sort -u | wc -l
   ```

2. [ ] **Update executive summary:**
   - Ensure FR count matches actual count
   - Ensure AC count matches actual count
   - Ensure status percentages are accurate

3. [ ] **Verify table consistency:**
   - Milestone rows should sum to total
   - Status (Implemented/Deferred) should match actual implementation

**Acceptance:**
- [ ] FR total matches actual unique FR count
- [ ] AC total matches actual unique AC count
- [ ] All milestone sums add up correctly
- [ ] Status percentages match counts

**Risk:** Low — documentation accuracy only

---

## 6. orchestrator.md — Model ID Validity

**Finding:** Model ID references may be invalid or outdated

**Files:** `.opencode/agents/orchestrator.md`

### Patch Steps:

1. [ ] **Verify all model IDs:**
   - Check against known valid models (Qwen 3.5-2B, Qwen 3.5-35B, etc.)
   - Update any deprecated model references

2. [ ] **Add model ID validation note:**
   ```markdown
   **Model ID Validation:**
   - Always use GGUF model IDs from official llama.cpp releases
   - Verify model architecture compatibility before use
   - Reference: https://github.com/ggerganov/llama.cpp/releases
   ```

3. [ ] **Update agent configuration examples:**
   - Ensure all example model paths point to valid files
   - Add placeholder guidance for custom models

**Acceptance:**
- [ ] All model IDs are valid and documented
- [ ] Agent configuration examples work with valid models
- [ ] Model validation guidance is clear

**Risk:** Low — configuration/documentation only

---

## 7. python-backend.md — Dataclass Default Example Issue

**Finding:** Dataclass default example shows mutable default argument anti-pattern

**Files:** `.opencode/agents/python-backend.md`

### Patch Steps:

1. [ ] **Fix mutable default example:**
   ```markdown
   **CORRECT:**
   ```python
   from dataclasses import dataclass, field
   from typing import Any
   
   @dataclass
   class Config:
       """Configuration dataclass."""
       settings: dict[str, Any] = field(default_factory=dict)
       tags: list[str] = field(default_factory=list)
   ```
   
   **WRONG (mutable default):**
   ```python
   @dataclass
   class Config:
       settings: dict[str, Any] = {}  # BAD: shared across instances
       tags: list[str] = []           # BAD: shared across instances
   ```
   ```

2. [ ] **Add explanation:**
   ```markdown
   > **Why?** Mutable defaults are evaluated once at definition time,
   > causing all instances to share the same object. Use `field(default_factory=...)` instead.
   ```

**Acceptance:**
- [ ] Example shows correct `field(default_factory=...)` pattern
- [ ] Anti-pattern is clearly marked as wrong
- [ ] Explanation is accurate and helpful

**Risk:** Very low — documentation-only fix

---

## Ordered Execution Sequence

| Order | Task | File | Risk | Time |
|-------|------|------|------|------|
| 1 | pyproject.toml wheel layout | `pyproject.toml` | Low | 10m |
| 2 | Lock timebase fix | `process_manager.py` | Medium | 15m |
| 3 | data-model.md type annotations | `data-model.md` | Very Low | 5m |
| 4 | research.md section numbering | `research.md` | Very Low | 10m |
| 5 | compliance-review.md totals | `prd-spec-001-compliance-review.md` | Low | 15m |
| 6 | orchestrator.md model IDs | `orchestrator.md` | Low | 10m |
| 7 | python-backend.md dataclass | `python-backend.md` | Very Low | 5m |

**Total estimated time:** ~70 minutes

---

## Validation Commands

```bash
# 1. pyproject.toml
uv build
uv pip install dist/*.whl
llm-runner --help

# 2. Lock timebase
uv run pytest src/tests/test_us1_lock_integrity.py -xvs
uv run pytest src/tests/test_us1_degraded_vs_full_block.py -xvs

# 3-7. Documentation checks
python -c "import markdown; markdown.markdown(open('specs/001-prd-mvp-spec/data-model.md').read())"
grep -c "^#" specs/001-prd-mvp-spec/research.md
grep -c "FR-[0-9]*" docs/reviews/prd-spec-001-compliance-review.md

# Full test suite
uv run pytest src/tests/ -x --tb=short

# Type check
uv run pyright

# Lint
uv run ruff check .
```

---

## Rollback Procedure

```bash
git revert HEAD~7..HEAD --no-edit
git reset --hard HEAD~7
```

---

## Notes

- **Task 2 (process_manager.py)** is the only high-risk change affecting runtime behavior
- **Tasks 3, 4, 6, 7** are documentation-only with minimal risk
- **Task 5** requires careful counting to ensure accuracy
- **Task 1** may affect CI if build configuration changes

**Recommended:** Complete tasks 3-7 first (quick wins), then tackle 1-2 (higher impact).
