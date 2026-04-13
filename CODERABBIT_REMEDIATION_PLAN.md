# CodeRabbit Findings Remediation Plan

**Total Findings:** 191  
**Source:** `/home/kmk/.local/share/opencode/tool-output/tool_d88c1f2b4001GLxoQrhcdBVahJ`  
**Generated:** 2026-04-14  
**Target Branch:** `001-prd-mvp-spec`

---

## Executive Summary

This plan provides a wave-based remediation strategy for all 191 CodeRabbit findings, organized by severity and impact to the codebase. The plan prioritizes critical runtime issues first, followed by major behavioral changes, then minor/trivial improvements, and finally documentation/specification updates.

---

## Wave 1: Critical Findings (7 findings)
**Goal:** Fix runtime-breaking issues that cause crashes, data corruption, or security vulnerabilities.

### Files to Modify

| File | Findings | Description |
|------|----------|-------------|
| `src/tests/test_fr003_007_011_contracts.py` | 101, 102 | Fix isinstance() calls with None in union types |
| `src/llama_cli/dry_run.py` | 125 | Handle None validation_results in build_dry_run_slot_payload |
| `src/llama_manager/gpu_stats.py` | 163 | Check subprocess returncode before json.loads |
| `src/llama_manager/process_manager.py` | 182, 183, 184, 185 | Fix lockfile TOCTOU, time.monotonic() usage, pid=0 placeholder |

### Conflict/Dependency Notes

- **Finding 183 (time.monotonic → time.time):** This affects `LockMetadata.started_at` which is used across multiple files. Must coordinate with:
  - `src/llama_manager/process_manager.py` (LockMetadata definition)
  - `src/tests/test_us1_lock_integrity.py` (test that uses time.monotonic)
  - `src/tests/test_us1_degraded_vs_full_block.py` (test that writes locks)

- **Finding 184/185 (TOCTOU fixes):** These are related atomic write fixes. Should be done together as they share the `_atomic_write_json` function.

### Acceptance Criteria

1. **Test fixes (101, 102):** All isinstance checks must use `isinstance(value, (str, type(None)))` or `value is None or isinstance(value, str)`
2. **Dry-run (125):** `build_dry_run_slot_payload` must accept `validation_results=None` and construct default object
3. **GPU stats (163):** subprocess must check `returncode != 0` and handle error gracefully
4. **Lockfile fixes (182-185):** 
   - `started_at` uses `time.time()` (wall clock)
   - pid=0 replaced with actual process PID
   - `_atomic_write_json` uses atomic temp+rename pattern
   - Lock creation uses O_CREAT|O_EXCL for TOCTOU prevention

### Validation Commands

```bash
# Run affected tests
uv run pytest src/tests/test_fr003_007_011_contracts.py::TestDryRunPayloadContract::test_openai_flag_bundle_field_types -xvs
uv run pytest src/tests/test_fr003_007_011_contracts.py::TestDryRunPayloadContract::test_hardware_notes_field_types -xvs
uv run pytest src/tests/test_us1_lock_integrity.py -xvs
uv run pytest src/llama_manager/gpu_stats.py -xvs

# Type check
uv run pyright src/llama_cli/dry_run.py src/llama_manager/process_manager.py src/llama_manager/gpu_stats.py
```

---

## Wave 2: Major Runtime/Code Behavior (40+ findings)
**Goal:** Fix behavioral issues that affect correctness, consistency, or user experience.

### Files to Modify

| File | Findings | Description |
|------|----------|-------------|
| `src/llama_manager/colors.py` | 23, 28 | Move to llama_cli, fix is_enabled() to avoid stdout inspection |
| `src/llama_manager/config.py` | 81, 116, 158, 159 | ModelSlot forward reference, ErrorCode casing, derived paths, mutable aliasing |
| `src/llama_manager/gpu_stats.py` | 134, 162 | Normalize schema, fix bare except |
| `src/llama_manager/process_manager.py` | 162, 180, 181, 186, 187, 188 | Exception handling, artifact collision, schema, redaction, ownership |
| `src/llama_manager/server.py` | 167, 171, 172, 173 | Type ignore, model_path validation, regex compilation, Config instantiation |
| `src/llama_cli/cli_parser.py` | 94, 96, 98, 104, 105 | Port validation, help examples, ports list validation, mode default, dry-run example |
| `src/llama_cli/server_runner.py` | 27, 91, 118, 119, 132, 153 | Return type, port defaulting, is_enabled usage, error printing, EOFError, broad except |
| `src/llama_cli/tui_app.py` | 147, 157, 160, 170 | Type hints, Live loop signal handling, Config caching, empty right column |
| `src/llama_cli/dry_run.py` | 144, 146, 164 | NoReturn type, error message, late import |
| `run_opencode_models.sh` | 38, 59, 122 | Comment accuracy, sampling flags, redaction pattern |
| `src/tests/*.py` | 9, 45, 46, 50, 55, 61, 82, 83, 84, 85, 88, 90, 92, 93, 97, 107, 108, 109, 111, 112, 124, 126, 127, 128, 129, 130, 131, 133, 135, 136, 137, 138, 139, 140, 141, 142, 145, 148, 149, 150, 151, 152, 154, 155, 156, 161, 165, 166 | Test improvements: type hints, duplicated helpers, psutil mocks, time.sleep, fixture usage |

### Conflict/Dependency Notes

- **Finding 23 (colors.py migration):** This is a cross-module refactor. Must update:
  - `src/llama_cli/colors.py` (new file)
  - All imports across `llama_cli/` to use `llama_cli.colors.Color`
  - Remove `src/llama_manager/colors.py`
  - Update `__init__.py` exports

- **Finding 158 (derived paths in Config):** Requires `__post_init__` implementation or `field(default_factory=...)` change. Must ensure no code assumes class-level defaults.

- **Test file fixes (multiple):** Many test fixes require coordinating `psutil` mocks. Group by test file to avoid conflicts.

### Acceptance Criteria

1. **colors.py migration:** `llama_manager` is pure library with no TUI/presentation code
2. **config.py fixes:** 
   - ModelSlot defined before functions that use it
   - ErrorCode values all UPPER_SNAKE_CASE
   - Derived paths computed in `__post_init__`
   - Mutable objects deep-copied in config builders
3. **gpu_stats.py:** Schema normalized, exceptions logged and re-raised
4. **process_manager.py:** Exception handling improved, artifact collision handled, ownership verified
5. **cli_parser.py:** All port validation, mode defaults, examples updated
6. **server_runner.py:** Return types added, EOFError handled, duplicated code extracted
7. **tui_app.py:** Live loop can exit via signal handlers, Config cached
8. **Test files:** Type hints on all fixtures, psutil mocks where needed, no time.sleep

### Validation Commands

```bash
# Run all tests
uv run pytest src/tests/ -x --tb=short

# Type check all modified files
uv run pyright src/llama_manager/ src/llama_cli/

# Lint
uv run ruff check src/llama_manager/ src/llama_cli/
uv run ruff format --check src/llama_manager/ src/llama_cli/
```

---

## Wave 3: Minor/Trivial Code+Tests (60+ findings)
**Goal:** Improve code quality, consistency, and testability.

### Files to Modify

| File | Findings | Description |
|------|----------|-------------|
| `src/tests/*.py` | 1, 2, 15, 16, 20, 22, 24, 26, 39, 41, 42, 44, 46, 48, 49, 55, 56, 59, 62, 63, 64, 69, 70, 71, 72, 73, 78, 79, 83, 84, 85, 87, 88, 90, 92, 93, 97, 107, 108, 109, 111, 112, 115, 124, 126, 127, 128, 129, 130, 131, 133, 135, 136, 137, 139, 140, 141, 142, 145, 148, 149, 150, 151, 152, 154, 155, 156, 161, 165, 166 | Test improvements: imports, type hints, fixtures, duplicates, mocks |
| `src/llama_manager/__init__.py` | 106, 113 | Import ordering, Color import position |
| `src/llama_manager/config_builder.py` | 159 | Mutable object aliasing |
| `src/llama_manager/server.py` | 167, 172, 173 | Type ignore, regex compilation, Config instantiation |
| `src/llama_manager/log_buffer.py` | 14, 30 | Return type hints, regex precompilation |
| `src/llama_manager/config.py` | 106, 110 | Import ordering, dict vs set |
| `src/llama_cli/__init__.py` | 57 | Module docstring |
| `src/llama_cli/server_runner.py` | 91, 118, 119 | Port defaulting, is_enabled usage, error printing |
| `src/llama_cli/cli_parser.py` | 94, 96, 98 | Port validation, help examples |
| `src/llama_cli/dry_run.py` | 144, 146, 156 | NoReturn, error message, redundant ternary |
| `src/llama_cli/tui_app.py` | 147, 157, 160, 170 | Type hints, signal handling, Config caching |

### Conflict/Dependency Notes

- **Import organization (multiple files):** Many findings relate to moving inline imports to module level. Should be done together using a consistent pattern.

- **Test helper extraction:** Multiple findings (44, 45, 64, 69, 70, 108, 127, 154) request extracting duplicated helpers. Coordinate to avoid conflicts.

### Acceptance Criteria

1. **All imports at module level:** No inline imports in test functions
2. **Type hints on all functions:** Including `__init__`, fixtures, helpers
3. **Duplicated code extracted:** Single source of truth for helpers
4. **Regex precompiled:** Module-level constants where appropriate
5. **Test fixtures:** Reusable fixtures for common patterns

### Validation Commands

```bash
# Check import ordering
uv run ruff check src/

# Check type hints
uv run pyright src/

# Run all tests
uv run pytest src/tests/ -x
```

---

## Wave 4: CI/Workflow/Config (6 findings)
**Goal:** Improve CI reliability, reproducibility, and security.

### Files to Modify

| File | Findings | Description |
|------|----------|-------------|
| `.github/workflows/ci.yml` | 10 | Coverage report step duplication with test job |
| `.github/workflows/copilot-setup-steps.yml` | 8 | uv version pinning |
| `.github/dependabot.yml` | 32, 52 | Commit message config, pytest-cov duplicate |
| `.markdownlint.json` | 11 | Redundant MD036 rule |
| `.gitignore` | 12 | Duplicated .vscode/.idea entries |
| `sonar-project.properties` | 66 | sonar.sources overlap with tests |

### Conflict/Dependency Notes

- **CI workflow changes:** Must ensure changes don't break existing CI. Test with dry-run first.

### Acceptance Criteria

1. **ci.yml:** Coverage step consumes test artifacts or re-run is documented
2. **copilot-setup-steps.yml:** uv version pinned to semantic version
3. **dependabot.yml:** Commit message templates added, pytest-cov duplicate removed
4. **markdownlint.json:** Redundant rules removed
5. **gitignore:** Duplicated entries consolidated
6. **sonar-project.properties:** Sources and tests don't overlap

### Validation Commands

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
python -c "import yaml; yaml.safe_load(open('.github/workflows/copilot-setup-steps.yml'))"
python -c "import yaml; yaml.safe_load(open('.github/dependabot.yml'))"

# Check markdownlint config
cat .markdownlint.json | python -c "import json,sys; json.load(sys.stdin)"

# Validate sonar config
cat sonar-project.properties
```

---

## Wave 5: Specs/Docs/Contracts (30+ findings)
**Goal:** Improve specification clarity, consistency, and completeness.

### Files to Modify

| File | Findings | Description |
|------|----------|-------------|
| `specs/001-prd-mvp-spec/contracts/observability-artifact-contract.md` | 18, 19, 21, 25, 40, 53 | Redaction rules, filename format, JSON schemas, usability checks, timestamp format |
| `specs/001-prd-mvp-spec/contracts/dry-run-canonical-contract.md` | 67, 68, 75, 76, 77 | vllm_eligibility schema, environment_redacted schema, openai_flag_bundle optional, validation_results schema |
| `specs/001-prd-mvp-spec/contracts/actionable-error-contract.md` | 89, 114, 120, 121, 122, 123 | Delivery Context, Format Conventions, error examples, why_blocked dual-purpose |
| `specs/001-prd-mvp-spec/data-model.md` | 36, 37, 47, 58 | Override interaction, slots/slot_id relationship, bracket errors |
| `specs/001-prd-mvp-spec/quickstart.md` | 100, 103, 143, 145, 122 | Permissions clarification, FR-005 inconsistency, FR code references, timestamp format |
| `specs/001-prd-mvp-spec/spec.md` | 193, 196 | Artifact naming collision policy |
| `specs/001-prd-mvp-spec/checklists/*.md` | 17, 31, 34, 35, 65 | Plan quality, implementation readiness, absolute paths |
| `specs/001-prd-mvp-spec/tasks.md` | 168 | Task description length |
| `docs/PRD.md` | 174, 175, 176, 177, 178 | Redundant FR-005, stream start ambiguity, AC-003 vLLM, vLLM contradiction, exit codes |
| `docs/plans/prd-spec-001-extension-patch.md` | 189, 190 | Typo, duplicated FR IDs |
| `docs/reviews/prd-spec-001-compliance-review.md` | 191, 195 | FR-014 milestone, heading duplication |
| `docs/spec-001-prd-coverage.md` | 192, 194 | Coverage numbers, backend abstraction contradiction |
| `.opencode/agents/*.md` | 29, 43, 46, 80, 99, 86 | Agent documentation updates |

### Conflict/Dependency Notes

- **Contract consistency:** Multiple contract files reference each other. Changes to one may require updates to others.

- **FR code references:** Many docs reference FR-XXX codes. Ensure consistency across all documents.

### Acceptance Criteria

1. **All contracts have explicit schemas:** JSON Schema or concrete examples for all objects
2. **FR code consistency:** All FR references are consistent across documents
3. **No absolute paths:** All paths are relative or repository-relative
4. **Clear semantics:** Ambiguous terms (e.g., "stream start", "optional") are explicitly defined
5. **Agent documentation:** All agent docs have complete guardrails and workflows

### Validation Commands

```bash
# Check markdown syntax
python -c "import markdown; markdown.markdown(open('specs/001-prd-mvp-spec/contracts/observability-artifact-contract.md').read())"

# Verify FR code consistency
grep -r "FR-[0-9][0-9]" specs/ docs/ | sort -u

# Check for absolute paths
grep -r "/home/kmk/llm-runner" specs/ docs/
```

---

## Wave 6: Security + Final Validation (10+ findings)
**Goal:** Address security concerns and perform final validation.

### Files to Modify

| File | Findings | Description |
|------|----------|-------------|
| `src/llama_manager/process_manager.py` | 187, 188 | Redaction of nested lists, lock ownership verification |
| `src/llama_manager/gpu_stats.py` | 162 | Exception handling (security: silent failures) |
| `src/llama_cli/dry_run.py` | 125 | validation_results=None handling |
| `src/llama_cli/server_runner.py` | 132 | EOFError handling (security: input validation) |
| Multiple test files | 133, 138, 161, 165 | psutil mocks for security-critical tests |

### Conflict/Dependency Notes

- **Security-critical tests:** Tests that verify security behavior (lock ownership, redaction) must be updated together to ensure they test the right conditions.

### Acceptance Criteria

1. **Redaction handles nested structures:** Lists, dicts recursively redacted
2. **Lock ownership verified:** Before write operations
3. **No silent failures:** Exceptions logged and handled appropriately
4. **Input validation:** All user inputs validated (EOFError, etc.)
5. **Security tests pass:** All security-related tests updated and passing

### Validation Commands

```bash
# Run security-related tests
uv run pytest src/tests/ -k "lock" -xvs
uv run pytest src/tests/ -k "redact" -xvs
uv run pytest src/tests/ -k "ownership" -xvs

# Run full test suite
uv run pytest src/tests/ --cov --cov-report=term-missing

# Run pre-commit hooks
uv run pre-commit run --all-files
```

---

## False Positive Analysis

### Potentially False Positives

1. **Finding 39 (test_validate_slot_id_rejects_invalid_chars):** The test name suggests rejection but the implementation normalizes. This is likely intentional - the test should be renamed to reflect actual behavior. **Not a false positive, just misleading naming.**

2. **Finding 167 (server.py type ignore):** The comment suggests this may be a mypy false positive if types already align. **Verify types first, may be safe to remove.**

3. **Finding 118 (server_runner.py Color.is_enabled() return value):** The finding suggests assigning to throwaway variable, but if the method is meant to have side effects only, this is documentation, not a bug. **Verify intent, may be intentional.**

4. **Finding 179 (process_manager.py _format_output color_code unused):** The color_code is fetched but not used. This might be intentional if colors are disabled by default. **Verify if this is intentional or oversight.**

---

## Implementation Order & Dependencies

```
Wave 1 (Critical) → Wave 2 (Major) → Wave 3 (Minor/Trivial) → Wave 4 (CI) → Wave 5 (Docs) → Wave 6 (Security)
     ↓                   ↓                    ↓                   ↓           ↓             ↓
  Runtime fixes     Behavioral fixes      Code quality       CI/CD       Specs         Security
  (crash prevention) (correctness)        (maintainability)  (reliability) (clarity)     (validation)
```

### Critical Path

1. **Wave 1, Finding 183:** `time.monotonic()` → `time.time()` in `LockMetadata`
   - Blocks: Wave 2 test fixes using locks, Wave 6 security tests

2. **Wave 1, Finding 184/185:** TOCTOU fixes in lock creation
   - Blocks: Wave 6 security validation

3. **Wave 2, Finding 23:** `colors.py` migration to `llama_cli`
   - Blocks: All imports of `llama_manager.colors.Color`

4. **Wave 2, Finding 158:** Config derived paths in `__post_init__`
   - Blocks: Any code relying on derived path values

---

## CI Quality Gates

All waves must pass these gates before merging:

| Gate | Command | When |
|------|---------|------|
| **Lint** | `uv run ruff check .` | After each wave |
| **Format** | `uv run ruff format --check .` | After each wave |
| **Type Check** | `uv run pyright` | After Waves 1-3 |
| **Tests** | `uv run pytest` | After each wave |
| **Coverage** | `uv run pytest --cov` | After Wave 3 |
| **Pre-commit** | `uv run pre-commit run --all-files` | Before final merge |

---

## Rollback Plan

If any wave fails:

1. **Wave 1 failure:** Revert to base commit, investigate critical path issue
2. **Wave 2 failure:** Isolate to specific file changes, revert affected files only
3. **Wave 3+ failure:** Can revert individual files without impacting runtime

---

## Estimated Timeline

| Wave | Files | Time Estimate |
|------|-------|---------------|
| 1 | 4 | 2-3 hours |
| 2 | 25+ | 6-8 hours |
| 3 | 20+ | 4-6 hours |
| 4 | 6 | 1 hour |
| 5 | 30+ | 4-6 hours |
| 6 | 5 | 2 hours |
| **Total** | **~100 files** | **~20-25 hours** |

---

## Specialist Agent Assignments

| Wave | Specialist | Focus Area |
|------|------------|------------|
| 1 | Runtime Specialist | Core library fixes |
| 2 | Core Library Specialist | llama_manager, llama_cli refactors |
| 3 | Test Specialist | Test improvements, fixtures |
| 4 | DevOps Specialist | CI/CD, workflows |
| 5 | Documentation Specialist | Specs, contracts, PRD |
| 6 | Security Specialist | Validation, security tests |

---

## Next Steps

1. **Review this plan** with the team
2. **Assign specialists** to each wave
3. **Begin Wave 1** immediately (critical issues)
4. **Update AGENTS.md** with progress tracking
5. **Create `br` issues** for each wave
6. **Sync to git** after each wave completion
