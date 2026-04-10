# Tasks: PRD M1 — Slot-First Launch & Dry-Run

**Input**: Design documents from `/specs/001-prd-mvp-spec/`  
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Test tasks are REQUIRED. Every user story includes automated tests for independent behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependency)
- **[Story]**: User story label (`[US1]`, `[US2]`, `[US3]`) for story-phase tasks only
- Every task includes an exact repository file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish shared M1 scaffolding and deterministic test helpers.

- [ ] T001 Add M1 schema and validation dataclass scaffolding with `ModelSlot` (slot_id, model_path, port) and `Config` dataclasses, including `error_code` enum and `ValidationResult` structure in `llama_manager/config.py`
- [ ] T002 Add runtime-directory resolution scaffolding with `resolve_runtime_dir()` fallback chain (`LLM_RUNNER_RUNTIME_DIR` env var then `$XDG_RUNTIME_DIR/llm-runner`) and path validation in `llama_manager/process_manager.py`
- [ ] T003 [P] Add deterministic validation sorting helper scaffolding with `sort_validation_errors()` function that orders by slot configuration sequence (slot_id iteration order); when tie-breaking errors, use `failed_check` ascending within slot in `llama_manager/server.py`
- [ ] T004 [P] Add runtime-dir and lock/artifact pytest fixtures (`tmp_runtime_dir`, `sample_lockfile`, `artifact_writer`) with cleanup teardown in `tests/conftest.py`
- [ ] T005 [P] Add deterministic comparison helpers (`assert_dicts_equal`, `assert_sorted_identically`, `normalize_output_for_diff`) for FR-003/FR-005 contract testing in `tests/helpers_determinism.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build core validation, contracts, and runtime state behavior required by all stories.

**⚠️ CRITICAL**: No user story implementation begins before this phase is complete.

- [ ] T006 Implement slot ID normalization with strict allowed character rule: strip whitespace, lowercase ASCII letters, allow only `a-z0-9_-`, reject any other character with `error_code="invalid_slot_id"` and duplicate detection with `error_code="duplicate_slot"` when multiple `ModelSlot` entries share same `slot_id` in `llama_manager/config.py`
- [ ] T007 Implement FR-006 precedence merge baseline with priority order `defaults < slot/workstation < profile < override`, merging `dict` fields via deep merge for `ServerConfig` construction in `llama_manager/config_builder.py`
- [ ] T008 Implement FR-005 structured error object builder with `ErrorDetail` (error_code, failed_check, why_blocked, how_to_fix, optional docs_ref) and `MultiValidationError` container with `errors: list[ErrorDetail]` and `sort_errors()` ordered by slot configuration sequence (slot_id iteration order), then by `failed_check` ascending within each slot in `llama_manager/server.py`
- [ ] T009 [P] Implement runtime path fallback chain: check `os.getenv("LLM_RUNNER_RUNTIME_DIR")` first, then fall back to `$XDG_RUNTIME_DIR/llm-runner`; when neither candidate usable, raise FR-005 actionable error with `error_code="runtime_dir_unavailable"`, `failed_check="runtime_dir_resolution"`, `why_blocked="neither LLM_RUNNER_RUNTIME_DIR env var nor XDG_RUNTIME_DIR/llm-runner directory exists and directory creation required"`, `how_to_fix="set LLM_RUNNER_RUNTIME_DIR to writable path or create directory structure"` in `llama_manager/process_manager.py`
- [ ] T010 [P] Implement FR-007 redaction helper with regex pattern matching environment variable key names (not values) containing `KEY|TOKEN|SECRET|PASSWORD|AUTH` case-insensitive, replacing matched tokens with `"[REDACTED]"` in `llama_manager/server.py`
- [ ] T011 Implement lockfile read/write metadata primitives with `LockMetadata` dataclass (`pid: int`, `port: int`, `started_at: float`) and integrity checks: process not found → stale (auto-clear), port mismatch/indeterminate owner → FR-005 actionable error with `error_code="LOCKFILE_INTEGRITY_FAILURE"`, `failed_check="lockfile_integrity"`, `why_blocked="indeterminate_owner: lock exists but ownership verification is not definitive"`, `how_to_fix="verify owning process and clear lock only after confirmed stale ownership"` in `llama_manager/process_manager.py`
- [ ] T012 Implement artifact record serializer with `json.dump()` for JSON artifacts, set file mode `0o600` for files and `0o700` for directories using `os.chmod()` after write, raise FR-005 actionable error with `error_code="ARTIFACT_PERSISTENCE_FAILURE"`, `failed_check="artifact_persistence"`, `why_blocked="artifact persistence failed to enforce required owner-only permissions"`, `how_to_fix="verify runtime path writability and filesystem permission support"` in `llama_manager/process_manager.py`
- [ ] T013 Add foundational regression tests for `resolve_runtime_dir()` fallback (LLM_RUNNER_RUNTIME_DIR then $XDG_RUNTIME_DIR/llm-runner), `redact_sensitive()` function, and `MultiValidationError` schema validation in `tests/test_foundation_contracts.py`

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Launch Models by Slot (Priority: P1) 🎯 MVP

**Goal**: Deliver deterministic slot-based launch flow with stale-lock handling, degraded one-slot launch, and full-block behavior.

**Independent Test**: With two configured slots, launch succeeds without collisions; with one unavailable slot, available slot launches with warning; with all slots blocked, launch returns FR-005 multi-error and starts none.

### Tests for User Story 1 (REQUIRED) ⚠️

> **NOTE: Write these tests first, ensure they fail before implementation**

- [ ] T014 [P] [US1] Add dual-slot success and collision launch tests covering: (1) both slots launch without lock collision, (2) second slot fails with `error_code="lock_conflict"` when first lock exists with matching port in `tests/test_us1_launch_flow.py`
- [ ] T015 [P] [US1] Add stale/live/indeterminate lock ownership tests: stale (PID not found → auto-clear), live (PID exists + port matches → block), indeterminate (PID exists + port mismatch → block with FR-005 `error_code="LOCKFILE_INTEGRITY_FAILURE"`) in `tests/test_us1_lock_integrity.py`
- [ ] T016 [P] [US1] Add degraded one-slot vs full-block behavior tests: one available slot returns success with `warnings` list, all slots blocked returns `MultiValidationError` with `error_count=2` and `launch_count=0` in `tests/test_us1_degraded_vs_full_block.py`

### Implementation for User Story 1

- [ ] T017 [US1] Implement lock owner evaluation with three checks: `os.path.exists(pid_file)` for pid existence, `psutil.pid_exists(metadata.pid)` for process existence, port ownership via `psutil.Process(metadata.pid).connections()` or `/proc/net/tcp` inspection for port binding, and `time.monotonic() - metadata.started_at < 300` for start-time match in `llama_manager/process_manager.py`
- [ ] T018 [US1] Implement stale-lock auto-clear when `psutil.pid_exists(pid) == False` and indeterminate-state blocking (PID exists but port mismatch) returns `error_code="LOCKFILE_INTEGRITY_FAILURE"` with `failed_check="lockfile_integrity"` and no auto-clear in `llama_manager/process_manager.py`
- [ ] T019 [US1] Implement launch decision engine that returns `LaunchResult(status="degraded", launched=[slot1], warnings=[...])` for one available slot or `LaunchResult(status="blocked", errors=[...])` with all errors aggregated when all slots unavailable in `llama_manager/process_manager.py`
- [ ] T020 [US1] Wire US1 launch outcomes and FR-005 blocking behavior into `llama_cli/server_runner.py` with `cli_main()` calling `ServerManager.launch_all_slots()`, checking `result.status` for "degraded" vs "blocked", and printing `MultiValidationError` details to stderr in `llama_cli/server_runner.py`
- [ ] T021 [US1] Add consistent degraded/full-block status rendering with acknowledgement-required/acknowledged and degraded/full-block outcomes in `llama_cli/tui_app.py`
- [ ] T022 [US1] Enforce per-slot lockfile lifecycle with `create_lock(slot_id, pid, port)`, `update_lock(slot_id, pid, port)` on heartbeat, `release_lock(slot_id)` on cleanup (delete file or write empty JSON), and `FileExistsError` if lock already exists in `llama_manager/process_manager.py`

**Checkpoint**: User Story 1 is fully functional and independently testable.

---

## Phase 4: User Story 2 - Resolve Launch Blocking Errors Early (Priority: P2)

**Goal**: Provide deterministic dry-run output, actionable FR-005 multi-error responses, and FR-007 observability artifacts.

**Independent Test**: Dry-run returns canonical FR-003 fields and deterministic ordering; known blockers return FR-005 fields with complete remediation; each run emits one redacted artifact or fails with `artifact_persistence`.

### Tests for User Story 2 (REQUIRED) ⚠️

> **NOTE: Write these tests first, ensure they fail before implementation**

- [ ] T023 [P] [US2] Add FR-005 single/multi-error schema and ordering tests verifying: (1) `MultiValidationError` has `errors: list[ErrorDetail]` with `error_count`, (2) ordering by slot configuration sequence (slot_id iteration order); when tie-breaking, use `failed_check` ascending within slot, (3) each `ErrorDetail` has `error_code`, `failed_check`, `why_blocked`, `how_to_fix`, optional `docs_ref` fields, (4) SC-002 denominator counts all `errors[n]` entries across runs (including multi-error responses) in `tests/test_us2_actionable_errors.py`
- [ ] T024 [P] [US2] Add FR-003 canonical dry-run schema and deterministic ordering tests validating: (1) payload includes per-slot required fields (`slot_id`, `binary_path`, `command_args`, `model_path`, `bind_address`, `port`, `environment_redacted`, `openai_flag_bundle`, `hardware_notes`, `vllm_eligibility`, `warnings`, `validation_results`), (2) slots ordered by slot configuration sequence (slot_id iteration order), (3) `validation_results.errors` ordered by slot configuration sequence with `failed_check` ascending tie-break within each slot in `tests/test_us2_dry_run_schema.py`
- [ ] T025 [P] [US2] Add FR-007 artifact persistence, redaction, and permission tests: (1) artifact contains `model_path`, `port`, `command` fields, (2) environment variable values for keys containing `KEY|TOKEN|SECRET|PASSWORD|AUTH` are redacted with `[REDACTED]` while filesystem paths like `model_path` are preserved, (3) file permissions verified via `stat.S_IMODE(st.st_mode) == 0o600` in `tests/test_us2_artifacts.py`

### Implementation for User Story 2

- [ ] T026 [US2] Implement deterministic FR-005 multi-error aggregation with `MultiValidationError(errors: list[ErrorDetail])` where errors are sorted by slot configuration sequence with tie-breaking by `failed_check` ascending within slot before return in `llama_manager/server.py`
- [ ] T027 [US2] Implement canonical M1 vllm non-eligibility mapping: `error_code="BACKEND_NOT_ELIGIBLE"`, `failed_check="vllm_launch_eligibility"`, `why_blocked="vllm is not launch-eligible in PRD M1"`, `how_to_fix="change backend to 'llama_cpp' for M1"` in `llama_manager/server.py`
- [ ] T028 [US2] Implement FR-003 canonical dry-run payload generation using per-slot entries that include all required fields, with `validation_results.errors` and per-slot `warnings` represented within each slot payload in `llama_cli/dry_run.py`
- [ ] T029 [US2] Implement FR-003 ordering guarantees: slots sorted by slot configuration sequence (slot_id iteration order), errors sorted by slot configuration sequence with tie-breaking by `failed_check` ascending within slot, warnings ordered by slot configuration sequence, assertions reference `validation_results.errors` in canonical payload rather than root `errors` in `llama_cli/dry_run.py`
- [ ] T030 [US2] Implement FR-007 artifact writing path: call `resolve_runtime_dir()` for directory, create `artifact-{timestamp}.json` file, write via `json.dump()` then `os.chmod(path, 0o600)`, raise FR-005 actionable error with `error_code="ARTIFACT_PERSISTENCE_FAILURE"`, `failed_check="artifact_persistence"`, `why_blocked="artifact persistence failed to enforce required permissions"`, `how_to_fix="verify runtime path and permission support before retry"` in `llama_manager/process_manager.py`
- [ ] T031 [US2] Ensure CLI and TUI output parity for FR-005/FR-003 semantics: both output `MultiValidationError` details in identical format (`error_code`, `failed_check`, `why_blocked`, `how_to_fix` per error), both render dry-run JSON with `json.dumps(result, indent=2)` in `llama_cli/server_runner.py`

**Checkpoint**: User Stories 1 and 2 both work independently.

---

## Phase 5: User Story 3 - Use Deterministic Overrides Safely (Priority: P3)

**Goal**: Enforce deterministic precedence with explicit acknowledgement gates for risky launch operations.

**Independent Test**: Conflicting values resolve using FR-006 precedence; risky operations block without acknowledgement and proceed only for current launch attempt when acknowledged.

### Tests for User Story 3 (REQUIRED) ⚠️

> **NOTE: Write these tests first, ensure they fail before implementation**

- [ ] T032 [P] [US3] Add precedence resolution tests verifying: (1) `overrides` wins over `profile`, (2) `profile` wins over `slot` and `workstation`, (3) `slot` and `workstation` win over `defaults`, (4) deep merge of `dict` fields like `model_params` in `tests/test_us3_precedence.py`
- [ ] T033 [P] [US3] Add risky-operation acknowledgement scope tests: (1) port < 1024 requires `--acknowledge-risky` flag, (2) `--bind=0.0.0.0` requires acknowledgement, (3) acknowledgement only valid for current `launch_attempt_id` not persisted across restarts in `tests/test_us3_risk_acknowledgement.py`
- [ ] T034 [P] [US3] Add SC-003 deterministic repeated-resolution evidence tests: (1) call `resolve_config()` twice with identical inputs, (2) verify `ServerConfig` fields match exactly via `==`, (3) no randomness in `port` assignment or `command` generation, (4) `json.dumps(config)` produces identical output, (5) repeated runs produce matching FR-007 artifact fields `resolved_command`, `validation_results`, and `warnings` in `tests/test_us3_determinism.py`

### Implementation for User Story 3

- [ ] T035 [US3] Implement profile guidance specificity with `profile_specifier` (e.g., "gpu" vs "gpu-high-memory"), merge semantics: `dict` fields deep-merge, scalar fields use higher-precedence source, `list` fields concatenate in `llama_manager/config_builder.py`
- [ ] T036 [US3] Implement profile validation timing: validate after all precedence levels merged, check `port` in range 1024-65535, `threads` > 0, `model_path` exists, override interaction: if `--override` sets `port`, profile default for `port` is ignored in `llama_manager/config_builder.py`
- [ ] T037 [US3] Implement risky-operation detection: flag `port < 1024` as `risky_operation="privileged_port"`, `bind_addr != "127.0.0.1"` as `risky_operation="non_loopback"`, and explicit manual override that bypasses warning as `risky_operation="warning_bypass"`; set `requires_acknowledgement: bool` on `ServerConfig` in `llama_manager/server.py`
- [ ] T038 [US3] Implement current-attempt-only acknowledgement state handling: store `acknowledgement_cache: dict[str, set[str]]` keyed by `launch_attempt_id` in process memory (not file), validate `ack_token` matches current `attempt_id` on each launch, clear cache on process exit in `llama_manager/process_manager.py`
- [ ] T039 [US3] Wire acknowledgement enforcement and operator prompts: argparse adds `--acknowledge-risky` flag, check `ServerConfig.requires_acknowledgement` before launch, call `input("Confirm risky operation [y/N]: ")` and validate response is "y" or "Y", raise `SystemExit(1)` with `error_code="acknowledgement_required"` if not acknowledged in `llama_cli/server_runner.py`
- [ ] T040 [US3] Render acknowledgement-required and acknowledged states: display acknowledgement panel if `requires_acknowledgement`, display acknowledgement confirmation panel after confirmation in `llama_cli/tui_app.py`

**Checkpoint**: All user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final quality, performance evidence, and cross-story hardening.

- [ ] T041 [P] Add SC-006 benchmark-style timing harness tests: measure `time.perf_counter()` around dry-run resolution and lock/port validation paths, assert dry-run single-slot p95 <=250ms, dry-run two-slot p95 <=400ms, and lock/port validation p95 <=150ms per slot across 100 iterations in `tests/test_sc006_performance.py`
- [ ] T042 [P] Add cross-story regression tests for FR-005/FR-003 parity: verify `MultiValidationError` fields match canonical dry-run `slot.validation_results.errors`, verify slot configuration sequence consistency between error output and dry-run payload, and verify `failed_check` ascending tie-break ordering within each slot in `tests/test_regression_m1_contracts.py`
- [ ] T043 [P] Add missing type hints and docstrings: add `-> Config`, `-> ServerConfig`, `-> list[str]` return types, document `create_*_cfg()` factory functions, `resolve_runtime_dir()`, `LockMetadata`, `create_lock()`, `release_lock()` with Google-style docstrings including Args, Returns, Raises sections in `llama_manager/config_builder.py` and `llama_manager/process_manager.py`
- [ ] T044 [P] Update operator verification steps and outcomes: document `llm-runner dry-run both` expected per-slot FR-003 canonical fields (including per-slot `warnings` and `validation_results`), document degraded launch warning message format, document full-block FR-005 multi-error format, and verify `artifact-{timestamp}.json` appears in runtime directory in `specs/001-prd-mvp-spec/quickstart.md`
- [ ] T045 Run lint checks using `pyproject.toml` against `llama_manager/`, `llama_cli/`, and `tests/`: execute `uv run ruff check llama_manager/ llama_cli/ tests/ --select D,E,F,W,I`, fix autofixable issues with `--fix`, ensure zero violations remain in `llama_manager/` and `llama_cli/` in `tests/`
- [ ] T046 Run formatting checks using `pyproject.toml` against `llama_manager/`, `llama_cli/`, and `tests/`: execute `uv run ruff format --check llama_manager/ llama_cli/ tests/`, ensure `would reformat: 0 files`, run `uv run ruff format llama_manager/ llama_cli/ tests/` to auto-fix any formatting issues
- [ ] T047 Run type checks using `pyproject.toml` for project modules: execute `uv run pyright llama_manager/ llama_cli/`, ensure zero type errors reported (`Found 0 errors, 0 warnings`), verify all functions have type hints, all dataclasses have field types
- [ ] T048 Run full pytest suite for `tests/` and resolve remaining failures: execute `uv run pytest tests/ --cov --cov-report=term-missing -x`, ensure 100% test pass rate, verify coverage thresholds (e.g., `llama_manager/` >= 90% coverage), document any skipped tests with reasons

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies; start immediately.
- **Phase 2 (Foundational)**: Depends on Phase 1; blocks all user stories.
- **Phase 3 (US1)**: Depends on Phase 2; MVP path starts here.
- **Phase 4 (US2)**: Depends on Phase 2; can run after or alongside US1 where no file conflicts exist.
- **Phase 5 (US3)**: Depends on Phase 2; may integrate with US1/US2 modules but remains independently testable.
- **Phase 6 (Polish)**: Depends on completion of all selected story phases.

### User Story Dependencies

- **US1 (P1)**: Independent after foundational completion; defines MVP.
- **US2 (P2)**: Independent after foundational completion; integrates with launch/dry-run flow.
- **US3 (P3)**: Independent after foundational completion; depends on shared precedence/validation primitives.

### Within Each User Story

- Write and run story tests first (expected failing).
- Implement core library behavior in `llama_manager/`.
- Integrate CLI/TUI behavior in `llama_cli/`.
- Re-run story tests and verify independent pass criteria.

### Parallel Opportunities

- Phase 1: T003/T004/T005 can run in parallel.
- Phase 2: T009/T010 can run in parallel; T013 can start after T008-T012.
- US1: T014/T015/T016 parallel; T020/T021 parallel after core launch logic.
- US2: T023/T024/T025 parallel; T028/T029 parallel after T026/T027.
- US3: T032/T033/T034 parallel; T039/T040 parallel after T037/T038.
- Polish: T041/T042/T043/T044 can run in parallel before final gate tasks T045-T048.

---

## Parallel Example: User Story 2

```bash
# Launch all US2 tests together:
Task: "T023 [P] [US2] Add FR-005 single/multi-error schema and ordering tests in tests/test_us2_actionable_errors.py"
Task: "T024 [P] [US2] Add FR-003 canonical dry-run schema and deterministic ordering tests in tests/test_us2_dry_run_schema.py"
Task: "T025 [P] [US2] Add FR-007 artifact persistence, redaction, and permission tests in tests/test_us2_artifacts.py"

# Launch parallel US2 implementation slices after contract core is in place:
Task: "T028 [US2] Implement FR-003 canonical dry-run payload generation in llama_cli/dry_run.py"
Task: "T030 [US2] Implement FR-007 artifact writing path with runtime fallback and permission enforcement in llama_manager/process_manager.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Complete US1 test tasks (T014-T016), then US1 implementation tasks (T017-T022).
3. Validate US1 independently against its acceptance criteria.
4. Demo MVP launch behavior (dual-slot success, degraded one-slot, full-block correctness).

### Incremental Delivery

1. Land US1 (MVP) first.
2. Add US2 to deliver deterministic dry-run + actionable errors + artifacts.
3. Add US3 to deliver deterministic overrides and risk acknowledgement gates.
4. Complete Phase 6 quality gates and regression hardening.

### Parallel Team Strategy

1. One contributor completes foundational core (Phase 2).
2. Then split by story: US1 launch flow, US2 contracts/dry-run/artifacts, US3 precedence/risk.
3. Rejoin for Phase 6 cross-cutting validation and quality gates.

---

## Notes

- `[P]` tasks are parallelizable by file and dependency boundaries.
- `[US#]` labels provide traceability to spec user stories.
- Story phases are independently testable by design.
- Keep `llama_manager/` pure library code and `llama_cli/` user I/O integration.
