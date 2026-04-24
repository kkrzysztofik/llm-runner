---

description: "Task list for M4: Operational Hardening and Smoke Verification"
---

# Tasks: M4 — Operational Hardening and Smoke Verification

**Input**: Design documents from `/specs/001-m4-op-hardening/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), data-model.md, contracts/smoke-api.md

**Tests**: Test tasks are REQUIRED. Every user story MUST include automated tests that validate independent behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Add `httpx` and `gguf` dependencies to `pyproject.toml` under `[project.dependencies]`
- [x] T002 [P] Create directory structure: `src/tests/fixtures/`, `src/scripts/`
- [x] T003 Update `pyproject.toml` test configuration to include fixtures path (runs after T001 — same file `pyproject.toml`, sequential execution required)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Add enums to `src/llama_manager/config.py`: `SlotState`, `SmokePhase`, `SmokeFailurePhase`, `SmokeProbeStatus`, `VRamRecommendation`, `DoctorCheckStatus`, `GgufParseError`
- [x] T005 Add `SmokeProbeConfiguration` dataclass to `src/llama_manager/config.py`
- [x] T006 Add smoke config fields to `Config` dataclass in `src/llama_manager/config.py`
- [x] T006b Add smoke config factory functions to `src/llama_manager/config_builder.py` (depends on T006 for Config fields)
- [x] T007 Create `src/llama_manager/smoke.py` with: `SmokeProbeResult`, `SmokeCompositeReport`, `ProvenanceRecord`, `ConsecutiveFailureCounter` dataclasses; `probe_slot()`, `resolve_provenance()`, `compute_overall_exit_code()` functions
- [x] T008 Create `src/llama_manager/metadata.py` with: `GGUFMetadataRecord` dataclass; `extract_gguf_metadata()`, `normalize_filename()` functions
- [x] T009 [AC-018] Create `src/scripts/generate_gguf_fixtures.py` to generate synthetic GGUF test fixtures in `src/tests/fixtures/`
- [x] T010 [AC-018] Create `src/tests/fixtures/README.md` documenting fixture generator script and usage
- [x] T011 [AC-018] Run `uv run python src/scripts/generate_gguf_fixtures.py` to generate fixture files
- [x] T011b [P] Validate fixture files: verify each fixture in `src/tests/fixtures/` is a valid binary file, non-empty, and under 10 KiB; confirm `gguf_v4_unsupported.gguf` triggers the expected unsupported-version error path in a dry-run metadata parse
- [x] T012 Run `uv run pip-audit` to verify new dependencies have no known CVEs

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

**Dependency notes**: T006b depends on T006 (Config dataclass fields must exist before factory functions reference them). T006b is marked [P] because it's in a different file (`config_builder.py` vs `config.py`) but MUST not start before T006 completes.

---

## Phase 3: User Story 1 — Launch, Monitor, and Shutdown Two Models Safely (Priority: P1) 🎯 MVP

**Goal**: Core operational loop: launch per-GPU-slot processes, show real-time health/logs/GPU telemetry in TUI, graceful shutdown with SIGTERM→SIGKILL escalation

**Independent Test**: Start both models from TUI, confirm per-slot status and logs are visible, then trigger shutdown and verify no `llama-server` processes remain.

### Tests for User Story 1 (REQUIRED) ⚠️

- [x] T013 [P] [US1] Write unit test for `SlotState` enum transitions in `src/tests/test_config.py`
- [x] T014 [US1] [AC-012] Write unit test for SIGTERM→SIGKILL shutdown flow in `src/tests/test_process_manager.py`
- [x] T015 Write unit test for lockfile acquisition and staleness detection in `src/tests/test_process_manager.py`
- [x] T016 [US1] Write unit test for rotating log appending with redaction in `src/tests/test_process_manager.py` (tests the `_append_audit_log` helper and secret pattern matching)
- [x] T016b [US1] Write unit test for `SlotRuntime` dataclass in `src/tests/test_process_manager.py`
- [x] T016c [US1] Write unit test for per-slot status display in `src/tests/test_tui.py`
- [x] T016d [US1] Write unit test for GPU telemetry panel update in `src/tests/test_tui.py` (runs after T016c — same file, sequential execution required)
- [x] T016e [US1] Write unit test for slot state transition handling in `src/tests/test_tui.py` (runs after T016d — same file, sequential execution required)
- [x] T016f [US1] Write unit test for graceful shutdown key handler (Ctrl+C) in `src/tests/test_tui.py` (runs after T016e — same file, sequential execution required)

### Implementation for User Story 1

- [x] T017 [US1] Add `SlotRuntime` dataclass to `src/llama_manager/process_manager.py`
- [x] T018 [US1] Add lockfile methods (`acquire_lock`, `release_lock`, `check_lock_stale`) to `src/llama_manager/process_manager.py`
- [x] T019 [US1] [AC-012] Add SIGTERM→SIGKILL shutdown implementation to `src/llama_manager/process_manager.py`
- [x] T020 [US1] Add rotating audit log implementation to `src/llama_manager/process_manager.py`
- [x] T021 [US1] [AC-011] Add per-slot status display (health, logs, GPU stats, backend label) to `src/llama_cli/tui_app.py`
- [x] T022 [US1] [AC-011] Add GPU telemetry panel update to `src/llama_cli/tui_app.py`
- [x] T023 [US1] [AC-013] Add slot state transition handling for launching→running, running→degraded, etc. in `src/llama_cli/tui_app.py`
- [x] T024 [US1] Add graceful shutdown key handler (Ctrl+C) in `src/llama_cli/tui_app.py`
- [x] T024b [US1] Write integration test for SC-001a: full launch + monitor + shutdown cycle completes in under 120 seconds (mocked servers, timing assertion)
- [x] T024c [US1] [AC-012] Write integration test for SC-001b: shutdown initiates within 1s of user request and completes without orphan processes within 30s; test scans for running `llama-server` processes owned by current user

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 — Verify Serving with OpenAI-Compatible Smoke Tests (Priority: P1)

**Goal**: Sequential verification that each slot responds to OpenAI-compatible endpoints. `smoke both` probes all slots sequentially.

**Independent Test**: Run `smoke both` against running servers and confirm it exits zero after probing each slot sequentially; run `smoke slot <id>` for partial probing.

### Tests for User Story 2 (REQUIRED) ⚠️

- [x] T025 [US2] Write unit test for Phase 1 listen/accept timeout in `src/tests/test_smoke.py`
- [x] T026 [US2] Write unit test for Phase 2 /v1/models response handling (success, empty, mismatch, 404, auth failure) in `src/tests/test_smoke.py`
- [x] T027 [US2] Write unit test for Phase 3 chat completion (success, timeout, auth failure) in `src/tests/test_smoke.py`
- [x] T028 [US2] Write unit test for slot crash detection (exit 19) in `src/tests/test_smoke.py`
- [x] T029 [US2] Write unit test for provenance resolution (SHA, version) in `src/tests/test_smoke.py`
- [x] T030 [US2] [AC-014] Write unit test for `compute_overall_exit_code()` in `src/tests/test_smoke.py`
- [x] T031 [US2] Write unit test for API key precedence (CLI > config > env) in `src/tests/test_smoke.py`
- [x] T032 [US2] [AC-014][FR-006] Write unit test for consecutive failure counter (FR-006) in `src/tests/test_smoke.py`
- [x] T033 [US2] [AC-014][AC-017] Write CLI test for `smoke both` and `smoke slot <id>` argument parsing in `src/tests/test_smoke_cli.py`
- [x] T034 [US2] Write test for human-readable and JSON output formatting in `src/tests/test_smoke.py`

### Implementation for User Story 2

- [x] T035 [US2] [AC-014][AC-017] Add `smoke` subcommand to `src/llama_cli/cli_parser.py` with `both` and `slot <id>` arguments
- [x] T036 [US2] [AC-017] Create `src/llama_cli/smoke_cli.py` with CLI entry point, output formatters
- [x] T037 [US2] [AC-017] Implement Phase 1 (listen/accept) in `src/llama_manager/smoke.py`
- [x] T037b [US2] Wire smoke command entry point in `src/llama_cli/server_runner.py` to call `smoke_cli.run_smoke()` (depends on T037 — Phase 1 must exist before wiring)
- [x] T038 [US2] [AC-017] Implement Phase 2 (/v1/models discovery) in `src/llama_manager/smoke.py`
- [x] T039 [US2] [AC-017] Implement Phase 3 (chat completion) in `src/llama_manager/smoke.py`
- [x] T040 [US2] [AC-017] Implement Phase 3b (slot crash detection) in `src/llama_manager/smoke.py`
- [x] T040b [US2] [AC-017] Implement model ID resolution chain (GGUF name → filename stem → catalog override → /v1/models match) in `src/llama_manager/smoke.py`
- [x] T041 [US2] [AC-017] Implement inter-slot delay and overall exit code computation in `src/llama_manager/smoke.py`
- [x] T042 [AC-017] Add `--api-key`, `--model-id`, `--max-tokens`, `--prompt`, `--delay`, `--timeout` flags to CLI in `src/llama_cli/cli_parser.py`
- [x] T043 [AC-017] Add `--json` output support to `src/llama_cli/smoke_cli.py`
- [x] T044 [US2] [FR-006] Implement consecutive failure counter in `src/llama_manager/smoke.py`
- [x] T044b [US2] [CA-003] Integrate smoke flag bundles into dry-run output in `src/llama_cli/dry_run.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 — Inspect Model Metadata Without Loading Weights (Priority: P2)

**Goal**: Extract GGUF metadata (architecture, context length, attention heads, etc.) without loading full weights. Support timeout and prefix cap.

**Independent Test**: Point the tool at a GGUF file and confirm metadata fields are extracted within a bounded time and memory budget; confirm timeout/corrupt handling produces clear errors.

### Tests for User Story 3 (REQUIRED) ⚠️

- [x] T045 [US3] [AC-010] Write unit test for valid GGUF v3 metadata extraction in `src/tests/test_metadata.py`
- [x] T046 [US3] [AC-010] Write unit test for missing general.name (use normalized filename stem) in `src/tests/test_metadata.py`
- [x] T047 [US3] [AC-010] Write unit test for corrupt file (bad magic bytes) in `src/tests/test_metadata.py`
- [x] T048 [US3] [AC-010] Write unit test for truncated file handling in `src/tests/test_metadata.py`
- [x] T049 [US3] [AC-010] Write unit test for GGUF v4 unsupported version error in `src/tests/test_metadata.py`
- [x] T050 [US3] [AC-010] Write unit test for parse timeout in `src/tests/test_metadata.py`
- [x] T051 [US3] [AC-010] Write unit test for filename NFKC normalization in `src/tests/test_metadata.py`

### Implementation for User Story 3

- [x] T052 [US3] [AC-010] Implement GGUF header parser in `src/llama_manager/metadata.py` using `gguf` library
- [x] T053 [US3] [AC-010] Add prefix cap enforcement (default 32 MiB) to `src/llama_manager/metadata.py`
- [x] T054 [US3] [AC-010] Add parse timeout enforcement (default 5s) to `src/llama_manager/metadata.py`
- [x] T055 [US3] [AC-010] Implement filename normalization (NFKC, whitespace replacement) in `src/llama_manager/metadata.py`
- [x] T056 [US3] [AC-010] Add error variants (CORRUPT_FILE, PARSE_TIMEOUT, UNSUPPORTED_VERSION, READ_ERROR) to metadata module
- [x] T057 [US3] Integrate GGUF metadata extraction with smoke model ID resolution chain in `src/llama_manager/smoke.py`

**Checkpoint**: At this point, User Stories 1, 2, AND 3 should all work independently

---

## Phase 6: User Story 4 — Acknowledge Hardware and VRAM Risks Before Launch (Priority: P2)

**Goal**: Hardware topology warnings requiring explicit acknowledgment. VRAM heuristic with confirmation path.

**Independent Test**: Simulate a non-standard GPU topology and verify warnings appear; simulate low free VRAM and verify launch/smoke blocks without explicit confirmation.

### Tests for User Story 4 (REQUIRED) ⚠️

- [x] T058 [US4] Write unit test for machine fingerprint computation in `src/tests/test_server.py`
- [x] T059 [US4] Write unit test for hardware allowlist check (match, mismatch, invalidated) in `src/tests/test_server.py`
- [x] T060 [US4] [AC-016] Write unit test for VRAM heuristic (proceed, warn, confirm-required) in `src/tests/test_server.py`
- [x] T061 [US4] Write unit test for hardware warning TUI key handler (y/n/q) in `src/tests/test_tui.py` (runs after T016f — same file, sequential execution required)
- [x] T061b [US4] Write unit test for VRAM risk confirmation TUI key handler (y/n) in `src/tests/test_tui.py` (runs after T061 — same file, sequential execution required)

### Implementation for User Story 4

- [x] T062 [US4] Add hardware fingerprint computation (lspci + sycl-ls) to `src/llama_manager/server.py`
- [x] T063 [US4] Add hardware allowlist read/write and session snooze file handling to `src/llama_manager/server.py`
- [x] T064 Add hardware mismatch warning to `src/llama_cli/tui_app.py` with key handler
- [x] T065 Add `--ack-nonstandard-hardware` CLI flag to `src/llama_cli/cli_parser.py`
- [x] T066 [US4] [AC-016] Implement VRAM heuristic formula in `src/llama_manager/server.py`
- [x] T067 [US4] [AC-016] Add VRAM risk assessment to `src/llama_manager/server.py`
- [x] T067b [US4] Add VRAM query method to `src/llama_manager/gpu_stats.py`
- [x] T068 Add VRAM warning to TUI in `src/llama_cli/tui_app.py` and `--confirm-vram-risk` CLI flag in `src/llama_cli/cli_parser.py` (runs after T064 — same file `tui_app.py`, sequential execution required)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T069 Add `DoctorCheckResult` and `DoctorReport` dataclasses and `--json` output logic to `src/llama_manager/server.py`
- [x] T070 Add exit code documentation (doctor 1-9, smoke 10-19, 130) to CLI help text in `src/llama_cli/cli_parser.py`
- [x] T071 Add rotating log file management (max size, retention) to `src/llama_manager/process_manager.py`
- [x] T072 Add report directory generation for smoke failures in `src/llama_manager/smoke.py`
- [x] T073 Add secret redaction patterns to audit log in `src/llama_manager/process_manager.py`
- [x] T074 Run `uv run ruff check .` and fix any lint errors
- [x] T075 Run `uv run ruff format --check .` and fix any formatting issues
- [x] T076 Run `uv run pyright` and fix any type errors
- [x] T077 Run `uv run pytest` and ensure all tests pass
- [x] T078 Run `uv run pytest --cov --cov-report=term-missing` and verify coverage
- [x] T079 [US1] Write state-machine integration test: verify full lifecycle (idle→launching→running→degraded→running→offline→idle) in TUI context with mocked process events
- [x] T080 [US2] Write CA-003 parity test: smoke results (TUI vs CLI) produce identical slot status and phase data for the same server state
- [x] T081 [US2] Write CA-003 parity test: `--json` output from smoke CLI matches the FR-020 schema defined in spec Appendix B and contracts/smoke-api.md Section 4
- [x] T082 [US4] Write CA-003 parity test: dry-run output includes OpenAI flag bundles and compatibility matrix rows in both TUI and CLI modes
- [x] T083 [US2] Write test for dry-run smoke flag bundle output: verify `dry-run` shows smoke-relevant flags (model ID, prompt, /v1/models probe, API key source) per contracts/smoke-api.md Appendix D

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2)

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational (Phase 2) - Depends on smoke.py core library
- **User Story 3 (P2)**: Can start after Foundational (Phase 2) - Depends on metadata.py core library
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Implementation Strategy

### MVP First (User Story 1 + User Story 2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Complete Phase 4: User Story 2
6. **STOP and VALIDATE**: Test User Story 2 independently
7. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Add User Story 4 → Test independently → Deploy/Demo
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3 + 4
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence