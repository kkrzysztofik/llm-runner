# Tasks: PRD M2 — Build Wizard + Setup Pipeline

**Input**: Design documents from `/specs/002-build-setup/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Test tasks are REQUIRED. Every user story MUST include automated tests that validate independent behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and M2-specific data models

**Note**: This project already has existing infrastructure from M1. Setup tasks focus on M2-specific additions.

- [X] T001 [P] Add ErrorCode extensions for M2 in `src/llama_manager/config.py` (TOOLCHAIN_MISSING, BUILD_LOCK_HELD, VENV_CORRUPT, PYTHON_NOT_FOUND, BUILD_FAILED, PREFLIGHT_FAILURE, GIT_CLONE_FAILED, GIT_CHECKOUT_FAILED, REPORT_WRITE_FAILURE, TOOL_VERSION_MISMATCH, CMAKE_INCOMPATIBLE)
- [X] T002 [P] Add Config extensions for M2 in `src/llama_manager/config.py` (xdg_cache_base, xdg_state_base, xdg_data_base, build_git_remote, build_git_branch, build_retry_attempts, build_retry_delay, build_max_reports, build_output_truncate_bytes, toolchain_timeout_seconds)
- [X] T003 [P] Create BuildConfig dataclass in `src/llama_manager/build_pipeline.py` with fields: backend, source_dir, build_dir, output_dir, git_remote_url, git_branch, retry_attempts, retry_delay, shallow_clone, jobs, plus class constants GGML_SYCL, GGML_CUDA, CMAKE_C_COMPILER, CMAKE_CXX_COMPILER
- [X] T004 [P] Create BuildArtifact dataclass in `src/llama_manager/build_pipeline.py` with fields: artifact_type, backend, created_at, git_remote_url, git_commit_sha, git_branch, build_command, build_duration_seconds, exit_code, binary_path, binary_size_bytes, build_log_path, failure_report_path
- [X] T005 [P] Create BuildProgress dataclass in `src/llama_manager/build_pipeline.py` with fields: stage, status, message, progress_percent, retries_remaining
- [X] T006 [P] Create ToolchainStatus dataclass in `src/llama_manager/toolchain.py` with fields: gcc, make, git, cmake, sycl_compiler, cuda_toolkit, nvtop
- [X] T007 [P] Create BuildLock dataclass in `src/llama_manager/build_pipeline.py` with fields: pid, started_at, backend
- [X] T008 [P] Create FailureReport dataclass in `src/llama_manager/reports.py` with fields: report_dir, timestamp, build_artifact_json, build_output_log, error_details_json
- [X] T009 [P] Create VenvResult dataclass in `src/llama_manager/setup_venv.py` with fields: venv_path, created, reused, activation_command
- [X] T010 [P] Create ToolchainHint dataclass in `src/llama_manager/toolchain.py` with fields: tool_name, install_command, install_url, required_for
- [X] T011 Add module constants in `src/llama_manager/toolchain.py`: SYCL_REQUIRED_TOOLS, CUDA_REQUIRED_TOOLS, CMAKE_MINIMUM_VERSION
- [X] T012 [P] Add MutatingActionLogEntry dataclass in `src/llama_manager/reports.py` with fields: command, timestamp, exit_code, truncated_output, redaction_applied

**Phase 1 Complete**: All 12 tasks implemented, tested (118 tests), and validated. Ready for Phase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T013 Implement ToolchainErrorDetail dataclass extending M1 ErrorDetail pattern in `src/llama_manager/toolchain.py` (ensure FR-005 compliance with error_code, failed_check, why_blocked, how_to_fix, docs_ref)
- [ ] T014 Implement redact_sensitive() utility function in `src/llama_manager/reports.py` (pattern: KEY|TOKEN|SECRET|PASSWORD|AUTH → [REDACTED])
- [ ] T015 Implement XDG path utilities in `src/llama_manager/config.py` for computing venv_path, builds_dir, reports_dir, build_lock_path from Config
- [ ] T016 Implement CMake version parser in `src/llama_manager/toolchain.py` (tuple comparison with suffix normalization)
- [ ] T017 [P] Implement detect_tool() in `src/llama_manager/toolchain.py` with subprocess.run and 30s timeout (configurable via Config.toolchain_timeout_seconds)
- [ ] T018 [P] Implement get_toolchain_hints() in `src/llama_manager/toolchain.py` returning list[ErrorDetail] for missing required tools with platform-specific install commands
- [ ] T019 [P] Implement get_venv_path() in `src/llama_manager/setup_venv.py` returning $XDG_CACHE_HOME/llm-runner/venv with fallback to ~/.cache/llm-runner/venv
- [ ] T020 [P] Implement create_venv() in `src/llama_manager/setup_venv.py` that creates venv at given path using venv module
- [ ] T021 [P] Implement check_venv_integrity() in `src/llama_manager/setup_venv.py` validating pyvenv.cfg existence and interpreter symlink
- [ ] T022 [P] Implement write_failure_report() in `src/llama_manager/reports.py` creating report directory with build-artifact.json, build-output.log (truncated and redacted), error-details.json, enforcing 0700/0600 permissions
- [ ] T023 Implement rotate_reports() in `src/llama_manager/reports.py` deleting oldest report directories when count exceeds Config.build_max_reports
- [ ] T024 [P] Write unit tests for dataclass validations in `src/tests/test_build_config.py`
- [ ] T025 [P] Write unit tests for redact_sensitive() in `src/tests/test_reports.py`
- [ ] T026 Add foundational regression tests for detect_toolchain(), get_toolchain_hints(), create_venv(), check_venv_integrity(), write_failure_report() in `src/tests/test_m2_foundation.py`

**Checkpoint**: Foundation ready — user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Build llama.cpp from Source (Priority: P1) 🎯 MVP

**Goal**: Deliver TUI build wizard with 5-stage pipeline (preflight → clone → configure → build → provenance), serialized backend execution, retry logic, build locking, and dry-run support

**Independent Test**: With a valid toolchain and source present, run `llm-runner build sycl` (TUI) and confirm: preflight passes, build executes, provenance is written, and the binary exists at the expected path. With a missing toolchain, confirm preflight fails with actionable FR-005 error. CLI `--json` output matches TUI success state.

### Tests for User Story 1 (REQUIRED) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T027 [P] [US1] Contract test for BuildArtifact JSON output in `src/tests/test_json_contract.py::test_build_artifact_contract`
- [ ] T028 [P] [US1] Test provenance fields compliance (SC-001) in `src/tests/test_build_pipeline.py::test_provenance_fields`
- [ ] T029 [P] [US1] Test serialized build order (SC-003) in `src/tests/test_build_pipeline.py::test_serialized_order`
- [ ] T030 [P] [US1] Test preflight failure detection (SC-006) in `src/tests/test_build_pipeline.py::test_preflight_failure`
- [ ] T031 [P] [US1] Test build lock behavior in `src/tests/test_build_pipeline.py::test_build_lock_held_error`
- [ ] T032 [P] [US1] Test retry logic with exponential backoff in `src/tests/test_build_pipeline.py::test_retry_behavior`
- [ ] T033 [P] [US1] Test retry applies only to failed targets in `src/tests/test_build_pipeline.py::test_retry_failed_only`
- [ ] T034 [P] [US1] Test clone branch behavior (existing vs fresh) in `src/tests/test_build_pipeline.py::test_clone_branch_behavior`
- [ ] T035 [P] [US1] Test cmake flag constants (FR-004.2a) in `src/tests/test_build_pipeline.py::test_cmake_flag_constants`
- [ ] T036 [P] [US1] Test branch ref and SHA recording in `src/tests/test_build_pipeline.py::test_branch_ref_and_sha`
- [ ] T037 [P] [US1] Test dry-run preflight only (FR-004.5) in `src/tests/test_build_pipeline.py::test_dry_run_preflight`
- [ ] T038 [P] [US1] Test atomic provenance write (FR-006.3) in `src/tests/test_build_pipeline.py::test_atomic_provenance_write`
- [ ] T039 [P] [US1] Test provenance write failure still succeeds (FR-006.3) in `src/tests/test_build_pipeline.py::test_provenance_failure_warning`
- [ ] T040 [P] [US1] Test predictable artifact paths (FR-006.2) in `src/tests/test_build_pipeline.py::test_artifact_paths`
- [ ] T041 [P] [US1] Test launch path does not trigger build (FR-006.2) in `src/tests/test_config.py::test_launch_no_autobuild`

### Implementation for User Story 1

- [ ] T042 [P] [US1] Implement BuildLock class in `src/llama_manager/build_pipeline.py` with acquire(), release(), is_stale() methods (file-based locking at Config.build_lock_path)
- [ ] T043 [P] [US1] Implement BuildPipeline class in `src/llama_manager/build_pipeline.py` with stage management (preflight → clone → configure → build → provenance)
- [ ] T044 [US1] Implement preflight stage in `src/llama_manager/build_pipeline.py` with toolchain validation, lock acquisition, source dir checks
- [ ] T045 [US1] Implement clone stage in `src/llama_manager/build_pipeline.py` with incremental update vs fresh clone logic, shallow clone support
- [ ] T046 [US1] Implement configure stage in `src/llama_manager/build_pipeline.py` with cmake flag derivation from BuildConfig class constants
- [ ] T047 [US1] Implement build stage in `src/llama_manager/build_pipeline.py` with make execution and progress parsing from [N/M] output
- [ ] T048 [US1] Implement provenance stage in `src/llama_manager/build_pipeline.py` with atomic JSON write to Config.builds_dir
- [ ] T049 [US1] Implement retry logic with exponential backoff for transient failures in `src/llama_manager/build_pipeline.py`
- [ ] T050 [US1] Implement serialized backend execution (SYCL first, then CUDA) in `src/llama_manager/build_pipeline.py` for 'both' backend
- [ ] T051 [US1] Create build_cli.py in `src/llama_cli/` with build command argument parsing (--dry-run, --retry-attempts, --full-clone, --jobs, --json)
- [ ] T052 [US1] Wire BuildPipeline to TUI progress updates in `src/llama_cli/tui_app.py` (Live progress panel, per-stage progress bar, retry feedback)
- [ ] T053 [US1] Add signal-safe cleanup in `src/llama_cli/tui_app.py` (Ctrl+C releases lock, preserves successful artifacts)
- [ ] T054 [US1] Implement BuildProgress event emission for TUI in `src/llama_manager/build_pipeline.py`

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Diagnose and Fix Missing Toolchains (Priority: P2)

**Goal**: Deliver toolchain diagnostics with FR-005 actionable errors and venv lifecycle management (create/reuse/integrity)

**Independent Test**: With a tool installed (e.g., cmake) and another missing (e.g., dpcpp), run `llm-runner setup --check` and confirm the present tool shows version and the missing tool shows an actionable FR-005 error with install instructions.

### Tests for User Story 2 (REQUIRED) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T055 [P] [US2] Contract test for ToolchainStatus JSON output in `src/tests/test_json_contract.py::test_toolchain_status_contract`
- [ ] T056 [P] [US2] Contract test for VenvResult JSON output in `src/tests/test_json_contract.py::test_venv_result_contract`
- [ ] T057 [P] [US2] Test toolchain errors with actionable hints (SC-002) in `src/tests/test_toolchain.py::test_missing_tool_error`
- [ ] T058 [P] [US2] Test venv lifecycle (SC-005) in `src/tests/test_setup_venv.py::test_venv_lifecycle`
- [ ] T059 [P] [US2] Test tool detection timeout (FR-005.4) in `src/tests/test_toolchain.py::test_detect_tool_timeout`
- [ ] T060 [P] [US2] Test cmake too old error (FR-005) in `src/tests/test_toolchain.py::test_cmake_too_old_error`
- [ ] T061 [P] [US2] Test setup --check skips venv integrity (per spec) in `src/tests/test_setup_cli.py::test_setup_check_skips_venv_integrity`
- [ ] T062 [P] [US2] Test venv integrity check detects corruption in `src/tests/test_setup_venv.py::test_venv_corruption_detection`
- [ ] T063 [P] [US2] Test venv path fallback to ~/.cache in `src/tests/test_setup_venv.py::test_venv_path_fallback`

### Implementation for User Story 2

- [ ] T064 [P] [US2] Implement check_toolchain(backend) in `src/llama_manager/toolchain.py` returning ToolchainStatus with None for missing tools
- [ ] T065 [P] [US2] Implement ToolchainHint generation with platform-specific install commands in `src/llama_manager/toolchain.py`
- [ ] T066 [US2] Create setup_cli.py in `src/llama_cli/` with --check and --json flags
- [ ] T067 [US2] Implement setup main path (no --check) in `src/llama_cli/setup_cli.py` with venv creation/reuse, integrity checks, toolchain validation
- [ ] T068 [US2] Implement CONFIRMATION_REQUIRED error for missing --yes in non-interactive mode in `src/llama_cli/setup_cli.py`

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Verify Build Provenance and Diagnose Failures (Priority: P3)

**Goal**: Deliver failure reports with redaction, rotation policy, and build provenance metadata

**Independent Test**: After a successful build, read the provenance JSON and confirm all required fields are present. After a failed build, confirm the report directory exists with redacted output.

### Tests for User Story 3 (REQUIRED) ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T069 [P] [US3] Test failure reports with redaction (SC-004) in `src/tests/test_reports.py::test_failure_report_redaction`
- [ ] T070 [P] [US3] Test report rotation (>50 reports) in `src/tests/test_reports.py::test_report_rotation`
- [ ] T071 [P] [US3] Test mutating-action log rotation in `src/tests/test_reports.py::test_mutating_action_log` (detailed: verify log rotates at Config.reports_dir, entries include command/timestamp/exit_code/truncated_output/redaction_applied, redaction applies to sensitive patterns)
- [ ] T072 [P] [US3] Test secrets redaction in build output in `src/tests/test_reports.py::test_secrets_redacted`
- [ ] T073 [P] [US3] Test report directory permissions (0700) in `src/tests/test_reports.py::test_report_permissions`
- [ ] T074 [P] [US3] Test offline-continue path when network unavailable but local clone exists in `src/tests/test_build_pipeline.py::test_offline_continue_path`

### Implementation for User Story 3

- [ ] T075 [P] [US3] Implement FailureReport class in `src/llama_manager/reports.py`
- [ ] T076 [P] [US3] Implement ReportManager class in `src/llama_manager/reports.py` with rotation policy (max 50, oldest-first deletion)
- [ ] T077 [P] [US3] Implement failure report generation in `src/llama_manager/reports.py` (build-artifact.json, build-output.log, error-details.json)
- [ ] T078 [P] [US3] Implement MutatingActionLog class in `src/llama_manager/reports.py` with rotation at Config.reports_dir path
- [ ] T079 [US3] Wire failure reports into build pipeline failure handlers in `src/llama_manager/build_pipeline.py`
- [ ] T080 [US3] Update tui_app.py to display failure report link on build failure

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: Doctor Repair & Cross-Cutting CLI (Priority: P3 - part of US1 extension)

**Goal**: Implement doctor --repair command for clearing failed staging directories

**Note**: doctor --repair scope is limited to failed-target staging/lock remediation (per FR-004.7)

### Tests for Doctor Repair

- [ ] T081 [P] Test doctor --repair clears failed staging in `src/tests/test_doctor_cli.py::test_doctor_repair_clears_failed`
- [ ] T082 [P] Test doctor --repair preserves successful artifacts in `src/tests/test_doctor_cli.py::test_doctor_repair_preserves_successful`
- [ ] T083 [P] Test doctor --repair handles stale locks in `src/tests/test_doctor_cli.py::test_doctor_repair_stale_lock`
- [ ] T084 [P] Test doctor success path (no repairs needed) in `src/tests/test_doctor_cli.py::test_doctor_success_no_repairs`

### Implementation for Doctor Repair

- [ ] T085 Create doctor_cli.py in `src/llama_cli/` with --repair flag
- [ ] T086 Implement doctor --repair logic in `src/llama_cli/doctor_cli.py` (clear failed staging dirs, validate/remove stale locks, preserve successful artifacts)
- [ ] T087 Implement DoctorRepairResult JSON output in `src/llama_cli/doctor_cli.py`

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, quality gates, and final validation

- [ ] T088 Update docstrings for all M2 modules (build_pipeline.py, toolchain.py, setup_venv.py, reports.py)
- [ ] T089 Update quickstart.md with M2 commands (build, setup, doctor)
- [ ] T090 Validate JSON contracts manually per contracts/cli-json-contract.md
- [ ] T091 Run ruff check: `uv run ruff check .` - must pass
- [ ] T092 Run ruff format check: `uv run ruff format --check .` - must pass
- [ ] T093 Run pyright: `uv run pyright` - must pass with 0 errors
- [ ] T094 Run pytest with coverage: `uv run pytest --cov=llama_manager,llama_cli --cov-report=term-missing` - must show 100% coverage on M2 modules
- [ ] T095 Run pip-audit: `uv run pip-audit` - no critical/high CVEs
- [ ] T096 Manual validation: `llm-runner build sycl --dry-run` shows preflight only
- [ ] T097 Manual validation: `llm-runner build both` runs serialized (SYCL first)
- [ ] T098 Manual validation: `llm-runner setup` creates venv at expected path
- [ ] T099 Manual validation: `llm-runner doctor --repair` clears failed staging dirs
- [ ] T100 Manual validation: TUI shows per-stage progress (SC-007)
- [ ] T101 Manual validation: Failure report contains redacted output (no secrets)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational phase completion
- **User Story 2 (Phase 4)**: Depends on Foundational phase completion; can run in parallel with US1 after T014 (redact_sensitive) and T015 (XDG paths)
- **User Story 3 (Phase 5)**: Depends on US1 Phase 2+ (failure paths from build pipeline)
- **Doctor Repair (Phase 6)**: Depends on US1 and US2 (needs build pipeline and lock logic)
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Independent of US1
- **User Story 3 (P3)**: Depends on US1 build pipeline failure paths being implemented

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Models/Config before services
- Services before CLI/TUI integration
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks (T001-T012) can run in parallel (different dataclasses/files)
- All Foundational tasks (T013-T026) can run in parallel after Setup
- Once Foundational phase completes:
  - US1 Tests (T027-T041) can run in parallel
  - US1 Implementation (T042-T054) can run in parallel with dependencies noted
  - US2 Tests (T055-T063) can run in parallel
  - US2 Implementation (T064-T068) can run in parallel
- US3 depends on US1 failure paths, but tests (T069-T074) can be written in parallel with US1 tests

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Contract test for BuildArtifact JSON output in src/tests/test_json_contract.py"
Task: "Test serialized build order (SC-003) in src/tests/test_build_pipeline.py"
Task: "Test build lock behavior in src/tests/test_build_pipeline.py"

# Launch models/pipeline components in parallel:
Task: "Implement BuildLock class in src/llama_manager/build_pipeline.py"
Task: "Implement BuildConfig dataclass in src/llama_manager/build_pipeline.py"
Task: "Implement BuildProgress dataclass in src/llama_manager/build_pipeline.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T012)
2. Complete Phase 2: Foundational (T013-T026) - CRITICAL
3. Complete Phase 3: User Story 1 (T027-T054)
4. **STOP and VALIDATE**: Test User Story 1 independently with quickstart.md scenarios 1-7
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP with build pipeline!)
3. Add User Story 2 → Test independently → Deploy/Demo (add toolchain/setup)
4. Add User Story 3 → Test independently → Deploy/Demo (add provenance/reports)
5. Add Doctor Repair → Test independently
6. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (build pipeline - highest priority)
   - Developer B: User Story 2 (toolchain/setup - can start immediately)
   - Developer C: User Story 3 (waiting for US1 failure paths)
3. Stories complete and integrate independently

---

## Summary

| Phase | Tasks | Focus |
| --- | --- | --------- |
| Phase 1: Setup | T001-T012 | Config extensions, ErrorCode additions, dataclass scaffolding |
| Phase 2: Foundational | T013-T026 | Toolchain detection, venv management, reporting infrastructure |
| Phase 3: User Story 1 (P1) | T027-T054 | Build pipeline, lock, retry, provenance, CLI, TUI integration |
| Phase 4: User Story 2 (P2) | T055-T068 | Toolchain diagnostics, venv creation, setup command |
| Phase 5: User Story 3 (P3) | T069-T080 | Failure reports, provenance, rotation, redaction |
| Phase 6: Doctor Repair | T081-T087 | Failed staging cleanup, lock remediation |
| Phase 7: Polish | T088-T101 | Documentation, quality gates, manual validation |
| **Total** | **101 tasks** | |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All subprocess calls in tests MUST be mocked (no real network/git/build)
- Build lock prevents concurrent builds; stale locks require doctor --repair (no auto-clear in M2)
- Provenance write failure does NOT fail the build (warning only)
- Report rotation maintains max 50 reports, deletes oldest first
- Redaction applies to build-output.log in failure reports and mutating-actions.log
- Venv integrity checks only verify pyvenv.cfg + interpreter symlink (package validation deferred post-MVP)
- Toolchain hints are Debian/Ubuntu focused (other platforms get generic "install X" messages)
- FR-004.7: doctor --repair confirmation UX is implementation-defined (optional in M2)
- FR-006.4 (offline-continue): If local clone exists and network unavailable, offer offline continue path

---

## Traceability Summary

| PRD Requirement | Task IDs | Verification |
|-----------------|----------|--------------|
| FR-004.1 (build subcommand) | T042-T054 | T027, T028, T036 |
| FR-004.2 (5-stage pipeline) | T044-T048 | T030, T034, T035 |
| FR-004.2a (cmake flag constants) | T003 | T035 |
| FR-004.3 (retry/backoff) | T049 | T032, T033 |
| FR-004.4 (build lock) | T042 | T031 |
| FR-004.5 (dry-run) | T051 | T037 |
| FR-004.6 (TUI + CLI) | T052, T053 | Manual SC-007 |
| FR-004.7 (doctor --repair) | T085-T087 | T081-T084 |
| FR-005.1 (setup --check) | T064, T065, T066 | T057, T059 |
| FR-005.2 (setup venv) | T019, T020, T067 | T058, T061, T063 |
| FR-005.3 (venv integrity) | T021 | T060, T062 |
| FR-005.4 (tool timeout) | T017 | T059 |
| FR-006.1 (provenance JSON) | T048 | T027, T028, T036 |
| FR-006.2 (predictable paths) | Config extensions | T040, T041 |
| FR-006.3 (atomic provenance) | T048 | T038, T039 |
| FR-006.4 (offline-continue) | T045 (clone stage) | T074 |
| FR-018.1-018.3 (failure reports) | T075-T079 | T069, T070, T073 |
| FR-018.4 (mutating-action log) | T078 | T071 |

---

## Changes Made - Architect Validation Fixes

### 1. Task ID Duplication Fix

**Before**: Task IDs T059-T064 appeared in both Phase 1 and Phase 5
**After**: Renumbered to unique sequential IDs:
- Phase 1 (Setup): T001-T012 (was T001-T006, T059-T064)
- Phase 2 (Foundational): T013-T026 (was T007-T010, T065-T074)
- Phase 3 (US1): T027-T054 (was T014-T041)
- Phase 4 (US2): T055-T068 (was T042-T055)
- Phase 5 (US3): T069-T080 (was T056-T066)
- Phase 6 (Doctor): T081-T087 (was T067-T072)
- Phase 7 (Polish): T088-T101 (was T073-T086)

### 2. Phase Structure Fix

**Before**: Overlapping task IDs between phases
**After**: Clean sequential ranges with no overlaps:
- Phase 1: T001-T012 (12 tasks)
- Phase 2: T013-T026 (14 tasks)
- Phase 3: T027-T054 (28 tasks)
- Phase 4: T055-T068 (14 tasks)
- Phase 5: T069-T080 (12 tasks)
- Phase 6: T081-T087 (7 tasks)
- Phase 7: T088-T101 (14 tasks)
- **Total**: 101 tasks

### 3. Missing Tests Added

**Added tasks**:
- T074 [US3] Test offline-continue path when network unavailable but local clone exists
- T084 [US3] Test doctor success path (no repairs needed)
- New test references for CMake version check (T060)
- New test references for build lock PID validation (T031)
- New test references for TUI progress events (T052, T100)

### 4. Test File Naming Fixes

**Before**: 
- T048: `test_setup_check_skips_venv_integrity.py` (standalone file)
- T028: `test_config.py::test_launch_no_autobuild` (incorrect test name)

**After**:
- T061: `test_setup_cli.py::test_setup_check_skips_venv_integrity`
- T041: `test_config.py::test_launch_no_autobuild`

### 5. Clarified Test Descriptions

**Before**:
- T058: "Test mutating-action log rotation in `src/tests/test_reports.py::test_mutating_action_log`"

**After**:
- T071: "Test mutating-action log rotation in `src/tests/test_reports.py::test_mutating_action_log` (detailed: verify log rotates at Config.reports_dir, entries include command/timestamp/exit_code/truncated_output/redaction_applied, redaction applies to sensitive patterns)"

### 6. Traceability Summary Updates

**Added**:
- FR-006.4 (offline-continue) → T045, T074
- FR-018.4 (mutating-action log) → T078, T071
- Updated all FR mappings to match new task IDs

### 7. [P] Marker Fixes

**Review**: All [P] markers now applied only when tasks truly have no dependencies:
- Setup phase: All tasks marked [P] (different dataclasses)
- Foundational phase: Core logic not marked [P], utility functions marked [P]
- US1 tests: All marked [P] (independent test files)
- US1 implementation: Only independent components marked [P]
- Doctor tests: All marked [P] (independent test cases)

### Summary Statistics

| Metric | Before | After |
|--------|--------|-------|
| Total tasks | 86 | 101 |
| Phase overlaps | 3 (T014-T018, T059-T064, T067-T072) | 0 |
| Missing tests | 4 | 0 |
| Naming issues | 2 | 0 |
| Traceability gaps | 2 FRs | 0 |
