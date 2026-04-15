# Tasks: PRD M2 — Build Wizard + Setup Pipeline

**Input**: Design documents from `/specs/002-build-setup/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Test tasks are REQUIRED. Every user story includes automated tests for independent behavior.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependency)
- **[Story]**: User story label (`[US1]`, `[US2]`, `[US3]`) for story-phase tasks only
- Every task includes an exact repository file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Extend Config with M2 fields, add new ErrorCode values, and create dataclass scaffolding.

- [ ] T001 Add XDG path fields to `Config` dataclass (`xdg_cache_base`, `xdg_state_base`, `xdg_data_base`) with environment variable fallbacks, and computed paths (`venv_path`, `builds_dir`, `reports_dir`, `build_lock_path`) in `src/llama_manager/config.py`
- [ ] T002 Add build configuration fields to `Config` dataclass (`build_git_remote`, `build_git_branch`, `build_retry_attempts`, `build_retry_delay`, `build_max_reports`, `build_output_truncate_bytes`) with defaults in `src/llama_manager/config.py`
- [ ] T003 Add new `ErrorCode` enum values (`TOOLCHAIN_MISSING`, `VENV_NOT_FOUND`, `VENV_CORRUPT`, `PYTHON_NOT_FOUND`, `BUILD_FAILED`, `BUILD_LOCK_HELD`, `GIT_CLONE_FAILED`, `GIT_CHECKOUT_FAILED`, `REPORT_WRITE_FAILURE`) in `src/llama_manager/config.py`
- [ ] T004 [P] Add `BuildConfig`, `BuildArtifact`, `BuildProgress` dataclasses with validation in `src/llama_manager/build_pipeline.py`
- [ ] T005 [P] Add `ToolchainStatus` dataclass and `ToolchainHint` dataclass with required fields in `src/llama_manager/toolchain.py`
- [ ] T006 [P] Add M2 pytest fixtures (`sample_build_config`, `sample_build_artifact`, `sample_toolchain_status`, `tmp_reports_dir`) in `src/tests/conftest.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build core toolchain detection, venv management, and reporting infrastructure required by all stories.

**⚠️ CRITICAL**: No user story implementation begins before this phase is complete.

- [ ] T007 Implement `detect_toolchain()` function that queries each tool's `--version` output via `subprocess.run` with 5-second timeout, returns `ToolchainStatus` with version strings or `None` for missing tools in `src/llama_manager/toolchain.py`
- [ ] T008 Implement `get_toolchain_hints()` function that takes `ToolchainStatus` and `backend` parameter, returns `list[ErrorDetail]` for missing required tools with platform-specific install commands in `src/llama_manager/toolchain.py`
- [ ] T009 [P] Implement `get_venv_path()` function returning `$XDG_CACHE_HOME/llm-runner/venv` with fallback to `~/.cache/llm-runner/venv` in `src/llama_manager/setup_venv.py`
- [ ] T010 [P] Implement `create_venv()` function that creates venv at given path using `venv` module if it doesn't exist, returns path; `check_venv_integrity()` that validates `pyvenv.cfg` existence and interpreter symlink, returns `ErrorDetail | None` in `src/llama_manager/setup_venv.py`
- [ ] T011 Implement `write_failure_report()` function that creates report directory with `build-artifact.json`, `build-output.log` (truncated and redacted), and `error-details.json`, enforcing `0700`/`0600` permissions in `src/llama_manager/reports.py`
- [ ] T012 Implement `rotate_reports()` function that deletes oldest report directories when count exceeds `Config.build_max_reports` in `src/llama_manager/reports.py`
- [ ] T013 Add foundational regression tests for `detect_toolchain()` (mocked subprocess), `get_toolchain_hints()` (missing tools → FR-005 errors), `create_venv()` (tmp_path), `check_venv_integrity()` (corrupted venv), and `write_failure_report()` (redaction, permissions) in `src/tests/test_m2_foundation.py`

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Build llama.cpp from Source (Priority: P1) 🎯 MVP

**Goal**: Deliver complete build pipeline with preflight, retry, lock, and provenance.

**Independent Test**: With valid toolchain, `build sycl` produces binary and provenance; with missing toolchain, build is blocked with FR-005 error.

### Tests for User Story 1 (REQUIRED) ⚠️

> **NOTE: Write these tests first, ensure they fail before implementation**

- [ ] T014 [P] [US1] Add build pipeline success tests: single-backend build completes all stages, produces binary and provenance JSON with all required fields in `src/tests/test_build_pipeline.py`
- [ ] T015 [P] [US1] Add build pipeline failure tests: missing toolchain blocks at preflight with FR-005 error, git clone failure retries then fails, build failure produces failure report in `src/tests/test_build_pipeline.py`
- [ ] T016 [P] [US1] Add build lock tests: concurrent build blocked with `BUILD_LOCK_HELD`, stale lock auto-cleared, lock released on completion/failure in `src/tests/test_build_pipeline.py`
- [ ] T017 [P] [US1] Add serialized build tests: `build both` runs SYCL first then CUDA, `--build-order cuda,sycl` reverses order, fail-fast on first backend failure in `src/tests/test_build_pipeline.py`

### Implementation for User Story 1

- [ ] T018 [US1] Implement `BuildPipeline.__init__()` and `BuildPipeline.run_preflight()` that checks toolchain (via `detect_toolchain` + `get_toolchain_hints`), source dir access, and build lock state; returns `list[ErrorDetail]` in `src/llama_manager/build_pipeline.py`
- [ ] T019 [US1] Implement `BuildPipeline._acquire_lock()` and `BuildPipeline._release_lock()` using file-based lock at `Config.build_lock_path` with PID and timestamp; stale lock detection (PID not running) auto-clears in `src/llama_manager/build_pipeline.py`
- [ ] T020 [US1] Implement `BuildPipeline._clone_source()` that runs `git clone` (shallow by default) with retry on `GIT_CLONE_FAILED`, and `BuildPipeline._checkout_branch()` that runs `git checkout` with `GIT_CHECKOUT_FAILED` on failure in `src/llama_manager/build_pipeline.py`
- [ ] T021 [US1] Implement `BuildPipeline._configure_build()` that runs cmake with backend-specific flags (SYCL: `-DGGML_SYCL=ON`, CUDA: `-DGGML_CUDA=ON`) and `BuildPipeline._build()` that runs `make -j N` with retry on transient failures, then verifies binary exists at expected path in `src/llama_manager/build_pipeline.py`
- [ ] T022 [US1] Implement `BuildPipeline._write_provenance()` that atomically writes provenance JSON to `Config.builds_dir` in `src/llama_manager/build_pipeline.py`
- [ ] T023 [US1] Implement `BuildPipeline.run()` that orchestrates all stages with progress tracking, retry logic, and failure report generation on error in `src/llama_manager/build_pipeline.py`
- [ ] T024 [US1] Implement `build_main()` CLI entry point in `src/llama_cli/build_cli.py` that parses build subcommand args and calls `BuildPipeline.run()` with appropriate `BuildConfig`
- [ ] T025 [US1] Update `cli_parser.py` to add `build` subcommand with `backend` positional arg (`sycl`|`cuda`|`both`), `--dry-run`, `--retry N`, `--verbose`, `--build-order`, `--full-clone`, `--jobs N` flags in `src/llama_cli/cli_parser.py`
- [ ] T026 [US1] Update `server_runner.py` main dispatch to route `build` subcommand to `build_main()` in `src/llama_cli/server_runner.py`

**Checkpoint**: User Story 1 is fully functional and independently testable.

---

## Phase 4: User Story 2 - Diagnose and Fix Missing Toolchains (Priority: P2)

**Goal**: Provide toolchain diagnostics and venv creation via CLI.

**Independent Test**: `setup --check` shows tool versions and actionable hints; `setup` creates venv and prints activation command.

### Tests for User Story 2 (REQUIRED) ⚠️

- [ ] T027 [P] [US2] Add toolchain detection tests: all tools present → versions shown, partial missing → FR-005 errors with install hints, backend-specific tool filtering (dpcpp only for sycl, nvcc only for cuda) in `src/tests/test_toolchain.py`
- [ ] T028 [P] [US2] Add venv creation tests: fresh creation, reuse existing, corrupted venv detection (missing pyvenv.cfg → `VENV_CORRUPT` error) in `src/tests/test_setup_venv.py`
- [ ] T029 [P] [US2] Add setup CLI tests: `setup --check` shows toolchain status, `setup` creates venv and prints activation command in `src/tests/test_setup_cli.py`
- [ ] T029.1 [P] [US2] Add doctor --repair tests: failed-target staging cleanup, preserve successful artifacts, lock-remediation behavior, confirmation UX (`--yes` vs interactive) in `src/tests/test_doctor_cli.py`

### Implementation for User Story 2

- [ ] T030 [US2] Implement `setup_main()` CLI entry point with `--check` flag that calls `detect_toolchain()` and renders status table, and default mode that calls `create_venv()` and prints activation command in `src/llama_cli/setup_cli.py`
- [ ] T030.5 [US2] Implement `doctor_main()` with `--repair` subcommand: failed-target staging cleanup, preserve successful artifacts, lock-remediation, confirmation UX (`--yes` or interactive) in `src/llama_cli/doctor_cli.py`
- [ ] T031 [US2] Update `cli_parser.py` to add `setup` subcommand with `--check` flag and `doctor` subcommand with `--repair` flag in `src/llama_cli/cli_parser.py`
- [ ] T032 [US2] Update `server_runner.py` main dispatch to route `setup` subcommand to `setup_main()` and `doctor` subcommand to `doctor_main()` in `src/llama_cli/server_runner.py`

**Checkpoint**: User Story 2 is fully functional and independently testable.

---

## Phase 5: TUI Build Wizard (Primary M2 Surface for FR-004) (Priority: P1)

**Goal**: Rich-based TUI with per-stage progress as the primary M2 interface for serialized builds, with explicit FR-004 visibility (stages/progress/retry-failed).

**Independent Test**: `uv run llm-runner build sycl` (TUI) shows per-stage progress (0% → 100%), retry countdown, failure report link; CLI parity `--json` output matches.

### Tests for TUI Build Wizard (REQUIRED) ⚠️

- [ ] T050 [P] [TUI] Add TUI progress event tests: `BuildProgress` events map to TUI updates, stage transitions (`pending/running/success/failed`), retry progress reset in `src/tests/test_tui_build.py`
- [ ] T051 [P] [TUI] Add TUI integration tests: `llm-runner build sycl` renders progress panel, Ctrl+C releases lock, failure report link visible, dry-run non-TUI output in `src/tests/test_tui_build.py`

### Implementation for TUI Build Wizard

- [ ] T052 [TUI] Update `tui_app.py` with Live progress panel (canonical TUI module): per-stage progress bar, percentage (0% → 100%), retry feedback ("Retry 1/3 in 2s..."), failure report link in `src/llama_cli/tui_app.py`
- [ ] T053 [TUI] Wire `BuildProgress` events to TUI updates: `BuildPipeline.run()` yields progress events, TUI consumes and renders in Live context in `src/llama_cli/tui_app.py`
- [ ] T054 [TUI] Add CLI fallback: `build_cli.py` for `llm-runner build` provides non-TUI output (minimal progress to stdout), `--json` flag for machine-readable output in `src/llama_cli/build_cli.py`
- [ ] T055 [TUI] Update `cli_parser.py` to add `build` subcommand with `--json` flag for TUI/CLI parity in `src/llama_cli/cli_parser.py`
- [ ] T056 [TUI] Update `server_runner.py` main dispatch to route `build` subcommand to `build_main()` (which delegates to TUI or CLI based on context) in `src/llama_cli/server_runner.py`

---

## Phase 6: User Story 3 - Verify Build Provenance and Diagnose Failures (Priority: P3)

**Goal**: Provenance records are complete and failure reports are properly structured with redaction.

**Independent Test**: After successful build, provenance JSON has all required fields; after failed build, report directory has redacted output.

### Tests for User Story 3 (REQUIRED) ⚠️

- [ ] T033 [P] [US3] Add provenance completeness tests: successful build → all required fields present, provenance write failure → warning emitted but build still succeeds in `src/tests/test_reports.py`
- [ ] T034 [P] [US3] Add failure report tests: report directory structure (3 files), output truncation (≤10 KiB), secret redaction (`API_KEY=secret123` → `API_KEY=[REDACTED]`), file/directory permissions (`0600`/`0700`) in `src/tests/test_reports.py`
- [ ] T035 [P] [US3] Add report rotation tests: more than 50 reports → oldest deleted, exactly 50 → no deletion in `src/tests/test_reports.py`
- [ ] T035.1 [P] [US3] Add mutating-action log tests: command/timestamp/exit code/truncated output/redaction rotation behavior for `setup` and other mutating actions in `src/tests/test_reports.py`

### Implementation for User Story 3

- [ ] T036 [US3] Implement provenance write failure handling: catch `OSError`/`PermissionError` during `_write_provenance()`, emit warning to stderr, continue build as successful in `src/llama_manager/build_pipeline.py`
- [ ] T037 [US3] Wire failure report generation into `BuildPipeline._handle_failure()` and `BuildPipeline.run()` exception paths in `src/llama_manager/build_pipeline.py`
- [ ] T038 [US3] Implement report rotation call after each `write_failure_report()` invocation in `src/llama_manager/reports.py`
- [ ] T038.1 [US3] Implement mutating-action log: rotating log entry with command, timestamp, exit code, truncated output, redaction for `setup` and other mutating actions in `src/llama_manager/reports.py`

**Checkpoint**: User Story 3 is fully functional and independently testable.

---

## Phase 6: Integration and Quality

**Purpose**: End-to-end integration, CLI parity, and quality gate validation.

- [ ] T039 Add integration test: `llm-runner build sycl --dry-run` runs preflight only without build in `src/tests/test_build_cli.py`
- [ ] T040 [P] Add integration test: `llm-runner setup --check` outputs toolchain status table in `src/tests/test_setup_cli.py`
- [ ] T041 [P] Add integration test: `llm-runner build both` with mocked pipeline runs serialized builds in `src/tests/test_build_cli.py`
- [ ] T057 [P] Add TUI integration test: `llm-runner build sycl` (TUI) renders progress panel, `--json` produces valid BuildArtifact JSON in `src/tests/test_tui_build.py`
- [ ] T058 [P] Add confirmatory UX tests: mutating setup actions (`setup`, `doctor --repair`) require `--yes` or interactive confirmation, no confirmation → exit 1 with message in `src/tests/test_setup_cli.py`, `src/tests/test_doctor_cli.py`
- [ ] T042 Run `uv run ruff check .` and `uv run ruff format --check .` — fix any violations in all new/modified files
- [ ] T043 Run `uv run pyright` — fix any type errors in all new/modified files
- [ ] T044 Run `uv run pytest --cov` — ensure all tests pass and new modules have ≥80% coverage
- [ ] T045 Run `uv run pip-audit` — ensure no new CVEs from dependency changes

---

## Summary

| Phase | Tasks | Focus |
| --- | --- | --------- |
| Phase 1: Setup | T001–T006 | Config extensions, dataclass scaffolding, fixtures |
| Phase 2: Foundational | T007–T013 | Toolchain detection, venv management, reporting infra |
| Phase 3: US1 Build Pipeline | T014–T026 | Build pipeline, lock, retry, provenance, CLI |
| Phase 4: US2 Toolchain/Setup | T027–T032 | Toolchain diagnostics, venv CLI, setup command, doctor --repair |
| Phase 5: TUI Build Wizard | T050–T056 | TUI progress panel, BuildProgress events, CLI/TUI parity |
| Phase 5 (renumbered): US3 Provenance/Reports | T033–T038.1 | Provenance completeness, failure reports, rotation, mutating-action logs |
| Phase 6: Integration | T039–T045, T057–T058 | End-to-end, TUI integration, confirmatory UX, quality gates |
| **Total** | **53 tasks** | |
