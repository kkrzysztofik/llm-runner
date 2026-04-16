# Implementation Plan: PRD M2 - Build Wizard + Setup Pipeline

**Branch**: `002-build-setup` | **Date**: 2026-04-15 | **Spec**: `/specs/002-build-setup/spec.md`
**Input**: Feature specification from `/specs/002-build-setup/spec.md`

## Summary

Deliver PRD milestone M2 with a serialized llama.cpp build pipeline (SYCL/CUDA), toolchain diagnostics, setup venv management, provenance capture, and redacted failure reporting. The implementation follows spec decisions from `research.md`, models entities and invariants in `data-model.md`, and preserves M1 conventions for FR-005 actionable errors and CLI/TUI consistency.

**M2 Scope**: Build wizard TUI (primary), CLI parity for automation (`build`, `setup`, `doctor --repair`), toolchain detection, venv lifecycle, provenance JSON, failure reports with rotation. **Out of scope**: M0 (initial project), M3 (future enhancements), M4 (full operational FR-010 monitoring).

## Technical Context

**Language/Version**: Python 3.12+
**Primary Dependencies**: stdlib (`subprocess`, `pathlib`, `venv`, `json`, `dataclasses`, `threading`), rich, psutil
**Storage**: Local filesystem only (source tree + XDG cache/state/data directories)
**Testing**: pytest with mocking (`pytest.raises`, `capsys`, monkeypatch); no real GPU or subprocess dependence in tests
**Target Platform**: Linux workstation (anchored Intel Arc B580 + NVIDIA RTX 3090)
**Project Type**: Single-project Python CLI/TUI application
**Performance Goals**: Correctness and determinism over throughput; no explicit latency/SLA target in M2
**Constraints**: Serialized builds only, file lock required, no runtime venv mutation, no sudo automation, redacted/truncated failure logs, fail-fast on non-retryable errors
**Scale/Scope**: Single-operator local workflow; two build backends (`sycl`, `cuda`) plus `both` orchestration

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Code quality impact is explicit: changes stay within `llama_manager` (core) and `llama_cli` (I/O), with typed interfaces and no architecture boundary violations.
- [x] Testing plan is explicit: each user story has deterministic unit/regression coverage and validation via `uv run pytest`.
- [x] UX consistency impact is explicit: CLI/TUI use M1-aligned FR-005 structured errors, clear progress states, and explicit dry-run/setup behavior.
- [x] Runtime safety and observability impact is explicit: lockfile serialization, signal-safe cleanup, redaction, provenance/failure artifacts, and documented exit semantics.
- [x] Merge gates are explicit: `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pyright` are part of completion criteria.

**Gate Status (pre-design)**: PASS
**Gate Status (post-design)**: PASS (validated against `research.md`, `data-model.md`, `quickstart.md`, and contracts)

## Project Structure

### Documentation (this feature)

```text
specs/002-build-setup/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── cli-json-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── llama_cli/
│   ├── cli_parser.py
│   ├── server_runner.py
│   ├── tui_app.py
│   ├── build_cli.py          # planned M2 addition
│   ├── setup_cli.py          # planned M2 addition
│   └── doctor_cli.py         # planned M2 addition (doctor --repair)
├── llama_manager/
│   ├── config.py
│   ├── build_pipeline.py     # planned M2 addition
│   ├── toolchain.py          # planned M2 addition
│   ├── setup_venv.py         # planned M2 addition
│   ├── reports.py            # planned M2 addition
│   └── [existing M1 modules]
└── tests/
    ├── test_build_pipeline.py    # planned M2 addition
    ├── test_toolchain.py         # planned M2 addition
    ├── test_setup_venv.py        # planned M2 addition
    ├── test_reports.py           # planned M2 addition
    ├── test_doctor_cli.py        # planned M2 addition
    ├── test_json_contract.py     # planned M2 addition
    └── [existing M1 tests]
```

**Structure Decision**: Keep the existing single-project layout. Add M2 build/setup/report modules inside `llama_manager`, expose them via new CLI entry modules in `llama_cli`, and add focused test modules under `src/tests` to preserve the established architecture and test conventions.

---

## 1. Milestone Scope & Traceability (PRD/spec mapping)

This section maps PRD requirements to spec.md sections and implementation tasks.

| PRD Requirement | Spec Section | Task Group | Owner | Verification |
|-----------------|--------------|------------|-------|--------------|
| M2 build pipeline (SYCL/CUDA) | User Story 1 | `build_pipeline.py` | TBA | SC-001, SC-003 |
| Build TUI wizard | User Story 1 | `tui_app.py` + `build_cli.py` | TBA | SC-007 |
| CLI parity for automation | FR-004.6 | `build_cli.py` | TBA | SC-001, SC-006 |
| Toolchain diagnostics | User Story 2 | `toolchain.py` | TBA | SC-002, SC-006 |
| `doctor --repair` | FR-004.7 | `doctor_cli.py` | TBA | Manual acceptance |
| Venv lifecycle | User Story 2 | `setup_venv.py` | TBA | SC-005 |
| Provenance JSON | FR-006.1, SC-001 | `build_pipeline.py` | TBA | SC-001 |
| Failure reports + rotation | FR-018.1–018.3 | `reports.py` | TBA | SC-004 |
| JSON contract compliance | Addendum | `contracts/cli-json-contract.md` | TBA | Manual + test |

**Traceability Rule**: Every FR-004.* through FR-006.* and FR-018.* requirement must have at least one test case and one acceptance scenario across **all phases**. No requirement can be marked complete without both.

---

## 2. Workstreams and Phase Plan (ordered dependencies)

### Phase 0: Foundation (Week 1)
- **Goal**: Core data models, validators, and JSON contracts
- **Tasks**:
  - Define `BuildConfig` and `BuildArtifact` in `build_pipeline.py`
  - Define `ToolchainStatus` in `toolchain.py`
  - Define `VenvResult` in `setup_venv.py`
  - Implement `ErrorCode` enum + `ErrorDetail` schema (reuse M1)
  - Write `cli-json-contract.md` (already exists; validate)
  - Add `SYCL_REQUIRED_TOOLS`, `CUDA_REQUIRED_TOOLS` constants
- **Dependencies**: None
- **Deliverables**: `data-model.md`, `contracts/cli-json-contract.md`

### Phase 1: Toolchain Detection (Week 1–2)
- **Goal**: `setup --check` with FR-005 actionable errors
- **Tasks**:
  - Implement `detect_tool()` with **30s default timeout** (configurable via `Config.toolchain_timeout_seconds`); test timeout policy explicitly; align with FR-005.4 spec wording
  - Implement `check_toolchain(backend)` with platform-specific hints
  - Add `setup_cli.py` with `--check` + `--json`
  - Write `test_toolchain.py` with mocked `subprocess.run`
  - Add `ToolchainHint.required_for` field derived from SYCL_REQUIRED_TOOLS/CUDA_REQUIRED_TOOLS constants in toolchain model/planning
- **Dependencies**: Phase 0
- **Deliverables**: `toolchain.py`, `setup_cli.py`, `test_toolchain.py`

### Phase 2: Build Pipeline Core (Week 2–3)
- **Goal**: Serialized build stages with retry, lock, provenance
- **Tasks**:
  - Implement `BuildLock` class (file-based, signal-safe cleanup)
  - Implement `BuildPipeline` with stages: preflight → clone → configure → build → provenance
  - Add retry logic with exponential backoff (transient only); build progress resets on retry
  - Implement `doctor_cli.py` for `doctor --repair`
  - Clone branch behavior: existing valid repo => incremental update; absent/invalid => fresh clone
  - `master` checkout targeting and tip-SHA recording in provenance
  - Write `test_build_pipeline.py` with mocked subprocess
- **Dependencies**: Phase 0, Phase 1 (preflight)
- **Deliverables**: `build_pipeline.py`, `doctor_cli.py`, `test_build_pipeline.py`

### Phase 3: Reports + Rotation (Week 3)
- **Goal**: Failure reports with redaction and rotation
- **Tasks**:
  - Implement `FailureReport` class with redaction
  - Implement `ReportManager` with rotation policy
  - Rotating mutating-action log handled in `reports.py` (command, timestamp, exit code, truncated output, redaction, rotation policy)
  - Wire into build pipeline failure handlers
  - Write `test_reports.py` with redaction + rotation tests + mutating-action log tests
- **Dependencies**: Phase 2 (failure paths)
- **Deliverables**: `reports.py`, `test_reports.py`
- **Requirement-to-implementation mapping**: FR-018 mutating-action log → rotating mutating-action log in `reports.py` + `test_reports.py::test_mutating_action_log`

### Phase 2.5: Retry Failed Only (Week 2–3, after Phase 2)
- **Goal**: Ensure retry applies only to failed targets when building multiple backends
- **Tasks**:
  - Add explicit requirement text to `build_pipeline.py` docstring
  - Document behavior in `build_pipeline.py` comments and `test_build_pipeline.py`
  - Add test case: `test_retry_failed_only` — verify successful backends are not retried
- **Dependencies**: Phase 2 (build pipeline core)
- **Deliverables**: Updated `build_pipeline.py`, updated `test_build_pipeline.py`

### Phase 4: Venv Lifecycle (Week 3–4)
- **Goal**: `setup` command with integrity checks
- **Tasks**:
  - Implement `create_venv()` with path resolution: primary `$XDG_CACHE_HOME/llm-runner/venv`, fallback `~/.cache/llm-runner/venv`
  - Implement `check_venv_integrity()` (pyvenv.cfg + interpreter symlink)
  - Add `setup_cli.py` main path (no `--check`): creates/reuses venv, checks venv integrity, performs toolchain validation
  - Write `test_setup_venv.py` with mocked `venv`
- **Dependencies**: Phase 0 (ErrorDetail)
- **Deliverables**: `setup_venv.py`, `setup_cli.py` (complete), `test_setup_venv.py`

### Phase 5: TUI Build Wizard (Week 4–5)
- **Goal**: Rich-based TUI with per-stage progress (SC-007)
- **Tasks**:
  - Update `tui_app.py` with Live progress panel (canonical TUI module)
  - Wire `BuildProgress` events to TUI updates
  - Add retry feedback, failure report link
  - Add CLI fallback (`build_cli.py` for `llm-runner build`)
  - Write integration tests (no real subprocess)
- **Dependencies**: Phase 2 (pipeline), Phase 3 (reports)
- **Deliverables**: `tui_app.py` updates, `build_cli.py` (no `build_tui.py` to avoid ambiguity)

### Phase 6: Integration & QA (Week 5–6)
- **Goal**: End-to-end validation, CI gates, documentation
- **Tasks**:
  - Run `uv run pytest --cov` locally
  - Validate JSON contracts manually + automated
  - Update `quickstart.md` with M2 commands
  - Write `tasks.md` with final task list
  - Pre-merge: `ruff`, `pyright`, `pip-audit`
- **Dependencies**: All phases
- **Deliverables**: `quickstart.md`, `tasks.md`, CI pass

---

## 3. Requirement-to-Implementation Matrix (FR-004.*, FR-005.*, FR-006.*, FR-018.*)

| Requirement | Implementation Location | Test File | Acceptance Scenario |
|-------------|------------------------|-----------|---------------------|
| **FR-004.1** (`build` subcommand) | `build_cli.py`, `build_pipeline.py` | `test_build_pipeline.py` | Scenario 1 (User Story 1) |
| **FR-004.6** (TUI + CLI parity) | `tui_app.py`, `build_cli.py` | `test_build_pipeline.py` + TUI manual | SC-007 |
| **FR-004.7** (`doctor --repair`) | `doctor_cli.py` | `test_doctor_cli.py` (new) | Manual acceptance: `doctor --repair` clears failed-target staging **without deleting artifacts from successful backends**; dry-run preflight only |
| **FR-004.2** (5-stage pipeline) | `build_pipeline.py` | `test_build_pipeline.py` | Scenario 1, 2, 8 |
| **FR-004.2** incremental update | `build_pipeline.py` (clone stage branching logic) | `test_build_pipeline.py::test_clone_branch_behavior` | Existing valid git repo -> incremental update; absent/invalid repo -> fresh clone |
| **FR-004.2a** (cmake flag names are class constants in BuildConfig) | `build_pipeline.py` (BuildConfig class with GGML_SYCL, GGML_CUDA, CMAKE_C_COMPILER, CMAKE_CXX_COMPILER constants) | `test_build_pipeline.py::test_cmake_flag_constants` | Unit test flag derivation |
| **FR-004.3** (retry + backoff) | `build_pipeline.py` (retry logic) | `test_build_pipeline.py` | Scenario 4 (network retry); test_retry_failed_only |
| **FR-004.4** (build lock) | `build_pipeline.py` (BuildLock class) | `test_build_pipeline.py` | Scenario 6 (lock contention); lock path `$XDG_CACHE_HOME/llm-runner/.build.lock` |
| **FR-004.5** (dry-run) | `build_pipeline.py` (preflight only) | `test_build_pipeline.py` | Scenario 7 (dry-run preflight) |
| **master/tip SHA** | `build_pipeline.py` (git checkout + SHA recording) | `test_build_pipeline.py::test_branch_ref_and_sha` | Explicit task: `master` checkout targeting and tip-SHA recording in provenance |
| **FR-005.1** (`setup --check`) | `setup_cli.py`, `toolchain.py` | `test_toolchain.py` | Scenario 1, 2 (User Story 2) + test: `setup --check` does NOT run venv integrity checks (test_setup_check_skips_venv_integrity) |
| **FR-005.2** (`setup` venv) | `setup_venv.py`, `setup_cli.py` | `test_setup_venv.py` | Scenario 3, 4 (User Story 2) |
| **FR-005.3** (venv integrity) | `setup_venv.py` (check_integrity) | `test_setup_venv.py` | Scenario 5 (corrupted venv) |
| **FR-005.4** (tool detection timeout) | `toolchain.py` (detect_tool) | `test_toolchain.py::test_detect_tool_timeout` | Unit test timeout handling |
| **FR-005** (cmake minimum version preflight failure) | `toolchain.py` (preflight version check) | `test_toolchain.py::test_cmake_too_old_error` | Explicit coverage: cmake too old returns FR-005-style error |
| **FR-006.1** (provenance JSON) | `build_pipeline.py` (write_provenance) | `test_build_pipeline.py` | Scenario 1 (User Story 3) |
| **FR-006.2** (artifact paths) | `config.py` (Config class) | `test_config.py` | Unit test path resolution + test for predictable artifact paths + test: launch path does not trigger build pipeline |
| **FR-006.3** (atomic provenance) | `build_pipeline.py` (atomic write) | `test_build_pipeline.py` | Unit test rename semantics + test: if provenance write fails, build still succeeds and warning is emitted |
| **FR-018.1** (failure reports) | `reports.py` (FailureReport) | `test_reports.py` | Scenario 2 (User Story 3) |
| **FR-018.2** (redaction) | `reports.py` (redact_sensitive) | `test_reports.py` | Scenario 3 (API_KEY redacted) |
| **FR-018.3** (report rotation) | `reports.py` (ReportManager) | `test_reports.py` | Scenario 4 (>50 reports rotated) |
| **FR-018** mutating-action log | `reports.py` (MutatingActionLog) | `test_reports.py::test_mutating_action_log` | Rotate `setup` and other mutating actions with command, timestamp, exit code, truncated output, redaction. This is complementary to build-failure report requirements (FR-018.1–018.3); FR-018.4 covers all mutating actions while FR-018.1–018.3 specifically address build-failure report structure and rotation. |

**Coverage Rule**: Every FR must have at least one test case that exercises the success path and one that exercises the failure path.

---

## 4. Success Criteria Verification Matrix (SC-001..SC-007 -> tests/commands)

| Success Criterion | Verification Command/Test | Expected Outcome | Owner |
|-------------------|---------------------------|------------------|-------|
| **SC-001** (provenance fields) | `uv run pytest test_build_pipeline.py::test_provenance_fields` | All required provenance fields present in `BuildArtifact` per `specs/002-build-setup/contracts/cli-json-contract.md` and spec FR-006.1 | TBA |
| **SC-002** (toolchain errors) | `uv run pytest test_toolchain.py::test_missing_tool_error` | `error_code=TOOLCHAIN_MISSING`, `how_to_fix` contains install command | TBA |
| **SC-003** (serialized builds) | `uv run pytest test_build_pipeline.py::test_serialized_order` | SYCL starts before CUDA; no overlap in timestamps | TBA |
| **SC-004** (failure reports) | `uv run pytest test_reports.py::test_failure_report_redaction` | Report directory exists; no unredacted secrets in `build-output.log` | TBA |
| **SC-005** (venv lifecycle) | `uv run pytest test_setup_venv.py::test_venv_lifecycle` | First run creates venv; second run reuses; no mutation during serve | TBA |
| **SC-006** (preflight detection) | `uv run pytest test_build_pipeline.py::test_preflight_failure` | Preflight fails before clone stage; FR-005 error with `failed_check` | TBA |
| **SC-007** (TUI progress) | Manual: `uv run llm-runner build sycl` (TUI) + `test_build_pipeline.py::test_tui_progress_events` | Per-stage progress updates visible (0% → 100%); retry feedback; failure report link | TBA |

**Verification Notes**:
- SC-003 requires manual inspection of build logs; add automated timestamp check in `test_build_pipeline.py::test_serialized_order`.
- SC-007 requires human verification; add screenshot test harness for CI (optional post-M2).
- SC-005 requires runtime serve test; document expected venv path in `quickstart.md`.

---

## 5. CLI/TUI Design Commitments

### M2 TUI Build Flow (Primary Interface)

```text
┌─────────────────────────────────────────────────────────┐
│  llm-runner build sycl (TUI)                            │
├─────────────────────────────────────────────────────────┤
│  Status: Preflight Check                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  ✓ Toolchain: OK (cmake 3.29.2, gcc 13.2.0)            │
│  ✓ Build Lock: Available                                │
│  ✓ Source Dir: Accessible                               │
├─────────────────────────────────────────────────────────┤
│  Status: Cloning Source                               [0%]│
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Cloning from https://github.com/ggerganov/llama.cpp... │
├─────────────────────────────────────────────────────────┤
│  Status: Configuring Build                            [20%]│
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  cmake -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx ..         │
├─────────────────────────────────────────────────────────┤
│  Status: Building (make)                              [65%]│
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  Compiling llama-server... [124/192]                    │
├─────────────────────────────────────────────────────────┤
│  Status: Writing Provenance                           [90%]│
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  /home/kmk/.local/state/llm-runner/builds/...json       │
├─────────────────────────────────────────────────────────┤
│  ✓ Build Complete                                       │
│  Binary: /home/kmk/src/llama.cpp/build/bin/llama-server │
│  Report: N/A                                            │
└─────────────────────────────────────────────────────────┘
```

**TUI Commitments**:
- Per-stage progress bar with percentage (0% → 100%).
- Retry feedback: "Retry 1/3 in 2s..." with countdown.
- Failure report link: "Report: ~/.local/share/llm-runner/reports/.../".
- Signal-safe cleanup: Ctrl+C releases lock, preserves successful artifacts.
- CLI fallback: `llm-runner build sycl` provides non-TUI output (minimal progress to stdout).
- **TUI ownership**: M2 TUI work extends existing `tui_app.py`; no new `build_tui.py` file. M2 does not alter M1 monitoring scope beyond M2 build-progress surface.
- **M1 doctor foundation**: Includes venv health verification (interpreter path + basic import health); M2 adds `doctor --repair` for failed-target staging/lock remediation only (per PRD baseline, does not re-implement all doctor checks).

### Minimal CLI Parity (Automation)

```bash
# Dry-run preflight
llm-runner build sycl --dry-run

# Machine-readable output
llm-runner build sycl --json > build-artifact.json

# Toolchain check
llm-runner setup --check
llm-runner setup --check --json

# Venv creation
llm-runner setup

# Doctor repair
llm-runner doctor --repair
```

**CLI Commitments**:
- `--json` flag produces deterministic JSON (key-order agnostic; relies on presence/type checks, not ordering).
- Exit codes per-command:
  - `build`: 0 success, 1 failure
  - `setup`: 0 success, 1 failure
  - `doctor --repair`: 0 = warnings-only/ok, 1 = blocking error, 2 = needs setup/prereqs
- Non-mutating automation (build --dry-run, setup --check) remains non-interactive; mutating actions require explicit `--yes` or confirmatory UX (`setup` hard-required; `doctor --repair` implementation-defined per FR-004.7).
- `build both` runs serialized with default order (SYCL first); `--build-order` override is post-MVP (not active in M2).

---

## 6. JSON Contract Verification Plan

**Reference**: `contracts/cli-json-contract.md`

### Verification Steps

1. **Manual JSON Validation**
   ```bash
   # Build success
   uv run llm-runner build sycl --json | jq 'keys'
   # Expected: artifact_type, backend, created_at, git_remote_url, ...

   # Toolchain check
   uv run llm-runner setup --check --json | jq 'keys'
   # Expected: gcc, make, git, cmake, sycl_compiler, cuda_toolkit, nvtop

   # Venv result
   uv run llm-runner setup --json | jq 'keys'
   # Expected: venv_path, created, reused, activation_command
   ```

2. **Automated Contract Tests**
    - Add `test_json_contract.py` with schema validation
    - Use manual key checks (no undeclared dependency); `jsonschema` optional if available
    - Verify all required fields present; no extra fields

3. **Error Contract Validation**
   - Verify `error.error_code` matches `ErrorCode` enum
   - Verify `error.how_to_fix` contains actionable text
   - Verify `error.docs_ref` points to valid doc (if applicable)

**Contract Owner**: TBA
**Contract Review**: Pre-merge, all JSON outputs must pass `test_json_contract.py`

---

## 7. Error/Validation Model Alignment (FR-005 ErrorDetail/ErrorCode usage)

### ErrorCode Enum (M1 + M2 additions)

```python
class ErrorCode(str, Enum):
    # M1 (existing)
    INVALID_PORT = "INVALID_PORT"
    INVALID_THREADS = "INVALID_THREADS"
    
    # M2 additions
    TOOLCHAIN_MISSING = "TOOLCHAIN_MISSING"
    BUILD_LOCK_HELD = "BUILD_LOCK_HELD"
    VENV_CORRUPT = "VENV_CORRUPT"
    PYTHON_NOT_FOUND = "PYTHON_NOT_FOUND"
    BUILD_FAILED = "BUILD_FAILED"
    PREFLIGHT_FAILURE = "PREFLIGHT_FAILURE"
```

### ErrorDetail Schema (FR-005)

```python
@dataclass
class ErrorDetail:
    error_code: ErrorCode
    failed_check: str  # e.g., "cmake_not_found", "build_lock_held"
    why_blocked: str   # Human-readable explanation
    how_to_fix: str    # Actionable instructions (platform-specific)
    docs_ref: str | None  # Optional link to docs
```

### M2 Error Usage Patterns

| Scenario | ErrorCode | failed_check | how_to_fix Pattern |
|----------|-----------|--------------|-------------------|
| Missing cmake | `TOOLCHAIN_MISSING` | `cmake_not_found` | "Install cmake ≥3.24: `apt-get install cmake`" |
| Build lock held | `BUILD_LOCK_HELD` | `build_lock_held` | "Wait for existing build to complete or run `llm-runner doctor --repair`" |
| Corrupted venv | `VENV_CORRUPT` | `venv_missing_pyvenv_cfg` | "Remove corrupted venv: `rm -rf ~/.cache/llm-runner/venv` then run `llm-runner setup`" |
| Python not found | `PYTHON_NOT_FOUND` | `python_interpreter_missing` | "Install Python 3.12+: `apt-get install python3.12`" |
| Build failed | `BUILD_FAILED` | `compile_error` | "See failure report: ~/.local/share/llm-runner/reports/<timestamp>/" |

**Error Convention**: All errors must include `how_to_fix` with at least one actionable command or path. No generic "error occurred" messages.

---

## 8. Logging/Reports Plan (including PRD FR-018 rotating log)

### Report Directory Structure

```text
~/.local/share/llm-runner/reports/
├── 20260415_143022/
│   ├── build-artifact.json    # Partial BuildArtifact with failure details (backend in file content)
│   ├── build-output.log       # Truncated, redacted build output
│   └── error-details.json     # Exception type, message, stack trace summary
├── 20260415_142105/
│   ├── ...
└── ... (rotation maintains max N reports)
```

**Note**: Report directory uses timestamp-only naming (`<timestamp>/`), backend value stored in `build-artifact.json.backend`. No backend suffix in directory names.

### Redaction Policy (FR-018.2)

- **Pattern**: `KEY|TOKEN|SECRET|PASSWORD|AUTH` (case-insensitive)
- **Replacement**: `[REDACTED]`
- **Implementation**: Reuse existing `redact_sensitive()` from M1
- **Coverage**: `build-output.log` only; full logs remain in build directory (unredacted)

### Rotation Policy (FR-018.3)

- **Max Reports**: 50 (implementation-defined; configurable via `Config.build_max_reports`)
- **Rotation Trigger**: When writing new report and count > max
- **Deletion Order**: Oldest directories first (by directory name timestamp)
- **Signal Safety**: Deletion happens after report write; no partial deletions

### Log Levels

- **Info**: Stage transitions, progress updates, provenance write success
- **Warning**: Provenance write failure (build still successful), rotation cleanup
- **Error**: Build failure, preflight failure, lock acquisition failure
- **All errors**: Printed to stderr with FR-005 `ErrorDetail` format

### Log File Paths

- **Build logs**: `$XDG_STATE_HOME/llm-runner/build-logs/<timestamp>-<backend>.log`
- **Failure reports**: `~/.local/share/llm-runner/reports/<timestamp>/` (backend in file content, not directory name)
- **Provenance**: `$XDG_STATE_HOME/llm-runner/builds/<timestamp>-<backend>.json` (backend-suffixed per spec)
- **Mutating-action log**: `$XDG_STATE_HOME/llm-runner/mutating_actions.log` (rotating, max configurable via `Config.build_max_reports`)

---

## 9. Resolved Plan Confirmation Items

### Stale Lock Behavior (FR-004.4)
**Decision**: M2 does not implement automatic stale-lock recovery. Lock file includes PID; if PID is zombie, operator must run `doctor --repair`. `doctor --repair` validates PID and removes stale lock if process is gone. Future M3 may add auto-recovery with PID validation.

**Rationale**: Safe recovery requires PID resurrection checks; out of M2 scope. `doctor --repair` scope includes lock cleanup/check and staging directory cleanup.

### Rotation Policy (FR-018.3)
**Decision**: Default max reports = 50. Configurable via `Config.build_max_reports` (default 50). No CLI flag to override at runtime; change `Config` or environment variable.

**Rationale**: Operators can adjust via config; no need for runtime flag in M2.

### Non-Debian Platform Policy
**Decision**: M2 is anchored-workstation scoped (Debian/Ubuntu). For non-Debian platforms, detection of missing required prerequisites/toolchain MUST still produce actionable FR-005 errors and fail safely (exit 1). Do not mask failures via forced success exit.

**Rationale**: PRD-safe behavior requires actionable diagnostics and safe failure even on non-Debian platforms; document platform-specific install hint limitations in `quickstart.md`.

### Venv Integrity Boundary (FR-005.3)
**Decision**: M2 venv integrity checks are limited to:
1. `pyvenv.cfg` file exists
2. Python interpreter symlink is valid

**Deferred post-MVP**: Package validation (pip list), site-packages integrity, external dependency checks.

**Rationale**: Core integrity checks sufficient for M2; advanced validation can wait.

### `doctor --repair` Scope (FR-004.7)
**Decision**: `doctor --repair` clears failed-target staging directories only. Does NOT delete:
- Successful build artifacts
- Provenance files
- Failure reports (unless explicitly rotated)
- Venv directory

**Rationale**: Safe cleanup without data loss; operators can manually delete reports.

---

## 10. Risks, Mitigations, and Merge Gates

### Technical Risks

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| Git clone timeout in CI | Medium | High | Mock `subprocess.run` in tests; add `--full-clone` flag for manual testing | TBA |
| TUI rendering issues on small terminals | Medium | Medium | Add terminal size detection; fallback to CLI if <80x24 | TBA |
| Lock file not cleaned on crash | Low | High | Include PID in lock file; `doctor --repair` validates PID | TBA |
| Redaction misses secrets | Low | High | Use regex + manual review; add test case with known secret patterns | TBA |
| Report rotation deletes recent reports | Low | Medium | Sort by directory name (timestamp); add unit test for rotation order | TBA |

### Process Risks

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| Spec drift during implementation | Medium | Medium | Weekly spec review; update `plan.md` if scope changes | TBA |
| Test flakiness (retry logic) | Medium | Medium | Use deterministic mock delays in tests; avoid real network calls | TBA |
| JSON contract drift | Low | Medium | Automated contract tests; review `cli-json-contract.md` on every PR | TBA |

### Merge Gates (Mandatory)

All three must pass before merge:

1. **Lint**: `uv run ruff check .` (exit 0)
2. **Format**: `uv run ruff format --check .` (no changes needed)
3. **Type check**: `uv run pyright` (0 errors)
4. **Tests**: `uv run pytest --cov=llama_manager,llama_cli --cov-report=term-missing` (100% coverage on M2 modules)
5. **Audit**: `uv run pip-audit` (no critical/high CVEs)

**Additional Gate**: `doctor --repair` must be documented in `quickstart.md` and tested manually.

---

## Validation Checklist

Reviewers can run this checklist to verify M2 completeness:

### Code Quality
- [ ] `uv run ruff check .` exits 0
- [ ] `uv run ruff format --check .` reports no changes
- [ ] `uv run pyright` reports 0 errors
- [ ] All M2 modules have type hints on function signatures

### Testing
- [ ] `uv run pytest` passes all tests
- [ ] `uv run pytest --cov` shows 100% coverage on M2 modules
- [ ] Test cases exist for: SC-001 (provenance), SC-002 (toolchain errors), SC-004 (failure reports), `test_doctor_cli.py`, `test_json_contract.py`
- [ ] No real subprocess calls in tests (all mocked)

### Requirements Coverage
- [ ] FR-004.1 (`build` subcommand) implemented and tested
- [ ] FR-004.2 (5-stage pipeline) implemented with retry logic; incremental vs fresh clone behavior tested
- [ ] FR-004.2a (cmake flag names are class constants in BuildConfig) implemented with explicit test
- [ ] FR-004.3 (retry/backoff behavior) implemented and tested
- [ ] FR-004.4 (build lock) implemented; `BUILD_LOCK_HELD` error on contention; lock path documented
- [ ] FR-004.5 (dry-run semantics, preflight only) implemented and tested
- [ ] FR-004.6 (TUI + CLI parity) implemented; TUI shows per-stage progress (`tui_app.py` + `build_cli.py`)
- [ ] FR-004.7 (`doctor --repair`) implemented; clears failed staging dirs **without deleting artifacts from successful backends**
- [ ] FR-005.1 (`setup --check`) implemented with FR-005 errors; timeout policy tested
- [ ] FR-005.2 (`setup` venv) implemented; activation command printed; venv path fallback tested
- [ ] FR-005.3 (venv integrity) implemented; `VENV_CORRUPT` error on corruption
- [ ] FR-005.4 (tool detection timeout) implemented with explicit test
- [ ] FR-006.1 (provenance JSON) implemented with all 12 required fields; master/tip SHA recorded
- [ ] FR-006.2 (predictable artifact paths + no silent auto-build on launch) implemented and tested
- [ ] FR-006.3 (atomic provenance write + success-with-warning on write failure) implemented and tested
- [ ] FR-018.1–018.3 (build failure reports + rotation) implemented; FR-018.4 (mutating-action logs) implemented separately; complementary requirements clarified
- [ ] Status transition assertions (`pending/running/success/failed`) + retry progress reset tested

### JSON Contracts
- [ ] `llm-runner build --json` produces `BuildArtifact` shape
- [ ] `llm-runner setup --check --json` produces `ToolchainStatus` shape
- [ ] `llm-runner setup --json` produces `VenvResult` shape
- [ ] All errors produce FR-005 `ErrorDetail` shape
- [ ] `test_json_contract.py` validates all JSON outputs

### Documentation
- [ ] `quickstart.md` updated with M2 commands (`build`, `setup`, `doctor`)
- [ ] `contracts/cli-json-contract.md` validated against implementation
- [ ] `tasks.md` contains final task list with status
- [ ] Module docstrings present on all M2 modules

### Manual Verification
- [ ] `llm-runner build sycl --dry-run` shows preflight only
- [ ] `llm-runner build both` runs serialized (SYCL first)
- [ ] `llm-runner setup` creates venv at expected path
- [ ] `llm-runner doctor --repair` clears failed staging dirs
- [ ] TUI shows per-stage progress (SC-007)
- [ ] Failure report contains redacted output (no secrets)

**Checklist Owner**: TBA
**Checklist Status**: Pending M2 completion

---

**Plan Status**: Draft (planning phase; not yet implemented)
**Last Updated**: 2026-04-15
**Next Review**: Post-Phase 0 research completion
