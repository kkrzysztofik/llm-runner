# Feature Specification: PRD M2 — Build Wizard + Setup Pipeline

**Feature Branch**: `002-build-setup`
**Created**: 2026-04-15
**Status**: Draft
**Input**: PRD.md v0.3 (M2 milestone), Spec 001 (M1 — completed)
**Milestone Scope**: M2 only — TUI build pipeline, toolchain diagnostics, setup venv, build provenance, failure reports. Not M0, M3, or M4.


## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build llama.cpp from Source (Priority: P1)

As a llama.cpp experimenter, I can trigger serialized SYCL and CUDA builds from the CLI with
preflight checks, progress tracking, and retry on transient failures, so I can produce verified
build artifacts without manual scripting.

**Why this priority**: The build pipeline is the core M2 value proposition — without buildable artifacts,
no downstream functionality (launch, smoke) can work.

**Independent Test**: With a valid toolchain and source present, run `llm-runner build sycl` and
confirm: preflight passes, build executes, provenance is written, and the binary exists at the
expected path. With a missing toolchain, confirm preflight fails with actionable FR-005 error.

**Acceptance Scenarios**:

1. **Given** a valid toolchain and source present, **When** I run `llm-runner build sycl`, **Then**
   the build completes successfully and the binary exists at the expected path with provenance metadata.
2. **Given** a valid toolchain and source present, **When** I run `llm-runner build both`, **Then**
   SYCL build runs first, then CUDA build runs sequentially, and both artifacts are produced with
   independent provenance records.
3. **Given** a missing cmake, **When** I run `llm-runner build sycl`, **Then** the build is blocked
   with FR-005 actionable error containing `error_code=TOOLCHAIN_MISSING`, `failed_check=cmake_not_found`,
   and `how_to_fix` with install instructions.
4. **Given** a network failure during git clone, **When** the build pipeline retries, **Then** the
   pipeline retries up to `--retry-attempts` times with exponential backoff.
5. **Given** the build fails after compilation starts, **When** the failure is processed, **Then** a
   failure report directory is written under `~/.local/share/llm-runner/reports/<timestamp>/` with
   redacted build output and error details.
6. **Given** a build is already in progress, **When** a second build is attempted, **Then** the second
   attempt is blocked with FR-005 error `failed_check=build_lock_held`.
7. **Given** `llm-runner build sycl --dry-run` is run, **When** preflight checks execute, **Then**
   only preflight results are displayed without starting the build.
**Flow Classification (User Story 1)**:

- **Primary flow**: Scenario 1 (single-backend build success).
- **Alternate flow**: Scenario 2 (both-backends build with default order).
- **Exception flow**: Scenarios 3, 4, 5, 6 (toolchain missing, network retry, compilation failure, lock contention).

---

### User Story 2 - Diagnose and Fix Missing Toolchains (Priority: P2)

As a systems user, I can run toolchain diagnostics and get actionable OS-level hints for missing
build tools, and use `setup` to create/reuse a venv, so I can resolve build blockers without guessing.

**Why this priority**: Actionable diagnostics are required before users can successfully build —
without them, build failures are opaque.

**Independent Test**: With a tool installed (e.g., cmake) and another missing (e.g., dpcpp), run
`llm-runner setup --check` and confirm the present tool shows version and the missing tool shows
an actionable FR-005 error with install instructions.

**Acceptance Scenarios**:

1. **Given** all required tools are present, **When** I run `llm-runner setup --check`, **Then**
   the toolchain status shows all tools with their detected versions and no errors.
2. **Given** cmake is missing, **When** I run `llm-runner setup --check`, **Then** the output
   includes `TOOLCHAIN_MISSING` error with `failed_check=cmake_not_found` and `how_to_fix` containing
   platform-specific install commands (e.g., `apt-get install cmake`).
3. **Given** no venv exists, **When** I run `llm-runner setup`, **Then** a venv is created at
   `$XDG_CACHE_HOME/llm-runner/venv` and the activation script path is printed.
4. **Given** a venv already exists, **When** I run `llm-runner setup`, **Then** the existing venv
   is reused and no new venv is created.
5. **Given** a corrupted venv (missing `pyvenv.cfg`), **When** I run `llm-runner setup`, **Then**
   an FR-005 error with `error_code=VENV_CORRUPT` is returned with `how_to_fix` instructing removal
   and re-creation.

---

### User Story 3 - Verify Build Provenance and Diagnose Failures (Priority: P3)

As an operator, I can verify that build artifacts have recorded provenance (git URL, commit SHA,
timestamps), and on build failure I get a structured report directory with redacted logs, so I can
audit and troubleshoot builds reliably.

**Why this priority**: Provenance and failure reporting are essential for operational trust but
depend on the build pipeline existing first.

**Independent Test**: After a successful build, read the provenance JSON and confirm all required
fields are present. After a failed build, confirm the report directory exists with redacted output.

**Acceptance Scenarios**:

1. **Given** a successful SYCL build, **When** I read the provenance file, **Then** it contains
   `artifact_type`, `backend`, `created_at`, `git_remote_url`, `git_commit_sha`, `git_branch`,
   `build_command`, `build_duration_seconds`, `exit_code`, `binary_path`, `binary_size_bytes`,
   and `build_log_path`.
2. **Given** a failed build, **When** I inspect the reports directory, **Then** it contains
   `build-artifact.json` with failure details, `build-output.log` with truncated redacted output,
   and `error-details.json` with the exception information.
3. **Given** build output contains `API_KEY=secret123`, **When** the failure report is written,
   **Then** the output log contains `API_KEY=[REDACTED]` and does NOT contain `secret123`.
4. **Given** more than 50 report directories exist, **When** a new failure report is written,
   **Then** the oldest reports are rotated (deleted) to maintain the maximum count.

### Edge Cases

- Serialized build order: SYCL first by default; `--build-order` override is post-MVP.
- `build both` fails fast on first backend failure; successful backends retain their artifacts.
- When building multiple targets, retry operation applies only to failed targets (not successful ones).
- Preflight checks run before any build execution; missing toolchain is always fatal (no retry).
- Network failures during git clone are retryable; compiler failures are fatal.
- Ctrl+C during build: clean up staging, preserve successful artifacts, release lock, no failure report.
- Venv integrity check detects missing `pyvenv.cfg` or broken interpreter symlink.
- `setup` does NOT auto-activate the venv; only prints the activation command.
- `setup --check` is toolchain-only; it does NOT check venv integrity. Venv integrity is checked during `setup` (without `--check`).
- `setup` does NOT install system packages or run sudo in M2; only creates venv and checks toolchain.
- Provenance files are written atomically where possible; partial writes are handled gracefully.
- Report rotation maintains a maximum number of reports; oldest are deleted first.
- Build output is truncated and redacted in failure reports; full logs remain in the build directory.
- Cmake flags are derived from `BuildConfig` dataclass fields; flag names are class constants.
- `build --json` outputs `BuildArtifact` JSON; `setup --json` outputs `VenvResult` JSON; `setup --check --json` outputs `ToolchainStatus` JSON.
- Toolchain install hints are platform-specific; non-Debian platforms run detection with graceful degradation but missing required build prerequisites/toolchain fails safely with actionable FR-005 diagnostics (no forced success status).
- No explicit performance targets; success criteria focus on correctness, not speed.
- `build --dry-run` runs preflight only; does not clone, configure, or build.
- Build-stage progress reporting is implementation-defined.
- `build --full-clone` (if supported) uses `git clone` without shallow clone.
- `build --jobs N` (if supported) overrides parallel make job count.
- On network loss during git clone, if local clone exists, `build` offers/allows offline continue path; otherwise fails with actionable diagnostics.
- `setup` venv creation requires network for package install; no offline fallback in M2.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-004.1**: System MUST provide a CLI `build` subcommand that accepts a backend argument
   (`sycl` | `cuda` | `both`) and executes the llama.cpp build pipeline for the specified backend(s).
   When `both` is specified, builds MUST run sequentially (never in parallel). Default build order is
   SYCL first; `--build-order` override is post-MVP. The `--json` flag produces machine-readable
   `BuildArtifact` JSON output (consistent with M1's `dry-run --json` pattern).
- **FR-004.6**: System MUST support M2 build workflow via TUI with minimal CLI parity as needed for automation.
  Full FR-010 operational monitoring remains M4.
- **FR-004.7**: The `doctor --repair` command clears failed-target staging directories without
   deleting artifacts from successful backends. Confirmation UX for `doctor --repair` is
   implementation-defined (optional in M2).

- **FR-004.2**: Build pipeline MUST execute the following stages in order: preflight → clone →
  configure → build → provenance (5 stages). Each stage MUST report its status (`pending` | `running` |
  `success` | `failed`) and progress percentage. During the `build` stage, the pipeline reports
  progress (implementation-defined method). Progress resets on retry. For other stages, progress is
  stage-level only (0.0 → 100.0 on completion). When the source directory already exists with a valid git
  repository, the clone stage performs an incremental update; otherwise a fresh clone is performed.
- **FR-004.2a**: The configure stage MUST derive cmake flags from `BuildConfig` dataclass fields
  rather than hardcoded strings. Flag names (e.g., `GGML_SYCL`, `GGML_CUDA`, compiler paths) MUST
  be defined as class constants. The specific flag set per backend:
  - SYCL: `GGML_SYCL=ON`, `CMAKE_C_COMPILER=icx`, `CMAKE_CXX_COMPILER=icpx`
  - CUDA: `GGML_CUDA=ON`
  These are the current upstream cmake flags (the older `LLAMA_*` names are deprecated); additional flags may be added via `BuildConfig` fields.
- **FR-004.3**: Build pipeline MUST implement retry logic for transient failures (network timeout,
  transient build tool errors). Retry-attempt count is configurable. Exponential backoff delay is
  configurable. Missing toolchain components are NOT retryable and MUST fail immediately with FR-005
  actionable error.
- **FR-004.4**: Build pipeline MUST enforce serialized execution via a file-based lock.
  A second build attempt while a lock is held MUST be blocked with FR-005 actionable error
  (`error_code=BUILD_LOCK_HELD`, `failed_check=build_lock_held`). Implementation-defined lock path.
- **FR-004.5**: Build pipeline MUST support `--dry-run` flag that runs preflight checks only without
  executing any build stages. MAY support `--full-clone` flag that disables shallow clone.
  MAY support `--jobs N` flag that overrides parallel make job count.
- **FR-005.1**: System MUST provide a `setup --check` subcommand that detects installed toolchain
  components (gcc, make, git, cmake, sycl_compiler, cuda_toolkit, nvtop) and returns their version
  strings, or `None` for missing tools. On supported platforms, missing tools produce FR-005 actionable
  errors with platform-specific install instructions. The `--json` flag produces machine-readable
  `ToolchainStatus` JSON output (consistent with M1's `dry-run --json` pattern).
  `setup --check` MUST NOT perform venv integrity checks — it is toolchain-only.
- **FR-005.2**: System MUST provide a `setup` subcommand (without `--check`) that creates or reuses
   a virtual environment at `$XDG_CACHE_HOME/llm-runner/venv` (fallback `~/.cache/llm-runner/venv`).
   The venv MUST NOT be auto-activated; the activation command MUST be printed to stdout.
   Runtime `llm-runner` MUST NOT mutate this venv during serve operations. When running `setup`
   without `--check`, the system MUST also check venv integrity (FR-005.3) and report any issues.
   Mutating actions under `setup` (without `--check`) MUST require confirmatory UX or explicit
   `--yes` equivalent before proceeding. The `--json` flag produces machine-readable `VenvResult` JSON.
- **FR-005.3**: System MUST detect corrupted venvs (missing `pyvenv.cfg`, broken interpreter symlink)
  and return FR-005 actionable error with `error_code=VENV_CORRUPT` and `how_to_fix` instructing
  removal and re-creation. Venv integrity is checked during `setup` (without `--check`), NOT during
  `setup --check` (which is toolchain-only).
- **FR-005.4**: Toolchain detection MUST use `subprocess.run` with timeout to query `--version`
   output. Detection failures (tool not found, timeout) return `None` for that tool and generate FR-005
   actionable error if the tool is required for the requested backend. **Timeout policy**: default 30s,
   configurable via `Config.toolchain_timeout_seconds`. The set of required tools per backend is defined
   as module-level constants (`SYCL_REQUIRED_TOOLS` and `CUDA_REQUIRED_TOOLS`) in `toolchain.py`.
   `ToolchainHint.required_for` fields are derived from these constants. Common tools (gcc, make, git,
   cmake) are required for both backends; cmake version is validated at preflight, emitting FR-005 error
   if below upstream llama.cpp requirements; backend-specific tools (dpcpp/icx/icpx for SYCL, nvcc for
   CUDA) are required only for their respective backend.
- **FR-006.1**: Every successful build MUST produce a provenance JSON file at
  `$XDG_STATE_HOME/llm-runner/builds/<timestamp>-<backend>.json` containing essential fields:
  `artifact_type`, `backend`, `created_at`, `git_remote_url`, `git_commit_sha`, `git_branch`,
  `build_command`, `exit_code`, `binary_path`.
- **FR-006.2**: Build artifacts MUST live at predictable paths determined by `Config`:
  Intel SYCL binary at `src/llama.cpp/build/bin/llama-server`, NVIDIA CUDA binary at
  `src/llama.cpp/build_cuda/bin/llama-server`. The system MUST NOT silently auto-build on launch.
- **FR-006.3**: Provenance files SHOULD be written atomically (write to temp file, then rename).
   If provenance write fails, the build is still considered successful but a warning is emitted
   indicating the provenance recording failure.
- **FR-006.4 (offline-continue)**: If a local clone exists and network is unavailable during
   `build`, the pipeline offers an offline-continue path (skips clone/update, proceeds to configure
   and build). If no local clone exists and network is unavailable, the pipeline fails with FR-005
   actionable error (`error_code=NETWORK_UNAVAILABLE`, `failed_check=git_clone_blocked`).
- **FR-018.1**: Build failures MUST produce a report directory at
  `~/.local/share/llm-runner/reports/<timestamp>/` containing: `build-artifact.json` (partial
  BuildArtifact with failure details), `build-output.log` (truncated, secrets redacted),
  and `error-details.json` (exception type, message, and stack trace summary).
- **FR-018.2**: All build output in reports MUST be redacted using the existing `redact_sensitive()`
  pattern (KEY|TOKEN|SECRET|PASSWORD|AUTH → `[REDACTED]`).
- **FR-018.3**: Report rotation MUST maintain a maximum number of report directories.
  When the limit is exceeded, the oldest reports (by directory timestamp) are deleted first.
  Default count is implementation-defined.
- **FR-018.4**: Mutating actions (e.g., `setup` venv creation) MUST produce a rotating log entry
   containing: command, timestamp, exit code, truncated output, and secret redaction.
   Rotation policy applies consistently (oldest entries deleted first when limit is exceeded).
   Default max entries is implementation-defined. **Path separation**: failure reports (FR-018.1–018.3)
   live under `~/.local/share/llm-runner/reports/<timestamp>/` (data directory), while mutating-action
   logs (FR-018.4) live under `$XDG_STATE_HOME/llm-runner/mutating_actions.log` (state directory);
   these are complementary artifacts serving different operational purposes. FR-018.4 covers all mutating
   actions while FR-018.1–018.3 specifically address build-failure report structure and rotation.

### Constitution Alignment *(mandatory)*

- **CA-001 Code Quality**: This feature MUST preserve core/CLI boundary integrity (`llama_manager/`
  is pure library; no `argparse`, no `Rich`, no module-level subprocess side effects) and pass all
  required static-quality gates (`ruff`, `pyright`) before merge.
- **CA-002 Testing**: This feature MUST include deterministic automated tests for success paths,
  failure paths, and regression cases for each user story. All subprocess calls in tests MUST be
  mocked.
- **CA-003 UX Consistency**: This feature MUST reuse the M1 FR-005 structured error format
  (`ErrorDetail` with `error_code`, `failed_check`, `why_blocked`, `how_to_fix`, `docs_ref`)
  for all build and setup errors. CLI output conventions MUST align with M1 patterns.
- **CA-004 Safety and Observability**: This feature MUST provide actionable operator diagnostics,
  clear failure attribution (toolchain → build → provenance → report), and redacted reporting
  outputs. Build lock prevents concurrent unsafe operations.

### Key Entities *(include if feature involves data)*

- **BuildConfig**: Per-build configuration (backend, source dir, build dir, output dir, git settings,
  retry settings).
- **BuildArtifact**: Provenance metadata for a completed build (success or failure).
- **BuildProgress**: Live progress tracking for TUI updates (stage, status, message, percent).
- **ToolchainStatus**: Detected toolchain component versions with `None` for missing tools.
- **BuildLock**: File-based lock preventing concurrent builds.
- **FailureReport**: Structured report directory with redacted logs and error details.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of successful builds produce provenance metadata with all required fields
  (`artifact_type`, `backend`, `created_at`, `git_remote_url`, `git_commit_sha`, `git_branch`,
  `build_command`, `build_duration_seconds`, `exit_code`, `binary_path`, `binary_size_bytes`,
  `build_log_path`).
- **SC-002**: 100% of toolchain detection failures (missing cmake, gcc, dpcpp, nvcc, git, make) return
  FR-005 actionable errors with `error_code=TOOLCHAIN_MISSING` and platform-specific `how_to_fix`
  containing install commands.
- **SC-003**: Serialized build order is enforced; `build both` always runs backends sequentially
  (never in parallel), with SYCL first by default.
- **SC-004**: 100% of build failures (non-interrupt) produce a report directory under
  `~/.local/share/llm-runner/reports/` with redacted output (no unredacted secrets).
- **SC-005**: `setup` venv creation succeeds and the venv is reusable; runtime `llm-runner`
  never mutates the venv during serve operations.
- **SC-006**: Build pipeline preflight detects all missing toolchain components before any build
  execution begins (clone, configure, build stages).
- **SC-007**: TUI build wizard displays per-stage progress updates in real-time (0% → 100%) during
  the `build` stage, with visible retry feedback and failure report link when applicable.

## Assumptions

- The target persona is a solo operator on the anchored workstation (Intel Arc + NVIDIA RTX 3090).
- M2 builds target `llama.cpp` `master` branch from the canonical GitHub repository.
- The operator has internet access for git clone (offline builds require pre-existing source).
- `sudo` operations are deferred from M2; `setup` focuses on venv creation and toolchain detection only.
- M1 slot-first infrastructure (Config, ServerConfig, ErrorDetail, ErrorCode, validators) is available
  and stable.
- The existing `Config.llama_cpp_root` path convention is used for source and build directories.
- Build artifacts (binaries) remain at paths defined by the existing llama.cpp build system.
- TUI build wizard/progress flow is included in M2 (target selection, stage progress, retry failed, report link). Full TUI monitoring (FR-010) remains M4.
- M1 doctor foundation includes venv health verification (interpreter path + basic import health); M2 adds `doctor --repair` for failed-target staging/lock remediation only (per PRD baseline, does not re-implement all doctor checks).

### Plan Confirmation Items

The following items require explicit confirmation during `/speckit.plan`:

- Build lock is mandatory in M2 (FR-004.4); planning to confirm stale-lock recovery behavior only.
- Report rotation maximum count remains configurable/implementation-defined in M2 plan (FR-018.3).
- Non-Debian platform behavior is out of M2 scope per anchored workstation narrative; broader platform policy is post-MVP.
- Venv integrity checks in M2 are limited to pyvenv.cfg + interpreter symlink checks (FR-005.3); package validation deferred post-MVP.

## Addendum — Definitions & Measurement Notes (M2)

### Canonical Terminology (M2)

| Term | Definition (M2 scope) |
| --- | --- |
| **build pipeline** | The orchestrated sequence of stages (preflight → clone → configure → build → provenance) that produces a llama.cpp build artifact. The `build` stage includes binary verification after `make` completes. |
| **serialized build** | Build execution where backends run one at a time, never in parallel. Required for `build both` in M2. |
| **preflight check** | Pre-build validation of toolchain availability, source directory access, and build lock state. Must pass before any build stage begins. |
| **retryable failure** | A transient failure (network timeout, build tool segfault, disk-full-after-cleanup) that the pipeline will automatically retry up to `--retry-attempts` times. |
| **fatal failure** | A non-transient failure (missing compiler, invalid git repo, permission denied) that the pipeline will not retry. Always produces an FR-005 actionable error. |
| **provenance** | Structured metadata recording the build's origin (git URL, commit SHA, branch), build parameters, timing, and output location. Written as JSON after successful or failed builds. |
| **failure report** | A directory under `~/.local/share/llm-runner/reports/<timestamp>/` containing redacted build output, artifact metadata, and error details for a failed build. |
| **build lock** | A file-based lock at `$XDG_CACHE_HOME/llm-runner/.build.lock` preventing concurrent build pipeline executions. |
| **toolchain** | The set of build tools required to compile llama.cpp: gcc, make, git, cmake, plus backend-specific compilers (dpcpp for SYCL, nvcc for CUDA). |
| **setup venv** | A dedicated Python virtual environment at `$XDG_CACHE_HOME/llm-runner/venv` for future Python package installations (post-MVP vLLM prep). Not mutated during serve. |
