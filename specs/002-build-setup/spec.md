# Feature Specification: PRD M2 — Build Wizard + Setup Pipeline

**Feature Branch**: `002-build-setup`
**Created**: 2026-04-15
**Status**: Draft
**Input**: PRD.md v0.3 (M2 milestone), Spec 001 (M1 — completed)
**Milestone Scope**: M2 only — TUI build pipeline, toolchain diagnostics, setup venv, build provenance, failure reports. Not M0, M3, or M4.

## Clarifications

### Session 2026-04-15

- Q: Should SYCL builds always precede CUDA builds? → A: Default SYCL first (lower resource contention on multi-GPU systems); allow override via `--build-order cuda,sycl`.
- Q: What constitutes a "retryable" failure vs "fatal"? → A: Retryable = network timeout, transient build tool errors, disk-full-after-cleanup. Fatal = missing compiler, missing git, invalid git repo, permission denied on source dir. Configurable via `--retry-attempts N` (default: 3).
- Q: Should `setup` activate the venv in the current shell? → A: Create and report path; do NOT auto-activate. Print `source $XDG_CACHE_HOME/llm-runner/venv/bin/activate` instruction. No `--activate` flag in M2 (deferred to M4+ shell integration).
- Q: Where should built binaries be stored? → A: Follow existing `Config` pattern — Intel SYCL at `src/llama.cpp/build/bin/llama-server`, NVIDIA CUDA at `src/llama.cpp/build_cuda/bin/llama-server`. Provenance metadata at `$XDG_STATE_HOME/llm-runner/builds/<timestamp>-<backend>.json`.
- Q: Should sudo be requested per-action or once at start? → A: Per-action with `sudo -v` credential caching (5-minute TTL). Show which action requires sudo before executing. No `--sudo` CLI flag in M2 (deferred; setup focuses on venv + toolchain detection only).
- Q: Should `build both` continue on first backend failure? → A: No — fail fast on first error. Record which backends succeeded and failed in the result. `--continue-on-error` is post-MVP.
- Q: What is the git clone depth? → A: Default `--depth 1` for speed. Add `--full-clone` flag for debugging/reproducibility.
- Q: Should `make -j N` be used? → A: Yes, default to `os.cpu_count()`. Add `--jobs N` flag for override.
- Q: How should concurrent build attempts be handled? → A: File-based lock at `$XDG_CACHE_HOME/llm-runner/.build.lock`; second build attempt blocks with FR-005 error until first completes.
- Q: Should build pipeline validate git remote URL? → A: Yes — only `https://github.com/ggerganov/llama.cpp` allowed by default. Configurable via `Config.build_git_remote` for forks.
- Q: What happens on build interrupt (Ctrl+C)? → A: Clean up staging directory, preserve any already-successful artifacts, release build lock. Interrupted build does NOT produce a failure report (operator cancelled intentionally).
- Q: When the source directory already exists with a git repo, what should the clone stage do? → A: Run `git fetch origin && git reset --hard origin/<branch>` to force the working tree to the remote tip (incremental update). Re-clone only if source directory does not exist.
- Q: How should cmake flags be specified for each backend? → A: Derive from `Config`/`BuildConfig` dataclass fields — no hardcoded flag strings. Flag names (`GGML_SYCL`, `GGML_CUDA`, compiler paths) defined as class constants on `BuildPipeline` or `BuildConfig`. The data model drives everything, consistent with M1 pattern.
- Q: Should `build` and `setup` support `--json` machine-readable output? → A: Yes — `--json` on both `build` and `setup`. `build --json` outputs `BuildArtifact` JSON; `setup --json` outputs `ToolchainStatus` JSON. Consistent with M1's `dry-run --json` pattern.
- Q: What platform scope for toolchain install hints? → A: Debian-derivatives only (apt-get). Matches target hardware (Ubuntu/Debian with Intel Arc SYCL + NVIDIA CUDA). Other platforms deferred post-MVP.
- Q: Should M2 define performance targets (max build time, preflight duration)? → A: No explicit targets — build times are hardware-dependent and non-deterministic. Success criteria focus on correctness, not speed.
- Q: Should cmake flags use `LLAMA_*` or `GGML_*` naming convention? → A: Use `GGML_SYCL` / `GGML_CUDA` (current upstream convention as of 2025+). The `LLAMA_*` names are deprecated in llama.cpp's cmake system.
- Q: How should build-stage progress be calculated during `make -j N`? → A: Parse make's `[N/M]` parallel output pattern to calculate real progress percentage (N÷M × 100). Reset progress to 0% on retry.
- Q: Should `setup --check` combine toolchain and venv integrity checks? → A: No — keep them separate. `setup --check` shows toolchain status only; venv integrity is checked when running `setup` (without `--check`). `--json` outputs the relevant structure for the active mode (`ToolchainStatus` for `--check`, venv path/integrity result for default mode).
- Q: How should per-backend required tool lists be defined? → A: Explicit required-tools list: define `SYCL_REQUIRED_TOOLS` and `CUDA_REQUIRED_TOOLS` as module-level constants in `toolchain.py`. `ToolchainHint.required_for` is derived from these constants, not the other way around. The authoritative "which tools are required" lives in one place.
- Q: Should the `install` pipeline stage remain separate? → A: No — remove `install` stage. Binary verification is merged into the `build` stage (verify binary exists after `make` completes). Pipeline stages become: preflight → clone → configure → build → provenance (5 stages, not 6).

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
   pipeline retries up to `--retry-attempts` times (default 3) with exponential backoff.
5. **Given** the build fails after compilation starts, **When** the failure is processed, **Then** a
   failure report directory is written under `~/.local/share/llm-runner/reports/<timestamp>/` with
   redacted build output and error details.
6. **Given** a build is already in progress, **When** a second build is attempted, **Then** the second
   attempt is blocked with FR-005 error `failed_check=build_lock_held`.
7. **Given** `llm-runner build sycl --dry-run` is run, **When** preflight checks execute, **Then**
   only preflight results are displayed without starting the build.
8. **Given** `llm-runner build both` with `--build-order cuda,sycl`, **When** the build pipeline
   executes, **Then** CUDA build runs first, then SYCL.

**Flow Classification (User Story 1)**:

- **Primary flow**: Scenario 1 (single-backend build success).
- **Alternate flow**: Scenarios 2 and 8 (both-backends build, custom order).
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
   platform-specific install commands (e.g., `apt-get install cmake` or `brew install cmake`).
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

- Serialized build order: SYCL first by default; `--build-order` overrides.
- If source directory already exists with a git repo, clone stage runs `git fetch origin && git reset --hard origin/<branch>` to force the working tree to the remote tip (incremental update); full clone only when source directory does not exist.
- `build both` fails fast on first backend failure; successful backends retain their artifacts.
- Preflight checks run before any build execution; missing toolchain is always fatal (no retry).
- Network failures during git clone are retryable; compiler failures are fatal.
- Build lock prevents concurrent builds; lock is released on completion, failure, or interrupt.
- Ctrl+C during build: clean up staging, preserve successful artifacts, release lock, no failure report.
- Venv integrity check detects missing `pyvenv.cfg` or broken interpreter symlink.
- `setup` does NOT auto-activate the venv; only prints the activation command.
- `setup --check` is toolchain-only; it does NOT check venv integrity. Venv integrity is checked during `setup` (without `--check`).
- `setup` does NOT install system packages or run sudo in M2; only creates venv and checks toolchain.
- Provenance files are written atomically; partial writes are treated as integrity failures.
- Report rotation keeps the N most recent reports (default 50); oldest are deleted first.
- Build output truncation: logs are capped at 10 KiB in failure reports; full logs remain in the build directory.
- Cmake flags are derived from `BuildConfig` dataclass fields; flag names are class constants, not hardcoded strings.
- `build --json` outputs `BuildArtifact` JSON; `setup --json` outputs `ToolchainStatus` JSON.
- Toolchain install hints are Debian-derivatives only (apt-get); other platforms deferred post-MVP.
- No explicit performance targets; success criteria focus on correctness, not speed.
- `build --dry-run` runs preflight only; does not clone, configure, or build.
- Build-stage progress is calculated by parsing make's `[N/M]` parallel output pattern; if no `[N/M]` pattern is detected, progress remains at 0% until completion.
- `build --full-clone` uses `git clone` without `--depth 1` for full history.
- `build --jobs N` overrides `os.cpu_count()` for parallel make.
- Git remote URL must match `Config.build_git_remote` (default: `https://github.com/ggerganov/llama.cpp`);
  mismatched URLs are blocked with FR-005 error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-004.1**: System MUST provide a CLI `build` subcommand that accepts a backend argument
  (`sycl` | `cuda` | `both`) and executes the llama.cpp build pipeline for the specified backend(s).
  When `both` is specified, builds MUST run sequentially (never in parallel). Default build order is
  SYCL first; `--build-order` flag allows override. The `--json` flag MUST produce machine-readable
  `BuildArtifact` JSON output (consistent with M1's `dry-run --json` pattern).
- **FR-004.2**: Build pipeline MUST execute the following stages in order: preflight → clone →
  configure → build → provenance (5 stages). Each stage MUST report its status (`pending` | `running` |
  `success` | `failed`) and progress percentage. During the `build` stage, the pipeline MUST parse
  make's `[N/M]` parallel output pattern to calculate progress as `N / M × 100`. Progress MUST reset
  to 0% on retry. For other stages, progress is stage-level only (0.0 → 100.0 on completion). When
  the source directory already exists with a valid git repository, the clone stage MUST run
  `git fetch origin && git reset --hard origin/<branch>` to force the working tree to the remote tip,
  instead of a full clone. If the source directory does not exist, a full `git clone` is performed
  (shallow by default; use `--full-clone` to request a full, non-shallow clone).
- **FR-004.2a**: The configure stage MUST derive cmake flags from `BuildConfig` dataclass fields
  rather than hardcoded strings. Flag names (e.g., `GGML_SYCL`, `GGML_CUDA`, compiler paths) MUST
  be defined as class constants. The specific flag set per backend:
  - SYCL: `GGML_SYCL=ON`, `CMAKE_C_COMPILER=icx`, `CMAKE_CXX_COMPILER=icpx`
  - CUDA: `GGML_CUDA=ON`
  These are the current upstream cmake flags (the older `LLAMA_*` names are deprecated); additional flags may be added via `BuildConfig` fields.
- **FR-004.3**: Build pipeline MUST implement retry logic for transient failures (network timeout,
  transient build tool errors). Retry-attempt count is configurable via `--retry-attempts` (default 3).
  Exponential backoff delay starts at `Config.build_retry_delay` seconds (default 5.0). Missing
  toolchain components are NOT retryable and MUST fail immediately with FR-005 actionable error.
- **FR-004.4**: Build pipeline MUST enforce serialized execution via a file-based lock at
  `$XDG_CACHE_HOME/llm-runner/.build.lock`. A second build attempt while a lock is held MUST be
  blocked with FR-005 actionable error (`error_code=BUILD_LOCK_HELD`,
  `failed_check=build_lock_held`).
- **FR-004.5**: Build pipeline MUST support `--dry-run` flag that runs preflight checks only without
  executing any build stages. MUST support `--full-clone` flag that disables shallow clone.
  MUST support `--jobs N` flag that overrides parallel make job count.
- **FR-005.1**: System MUST provide a `setup --check` subcommand that detects installed toolchain
  components (gcc, make, git, cmake, sycl_compiler, cuda_toolkit, nvtop) and returns their version
  strings, or `None` for missing tools. Missing tools MUST produce FR-005 actionable errors with
  Debian-specific install instructions (apt-get only). The `--json` flag MUST produce machine-readable
  `ToolchainStatus` JSON output (consistent with M1's `dry-run --json` pattern). `setup --check`
  MUST NOT perform venv integrity checks — it is toolchain-only. Platform scope is
  Debian-derivatives only in M2; other platforms deferred post-MVP.
  **Non-Debian behavior**: When `setup --check` is run on a non-Debian system, the command MUST
  detect the platform and still perform toolchain detection, but MUST omit Debian-specific install
  hints. The `--json` output MUST include a `ToolchainStatus` JSON with the following shape:

  ```json
  {
    "platform": "<os>",
    "debianCompatible": false,
    "supported": false,
    "message": "Platform not supported in M2; install hints deferred post-MVP"
  }
  ```

  `setup --check` on a non-Debian platform MUST exit with code 0 (success) and MUST NOT emit
  FR-005 errors. FR-005 (`TOOLCHAIN_MISSING`) is reserved for explicit `--install` attempts or
  when a missing tool blocks a requested build. The `supported: false` field in `ToolchainStatus`
  communicates the platform limitation to callers without triggering an error exit.
- **FR-005.2**: System MUST provide a `setup` subcommand (without `--check`) that creates or reuses
  a virtual environment at `$XDG_CACHE_HOME/llm-runner/venv` (fallback `~/.cache/llm-runner/venv`).
  The venv MUST NOT be auto-activated; the activation command MUST be printed to stdout.
  Runtime `llm-runner` MUST NOT mutate this venv during serve operations. When running `setup`
  without `--check`, the system MUST also check venv integrity (FR-005.3) and report any issues.
  The `--json` flag MUST produce machine-readable output with venv path and integrity result.
- **FR-005.3**: System MUST detect corrupted venvs (missing `pyvenv.cfg`, broken interpreter symlink)
  and return FR-005 actionable error with `error_code=VENV_CORRUPT` and `how_to_fix` instructing
  removal and re-creation. Venv integrity is checked during `setup` (without `--check`), NOT during
  `setup --check` (which is toolchain-only).
- **FR-005.4**: Toolchain detection MUST use `subprocess.run` with timeout (5 seconds per tool) to
  query `--version` output. Detection failures (tool not found, timeout) MUST return `None` for that
  tool and generate FR-005 actionable error if the tool is required for the requested backend. The
  set of required tools per backend MUST be defined as module-level constants (`SYCL_REQUIRED_TOOLS`
  and `CUDA_REQUIRED_TOOLS`) in `toolchain.py`. `ToolchainHint.required_for` fields MUST be derived
  from these constants. Common tools (gcc, make, git, cmake) are required for both backends;
  backend-specific tools (dpcpp/icx/icpx for SYCL, nvcc for CUDA) are required only for their
  respective backend.
- **FR-006.1**: Every successful build MUST produce a provenance JSON file at
  `$XDG_STATE_HOME/llm-runner/builds/<timestamp>-<backend>.json` containing: `artifact_type`,
  `backend`, `created_at`, `git_remote_url`, `git_commit_sha`, `git_branch`, `build_command`,
  `build_duration_seconds`, `exit_code`, `binary_path`, `binary_size_bytes`, `build_log_path`.
- **FR-006.2**: Build artifacts MUST live at predictable paths determined by `Config`:
  Intel SYCL binary at `src/llama.cpp/build/bin/llama-server`, NVIDIA CUDA binary at
  `src/llama.cpp/build_cuda/bin/llama-server`. The system MUST NOT silently auto-build on launch.
- **FR-006.3**: Provenance files MUST be written atomically (write to temp file, then rename).
  If provenance write fails, the build MUST still be considered successful but a warning MUST be
  emitted indicating the provenance recording failure.
- **FR-018.1**: Build failures MUST produce a report directory at
  `~/.local/share/llm-runner/reports/<timestamp>/` containing: `build-artifact.json` (partial
  BuildArtifact with failure details), `build-output.log` (truncated to 10 KiB, secrets redacted),
  and `error-details.json` (exception type, message, and stack trace summary).
- **FR-018.2**: All build output in reports MUST be redacted using the existing `redact_sensitive()`
  pattern (KEY|TOKEN|SECRET|PASSWORD|AUTH → `[REDACTED]`). Report directories MUST use owner-only
  permissions (`0700`), files within MUST use `0600`.
- **FR-018.3**: Report rotation MUST maintain a maximum of `Config.build_max_reports` (default 50)
  report directories. When the limit is exceeded, the oldest reports (by directory timestamp) MUST be
  deleted first.

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

## Assumptions

- The target persona is a solo operator on the anchored workstation (Intel Arc + NVIDIA RTX 3090).
- M2 builds target `llama.cpp` `master` branch from the canonical GitHub repository.
- The operator has internet access for git clone (offline builds require pre-existing source).
- `sudo` operations are deferred from M2; `setup` focuses on venv creation and toolchain detection only.
- M1 slot-first infrastructure (Config, ServerConfig, ErrorDetail, ErrorCode, validators) is available
  and stable.
- The existing `Config.llama_cpp_root` path convention is used for source and build directories.
- Build artifacts (binaries) remain at paths defined by the existing llama.cpp build system.
- TUI build visualization is deferred to M4; M2 provides CLI-only build output.

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
