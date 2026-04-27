# Quickstart — Validate PRD M2 Build Wizard + Setup Pipeline

## Scope & Compliance Notice

**CRITICAL:** This branch (`002-build-setup`) implements **PRD Milestone M2 only** — not the full PRD MVP.

**M2 Scope (Target Behavior):**
- TUI build pipeline for llama.cpp (FR-004)
- Toolchain diagnostics and actionable hints (FR-005)
- Setup venv creation and integrity checking (FR-005)
- Build provenance and artifact tracking (FR-006)
- Failure reports with redaction and rotation (FR-018)

**Deferred to Future Milestones:**
- M0: Documentation generation (FR-019)
- M1: Slot-first launch, dry-run, lockfiles (planned in Spec 001)
- M3: Profiling and presets (FR-007, FR-008, FR-009)
- M4: Smoke tests, TUI monitoring, shutdown, GGUF parsing, hardware ack
- Exit codes: `doctor` exit-code contract is already in scope (M1/MVP baseline); 002 quickstart references it rather than deferring.

**Do not claim full PRD completion.** This branch is a milestone delivery, not MVP completion.

## Prerequisites

- Python 3.12 environment set up (`uv sync --extra dev`)
- Feature branch: `002-build-setup`
- Spec inputs prepared in `specs/002-build-setup/spec.md`
- M1 infrastructure available and tested (441+ tests passing)

## 1) Run build pipeline preflight checks (TUI or CLI)

```bash
uv run llm-runner build sycl --dry-run
```

**Meaning**: Runs preflight toolchain checks for SYCL backend without starting the build.
TUI wizard is primary M2 interface; CLI provides parity for automation.

Expected outcomes:

- Output lists each required tool with its detected version or `MISSING` status
- If a tool is missing, output includes FR-005 actionable error with `error_code=TOOLCHAIN_MISSING`,
  `failed_check=<tool>_not_found`, `why_blocked`, and `how_to_fix` with install instructions
- If all tools are present, output confirms "Preflight checks passed"

## 2) Verify toolchain detection

```bash
uv run llm-runner setup --check
```

**Meaning:** Detects installed toolchain components and reports status.

Expected outcomes:

- Output shows toolchain status table with columns: Tool, Status, Version
- Present tools show version string (e.g., `cmake 3.28.0`)
- Missing tools show `MISSING` with platform-specific install hint
- Backend-specific requiredness is informational via `required_for` hints in error messages
- Common tools (gcc, make, git, cmake) are required for both backends

## 3) Run setup venv creation

```bash
uv run llm-runner setup --yes
```

**Meaning**: Creates or reuses the Python virtual environment for build tools.
M2 `setup` is venv lifecycle + toolchain checks; does NOT install packages (network not required).

**Expected outcomes:**

- First run creates venv at `$XDG_CACHE_HOME/llm-runner/venv`
- Output prints activation command: `source $XDG_CACHE_HOME/llm-runner/venv/bin/activate`
- Subsequent runs reuse existing venv and print "Venv already exists at <path>"
- Corrupted venv (missing pyvenv.cfg) produces FR-005 error with `error_code=VENV_CORRUPT`
- Without `--yes` in non-interactive mode, returns `CONFIRMATION_REQUIRED` error with actionable hint

## 4) Execute a single-backend build

```bash
uv run llm-runner build sycl
```

**Meaning:** Executes the full build pipeline for the SYCL backend.

Expected outcomes (with valid toolchain):

- Preflight checks pass
- Build stages execute sequentially: preflight → clone → configure → build → provenance
- Each stage reports status and progress
- On success: binary at `$XDG_CACHE_HOME/llm-runner/llama.cpp/build/bin/llama-server`
  (or `~/.cache/llm-runner/llama.cpp/build/bin/llama-server` when XDG_CACHE_HOME is unset;
  also checks `<LLAMA_CPP_ROOT>/build/bin/llama-server` / `<source-dir>/build/bin/llama-server` when overridden)
- On success: provenance JSON at `~/.local/state/llm-runner/builds/<timestamp>-sycl.json`
- Provenance contains: `artifact_type`, `backend`, `created_at`, `git_remote_url`,
  `git_commit_sha`, `git_branch`, `build_command`, `build_duration_seconds`, `exit_code`,
  `binary_path`, `binary_size_bytes`, `build_log_path`

Expected outcomes (with missing toolchain):

- Build blocked with FR-005 error at preflight stage
- Error includes `error_code=TOOLCHAIN_MISSING` and actionable `how_to_fix`

## 5) Verify serialized both-backends build

```bash
uv run llm-runner build both
```

**Meaning:** Builds SYCL then CUDA sequentially with per-target semantics.

**Expected outcomes:**

- SYCL build completes first (success or failure)
- CUDA build starts next regardless of SYCL outcome (independent status)
- Each backend tracks its own success/failure state independently
- Failed backends may be retried without re-running successful ones
- Both backends produce independent provenance records

## 6) Verify failure reporting

Trigger a build failure (e.g., with intentionally broken toolchain or source) and verify:

- Report directory created at `~/.local/share/llm-runner/reports/<timestamp>/`
- Directory contains `build-artifact.json`, `build-output.log`, `error-details.json`
- `build-output.log` is truncated to ≤10 KiB
- `build-output.log` has secrets redacted (`API_KEY=secret123` → `API_KEY=[REDACTED]`)
- Directory permissions: `0700`; file permissions: `0600`

## 7) Verify build lock behavior

Run two concurrent build attempts and verify:

- Second attempt is blocked with FR-005 error (`error_code=BUILD_LOCK_HELD`)
- Lock file at `$XDG_CACHE_HOME/llm-runner/.build.lock` contains PID and timestamp
- Stale lock (PID not running) requires manual remediation via `llm-runner doctor --repair` (no auto-clear in M2; future post-MVP may add auto-recovery)
- Lock is released after build completes (success or failure)

## 8) Verify offline-continue on network loss

Simulate network loss (e.g., disconnect internet) and verify:

- If local clone exists, `build` offers/allows offline continue path
- If no local clone exists, `build` fails with actionable diagnostics (FR-005)
- Error message indicates how to proceed (clone source first, then retry)

## 9) Verify mutating-action rotating logs

Run `uv run llm-runner setup --yes` and verify:

- Rotating log entry created with: command, timestamp, exit code, truncated output, redaction
- Log follows rotation policy (oldest entries deleted first when limit exceeded)
- Secrets in output are redacted (`API_KEY=secret123` → `API_KEY=[REDACTED]`)

Note: `doctor --repair` confirmation UX is implementation-defined (optional in M2 per FR-004.7); may or may not require `--yes`.

## 10) Run automated test suite

```bash
uv run pytest --tb=short -q
```

Expected outcomes:

- All existing M1 tests continue to pass
- New M2 tests for build pipeline, toolchain, venv, reports pass
- No subprocess calls in CI (all mocked)
- `uv run ruff check .` passes with no errors
- `uv run pyright` passes with no errors

## Quality Gates

Before considering M2 complete, all must pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov --cov-report=term-missing
uv run pip-audit
```
