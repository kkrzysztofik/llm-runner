# Phase 0 Research — PRD M2 Build Wizard + Setup Pipeline

## 1) Build serialization order

- **Decision**: Default SYCL first, then CUDA; `--build-order` override is deferred post-MVP (not active in M2).
- **Rationale**: SYCL builds typically have lower GPU resource contention on the anchored workstation (Arc B580 is secondary GPU). Building SYCL first avoids potential CUDA driver interference during Intel oneAPI compilation.
- **Alternatives considered**:
  - Always CUDA first (rejected: CUDA builds are longer; SYCL success is more fragile and should be validated first).
  - Parallel builds (rejected: PRD explicitly requires serialized builds for MVP; resource contention risk).
  - User-config-only order without default (rejected: poor UX for common case).

## 2) Retryable vs fatal failure classification

- **Decision**: Retryable = network timeout, transient build tool errors, disk-full-after-cleanup.
  Fatal = missing compiler, missing git, invalid git repo, permission denied on source dir.
- **Rationale**: Transient failures are expected in network-dependent operations and may succeed on retry.
  Missing system components cannot self-resolve and require operator action (FR-005 actionable error).
- **Alternatives considered**:
  - Retry all failures (rejected: wastes time on irrecoverable failures like missing compilers).
  - No retry at all (rejected: network flakiness is common and retrying is standard practice).
  - Classify by exit code ranges (rejected: exit codes vary across tools and are unreliable classifiers).

## 3) Setup venv activation policy

- **Decision**: `setup` creates/reuses venv and prints activation command; does NOT auto-activate.
- **Rationale**: Auto-activation requires shell integration (eval, source) that is fragile across shells
  (bash, zsh, fish) and can have unexpected side effects. Printing the activation command is safe and
  transparent. The venv is for future post-MVP use (vLLM prep); M2 runtime does not need it.
- **Alternatives considered**:
  - Auto-activate with `--activate` flag (rejected: requires shell `eval` integration; deferred).
  - Add venv to PATH via wrapper script (rejected: over-engineering for M2 scope).

## 4) Build artifact storage paths

- **Decision**: Follow existing `Config` convention — SYCL binary at `src/llama.cpp/build/bin/llama-server`,
  CUDA binary at `src/llama.cpp/build_cuda/bin/llama-server`. Provenance at
  `$XDG_STATE_HOME/llm-runner/builds/<timestamp>-<backend>.json`.
- **Rationale**: Preserves backward compatibility with existing `Config.llama_server_bin_intel` and
  `Config.llama_server_bin_nvidia` computed paths. Provenance in XDG_STATE_HOME follows the
  XDG Base Directory Specification for application state data.
- **Alternatives considered**:
  - Store binaries in XDG directories (rejected: breaks existing Config paths and run scripts).
  - Provenance alongside binaries (rejected: mixes build output with metadata; harder to rotate/clean).

## 5) Sudo handling in M2

- **Decision**: No `sudo` operations in M2. `setup` focuses on venv creation and toolchain detection
  only. Sudo-requiring package installation steps are deferred post-MVP; `doctor --repair`
  remains M2 and sudo-free.
- **Rationale**: Sudo requires interactive confirmation and credential caching that adds significant
  UX complexity. M2's primary value is the build pipeline; sudo-free setup is safer and simpler.
  Toolchain detection can tell users what to install without performing the installation.
- **Alternatives considered**:
  - `--sudo` flag per action (rejected: adds complexity without clear M2 value).
  - Automatic sudo for apt-get (rejected: security risk; requires per-action confirmation UX).

## 6) Build lock mechanism

- **Decision**: File-based lock at `$XDG_CACHE_HOME/llm-runner/.build.lock` containing PID and
  timestamp. Lock is acquired before build starts, released on completion/failure/interrupt.
  Stale locks (PID not running) require manual remediation via `doctor --repair` in M2.
- **Rationale**: File-based locks are simple, visible to operators, and survive process crashes
  (can be inspected and cleaned up manually). Stale lock detection uses the same pattern as M1
  lockfile ownership checks.
- **Alternatives considered**:
  - fcntl file locking (rejected: not portable; harder to debug).
  - No lock / allow concurrent builds (rejected: PRD requires serialized builds; concurrent builds
    would risk resource contention and build corruption).

## 7) Git remote URL validation

- **Decision**: Validate git remote URL against `Config.build_git_remote` (default:
  `https://github.com/ggerganov/llama.cpp`). Mismatched URLs are blocked with FR-005 error.
  Configurable for forks.
- **Rationale**: Prevents accidental builds from untrusted sources. The canonical repository is
  well-known and trusted. Fork support via config allows advanced users without compromising safety.
- **Alternatives considered**:
  - No URL validation (rejected: security risk; could build from malicious source).
  - Hardcoded URL only (rejected: prevents fork usage; overly rigid).

## 8) Build output truncation and redaction in reports

- **Decision**: Build output in failure reports is truncated to 10 KiB. Full build logs remain in
  the build directory. Redaction uses existing `redact_sensitive()` function from M1.
- **Rationale**: 10 KiB captures enough context (last ~200 lines) for most build failures while
  keeping report size manageable. Full logs in the build directory provide complete output for
  detailed debugging. Redaction is consistent with M1 FR-007 policy.
- **Alternatives considered**:
  - No truncation (rejected: build logs can be hundreds of MB; report size unbounded).
  - Fixed line count instead of byte size (rejected: lines vary in length; byte cap is more predictable).

## 9) Shallow clone vs full clone

- **Decision**: Default to `git clone --depth 1` for speed. `--full-clone` flag disables shallow
  clone for debugging or reproducibility needs.
- **Rationale**: Shallow clones are significantly faster and smaller. Full history is rarely needed
  for build purposes. The `--full-clone` escape hatch covers edge cases.
- **Alternatives considered**:
  - Always full clone (rejected: slow, wasteful for most use cases).
  - `--depth 50` compromise (rejected: arbitrary; `--depth 1` is sufficient for building from tip).

## 10) Provenance write failure handling

- **Decision**: If provenance JSON write fails, the build is still considered successful (binary
  exists), but a warning is emitted. Provenance is a best-effort observability feature, not a
  build-critical operation.
- **Rationale**: A successful build that produces a working binary is still valuable even if
  provenance recording fails (e.g., XDG_STATE_HOME not writable). Warning the operator allows
  them to fix the path issue without losing the build result.
- **Alternatives considered**:
  - Fail the build on provenance write failure (rejected: too strict; working binary is still useful).
  - Silently skip provenance (rejected: operator should know provenance was not recorded).
