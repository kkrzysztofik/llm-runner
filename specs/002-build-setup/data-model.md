# Data Model — PRD M2 Build Wizard + Setup Pipeline

## Entity: BuildConfig

- **Fields**:
  - `backend: Literal["sycl", "cuda"]` — M2 backend literals; maps to PRD artifact labels: `sycl` → `intel-sycl`, `cuda` → `nvidia-cuda`
  - `source_dir: Path` — llama.cpp source root (default: `Config.llama_cpp_root`)
  - `build_dir: Path` — cmake build directory (e.g., `src/llama.cpp/build` or `src/llama.cpp/build_cuda`)
  - `output_dir: Path` — binary output directory (e.g., `src/llama.cpp/build/bin`)
  - `git_remote_url: str` — upstream repository URL (default: `Config.build_git_remote`)
  - `git_branch: str` — branch to checkout (default: `Config.build_git_branch` = `"master"`)
  - `retry_attempts: int` — max retry count for transient failures (default: `Config.build_retry_attempts` = 3)
  - `retry_delay: float` — initial backoff delay in seconds (default: `Config.build_retry_delay` = 5.0)
  - `shallow_clone: bool` — use `--depth 1` (default: True)
  - `jobs: int` — parallel make job count (default: `os.cpu_count()`)
  - **Class constants**: `GGML_SYCL = "GGML_SYCL"`, `GGML_CUDA = "GGML_CUDA"`, `CMAKE_C_COMPILER = "icx"`, `CMAKE_CXX_COMPILER = "icpx"` — CMake flag names follow `GGML_*` convention matching upstream; version policy documented below.
- **Validation rules**:
  - `backend` must be `"sycl"` or `"cuda"`
  - `retry_attempts` must be >= 0
  - `retry_delay` must be > 0
  - `jobs` must be >= 1
  - `git_remote_url` must match `Config.build_git_remote` unless explicitly overridden
  - CMake version policy: minimum version is `CMAKE_MINIMUM_VERSION` (3.24.0). Version comparison uses tuple comparison with normalized suffix handling (e.g., "3.24.0" >= (3, 24, 0)).

## Entity: BuildArtifact

- **Fields**:
  - `artifact_type: Literal["llama-server"]`
  - `backend: Literal["sycl", "cuda"]`
  - `created_at: float` — Unix timestamp of build completion
  - `git_remote_url: str` — URL of the git repository
  - `git_commit_sha: str` — tip SHA of the checked-out commit
  - `git_branch: str` — branch name
  - `build_command: list[str]` — the cmake/make command that was executed
  - `build_duration_seconds: float` — wall-clock build time
  - `exit_code: int` — subprocess exit code (0 = success)
  - `binary_path: Path` — absolute path to the produced binary
  - `binary_size_bytes: int | None` — file size of the binary, or None if not produced
  - `build_log_path: Path` — path to the full build log
  - `failure_report_path: Path | None` — path to failure report directory, or None on success
- **Validation rules**:
  - `exit_code` of 0 implies `binary_path` exists and `binary_size_bytes` is not None
  - `exit_code` != 0 implies `failure_report_path` is not None
  - `git_commit_sha` must be a 40-character hex string
  - `build_duration_seconds` must be >= 0
  - Provenance write failure does NOT fail the build; build remains successful with warning logged.

## Entity: BuildProgress

- **Fields**:
  - `stage: Literal["preflight", "clone", "configure", "build", "provenance"]`
  - `status: Literal["pending", "running", "success", "failed", "skipped", "retrying"]`
  - `message: str` — human-readable progress description
  - `progress_percent: float` — 0.0 to 100.0
  - `retries_remaining: int` — remaining retry attempts for current stage
- **Validation Rules**:
  - `progress_percent` must be between 0.0 and 100.0
  - `retries_remaining` must be >= 0
  - Stage literal set: exactly one of `preflight`, `clone`, `configure`, `build`, `provenance`
  - Status allowed values: exactly one of `pending`, `running`, `success`, `failed`, `skipped`, `retrying`
  - Stage/status consistency: each stage transitions `pending` → `running` → (`success`|`failed`); pipeline terminates on first `failed`, or after `provenance=success`
  - `skipped` is a terminal status (no further transitions allowed)
  - `retrying` is treated like `running` for transition purposes (can transition to `success` or `failed`)
- **Behavioral Notes**:
  - Stage ordering: preflight → clone → configure → build → provenance
  - The `build` stage includes binary verification after `make` completes
  - During `build` stage, `progress_percent` is derived from make's `[N/M]` output pattern (N÷M×100)
  - If make output contains no `[N/M]` pattern, `progress_percent` remains 0.0 until stage completion
  - Progress resets to 0.0 on retry (exponential backoff applies)
  - Pipeline terminates on first `failed` stage, OR continues to next backend in `build both` mode with per-target retry semantics per spec.md Edge Cases.

## Entity: ToolchainStatus

- **Fields**:
  - `gcc: str | None` — detected version string (e.g., "11.4.0") or None
  - `make: str | None` — detected version string or None
  - `git: str | None` — detected version string or None
  - `cmake: str | None` — detected version string or None
  - `sycl_compiler: str | None` — detected oneAPI version (e.g., "oneAPI 2024.1") or None
  - `cuda_toolkit: str | None` — detected CUDA toolkit version (e.g., "12.3") or None
  - `nvtop: str | None` — detected runtime diagnostics tool version or None (informational only)
- **Runtime diagnostic tools**: `nvtop` is a runtime diagnostic tool, not a build-time requirement. If detected, include in status but do not list as required.
- **Validation rules**:
  - Version strings must be non-empty when not None
  - None indicates the tool was not found or timed out during detection
- **Timeout policy (FR-005.4)**: Tool detection uses `subprocess.run` with **30s default timeout** (configurable via `Config.toolchain_timeout_seconds`). Timeout or non-zero exit returns `None` for that tool.

## Entity: BuildLock

- **Fields**:
  - `pid: int` — PID of the process holding the lock (positive integer)
  - `started_at: float` — Unix timestamp when lock was acquired (not in future)
  - `backend: Literal["sycl", "cuda"]` — backend being built
- **Validation Rules**:
  - `pid` must be positive integer (>0)
  - `started_at` must not be in future (relative to current time)
  - `backend` must be `"sycl"` or `"cuda"`
- **File Format & Safety Notes**:
  - JSON shape: `{"pid": int, "started_at": float, "backend": str}`
  - Lock file path: `$XDG_CACHE_HOME/llm-runner/.build.lock`
  - Lock file permissions: `0600` (owner read/write only)
  - Lock directory permissions: `0700`
  - Stale-lock check: verify process existence via PID; safety consideration: only clear if same user owns the lock file

## Entity: FailureReport

- **Fields**:
  - `report_dir: Path` — directory containing report files
  - `timestamp: str` — ISO 8601 timestamp used as directory name
  - `build_artifact_json: Path` — partial BuildArtifact JSON
  - `build_output_log: Path` — truncated, redacted build output (max 10 KiB)
  - `error_details_json: Path` — exception type, message, and stack trace summary
- **Validation rules**:
  - Report directory at `~/.local/share/llm-runner/reports/<timestamp>/`
  - Directory permissions: `0700`
  - File permissions: `0600`
  - Build output MUST redact obvious secret patterns (environment variables or values containing KEY/TOKEN/SECRET/PASSWORD/AUTH etc.)
  - Path/PII redaction is optional or implementation-defined hardening (not a default mandatory requirement)
  - Maximum report count: `Config.build_max_reports` (default 50)
- **Rotation Policy**:
  - Deterministic rotation: oldest-first by directory name (timestamp-based)
  - Preserve failure evidence: keep all reports until rotation exceeds max count
  - Deletion is atomic per report directory

## Entity: MutatingActionLogEntry

- **Fields**:
  - `command: list[str]` — the exact command that was executed
  - `timestamp: float` — Unix timestamp of execution start
  - `exit_code: int | None` — subprocess exit code (None if not yet completed or not applicable)
  - `truncated_output: str` — truncated stdout/stderr (max 10 KiB)
  - `redaction_applied: bool` — whether redaction was applied to output
- **Usage**:
  - Entries are appended to rotating log in `reports.py`
  - Log file path: `$XDG_STATE_HOME/llm-runner/mutating_actions.log`
  - Rotation policy: count-based rotation (implementation-defined max, configurable via `Config.build_max_reports`)

## Entity: ToolchainHint

- **Fields**:
  - `tool_name: str` — name of the missing tool (e.g., "cmake", "dpcpp")
  - `install_command: str` — apt-get command for Debian-derivatives (M2 scope; other platforms deferred)
  - `install_url: str` — documentation URL for manual installation
  - `required_for: list[Literal["sycl", "cuda"]]` — which backends require this tool (derived from module constants)
- **Validation rules**:
  - At least one of `install_command` or `install_url` must be non-empty
  - `required_for` must not be empty
  - `required_for` values SHOULD align with `SYCL_REQUIRED_TOOLS` and `CUDA_REQUIRED_TOOLS` constants (authoritative source)

## Module Constants: Required Tool Lists

- **`SYCL_REQUIRED_TOOLS: frozenset[str]`** — tools required for SYCL backend builds:
  - Common: `gcc`, `make`, `git`, `cmake` (>= 3.24)
  - SYCL-specific: `dpcpp`, `icx`, `icpx` (oneAPI compilers)
- **`CUDA_REQUIRED_TOOLS: frozenset[str]`** — tools required for CUDA backend builds:
  - Common: `gcc`, `make`, `git`, `cmake` (>= 3.24)
  - CUDA-specific: `nvcc` (CUDA toolkit compiler)
- **`CMAKE_MINIMUM_VERSION: tuple[int, int, int]`** — `(3, 24, 0)` — minimum CMake version policy (upstream llama.cpp baseline; project may enforce stricter versioning)
  - **Version parsing**: Compare as tuple of integers; normalize suffix handling (e.g., "3.24.0" → (3, 24, 0), "3.24.0-alpha" → (3, 24, 0))
  - **Comparison**: `(3, 29, 2) >= (3, 24, 0)` evaluates to True
- These constants are the authoritative source for "which tools are required per backend".
  `ToolchainHint.required_for` is derived from these, not the other way around.

## Config Extensions (added to existing Config dataclass)

- **New fields**:
  - `xdg_cache_base: str` — `$XDG_CACHE_HOME` or `~/.cache` if unset
  - `xdg_state_base: str` — `$XDG_STATE_HOME` or `~/.local/state` if unset
  - `xdg_data_base: str` — `$XDG_DATA_HOME` or `~/.local/share` if unset
  - `build_git_remote: str` — default: `"https://github.com/ggerganov/llama.cpp"`
  - `build_git_branch: str` — default: `"master"`
  - `build_retry_attempts: int` — default: 3
  - `build_retry_delay: float` — default: 5.0
  - `build_max_reports: int` — default: 50
  - `build_output_truncate_bytes: int` — default: 10240 (10 KiB)
  - `toolchain_timeout_seconds: int` — default: 30 (seconds for tool detection subprocess timeout, per FR-005.4)
- **Computed paths** (in `__post_init__`):
  - `venv_path: Path` — `$xdg_cache_base/llm-runner/venv`
  - `builds_dir: Path` — `$xdg_state_base/llm-runner/builds`
  - `reports_dir: Path` — `~/.local/share/llm-runner/reports` (fixed M2 path per spec/PRD)
  - `build_lock_path: Path` — `$xdg_cache_base/llm-runner/.build.lock`

## Entity: VenvResult

- **Fields**:
  - `venv_path: Path` — absolute path to virtual environment
  - `created: bool` — `True` if venv was created in this run
  - `reused: bool` — `True` if existing venv was reused
  - `activation_command: str` — shell command to activate venv (e.g., `source /path/to/venv/bin/activate`)
- **Validation rules**:
  - `venv_path` must be absolute path
  - Exactly one of `created`/`reused` must be `True`
  - `activation_command` must be non-empty string

## ErrorCode Extensions (added to existing ErrorCode StrEnum)

- **New values**:
  - `PREFLIGHT_FAILURE = "PREFLIGHT_FAILURE"` — preflight check failed (generic preflight error)
  - `TOOLCHAIN_MISSING = "TOOLCHAIN_MISSING"` — a required build tool is not installed
  - `VENV_NOT_FOUND = "VENV_NOT_FOUND"` — setup venv does not exist (informational warning, not blocking)
  - `VENV_CORRUPT = "VENV_CORRUPT"` — setup venv exists but is unusable
  - `PYTHON_NOT_FOUND = "PYTHON_NOT_FOUND"` — Python interpreter not found for venv creation
  - `BUILD_FAILED = "BUILD_FAILED"` — build subprocess exited non-zero
  - `BUILD_LOCK_HELD = "BUILD_LOCK_HELD"` — another build is already in progress
  - `GIT_CLONE_FAILED = "GIT_CLONE_FAILED"` — git clone failed (may be retryable)
  - `GIT_CHECKOUT_FAILED = "GIT_CHECKOUT_FAILED"` — git checkout failed
  - `REPORT_WRITE_FAILURE = "REPORT_WRITE_FAILURE"` — failure report could not be written
  - `TOOL_VERSION_MISMATCH = "TOOL_VERSION_MISMATCH"` — tool version incompatible with requirements
  - `CMAKE_INCOMPATIBLE = "CMAKE_INCOMPATIBLE"` — CMake version below minimum required
