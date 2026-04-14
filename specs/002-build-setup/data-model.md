# Data Model — PRD M2 Build Wizard + Setup Pipeline

## Entity: BuildConfig

- **Fields**:
  - `backend: Literal["sycl", "cuda"]`
  - `source_dir: Path` — llama.cpp source root (default: `Config.llama_cpp_root`)
  - `build_dir: Path` — cmake build directory (e.g., `src/llama.cpp/build` or `src/llama.cpp/build_cuda`)
  - `output_dir: Path` — binary output directory (e.g., `src/llama.cpp/build/bin`)
  - `git_remote_url: str` — upstream repository URL (default: `Config.build_git_remote`)
  - `git_branch: str` — branch to checkout (default: `Config.build_git_branch` = `"master"`)
  - `retry_attempts: int` — max retry count for transient failures (default: `Config.build_retry_attempts` = 3)
  - `retry_delay: float` — initial backoff delay in seconds (default: `Config.build_retry_delay` = 5.0)
  - `shallow_clone: bool` — use `--depth 1` (default: True)
  - `jobs: int` — parallel make job count (default: `os.cpu_count()`)
- **Validation rules**:
  - `backend` must be `"sycl"` or `"cuda"`
  - `retry_attempts` must be >= 0
  - `retry_delay` must be > 0
  - `jobs` must be >= 1
  - `git_remote_url` must match `Config.build_git_remote` unless explicitly overridden

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
  - `binary_path: str` — absolute path to the produced binary
  - `binary_size_bytes: int | None` — file size of the binary, or None if not produced
  - `build_log_path: Path` — path to the full build log
  - `failure_report_path: Path | None` — path to failure report directory, or None on success
- **Validation rules**:
  - `exit_code` of 0 implies `binary_path` exists and `binary_size_bytes` is not None
  - `exit_code` != 0 implies `failure_report_path` is not None
  - `git_commit_sha` must be a 40-character hex string
  - `build_duration_seconds` must be >= 0

## Entity: BuildProgress

- **Fields**:
  - `stage: Literal["preflight", "clone", "configure", "build", "install", "provenance"]`
  - `status: Literal["pending", "running", "success", "failed"]`
  - `message: str` — human-readable progress description
  - `progress_percent: float` — 0.0 to 100.0
  - `retries_remaining: int` — remaining retry attempts for current stage
- **Validation rules**:
  - `progress_percent` must be between 0.0 and 100.0
  - `retries_remaining` must be >= 0
  - Stage ordering: preflight → clone → configure → build → install → provenance

## Entity: ToolchainStatus

- **Fields**:
  - `gcc: str | None` — detected version string (e.g., "11.4.0") or None
  - `make: str | None` — detected version string or None
  - `git: str | None` — detected version string or None
  - `cmake: str | None` — detected version string or None
  - `sycl_compiler: str | None` — detected oneAPI version (e.g., "oneAPI 2024.1") or None
  - `cuda_toolkit: str | None` — detected CUDA toolkit version (e.g., "12.3") or None
  - `nvtop: str | None` — detected version string or None
- **Validation rules**:
  - Version strings must be non-empty when not None
  - None indicates the tool was not found or timed out during detection

## Entity: BuildLock

- **Fields**:
  - `pid: int` — PID of the process holding the lock
  - `started_at: float` — Unix timestamp when lock was acquired
  - `backend: str` — backend being built (e.g., "sycl" or "cuda")
- **Validation rules**:
  - Lock file at `$XDG_CACHE_HOME/llm-runner/.build.lock`
  - Lock is stale if PID is not running (auto-clear)
  - Lock file permissions: `0600`
  - Lock directory permissions: `0700`

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
  - Build output MUST be redacted using `redact_sensitive()`
  - Maximum report count: `Config.build_max_reports` (default 50)

## Entity: ToolchainHint

- **Fields**:
  - `tool_name: str` — name of the missing tool (e.g., "cmake", "dpcpp")
  - `install_command: str` — apt-get command for Debian-derivatives (M2 scope; other platforms deferred)
  - `install_url: str` — documentation URL for manual installation
  - `required_for: list[Literal["sycl", "cuda"]]` — which backends require this tool
- **Validation rules**:
  - At least one install instruction must be non-empty
  - `required_for` must not be empty

## Config Extensions (added to existing Config dataclass)

- **New fields**:
  - `xdg_cache_base: str` — `$XDG_CACHE_HOME` or `~/.cache`
  - `xdg_state_base: str` — `$XDG_STATE_HOME` or `~/.local/state`
  - `xdg_data_base: str` — `$XDG_DATA_HOME` or `~/.local/share`
  - `build_git_remote: str` — default: `"https://github.com/ggerganov/llama.cpp"`
  - `build_git_branch: str` — default: `"master"`
  - `build_retry_attempts: int` — default: 3
  - `build_retry_delay: float` — default: 5.0
  - `build_max_reports: int` — default: 50
  - `build_output_truncate_bytes: int` — default: 10240 (10 KiB)
- **Computed paths** (in `__post_init__`):
  - `venv_path: Path` — `$xdg_cache_base/llm-runner/venv`
  - `builds_dir: Path` — `$xdg_state_base/llm-runner/builds`
  - `reports_dir: Path` — `$xdg_data_base/llm-runner/reports`
  - `build_lock_path: Path` — `$xdg_cache_base/llm-runner/.build.lock`

## ErrorCode Extensions (added to existing ErrorCode StrEnum)

- **New values**:
  - `TOOLCHAIN_MISSING = "TOOLCHAIN_MISSING"` — a required build tool is not installed
  - `VENV_NOT_FOUND = "VENV_NOT_FOUND"` — setup venv does not exist (informational, not blocking)
  - `VENV_CORRUPT = "VENV_CORRUPT"` — setup venv exists but is unusable
  - `PYTHON_NOT_FOUND = "PYTHON_NOT_FOUND"` — Python interpreter not found for venv creation
  - `BUILD_FAILED = "BUILD_FAILED"` — build subprocess exited non-zero
  - `BUILD_LOCK_HELD = "BUILD_LOCK_HELD"` — another build is already in progress
  - `GIT_CLONE_FAILED = "GIT_CLONE_FAILED"` — git clone failed (may be retryable)
  - `GIT_CHECKOUT_FAILED = "GIT_CHECKOUT_FAILED"` — git checkout failed
  - `REPORT_WRITE_FAILURE = "REPORT_WRITE_FAILURE"` — failure report could not be written
