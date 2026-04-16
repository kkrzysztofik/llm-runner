# CLI JSON Contracts - PRD M2 Build + Setup

## Scope

Defines machine-readable output contracts for M2 commands that support `--json`.

### Output Shapes

- `BuildArtifact`: Successful build provenance (FR-006.1)
- `ToolchainStatus`: Detected toolchain versions (FR-005.1)
- `VenvResult`: Venv creation result (FR-005.2)
- `DoctorRepairResult`: Doctor repair action result (FR-004.7)
- `ErrorDetail`: FR-005 structured error (all commands)

- `llm-runner build <backend> --json`
- `llm-runner setup --check --json`
- `llm-runner setup --json`
- `llm-runner doctor --repair --json`

All errors use existing FR-005 `ErrorDetail` shape and retain deterministic keys.

## Input Semantics: Confirmatory UX for Mutating Actions

- **`--yes` flag**: Required for `setup` (without `--check`) mutating actions. When omitted, command requires either interactive confirmation OR `--yes` flag. `doctor --repair` confirmation UX is implementation-defined (optional in M2); may support `--yes` but not required.
- **Behavior when omitted**: `setup` (without `--check`) requires explicit confirmation via `--yes` flag or interactive TUI prompt. Non-mutating actions (`--check`, `--dry-run`) do not require confirmation. In non-interactive mode without `--yes`, return `CONFIRMATION_REQUIRED` error. `doctor --repair` confirmation is optional in M2.
- **Error contract**: When `setup` mutating action is blocked due to missing confirmation, return FR-005 `ErrorDetail` with `error_code = "CONFIRMATION_REQUIRED"`, `failed_check = "missing_confirmation"`, `how_to_fix = "Run with --yes flag or confirm interactively"` in JSON output. `doctor --repair` may implement its own confirmation policy (implementation-defined).

## Command: `llm-runner doctor --repair --json`

### Doctor Repair Input

- `--repair`: Required flag to enable repair mode
- Optional: `--yes` (confirmation flag; optional in M2 per FR-004.7)

### Doctor Repair Success Output (`DoctorRepairResult`)

```json
{
  "action": "doctor_repair",
  "staging_dirs_cleared": [
    "/abs/path/.local/state/llm-runner/builds/sycl-failed"
  ],
  "successful_artifacts_preserved": [
    "/abs/path/src/llama.cpp/build/bin/llama-server-cuda"
  ],
  "lock_file_remediated": true,
  "warnings": []
}
```

### Doctor Repair Error Output

- `doctor --repair` confirmation policy is implementation-defined (optional in M2); may not require `--yes` flag.
- Lock file not removable returns FR-005 `ErrorDetail` with `error_code = "LOCK_REMOVAL_FAILED"`.
- Staging directory cleanup failure returns FR-005 `ErrorDetail` with `error_code = "STAGING_CLEANUP_FAILED"`.

---

## Command: `llm-runner build <backend> --json`

### Build Input

- `backend`: `sycl | cuda | both`
- Optional: `--dry-run`, `--retry-attempts <int>`, `--full-clone`, `--jobs <int>`
- **Deferred (post-MVP)**: `--build-order <csv>` — explicit build ordering (e.g., `sycl,cuda`) is post-MVP/deferred in M2

### Build Success Output (`BuildArtifact`)

**Single backend (`sycl` or `cuda`)**: Outputs single `BuildArtifact` JSON object.

**Both backends (`both`)**: Outputs **NDJSON stream** — one `BuildArtifact` per line in execution order
(SYCL first, then CUDA). Each line is a complete, valid JSON object. Consumer should process line-by-line.

```json
{
  "artifact_type": "llama-server",
  "backend": "sycl",
  "created_at": 1760000000.123,
  "git_remote_url": "https://github.com/ggerganov/llama.cpp",
  "git_commit_sha": "0123456789abcdef0123456789abcdef01234567",
  "git_branch": "master",
  "build_command": ["cmake", "-DGGML_SYCL=ON", ".."],
  "build_duration_seconds": 123.45,
  "exit_code": 0,
  "binary_path": "/abs/path/src/llama.cpp/build/bin/llama-server",
  "binary_size_bytes": 123456789,
  "build_log_path": "/abs/path/.local/state/llm-runner/build-logs/sycl.log",
  "failure_report_path": null
}
```

### Build Error Output

- Non-zero process exit code.
- JSON payload includes `error` as FR-005 `ErrorDetail` object.
- For lock contention: `error.error_code = "BUILD_LOCK_HELD"`.
- For missing toolchain: `error.error_code = "TOOLCHAIN_MISSING"`.

## Command: `llm-runner setup --check --json`

### Setup Check Success Output (`ToolchainStatus`)

```json
{
  "gcc": "13.2.0",
  "make": "4.3",
  "git": "2.43.0",
  "cmake": "3.29.2",
  "sycl_compiler": null,
  "cuda_toolkit": "12.4",
  "nvtop": "3.1.0"
}
```

### Setup Check Error Output

- Missing required tool(s) still returns structured FR-005 errors (single or list).
- For required-missing tools: `error.error_code = "TOOLCHAIN_MISSING"`.

## Command: `llm-runner setup --json`

### Setup Success Output (`VenvResult`)

```json
{
  "venv_path": "/abs/path/.cache/llm-runner/venv",
  "created": true,
  "reused": false,
  "activation_command": "source /abs/path/.cache/llm-runner/venv/bin/activate"
}
```

### Setup Error Output

- Corrupt venv returns FR-005 shape with `error.error_code = "VENV_CORRUPT"`.
- Missing Python interpreter returns FR-005 shape with `error.error_code = "PYTHON_NOT_FOUND"`.
- Missing confirmation (no `--yes` flag in non-interactive mode) returns FR-005 shape with `error.error_code = "CONFIRMATION_REQUIRED"`, `failed_check = "missing_confirmation"`, `how_to_fix = "Run with --yes flag or confirm interactively"`.

## Behavioral Guarantees

- JSON output is stable and key-order agnostic.
- Timestamp/path values are runtime-dependent but required keys are always present.
- `build both --json` emits NDJSON stream: one `BuildArtifact` per line in serialized execution order
  (SYCL first, then CUDA). Never parallel.
- Mutating actions (`setup` without `--check`) require `--yes` flag or interactive confirmation;
  in non-interactive mode without `--yes`, returns FR-005 `ErrorDetail` with `error_code = "CONFIRMATION_REQUIRED"`.
  `doctor --repair` confirmation UX is implementation-defined (optional in M2).

---

## Summary

| Command | Success Shape | Error Codes | Confirmation Required |
| --- | --------- | --------- | --------- |
| `build <backend> --json` | `BuildArtifact` (single) or NDJSON stream (both) | `BUILD_LOCK_HELD`, `TOOLCHAIN_MISSING`, `BUILD_FAILED` | No |
| `setup --check --json` | `ToolchainStatus` | `TOOLCHAIN_MISSING` | No |
| `setup --json` | `VenvResult` | `VENV_CORRUPT`, `PYTHON_NOT_FOUND`, `CONFIRMATION_REQUIRED` | Yes (`--yes` or interactive) |
| `doctor --repair --json` | `DoctorRepairResult` | `LOCK_REMOVAL_FAILED`, `STAGING_CLEANUP_FAILED` | Optional (implementation-defined in M2) |
