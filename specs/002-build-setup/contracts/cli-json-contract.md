# CLI JSON Contracts - PRD M2 Build + Setup

## Scope

Defines machine-readable output contracts for M2 commands that support `--json`.

- `llm-runner build <backend> --json`
- `llm-runner setup --check --json`
- `llm-runner setup --json`

All errors use existing FR-005 `ErrorDetail` shape and retain deterministic keys.

## Command: `llm-runner build <backend> --json`

### Input

- `backend`: `sycl | cuda | both`
- Optional: `--dry-run`, `--retry-attempts <int>`, `--build-order <csv>`, `--full-clone`, `--jobs <int>`

### Success Output (`BuildArtifact`)

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
  "build_log_path": "/abs/path/.cache/llm-runner/build-logs/sycl.log",
  "failure_report_path": null
}
```

### Error Output

- Non-zero process exit code.
- JSON payload includes `error` as FR-005 `ErrorDetail` object.
- For lock contention: `error.error_code = "BUILD_LOCK_HELD"`.
- For missing toolchain: `error.error_code = "TOOLCHAIN_MISSING"`.

## Command: `llm-runner setup --check --json`

### Success Output (`ToolchainStatus`)

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

### Error Output

- Missing required tool(s) still returns structured FR-005 errors (single or list).
- For required-missing tools: `error.error_code = "TOOLCHAIN_MISSING"`.

## Command: `llm-runner setup --json`

### Success Output (`VenvResult`)

```json
{
  "venv_path": "/abs/path/.cache/llm-runner/venv",
  "created": true,
  "reused": false,
  "activation_command": "source /abs/path/.cache/llm-runner/venv/bin/activate"
}
```

### Error Output

- Corrupt venv returns FR-005 shape with `error.error_code = "VENV_CORRUPT"`.
- Missing Python interpreter returns FR-005 shape with `error.error_code = "PYTHON_NOT_FOUND"`.

## Behavioral Guarantees

- JSON output is stable and key-order agnostic.
- Timestamp/path values are runtime-dependent but required keys are always present.
- `build both --json` emits one backend result at a time in serialized order; never parallel.
