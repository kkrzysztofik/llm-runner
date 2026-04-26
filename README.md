# llm-runner

`llm-runner` is a local-first control plane for launching, validating, building,
and probing `llama.cpp` server instances on a mixed Intel SYCL + NVIDIA CUDA
workstation.

It ships a CLI for launch, dry-run, build, setup, doctor, smoke, and profile
flows, plus a Rich-based TUI for live logs and GPU telemetry.

## Features

- Launch one or two llama-server instances with explicit slot/port mapping
- Preview exact commands with deterministic dry-run output
- Run build and setup flows for SYCL and CUDA backends
- Inspect toolchain, lockfiles, reports, and profile staleness with doctor
- Smoke-test live servers and capture JSON reports
- Benchmark GPU performance and persist profile guidance
- Redact sensitive values in logs, artifacts, and reports

## Requirements

- Python 3.12+
- `uv`
- A local `llama.cpp` checkout managed under XDG cache by default
- Model files for the configured slots
- Intel and/or NVIDIA tooling for the backends you plan to use

## Install

```bash
uv sync --extra dev
```

Optional shell activation:

```bash
source .venv/bin/activate
```

## Configuration

The codebase reads these common environment variables:

- `LLAMA_CPP_ROOT` — path to the llama.cpp checkout (overrides the XDG cache default)
- `MODEL_SUMMARY_BALANCED`
- `MODEL_SUMMARY_FAST`
- `MODEL_QWEN35`
- `MODEL_QWEN35_BOTH`
- `XDG_CACHE_HOME`
- `XDG_STATE_HOME`
- `XDG_DATA_HOME`
- `XDG_RUNTIME_DIR`
- `LLM_RUNNER_RUNTIME_DIR`

Defaults place:

- llama.cpp source at `$XDG_CACHE_HOME/llm-runner/llama.cpp`
- SYCL/CUDA binaries under that source root at `build/bin/llama-server` and
  `build_cuda/bin/llama-server`
- venv at `$XDG_CACHE_HOME/llm-runner/venv`
- build provenance at `$XDG_STATE_HOME/llm-runner/builds`
- reports at `$XDG_DATA_HOME/llm-runner/reports`
- runtime locks/artifacts under `$LLM_RUNNER_RUNTIME_DIR` or
  `$XDG_RUNTIME_DIR/llm-runner`

## Usage

### Launch modes

```bash
uv run llm-runner summary-balanced [port]
uv run llm-runner summary-fast [port]
uv run llm-runner qwen35 [port]
uv run llm-runner both [summary_port qwen35_port]
```

Default ports:

- `summary-balanced` → `8080`
- `summary-fast` → `8082`
- `qwen35` → `8081`

### Dry-run

```bash
uv run llm-runner dry-run both
uv run llm-runner dry-run summary-balanced 8080
```

### Build

```bash
uv run llm-runner build sycl
uv run llm-runner build cuda
uv run llm-runner build both
```

Use `LLAMA_CPP_ROOT=/path/to/llama.cpp` or `--source-dir /path/to/llama.cpp`
to build from an explicit checkout instead of the XDG-managed default.

### Setup

```bash
uv run llm-runner setup check
uv run llm-runner setup venv
uv run llm-runner setup clean-venv --yes
```

### Doctor

```bash
uv run llm-runner doctor check
uv run llm-runner doctor repair --dry-run
```

### Smoke

```bash
uv run llm-runner smoke both
uv run llm-runner smoke slot summary-balanced
```

### Profile

```bash
uv run llm-runner profile summary-balanced balanced
uv run llm-runner profile qwen35 quality --json
```

## TUI

The legacy Rich TUI launcher is still available:

```bash
uv run python src/run_models_tui.py both
uv run python src/run_models_tui.py summary-balanced --port 8080
```

It shows live logs, configuration, and GPU stats for the configured slots.

## Safety Notes

- All servers bind to `127.0.0.1` by default
- Risky launches require explicit acknowledgement
- Sensitive environment values are redacted by key name (`KEY`, `TOKEN`,
  `SECRET`, `PASSWORD`, `AUTH`)
- Lockfiles and artifacts are written into runtime directories with strict
  permissions

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

See `QUICKSTART.md` for a guided first-run sequence.
