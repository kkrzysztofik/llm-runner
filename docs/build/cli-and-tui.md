# CLI and TUI build interfaces

## CLI

**Command:** `uv run llm-runner build <sycl|cuda|both> [options]`

**Handler:** `llama_cli/commands/build.py` → `BuildPipeline` or sequential single-backend runs for `both`.

### Examples

```bash
uv run llm-runner build sycl
uv run llm-runner build cuda -j 16
uv run llm-runner build both --dry-run
uv run llm-runner build sycl --json
LLAMA_CPP_ROOT=/path/to/llama.cpp uv run llm-runner build cuda
uv run llm-runner build sycl --source-dir /path/to/llama.cpp --no-update-sources
uv run llm-runner build sycl --git-commit abc1234def
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `backend` | (required) | `sycl`, `cuda`, or `both` |
| `--source-dir` | `$LLAMA_CPP_ROOT` or XDG cache clone | llama.cpp root |
| `--build-dir` | `build` / `build_cuda` under source | If set, actual dir is `<build-dir>/<backend>` |
| `--output-dir` | `$XDG_STATE_HOME/llm-runner/builds` | Provenance parent; uses `<output-dir>/<backend>/` |
| `--git-remote` | `https://github.com/ggerganov/llama.cpp.git` | Clone URL |
| `--git-branch` | `master` | Branch to clone/checkout |
| `--no-shallow-clone` | shallow on | Full history clone |
| `--no-update-sources` | update on | Skip fetch; may skip configure if cache exists |
| `--git-commit` | none | Pin SHA after clone/update |
| `-j`, `--jobs` | `os.cpu_count()` | Parallel compile jobs |
| `--retry-attempts` | `2` | Per-stage retries (CLI; library default 3) |
| `--retry-delay` | `5` | Base backoff seconds |
| `--dry-run` | off | Each stage prints “Would run …” without subprocess side effects (lock skipped) |
| `--json` | off | Emit artifact JSON on success or error array on failure |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All requested backends succeeded |
| `1` | One or more backends failed |
| `130` | Keyboard interrupt |

### `both` behavior

Loops backends in order: **SYCL**, then **CUDA**. Each iteration:

1. Resolves `build_dir` and `output_dir` for that backend.
2. Constructs `BuildConfig` and `BuildPipeline`.
3. Calls `pipeline.run()` (full lock + five stages).

Prints per-backend summary (binary path, size, duration, commit, log path).

## TUI

**Components:**

- `llama_cli/tui/components/build.py` — `BuildModalScreen` wizard UI
- `llama_cli/tui/controller.py` — `_run_build_background`, `run_build_for_backend`

**Orchestration:** `llama_manager/build_pipeline/orchestration.py`

```python
run_build_for_backend("sycl" | "cuda", config=Config(), progress_callback=..., dry_run=...)
```

Paths match CLI defaults:

- `source_dir = Path(config.llama_cpp_root)`
- `build_dir = source / ("build_cuda" if cuda else "build")`
- `output_dir = config.builds_dir / backend`

The controller runs selected backends **sequentially** in a worker thread, forwarding `BuildProgress` (including compile `output_line`) to the modal. It keeps a reference to `BuildPipeline` for cancellation/signal handling.

Wizard dry-run sets `pipeline.dry_run = True` on the same code path.

## `setup` vs `build`

| Command | Role |
|---------|------|
| `llm-runner setup check` | Toolchain table; FR-005 hints for missing tools |
| `llm-runner setup venv` | Create `$XDG_CACHE_HOME/llm-runner/venv` (Python env for llm-runner, not llama.cpp compile) |
| `llm-runner build …` | Full pipeline including clone/configure/compile |

Run `setup check` before first `build` on a new machine.

## Dry-run semantics

`--dry-run` sets `BuildPipeline.dry_run = True`:

| Stage | Dry-run behavior |
|-------|------------------|
| preflight | **Runs** real toolchain detection |
| clone | Prints would-run clone command; no git |
| configure | Prints would-run cmake; no cmake |
| build | Prints would-run `cmake --build`; no compile |
| finalize | Skipped when build stage did not actually compile |

So dry-run is **not** preflight-only; it walks all stages with simulated clone/configure/build. Lock acquisition is skipped.

## JSON output shape

**Success:**

```json
{
  "success": true,
  "artifacts": [ { "...BuildArtifact fields as strings..." } ]
}
```

**Failure:**

```json
{
  "success": false,
  "errors": [
    {
      "backend": "sycl",
      "error": "...",
      "progress": { "stage": "configure", "status": "failed", "message": "..." },
      "build_log_path": "...",
      "failure_report_path": "..."
    }
  ]
}
```

## Status inspection (TUI / automation)

`get_build_status(BuildBackend, config)` in `build_pipeline/status.py`:

- Reads `builds_dir/<backend>/build-artifact.json`
- Probes git remote/branch in `source_dir`
- Falls back to version probe on default `llama_server_bin_*` if no JSON

Used to populate build panels without re-running the pipeline. The TUI Build Wizard
fetches SYCL and CUDA status on a background worker (parallel probes) and refreshes
per-backend status cards in place after the modal is shown. Each card shows a
**Current** or **Needs update** badge (or **Missing** when sources are absent) plus
binary, source, and remote detail lines; cards use a loading indicator until data arrives.
