# Troubleshooting builds

## Quick diagnosis

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `Missing SYCL tools: …` at preflight | oneAPI or cmake not on PATH | [workstation-setup.md](workstation-setup.md) §2 |
| `Missing CUDA tools: …` | nvcc not installed | [workstation-setup.md](workstation-setup.md) §3 |
| `Failed to acquire build lock` | Another build running | Wait, or remove stale lock (below) |
| Configure fails: `icpx` not found | `setvars.sh` not sourced and not wrapped | Install oneAPI; confirm `/opt/intel/oneapi/setvars.sh` exists |
| Configure fails: exit 3, `setvars.sh` usage in stdout | Parent env has `SETVARS_COMPLETED=1`; setvars refuses re-source | Fixed in pipeline via `source setvars.sh --force`; or `unset SETVARS_COMPLETED` before building |
| Clone fails, then “sources already exist” | Offline continue with partial tree | Fix network or use valid existing clone |
| Build succeeds but launch can't find binary | Wrong `LLAMA_CPP_ROOT` vs actual build dir | [paths-and-artifacts.md](paths-and-artifacts.md) |
| Second build behaves oddly | Stale `CMakeCache.txt` with `--no-update-sources` | Remove `build/` or `build_cuda/` |

Always collect:

```bash
uv run llm-runner setup check
cat ~/.local/state/llm-runner/builds/sycl/build-artifact.json 2>/dev/null
ls -la ~/.local/state/llm-runner/reports/
```

---

## Build lock (`.build.lock`)

**Path:** `$XDG_CACHE_HOME/llm-runner/.build.lock` (default `~/.cache/llm-runner/.build.lock`)

**Format:** JSON with `pid`, `started_at`, `backend`.

A new build fails if:

- Lock file exists, and
- Holder PID is still running, and
- Lock age ≤ **3600 seconds**

### Stale lock recovery

If no build process is running but lock remains:

1. Confirm no `llm-runner build` or TUI build is active: `ps aux | grep -E 'llm-runner|cmake'`
2. If the PID in the lock file is dead, the next build should auto-replace a stale lock. If not, remove manually:

```bash
rm ~/.cache/llm-runner/.build.lock
```

Only remove the lock when you are certain no build is in progress.

---

## Preflight failures

Preflight does **not** check disk space or GPU driver versions — only toolchain binaries.

### SYCL

Required: `gcc`, `make`, `git`, `cmake`, and one of `icpx` / `icx` / `dpcpp`.

```bash
source /opt/intel/oneapi/setvars.sh
which icpx cmake gcc git make
```

### CUDA

Required for compile: `gcc`, `make`, `git`, `cmake`, `nvcc`.

`setup check` may also warn about missing `nvtop` for monitoring.

---

## Clone stage

### “Source directory exists but is not a git repository”

`source_dir` is non-empty and has no `.git`. Remove the directory or point `--source-dir` / `LLAMA_CPP_ROOT` at a valid clone.

### Clone timeout

Default clone subprocess timeout is **120 seconds**. Slow networks: clone manually then reuse:

```bash
git clone --branch master --depth 1 https://github.com/ggerganov/llama.cpp.git ~/.cache/llm-runner/llama.cpp
uv run llm-runner build sycl --no-update-sources
```

### Pin commit fails

Ensure `git fetch` ran (don't use `--no-update-sources` on first fetch of old shallow clone). Full clone: `--no-shallow-clone`.

---

## Configure stage

### Skipped configure unexpectedly

With `--no-update-sources` and existing `CMakeCache.txt`, configure is skipped. Delete cache to re-run:

```bash
rm -rf ~/.cache/llm-runner/llama.cpp/build/CMakeCache.txt
# or entire build tree:
rm -rf ~/.cache/llm-runner/llama.cpp/build
```

### Configure timeout

Default **3600s** per configure invocation. Huge projects on slow disks may need a code change to `build_timeout_seconds` in `BuildConfig` (library) — not exposed on CLI today.

### SYCL cmake errors without oneAPI

If `setvars.sh` is missing, llm-runner runs plain `cmake` and Intel compilers may be absent. Install oneAPI or export compiler paths before building.

---

## Build (compile) stage

Failures after configure write:

- `build-artifact.json` with non-zero `exit_code`
- `failure_report_path` directory under `reports/<timestamp>/`
- `build_log_path` with redacted stdout/stderr tails

Inspect the log tail in the CLI error message or:

```bash
ls -lt ~/.local/state/llm-runner/reports/*.log | head -3
less <path-from-artifact>
```

Common compile issues:

- Out of memory during link — reduce `-j`
- CUDA arch mismatch — set upstream `CMAKE_CUDA_ARCHITECTURES` manually in the build dir
- SYCL header not found — oneAPI environment not loaded

---

## Dry-run vs spec quickstart

`specs/002-build-setup/quickstart.md` sometimes describes dry-run as “preflight only.” **Actual behavior:** all stages run in simulation except finalize; preflight still executes real toolchain detection. See [cli-and-tui.md](cli-and-tui.md).

---

## Provenance / binary mismatch

`build-artifact.json` records `binary_path` under the build tree. Launch uses `Config.llama_server_bin_intel` / `llama_server_bin_nvidia` derived from `llama_cpp_root`.

If you build with `--source-dir` but launch without `LLAMA_CPP_ROOT`, launch may point at the XDG default while artifacts live elsewhere.

**Fix:** Export `LLAMA_CPP_ROOT` consistently or symlink binaries into the expected tree.

---

## `build both` partial success

CLI returns exit code **1** if either backend failed. Check each backend block in text output or JSON `errors` array. Re-run only the failed backend:

```bash
uv run llm-runner build cuda   # after fixing CUDA toolchain
```

Successful backend provenance is not rolled back.

---

## Getting help from logs

| Log | Location |
|-----|----------|
| llm-runner internal | `~/.local/share/llm-runner/llm-runner.log` (loguru) |
| Per-build command capture | `~/.local/state/llm-runner/reports/*.log` |
| Failure bundle | `~/.local/state/llm-runner/reports/<timestamp>/` |

Build output redacts API keys and credential-like URL segments before writing to disk.
